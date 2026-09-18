//! [`LiteLlmHost`]: an [`OcrHost`] that talks to a LiteLLM proxy's `/v1/ocr` endpoint.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use doc_router::{remap_pages, split_pdf, Error, OcrHost, OcrResult, Page};
use serde::Deserialize;

/// Default per-call timeout. OCR is slow; two minutes is a realistic ceiling.
pub const DEFAULT_TIMEOUT_SECONDS: f64 = 120.0;

/// Environment variables consulted for the proxy key, in order.
pub const API_KEY_ENV_VARS: [&str; 2] = ["LITELLM_API_KEY", "LITELLM_PROXY_API_KEY"];

/// Largest response body we will read into memory (128 MiB).
const MAX_RESPONSE_BYTES: u64 = 128 * 1024 * 1024;

/// How many characters of a failing response body land in the error message.
const ERROR_BODY_CHARS: usize = 300;

/// One `/v1/ocr` call as it happened, for the CLI's per-leg summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallRecord {
    /// The model the call asked for.
    pub model: String,
    /// The 0-indexed page subset requested, or `None` for the whole document.
    pub pages: Option<Vec<u32>>,
    /// Wall-clock duration of the HTTP round trip, in milliseconds.
    pub elapsed_ms: u128,
    /// The HTTP status the proxy returned, when a response arrived at all.
    pub status: Option<u16>,
}

/// An [`OcrHost`] backed by a LiteLLM proxy (or anything else exposing `/v1/ocr`).
///
/// Requests are blocking on purpose: the core runs its legs on scoped threads, so a
/// blocking client is both simpler and correct here.
pub struct LiteLlmHost {
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
    split_subset: bool,
    calls: Mutex<Vec<CallRecord>>,
}

impl std::fmt::Debug for LiteLlmHost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LiteLlmHost")
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("split_subset", &self.split_subset)
            .finish()
    }
}

impl LiteLlmHost {
    /// A host pointed at `base_url` (e.g. `http://localhost:4000`), with the default timeout.
    pub fn new(base_url: impl Into<String>) -> Self {
        Self::with_timeout(base_url, DEFAULT_TIMEOUT_SECONDS)
    }

    /// A host with an explicit per-call timeout in seconds.
    ///
    /// A non-finite or non-positive timeout is treated as "no timeout".
    pub fn with_timeout(base_url: impl Into<String>, timeout_seconds: f64) -> Self {
        let timeout = (timeout_seconds.is_finite() && timeout_seconds > 0.0)
            .then(|| Duration::from_secs_f64(timeout_seconds));
        let config = ureq::Agent::config_builder()
            .timeout_global(timeout)
            // Keep 4xx/5xx as ordinary responses so the body can go into the error.
            .http_status_as_error(false)
            .build();
        LiteLlmHost {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key: None,
            agent: config.new_agent(),
            split_subset: false,
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Send `Authorization: Bearer <key>`. An empty or whitespace-only key is ignored.
    #[must_use]
    pub fn with_api_key(mut self, api_key: Option<impl Into<String>>) -> Self {
        self.api_key = api_key.map(Into::into).filter(|key| !key.trim().is_empty());
        self
    }

    /// For providers that ignore the `pages` field: send a split subset PDF instead.
    ///
    /// When enabled and a page subset is requested, the host uploads
    /// [`split_pdf`]`(document, pages)` with **no** `pages` field and then runs
    /// [`remap_pages`] over the response so page indices stay original.
    #[must_use]
    pub fn with_split_subset(mut self, split_subset: bool) -> Self {
        self.split_subset = split_subset;
        self
    }

    /// The proxy endpoint this host posts to.
    pub fn endpoint(&self) -> String {
        format!("{}/v1/ocr", self.base_url)
    }

    /// Every call this host has made, oldest first.
    pub fn calls(&self) -> Vec<CallRecord> {
        self.calls.lock().expect("call log mutex").clone()
    }

    /// The request body for `model` over `document`, exactly as it goes on the wire.
    fn request_body(model: &str, document: &[u8], pages: Option<&[u32]>) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": model,
            "document": {
                "type": "document_url",
                "document_url": format!("data:application/pdf;base64,{}", BASE64.encode(document)),
            },
        });
        if let Some(pages) = pages {
            body["pages"] = serde_json::json!(pages);
        }
        body
    }

    fn post(
        &self,
        model: &str,
        document: &[u8],
        pages: Option<&[u32]>,
    ) -> Result<OcrResult, Error> {
        let url = self.endpoint();
        let body = Self::request_body(model, document, pages);

        let mut request = self.agent.post(&url);
        if let Some(key) = &self.api_key {
            request = request.header("Authorization", format!("Bearer {key}"));
        }

        let started = Instant::now();
        let outcome = request.send_json(&body);
        let elapsed_ms = started.elapsed().as_millis();

        let record = |status: Option<u16>| CallRecord {
            model: model.to_string(),
            pages: pages.map(<[u32]>::to_vec),
            elapsed_ms,
            status,
        };

        let mut response = match outcome {
            Ok(response) => response,
            Err(e) => {
                self.calls
                    .lock()
                    .expect("call log mutex")
                    .push(record(None));
                return Err(Error::Host(format!("POST {url} failed: {e}")));
            }
        };
        let status = response.status().as_u16();
        self.calls
            .lock()
            .expect("call log mutex")
            .push(record(Some(status)));

        let text = response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_string()
            .map_err(|e| {
                Error::Host(format!(
                    "POST {url} returned HTTP {status}, unreadable body: {e}"
                ))
            })?;

        if !(200..300).contains(&status) {
            return Err(Error::Host(format!(
                "POST {url} returned HTTP {status}: {}",
                truncate(&text, ERROR_BODY_CHARS)
            )));
        }

        let parsed: OcrResponse = serde_json::from_str(&text).map_err(|e| {
            Error::Host(format!(
                "POST {url} returned HTTP {status} with unparseable JSON ({e}): {}",
                truncate(&text, ERROR_BODY_CHARS)
            ))
        })?;
        Ok(parsed.into_result(model))
    }
}

impl OcrHost for LiteLlmHost {
    fn ocr(&self, model: &str, document: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error> {
        match (self.split_subset, pages) {
            (true, Some(pages)) => {
                let subset = split_pdf(document, pages)?;
                let mut result = self.post(model, &subset, None)?;
                remap_pages(&mut result, pages);
                Ok(result)
            }
            _ => self.post(model, document, pages),
        }
    }
}

/// Clip `text` to `max` characters (not bytes), marking the cut.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}…")
}

/// LiteLLM's `OCRResponse`, reduced to the fields the router uses.
#[derive(Debug, Deserialize)]
struct OcrResponse {
    #[serde(default)]
    pages: Vec<OcrResponsePage>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    usage_info: Option<UsageInfo>,
}

#[derive(Debug, Deserialize)]
struct OcrResponsePage {
    index: u32,
    #[serde(default)]
    markdown: String,
}

#[derive(Debug, Deserialize)]
struct UsageInfo {
    #[serde(default)]
    pages_processed: Option<u32>,
    #[serde(default)]
    doc_size_bytes: Option<u64>,
}

impl OcrResponse {
    fn into_result(self, requested_model: &str) -> OcrResult {
        let model = self
            .model
            .filter(|m| !m.trim().is_empty())
            .unwrap_or_else(|| requested_model.to_string());
        let pages: Vec<Page> = self
            .pages
            .into_iter()
            .map(|page| Page::new(page.index, page.markdown, &model))
            .collect();
        let usage = self.usage_info;
        OcrResult {
            pages_processed: usage
                .as_ref()
                .and_then(|u| u.pages_processed)
                .unwrap_or(pages.len() as u32),
            doc_size_bytes: usage.and_then(|u| u.doc_size_bytes),
            pages,
            model,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_body_omits_pages_when_none() {
        let body = LiteLlmHost::request_body("mistral-ocr", b"%PDF-1.4", None);
        assert_eq!(body["model"], "mistral-ocr");
        assert_eq!(body["document"]["type"], "document_url");
        assert_eq!(
            body["document"]["document_url"],
            format!("data:application/pdf;base64,{}", BASE64.encode(b"%PDF-1.4"))
        );
        assert!(body.get("pages").is_none());
    }

    #[test]
    fn request_body_includes_a_page_subset() {
        let body = LiteLlmHost::request_body("mistral-ocr", b"%PDF-1.4", Some(&[1, 3]));
        assert_eq!(body["pages"], serde_json::json!([1, 3]));
    }

    #[test]
    fn response_falls_back_to_the_requested_model_and_page_count() {
        let parsed: OcrResponse =
            serde_json::from_str(r#"{"pages":[{"index":0,"markdown":"hi"}]}"#).unwrap();
        let result = parsed.into_result("mistral-ocr");
        assert_eq!(result.model, "mistral-ocr");
        assert_eq!(result.pages_processed, 1);
        assert_eq!(result.pages[0].model, "mistral-ocr");
        assert_eq!(result.doc_size_bytes, None);
    }

    #[test]
    fn response_prefers_its_own_model_and_usage_info() {
        let parsed: OcrResponse = serde_json::from_str(
            r#"{"pages":[{"index":2,"markdown":"hi","images":[]}],
                "model":"mistral-ocr-2505",
                "usage_info":{"pages_processed":7,"doc_size_bytes":42}}"#,
        )
        .unwrap();
        let result = parsed.into_result("mistral-ocr");
        assert_eq!(result.model, "mistral-ocr-2505");
        assert_eq!(result.pages[0].model, "mistral-ocr-2505");
        assert_eq!(result.pages_processed, 7);
        assert_eq!(result.doc_size_bytes, Some(42));
    }

    #[test]
    fn api_keys_that_are_blank_are_dropped() {
        let host = LiteLlmHost::new("http://localhost:4000/").with_api_key(Some("  "));
        assert!(host.api_key.is_none());
        assert_eq!(host.endpoint(), "http://localhost:4000/v1/ocr");
    }

    #[test]
    fn truncate_marks_the_cut() {
        assert_eq!(truncate("abc", 10), "abc");
        assert_eq!(truncate("abcdef", 3), "abc…");
    }
}
