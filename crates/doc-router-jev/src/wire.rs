//! Everything that goes on the wire to `POST /v1/systemone`, and comes back.
//!
//! This module is deliberately free of HTTP and of [`JevJudge`](crate::JevJudge):
//! it turns `&[PageEvidence]` into request bodies and turns a response back into
//! a probability for a named page, and nothing else. That is what makes the
//! interesting parts — the state builder, the chunker, the answer matcher —
//! testable without a socket, which is most of this crate's test suite.
//!
//! # The protocol, in one paragraph
//!
//! A System One request carries a `state` (what the model should look at) and a
//! map of named `questions` about it. Each answer comes back under **the same
//! name**, which is the whole reason this crate can batch a document into one
//! call: `state` is the array of pages, `questions` has one entry per page keyed
//! [`question_key`]`(page)`, and the answers are matched back by that key. A
//! `noul` question answers with a single calibrated probability in 0..1 — the
//! model's belief that the statement in `instructions` is true — which is
//! exactly the shape [`PageVerdict::confidence`](doc_router::PageVerdict) wants
//! and exactly what [`HeuristicJudge`](doc_router::HeuristicJudge) has none of.
//!
//! # Why the answers are never read positionally
//!
//! `answers` is a JSON object. Nothing in the protocol promises the server emits
//! its keys in the order the request listed them, and this crate additionally
//! splits long documents across several requests, so "the third answer" is not a
//! page number under any reading. [`SystemOneResponse::noul_for`] takes a page
//! and looks its key up; a key that is absent, answered with the wrong question
//! type, or carrying a value that is not a number is a [`ProtocolError`], never
//! a quietly-defaulted `false`. A judge that invents `false` for a page the
//! vendor did not answer is a judge that silently routes scans to the local text
//! extractor, which is the exact failure this crate exists to catch.

use std::collections::BTreeMap;

use doc_router::PageEvidence;
use serde::Deserialize;

/// TypeSafe's hosted API. Override with `TYPESAFE_BASE_URL` for a proxy or a
/// mock server.
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";

/// The model asked for when nothing else is configured.
pub const DEFAULT_MODEL: &str = "jev-latest";

/// The one endpoint this crate posts to, appended to the base URL.
pub const ENDPOINT_PATH: &str = "/v1/systemone";

/// How many characters of one page's text are sent.
///
/// A page's text is evidence, not the document: the question is whether the text
/// layer means anything, and the first couple of thousand characters answer that
/// as well as the whole page would while keeping a 500-page document's request
/// bounded. Pages cut here say so in their state object (`text_truncated`), so
/// the model is never told a fragment is the whole page.
pub const MAX_TEXT_CHARS: usize = 2000;

/// How many pages ride in one request before the document is split across
/// several.
pub const MAX_PAGES_PER_REQUEST: usize = 50;

/// Soft ceiling on one request's serialised size, in bytes.
///
/// Soft because a single page is never split: a chunk always carries at least
/// one page. In practice [`MAX_TEXT_CHARS`] already bounds a page at a few
/// kilobytes, so this second bound only ever fires for multi-byte text — 50
/// pages of Latin-1 fit comfortably, 50 pages of CJK or of mojibake do not.
pub const MAX_REQUEST_BYTES: usize = 256 * 1024;

/// The prefix every question name carries, so a `page_<n>` key is recognisable
/// on the wire and in a captured request body.
pub const QUESTION_KEY_PREFIX: &str = "page_";

/// The question name for a page. `page` is 0-indexed, like everywhere else.
#[must_use]
pub fn question_key(page: u32) -> String {
    format!("{QUESTION_KEY_PREFIX}{page}")
}

/// One page as the model sees it.
///
/// Both halves of the evidence go in: the page's text, and what pdf-inspector
/// thought of it. Withholding the structural flags would throw away a signal the
/// model can use, and it is the one signal known to be right about the easy
/// cases (a page with no text layer at all); the point of asking is the cases it
/// gets wrong, not replacing it.
///
/// `text` is `null` when the text layer could not be extracted, which is itself
/// strong evidence and is distinct from an empty string (a page that extracted
/// cleanly to nothing).
#[must_use]
pub fn page_state(evidence: &PageEvidence) -> serde_json::Value {
    let (text, chars, truncated) = match &evidence.text {
        None => (serde_json::Value::Null, 0, false),
        Some(text) => {
            let chars = text.chars().count();
            let clipped = if chars > MAX_TEXT_CHARS {
                text.chars().take(MAX_TEXT_CHARS).collect()
            } else {
                text.clone()
            };
            (
                serde_json::Value::String(clipped),
                chars,
                chars > MAX_TEXT_CHARS,
            )
        }
    };
    serde_json::json!({
        "page": evidence.page,
        "text": text,
        "text_chars": chars,
        "text_truncated": truncated,
        "inspector_flagged": evidence.flagged_by_inspector,
        "inspector_reasons": evidence.reasons,
        "has_tables": evidence.has_tables,
        "has_columns": evidence.has_columns,
        "has_encoding_issues": evidence.has_encoding_issues,
    })
}

/// The `noul` question asked about one page.
///
/// The statement is phrased so that **true means the page needs OCR**, because
/// that is the direction [`PageVerdict::needs_ocr`](doc_router::PageVerdict)
/// reads; flipping it here would mean flipping it again in the judge, twice as
/// many places to get a `1.0 -` wrong. `criteria` is optional for `noul`, but it
/// is supplied anyway: it is what pins the model's probability to a definition
/// of "needs OCR" that matches the corpus's labels instead of to whatever the
/// phrase suggests on its own.
#[must_use]
pub fn page_question(page: u32) -> serde_json::Value {
    serde_json::json!({
        "type": "noul",
        "instructions": format!(
            "Look only at the entry in the state array whose `page` field is {page}. \
             That page needs to be re-read with OCR: the text it carries is missing, \
             or it is present but does not faithfully represent what is printed on the \
             page."
        ),
        "criteria": {
            "true": "The text is absent, is mojibake or nonsense from a broken font \
                     encoding, is a bad pre-existing OCR layer, is only a watermark, \
                     header or footer, or otherwise does not correspond to a page of \
                     readable content. Running OCR would recover content that is \
                     currently lost.",
            "false": "The text is coherent, readable prose, tabular or form content \
                      that plausibly reflects the whole printed page. Running OCR \
                      would cost money and return what is already in hand.",
        },
    })
}

/// The request body for one chunk of pages, exactly as it goes on the wire.
#[must_use]
pub fn request_body(model: &str, chunk: &[PageEvidence]) -> serde_json::Value {
    let state: Vec<serde_json::Value> = chunk.iter().map(page_state).collect();
    let questions: serde_json::Map<String, serde_json::Value> = chunk
        .iter()
        .map(|page| (question_key(page.page), page_question(page.page)))
        .collect();
    serde_json::json!({
        "model": model,
        "state": state,
        "questions": questions,
    })
}

/// Split a document into the requests it will be sent as, in page order.
///
/// Greedy, and bounded on both axes that actually break a request: the page
/// count ([`MAX_PAGES_PER_REQUEST`]) and the serialised size
/// ([`MAX_REQUEST_BYTES`]). Chunking is invisible to the rest of the crate
/// because the question keys are derived from [`PageEvidence::page`] and never
/// from a position within a chunk — page 73 is `page_73` whether it rides in the
/// first request or the fourth.
///
/// The size bound is measured by actually serialising each page's state and
/// question rather than estimating from the character cap, because
/// [`MAX_TEXT_CHARS`] is a *character* cap and a page of CJK or of mojibake
/// costs several bytes per character. Serialising twice is cheap next to the
/// round trip it is sizing.
#[must_use]
pub fn chunks(evidence: &[PageEvidence]) -> Vec<&[PageEvidence]> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut bytes = 0;
    for (index, page) in evidence.iter().enumerate() {
        let size = wire_size(page);
        let too_many = index - start >= MAX_PAGES_PER_REQUEST;
        // `index > start` keeps every chunk non-empty: one oversized page still
        // gets its own request rather than an infinite sequence of empty ones.
        let too_big = index > start && bytes + size > MAX_REQUEST_BYTES;
        if too_many || too_big {
            out.push(&evidence[start..index]);
            start = index;
            bytes = 0;
        }
        bytes += size;
    }
    if start < evidence.len() {
        out.push(&evidence[start..]);
    }
    out
}

/// Roughly what one page contributes to a request body, in bytes.
fn wire_size(evidence: &PageEvidence) -> usize {
    page_state(evidence).to_string().len() + page_question(evidence.page).to_string().len()
}

/// A response that arrived and parsed but does not answer the question asked.
///
/// Separate from a transport failure because the judge treats the two
/// differently in its reason strings: `jev_fallback_protocol` means the vendor
/// answered and the answer was unusable, which is worth telling apart from the
/// vendor being unreachable.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProtocolError {
    /// No answer came back under this page's question key.
    #[error("the response has no answer for {key}")]
    MissingAnswer {
        /// The question key that went unanswered.
        key: String,
    },
    /// The answer came back, but as some other question type.
    #[error("the answer for {key} has type {found:?}, not \"noul\"")]
    WrongType {
        /// The question key.
        key: String,
        /// The `type` the server reported.
        found: String,
    },
    /// A `noul` answer with no `noul` field in it.
    #[error("the answer for {key} carries no noul value")]
    MissingValue {
        /// The question key.
        key: String,
    },
    /// A `noul` value that is NaN or infinite.
    ///
    /// Clamping this the way an out-of-range number is clamped would turn "the
    /// vendor sent nonsense" into a confident verdict, so it is an error.
    #[error("the answer for {key} is {value}, which is not a probability")]
    NotFinite {
        /// The question key.
        key: String,
        /// The value as it arrived.
        value: f64,
    },
}

/// One answer in a System One response.
///
/// Only the fields a `noul` question produces are modelled. `choice` and `score`
/// answers deserialise into this shape too, with `noul` absent — which is
/// exactly the case [`SystemOneResponse::noul_for`] reports as
/// [`ProtocolError::WrongType`].
#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    /// The question type the server says it answered.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// The probability, for a `noul` answer.
    #[serde(default)]
    pub noul: Option<f64>,
}

/// Token usage, as reported. Recorded for cost reporting, never acted on.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct Usage {
    /// Input tokens billed.
    #[serde(default)]
    pub input_tokens: u64,
    /// Output tokens billed.
    #[serde(default)]
    pub output_tokens: u64,
}

/// `POST /v1/systemone`'s response, reduced to the fields this crate uses.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct SystemOneResponse {
    /// The model that actually answered, which may be a pinned version of the
    /// alias that was asked for.
    #[serde(default)]
    pub model: Option<String>,
    /// Answers by question name. A `BTreeMap` rather than a `Vec` because the
    /// name is the only thing that identifies which page an answer is about.
    #[serde(default)]
    pub answers: BTreeMap<String, Answer>,
    /// Token usage, when reported.
    #[serde(default)]
    pub usage: Option<Usage>,
}

impl SystemOneResponse {
    /// The probability this response reports for `page`, clamped into 0..=1.
    ///
    /// Clamping rather than erroring on an out-of-range number is the one piece
    /// of leniency here, and it is deliberate: a `1.0000000000000002` from a
    /// float round trip is not a protocol violation, it is a float. A value that
    /// is not a number at all is, and is refused.
    pub fn noul_for(&self, page: u32) -> Result<f32, ProtocolError> {
        let key = question_key(page);
        let answer = self
            .answers
            .get(&key)
            .ok_or_else(|| ProtocolError::MissingAnswer { key: key.clone() })?;
        if let Some(kind) = &answer.kind {
            if kind != "noul" {
                return Err(ProtocolError::WrongType {
                    key,
                    found: kind.clone(),
                });
            }
        }
        let value = answer
            .noul
            .ok_or_else(|| ProtocolError::MissingValue { key: key.clone() })?;
        if !value.is_finite() {
            return Err(ProtocolError::NotFinite { key, value });
        }
        Ok((value as f32).clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(page: u32, text: Option<&str>) -> PageEvidence {
        PageEvidence {
            page,
            text: text.map(str::to_string),
            reasons: vec!["garbled_text".to_string()],
            flagged_by_inspector: true,
            has_tables: false,
            has_columns: true,
            has_encoding_issues: false,
        }
    }

    fn parse(json: &str) -> SystemOneResponse {
        serde_json::from_str(json).expect("a well-formed response")
    }

    #[test]
    fn question_keys_are_the_zero_indexed_page_number() {
        assert_eq!(question_key(0), "page_0");
        assert_eq!(question_key(73), "page_73");
    }

    #[test]
    fn page_state_carries_both_the_text_and_the_inspector_flags() {
        let state = page_state(&evidence(4, Some("hello")));
        assert_eq!(state["page"], 4);
        assert_eq!(state["text"], "hello");
        assert_eq!(state["text_chars"], 5);
        assert_eq!(state["text_truncated"], false);
        assert_eq!(state["inspector_flagged"], true);
        assert_eq!(
            state["inspector_reasons"],
            serde_json::json!(["garbled_text"])
        );
        assert_eq!(state["has_columns"], true);
        assert_eq!(state["has_tables"], false);
        assert_eq!(state["has_encoding_issues"], false);
    }

    #[test]
    fn missing_text_is_null_and_distinct_from_empty_text() {
        assert_eq!(
            page_state(&evidence(0, None))["text"],
            serde_json::Value::Null
        );
        assert_eq!(page_state(&evidence(0, Some("")))["text"], "");
    }

    #[test]
    fn long_text_is_clipped_and_says_so() {
        let long = "x".repeat(MAX_TEXT_CHARS + 500);
        let state = page_state(&evidence(0, Some(&long)));
        assert_eq!(state["text"].as_str().expect("text").len(), MAX_TEXT_CHARS);
        assert_eq!(state["text_truncated"], true);
        // The *original* length is reported, so the model knows what it is missing.
        assert_eq!(state["text_chars"], MAX_TEXT_CHARS + 500);
    }

    #[test]
    fn text_is_clipped_on_a_character_boundary() {
        // Every character is 3 bytes, so a byte-wise cut would produce invalid UTF-8.
        let long = "あ".repeat(MAX_TEXT_CHARS + 10);
        let state = page_state(&evidence(0, Some(&long)));
        let clipped = state["text"].as_str().expect("text");
        assert_eq!(clipped.chars().count(), MAX_TEXT_CHARS);
        assert_eq!(clipped.len(), MAX_TEXT_CHARS * 3);
    }

    #[test]
    fn the_request_body_asks_one_noul_question_per_page() {
        let pages = vec![evidence(0, Some("a")), evidence(7, Some("b"))];
        let body = request_body("jev-latest", &pages);
        assert_eq!(body["model"], "jev-latest");
        assert_eq!(body["state"].as_array().expect("state").len(), 2);
        assert_eq!(body["state"][1]["page"], 7);
        let questions = body["questions"].as_object().expect("questions");
        assert_eq!(questions.len(), 2);
        assert_eq!(questions["page_0"]["type"], "noul");
        assert_eq!(questions["page_7"]["type"], "noul");
        // The question names the page it is about, since state carries several.
        assert!(questions["page_7"]["instructions"]
            .as_str()
            .expect("instructions")
            .contains("page` field is 7"));
    }

    #[test]
    fn a_short_document_is_one_chunk() {
        let pages: Vec<_> = (0..MAX_PAGES_PER_REQUEST as u32)
            .map(|p| evidence(p, Some("short")))
            .collect();
        let chunks = chunks(&pages);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), MAX_PAGES_PER_REQUEST);
    }

    #[test]
    fn a_long_document_splits_on_the_page_count_and_keeps_page_identity() {
        let pages: Vec<_> = (0..(MAX_PAGES_PER_REQUEST as u32 * 2 + 3))
            .map(|p| evidence(p, Some("short")))
            .collect();
        let chunks = chunks(&pages);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].len(), MAX_PAGES_PER_REQUEST);
        assert_eq!(chunks[1].len(), MAX_PAGES_PER_REQUEST);
        assert_eq!(chunks[2].len(), 3);
        // The keys a chunk produces are the original page numbers, not offsets.
        let body = request_body("jev-latest", chunks[1]);
        let questions = body["questions"].as_object().expect("questions");
        assert!(questions.contains_key(&question_key(MAX_PAGES_PER_REQUEST as u32)));
        assert!(!questions.contains_key("page_0"));
        // Every page appears exactly once, in order.
        let seen: Vec<u32> = chunks
            .iter()
            .flat_map(|chunk| chunk.iter().map(|p| p.page))
            .collect();
        assert_eq!(seen, (0..pages.len() as u32).collect::<Vec<_>>());
    }

    #[test]
    fn a_heavy_document_splits_on_size_before_the_page_count() {
        // Three bytes per character, so a full page costs three times what the
        // character cap suggests and the byte bound bites before the page bound.
        let fat = "\u{3042}".repeat(MAX_TEXT_CHARS);
        let pages: Vec<_> = (0..MAX_PAGES_PER_REQUEST as u32)
            .map(|p| evidence(p, Some(&fat)))
            .collect();
        let chunks = chunks(&pages);
        assert!(
            chunks.len() > 1,
            "a full page of CJK times {MAX_PAGES_PER_REQUEST}"
        );
        let mut seen = Vec::new();
        for chunk in &chunks {
            assert!(!chunk.is_empty());
            assert!(chunk.len() < MAX_PAGES_PER_REQUEST);
            // Within the bound, allowing the one page that crossed it.
            let body = request_body("jev-latest", chunk).to_string().len();
            assert!(body <= MAX_REQUEST_BYTES + wire_size(&pages[0]));
            seen.extend(chunk.iter().map(|p| p.page));
        }
        assert_eq!(seen, (0..pages.len() as u32).collect::<Vec<_>>());
    }

    #[test]
    fn no_pages_is_no_chunks() {
        assert!(chunks(&[]).is_empty());
    }

    #[test]
    fn answers_are_found_by_key_not_by_position() {
        // Deliberately out of order, and page 1 is absent from the middle.
        let response = parse(
            r#"{"answers":{"page_9":{"type":"noul","noul":0.9},
                           "page_0":{"type":"noul","noul":0.1}}}"#,
        );
        assert_eq!(response.noul_for(0).expect("page 0"), 0.1);
        assert_eq!(response.noul_for(9).expect("page 9"), 0.9);
    }

    #[test]
    fn a_missing_answer_is_an_error_not_a_false() {
        let response = parse(r#"{"answers":{"page_0":{"type":"noul","noul":0.1}}}"#);
        assert_eq!(
            response.noul_for(1),
            Err(ProtocolError::MissingAnswer {
                key: "page_1".to_string()
            })
        );
    }

    #[test]
    fn an_answer_of_the_wrong_question_type_is_an_error() {
        let response =
            parse(r#"{"answers":{"page_0":{"type":"choice","choice":"yes","confidence":0.4}}}"#);
        assert_eq!(
            response.noul_for(0),
            Err(ProtocolError::WrongType {
                key: "page_0".to_string(),
                found: "choice".to_string()
            })
        );
    }

    #[test]
    fn a_noul_answer_with_no_value_is_an_error() {
        let response = parse(r#"{"answers":{"page_0":{"type":"noul"}}}"#);
        assert_eq!(
            response.noul_for(0),
            Err(ProtocolError::MissingValue {
                key: "page_0".to_string()
            })
        );
    }

    #[test]
    fn an_untyped_answer_carrying_a_noul_is_accepted() {
        let response = parse(r#"{"answers":{"page_0":{"noul":0.25}}}"#);
        assert_eq!(response.noul_for(0).expect("page 0"), 0.25);
    }

    #[test]
    fn out_of_range_values_are_clamped_rather_than_refused() {
        let response = parse(
            r#"{"answers":{"page_0":{"type":"noul","noul":1.0000000000000002},
                           "page_1":{"type":"noul","noul":-0.0000001}}}"#,
        );
        assert_eq!(response.noul_for(0).expect("page 0"), 1.0);
        assert_eq!(response.noul_for(1).expect("page 1"), 0.0);
    }

    #[test]
    fn a_value_that_is_not_a_number_is_refused() {
        // serde_json will not produce NaN from text, so build the response directly.
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut response = SystemOneResponse::default();
            response.answers.insert(
                question_key(0),
                Answer {
                    kind: Some("noul".to_string()),
                    noul: Some(value),
                },
            );
            assert!(matches!(
                response.noul_for(0),
                Err(ProtocolError::NotFinite { .. })
            ));
        }
    }

    #[test]
    fn a_response_parses_without_the_optional_fields() {
        let response = parse(r#"{"answers":{"page_0":{"type":"noul","noul":0.5}}}"#);
        assert!(response.model.is_none());
        assert!(response.usage.is_none());

        let response = parse(
            r#"{"model":"jev-2026-01","answers":{"page_0":{"type":"noul","noul":0.5}},
                "usage":{"input_tokens":12,"output_tokens":3}}"#,
        );
        assert_eq!(response.model.as_deref(), Some("jev-2026-01"));
        assert_eq!(response.usage.expect("usage").input_tokens, 12);
    }
}
