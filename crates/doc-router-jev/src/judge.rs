//! [`JevJudge`]: the [`PageJudge`] itself, and the policy around it.
//!
//! [`wire`](crate::wire) knows the protocol; this module knows what to do about
//! it. Three decisions live here and nowhere else:
//!
//! * **What `confidence` means** — see [`JevJudge::verdict`], which is the one
//!   place a probability becomes a [`PageVerdict`].
//! * **When to call at all** — [`JevMode::Gated`] and [`is_ambiguous`].
//! * **What to do when the call fails** — the strict flag and the circuit
//!   breaker, below.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use doc_router::{Error, HeuristicJudge, PageEvidence, PageJudge, PageVerdict};

use crate::wire::{self, ProtocolError, SystemOneResponse};

/// The registry name, and [`PageJudge::name`], for [`JevMode::Always`].
pub const JUDGE_NAME_ALWAYS: &str = "jev";

/// The registry name, and [`PageJudge::name`], for [`JevMode::Gated`].
pub const JUDGE_NAME_GATED: &str = "jev_gated";

/// Default per-request timeout. A judgement is a fast call by OCR standards;
/// waiting two minutes for one would cost more than the OCR it saves.
pub const DEFAULT_TIMEOUT_SECONDS: f64 = 30.0;

/// Default probability at or above which a page is routed to OCR.
///
/// 0.5 because the answer is a calibrated probability and nothing about this
/// problem is asymmetric enough to justify a thumb on the scale by default.
/// Callers who would rather over- than under-OCR can move it with
/// [`JevJudge::with_threshold`].
pub const DEFAULT_THRESHOLD: f32 = 0.5;

/// Consecutive failures that open the circuit breaker.
pub const DEFAULT_FAILURE_THRESHOLD: u32 = 3;

/// How long the breaker stays open before the next call is allowed through.
pub const DEFAULT_COOL_OFF_SECONDS: f64 = 30.0;

/// Environment variables consulted for the API key, in order.
pub const API_KEY_ENV_VARS: [&str; 2] = ["TYPESAFE_API_KEY", "JEV_API_KEY"];

/// Environment variable that overrides [`DEFAULT_BASE_URL`](crate::DEFAULT_BASE_URL).
pub const BASE_URL_ENV_VAR: &str = "TYPESAFE_BASE_URL";

/// Environment variable that overrides [`DEFAULT_MODEL`](crate::DEFAULT_MODEL).
pub const MODEL_ENV_VAR: &str = "TYPESAFE_MODEL";

/// Recorded on a page Jev says needs OCR.
pub const REASON_NEEDS_OCR: &str = "jev_needs_ocr";

/// Recorded on a page Jev says is fine as extracted.
pub const REASON_CLEAR: &str = "jev_clear";

/// Recorded on every page of a [`JevMode::Gated`] document the gate did not fire
/// for. The verdicts are the heuristic's; the reason says so, so a report never
/// shows a page as judged by Jev when Jev was never asked.
pub const REASON_NOT_ESCALATED: &str = "jev_not_escalated";

/// Recorded when a non-strict judge fell back because the request timed out.
pub const REASON_FALLBACK_TIMEOUT: &str = "jev_fallback_timeout";

/// Recorded when a non-strict judge fell back because of a transport failure or
/// a non-2xx status.
pub const REASON_FALLBACK_HTTP: &str = "jev_fallback_http";

/// Recorded when a non-strict judge fell back because the response arrived and
/// did not answer the question asked.
pub const REASON_FALLBACK_PROTOCOL: &str = "jev_fallback_protocol";

/// Recorded when a non-strict judge fell back without calling at all, because
/// the circuit breaker is open.
pub const REASON_FALLBACK_BREAKER_OPEN: &str = "jev_fallback_breaker_open";

/// Largest response body read into memory. Answers are small; a body this size
/// is a misconfiguration, not a document.
const MAX_RESPONSE_BYTES: u64 = 4 * 1024 * 1024;

/// How many characters of a failing response body land in the error message.
const ERROR_BODY_CHARS: usize = 300;

/// When [`JevJudge`] calls out to the hosted model.
///
/// The two variants are the same judge with a different admission policy, not
/// two judges, which is why they share every other setting and differ only in
/// [`PageJudge::name`]. The benchmark registers both so the cost of gating shows
/// up as a row next to the cost of not gating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum JevMode {
    /// Every document goes to Jev, every page of it. The most accurate mode and
    /// the most expensive one; this is the mode the benchmark scores when it
    /// wants to know what the model can do.
    #[default]
    Always,
    /// Only documents [`is_ambiguous`] calls ambiguous go to Jev. Documents the
    /// inspector is internally consistent about keep the heuristic's verdicts
    /// and cost nothing.
    Gated,
}

impl JevMode {
    /// The stable name this mode is registered and reported under.
    #[must_use]
    pub fn judge_name(self) -> &'static str {
        match self {
            JevMode::Always => JUDGE_NAME_ALWAYS,
            JevMode::Gated => JUDGE_NAME_GATED,
        }
    }
}

/// Is this document worth paying a hosted model to look at?
///
/// The gate has to be computable from `&[PageEvidence]` alone, because it runs
/// before any call is made and there is nothing else to run it on. Each clause
/// is a different way for pdf-inspector's own answer to be untrustworthy:
///
/// * **Any page has `has_encoding_issues`.** It is a document-level flag, so
///   this fires for the whole document as soon as one page carries it. Broken
///   font encodings are the canonical case of a text layer that exists and is
///   worthless — the extractor will happily return mojibake and the structural
///   check will happily call it text.
/// * **The inspector flagged some pages but not all of them.** This is the
///   evidence-level shape of a `Mixed` document: a file that is part born-digital
///   and part scanned. The page-by-page boundary is exactly where the structural
///   heuristic is least reliable, and it is also where a wrong answer is most
///   expensive, because the alternative to OCR-ing the scan is silently dropping
///   its content. An all-flagged or all-clear document is one the inspector was
///   internally consistent about; those are usually right and always cheap to be
///   wrong about in only one direction.
/// * **Some flagged page carries no reason.** [`PageEvidence::reasons`] is empty
///   for a flagged page when the analysis pass was skipped or failed, so the
///   flag is there without anything behind it. That is a page the inspector
///   flagged and cannot say why, which is the definition of a decision worth a
///   second opinion.
///
/// Deliberately *not* in the gate: `has_tables` and `has_columns`. Both describe
/// layout complexity, not text-layer quality, and a complex layout that extracts
/// correctly needs no OCR. Gating on them would send well-behaved reports and
/// financial statements to a paid model for no reason.
#[must_use]
pub fn is_ambiguous(evidence: &[PageEvidence]) -> bool {
    let flagged = evidence
        .iter()
        .filter(|page| page.flagged_by_inspector)
        .count();
    evidence.iter().any(|page| page.has_encoding_issues)
        || (flagged > 0 && flagged < evidence.len())
        || evidence
            .iter()
            .any(|page| page.flagged_by_inspector && page.reasons.is_empty())
}

/// One `/v1/systemone` call as it happened, for cost and latency reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CallRecord {
    /// The 0-indexed pages this call asked about.
    pub pages: Vec<u32>,
    /// Wall-clock duration of the HTTP round trip, in milliseconds.
    pub elapsed_ms: u128,
    /// The HTTP status returned, when a response arrived at all.
    pub status: Option<u16>,
    /// Input tokens the response reported, when it reported any.
    pub input_tokens: Option<u64>,
    /// Output tokens the response reported, when it reported any.
    pub output_tokens: Option<u64>,
}

/// A [`JevJudge`]'s call log, shared with whoever is counting.
///
/// The judge writes to it as calls happen; a holder of a second handle reads it
/// afterwards. It is the judge's own log, not a copy: there is nothing to keep
/// in step.
pub type CallLog = Arc<Mutex<Vec<CallRecord>>>;

thread_local! {
    /// The call logs of judges built inside the innermost [`capturing`] call, or
    /// `None` when nobody is capturing -- which is every caller but the
    /// benchmark harness.
    static CAPTURE: RefCell<Option<Vec<CallLog>>> = const { RefCell::new(None) };
}

/// Restores the enclosing capture even if `build` panics, so a panicking caller
/// cannot leave this thread capturing into a `Vec` nobody will read.
struct CaptureGuard(Option<Vec<CallLog>>);

impl Drop for CaptureGuard {
    fn drop(&mut self) {
        CAPTURE.with(|slot| slot.replace(self.0.take()));
    }
}

/// Run `build`, and hand back the call log of every [`JevJudge`] constructed
/// while it ran.
///
/// # Why construction is the seam
///
/// A caller that builds its judges itself can simply keep them and call
/// [`JevJudge::calls`]. This exists for the one that cannot: the benchmark
/// harness resolves `--judge` through a shared registry that hands back
/// `Box<dyn PageJudge>`, and a trait object cannot be asked what it spent --
/// [`PageJudge`] is a routing decision, and the core crate that defines it makes
/// no calls and has no tokens to report. Widening that trait, or teaching the
/// registry about billing, would push a vendor's concern into two places that
/// are deliberately free of it. Capturing at construction keeps it in the crate
/// that actually makes the calls.
///
/// The capture is scoped and per-thread, not a process-wide meter: each
/// `capturing` call sees only the judges built inside it, so two judges built
/// separately are counted separately, and a caller who never asks is unaffected.
/// Judges built on another thread inside `build` are not captured; nothing in
/// this workspace builds one that way.
///
/// ```
/// use doc_router_jev::{capturing, JevJudge};
///
/// let (judge, logs) = capturing(|| JevJudge::new("key"));
/// assert_eq!(logs.len(), 1);
/// // Nothing has been called yet, and the log says so rather than guessing.
/// assert!(logs[0].lock().expect("call log mutex").is_empty());
/// let _ = judge;
/// ```
///
/// [`PageJudge`]: doc_router::PageJudge
pub fn capturing<T>(build: impl FnOnce() -> T) -> (T, Vec<CallLog>) {
    let guard = CaptureGuard(CAPTURE.with(|slot| slot.replace(Some(Vec::new()))));
    let value = build();
    let logs = CAPTURE
        .with(|slot| slot.borrow_mut().take())
        .unwrap_or_default();
    drop(guard);
    (value, logs)
}

/// How a call failed, in the granularity the reason strings distinguish.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FailureClass {
    Timeout,
    Http,
    Protocol,
    BreakerOpen,
}

impl FailureClass {
    fn reason(self) -> &'static str {
        match self {
            FailureClass::Timeout => REASON_FALLBACK_TIMEOUT,
            FailureClass::Http => REASON_FALLBACK_HTTP,
            FailureClass::Protocol => REASON_FALLBACK_PROTOCOL,
            FailureClass::BreakerOpen => REASON_FALLBACK_BREAKER_OPEN,
        }
    }
}

/// A failed call: what class it was, and a message that names no credentials.
#[derive(Debug, Clone)]
struct Failure {
    class: FailureClass,
    detail: String,
}

/// Consecutive-failure state, shared across threads.
#[derive(Debug, Default)]
struct Breaker {
    consecutive_failures: u32,
    opened_at: Option<Instant>,
}

/// A [`PageJudge`] that asks TypeSafe's hosted System One model about each page.
///
/// One HTTP call per document (per [`wire::chunks`] chunk of it, for long ones),
/// not one per page: a page's verdict is not independent of its neighbours'
/// — "every other page in this file is a clean scan" is evidence about this page
/// — and 99 round trips for a 99-page document would cost more in latency than
/// the OCR it is trying to avoid.
///
/// Requests are blocking on purpose, for the same reason
/// [`OcrHost`](doc_router::OcrHost)'s are: [`PageJudge::judge`] is a synchronous
/// call and wrapping an async client in a runtime here would buy nothing.
pub struct JevJudge {
    base_url: String,
    api_key: String,
    model: String,
    agent: ureq::Agent,
    mode: JevMode,
    threshold: f32,
    strict: bool,
    failure_threshold: u32,
    cool_off: Duration,
    breaker: Mutex<Breaker>,
    calls: CallLog,
}

/// Redacting by hand rather than deriving, because the whole struct is one
/// `dbg!` away from a key in a log file. Same shape as the CLI's `LiteLlmHost`.
impl std::fmt::Debug for JevJudge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JevJudge")
            .field("base_url", &self.base_url)
            .field("api_key", &"<redacted>")
            .field("model", &self.model)
            .field("mode", &self.mode)
            .field("threshold", &self.threshold)
            .field("strict", &self.strict)
            .field("failure_threshold", &self.failure_threshold)
            .field("cool_off", &self.cool_off)
            .finish_non_exhaustive()
    }
}

impl JevJudge {
    /// A judge authenticating with `api_key`, against the hosted API, in
    /// [`JevMode::Always`] and non-strict.
    ///
    /// Non-strict is the right default for everything except the benchmark: see
    /// [`JevJudge::with_strict`].
    #[must_use]
    pub fn new(api_key: impl Into<String>) -> Self {
        let judge = JevJudge {
            base_url: crate::DEFAULT_BASE_URL.to_string(),
            api_key: api_key.into(),
            model: crate::DEFAULT_MODEL.to_string(),
            agent: build_agent(DEFAULT_TIMEOUT_SECONDS),
            mode: JevMode::Always,
            threshold: DEFAULT_THRESHOLD,
            strict: false,
            failure_threshold: DEFAULT_FAILURE_THRESHOLD,
            cool_off: Duration::from_secs_f64(DEFAULT_COOL_OFF_SECONDS),
            breaker: Mutex::new(Breaker::default()),
            calls: CallLog::default(),
        };
        CAPTURE.with(|slot| {
            if let Some(logs) = slot.borrow_mut().as_mut() {
                logs.push(Arc::clone(&judge.calls));
            }
        });
        judge
    }

    /// A judge configured from the environment, or `None` when no key is set.
    ///
    /// `None` means exactly one thing — neither [`API_KEY_ENV_VARS`] entry holds
    /// a non-blank value — and callers are expected to say so in those words.
    /// "No credentials" and "no such judge" are different sentences to a user
    /// staring at a CLI, and conflating them turns a one-line fix into a bug
    /// report.
    ///
    /// [`BASE_URL_ENV_VAR`] and [`MODEL_ENV_VAR`] are honoured when set and
    /// non-blank, which is how the integration tests point a real judge at a
    /// mock server.
    #[must_use]
    pub fn from_env() -> Option<Self> {
        let api_key = API_KEY_ENV_VARS
            .iter()
            .filter_map(|name| std::env::var(name).ok())
            .find(|key| !key.trim().is_empty())?;
        let mut judge = JevJudge::new(api_key);
        if let Some(base_url) = env_non_blank(BASE_URL_ENV_VAR) {
            judge = judge.with_base_url(base_url);
        }
        if let Some(model) = env_non_blank(MODEL_ENV_VAR) {
            judge = judge.with_model(model);
        }
        Some(judge)
    }

    /// Point the judge at another origin — a proxy, a staging endpoint, or a
    /// test server. A trailing slash is trimmed so [`JevJudge::endpoint`] never
    /// produces a double slash.
    #[must_use]
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into().trim_end_matches('/').to_string();
        self
    }

    /// Ask for a specific model rather than the `jev-latest` alias. Pinning is
    /// what makes two benchmark runs comparable.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// An explicit per-request timeout in seconds. A non-finite or non-positive
    /// value is treated as "no timeout", as in the CLI's host.
    #[must_use]
    pub fn with_timeout(mut self, timeout_seconds: f64) -> Self {
        self.agent = build_agent(timeout_seconds);
        self
    }

    /// Choose [`JevMode::Always`] or [`JevMode::Gated`]. This also changes
    /// [`PageJudge::name`].
    #[must_use]
    pub fn with_mode(mut self, mode: JevMode) -> Self {
        self.mode = mode;
        self
    }

    /// The probability at or above which a page is routed to OCR. Clamped into
    /// 0..=1; a non-finite value leaves the threshold unchanged.
    #[must_use]
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        if threshold.is_finite() {
            self.threshold = threshold.clamp(0.0, 1.0);
        }
        self
    }

    /// Turn vendor failures into [`Error::Judge`] instead of heuristic verdicts.
    ///
    /// The default is `false`, because routing a document must not stop because
    /// a vendor is having an afternoon: a fallback to the heuristic is the same
    /// answer the router would have given before this crate existed, which is a
    /// worse answer but never a broken one. Each fallback records a reason
    /// naming the failure class, so a fallback is visible in the output rather
    /// than silent.
    ///
    /// **The benchmark must construct this strict.** A silent fallback in a
    /// benchmark means the row labelled `jev` is reporting the heuristic's
    /// score under Jev's name — a fabricated comparison, and one that would
    /// look *better* the more often the vendor was down, since the heuristic is
    /// the thing the corpus was built to be easy for. A benchmark that cannot
    /// reach the model must fail loudly and be re-run, not quietly measure the
    /// baseline twice.
    #[must_use]
    pub fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    /// How many consecutive failures open the breaker. Zero disables it.
    #[must_use]
    pub fn with_failure_threshold(mut self, failures: u32) -> Self {
        self.failure_threshold = failures;
        self
    }

    /// How long an open breaker stays open. A non-finite or negative value is
    /// treated as zero.
    #[must_use]
    pub fn with_cool_off_seconds(mut self, seconds: f64) -> Self {
        self.cool_off = if seconds.is_finite() && seconds > 0.0 {
            Duration::from_secs_f64(seconds)
        } else {
            Duration::ZERO
        };
        self
    }

    /// The endpoint this judge posts to.
    #[must_use]
    pub fn endpoint(&self) -> String {
        format!("{}{}", self.base_url, wire::ENDPOINT_PATH)
    }

    /// The mode this judge runs in.
    #[must_use]
    pub fn mode(&self) -> JevMode {
        self.mode
    }

    /// The probability at or above which this judge routes a page to OCR.
    #[must_use]
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Whether this judge propagates failures rather than falling back.
    #[must_use]
    pub fn is_strict(&self) -> bool {
        self.strict
    }

    /// Every call this judge has made, oldest first.
    #[must_use]
    pub fn calls(&self) -> Vec<CallRecord> {
        self.calls.lock().expect("call log mutex").clone()
    }

    /// A second handle on the same call log, for a caller that will not be
    /// holding the judge when it wants to read it. See [`capturing`].
    #[must_use]
    pub fn call_log(&self) -> CallLog {
        Arc::clone(&self.calls)
    }

    /// Turn one page's probability into a verdict.
    ///
    /// # What `confidence` means here
    ///
    /// `p` is the model's probability that the page **needs OCR**.
    /// [`PageVerdict::confidence`] is not that number. The benchmark's
    /// `confidence_report` reads `confidence` as the judge's stated probability
    /// that **its own verdict** is right, which is the only reading the trait
    /// supports — the field sits on a struct that has already decided
    /// `needs_ocr`. So:
    ///
    /// ```text
    /// needs_ocr  = p >= threshold
    /// confidence = if needs_ocr { p } else { 1.0 - p }
    /// ```
    ///
    /// A `p` of 0.02 is a *confident* verdict of "no OCR" (confidence 0.98), not
    /// an unconfident one. Forwarding `p` raw would report that page as 2%
    /// confident while being right about it, which the Brier score would punish
    /// exactly as hard as being wrong — and the reliability table would fill its
    /// bottom bucket with correct answers.
    ///
    /// Unlike [`HeuristicJudge`]'s constant 1.0, these are real calibrated
    /// numbers: this is the first judge whose confidence column the benchmark's
    /// Brier score and reliability buckets can say anything about at all.
    fn verdict(&self, page: u32, p: f32) -> PageVerdict {
        let p = p.clamp(0.0, 1.0);
        let needs_ocr = p >= self.threshold;
        PageVerdict {
            page,
            needs_ocr,
            confidence: if needs_ocr { p } else { 1.0 - p },
            reason: if needs_ocr {
                REASON_NEEDS_OCR.to_string()
            } else {
                REASON_CLEAR.to_string()
            },
        }
    }

    /// Replace every verdict's reason, keeping `needs_ocr` and `confidence`.
    ///
    /// Used for both the gated no-fire case and every fallback, so a verdict
    /// this judge did not actually form is always labelled as such.
    fn relabel(verdicts: Vec<PageVerdict>, reason: &str) -> Vec<PageVerdict> {
        verdicts
            .into_iter()
            .map(|verdict| PageVerdict {
                reason: reason.to_string(),
                ..verdict
            })
            .collect()
    }

    /// Apply the failure policy: propagate when strict, relabel the heuristic's
    /// verdicts when not.
    fn fall_back(
        &self,
        failure: &Failure,
        heuristic: Vec<PageVerdict>,
    ) -> Result<Vec<PageVerdict>, Error> {
        if self.strict {
            return Err(Error::Judge(failure.detail.clone()));
        }
        Ok(Self::relabel(heuristic, failure.class.reason()))
    }

    /// True when the breaker is open and the cool-off has not yet elapsed.
    ///
    /// Reading it also closes an expired breaker, so the next call goes through
    /// and either succeeds (resetting the count) or fails (re-opening it). That
    /// is the whole half-open dance: one probe per cool-off period.
    fn breaker_is_open(&self) -> bool {
        let mut breaker = self.breaker.lock().expect("breaker mutex");
        match breaker.opened_at {
            None => false,
            Some(opened_at) if opened_at.elapsed() >= self.cool_off => {
                breaker.opened_at = None;
                breaker.consecutive_failures = 0;
                false
            }
            Some(_) => true,
        }
    }

    fn record_success(&self) {
        let mut breaker = self.breaker.lock().expect("breaker mutex");
        breaker.consecutive_failures = 0;
        breaker.opened_at = None;
    }

    fn record_failure(&self) {
        let mut breaker = self.breaker.lock().expect("breaker mutex");
        breaker.consecutive_failures = breaker.consecutive_failures.saturating_add(1);
        if self.failure_threshold > 0 && breaker.consecutive_failures >= self.failure_threshold {
            breaker.opened_at = Some(Instant::now());
        }
    }

    /// One `POST /v1/systemone`, with no retry and no interpretation.
    fn post(&self, chunk: &[PageEvidence]) -> Result<SystemOneResponse, Failure> {
        let url = self.endpoint();
        let body = wire::request_body(&self.model, chunk);
        let pages: Vec<u32> = chunk.iter().map(|page| page.page).collect();

        let started = Instant::now();
        let outcome = self
            .agent
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .send_json(&body);
        let elapsed_ms = started.elapsed().as_millis();

        let mut response = match outcome {
            Ok(response) => response,
            Err(e) => {
                self.log(CallRecord {
                    pages,
                    elapsed_ms,
                    status: None,
                    input_tokens: None,
                    output_tokens: None,
                });
                // `ureq` reports a timeout as its own variant, which is worth
                // keeping distinct: "the vendor is slow" and "the vendor is
                // unreachable" want different operational responses.
                let class = if matches!(e, ureq::Error::Timeout(_)) {
                    FailureClass::Timeout
                } else {
                    FailureClass::Http
                };
                return Err(Failure {
                    class,
                    // `e` carries the URL at most; the key is a header and never
                    // appears in it.
                    detail: format!("POST {url} failed: {e}"),
                });
            }
        };
        let status = response.status().as_u16();

        let text = match response
            .body_mut()
            .with_config()
            .limit(MAX_RESPONSE_BYTES)
            .read_to_string()
        {
            Ok(text) => text,
            Err(e) => {
                self.log(CallRecord {
                    pages,
                    elapsed_ms,
                    status: Some(status),
                    input_tokens: None,
                    output_tokens: None,
                });
                return Err(Failure {
                    class: FailureClass::Http,
                    detail: format!("POST {url} returned HTTP {status}, unreadable body: {e}"),
                });
            }
        };

        if !(200..300).contains(&status) {
            self.log(CallRecord {
                pages,
                elapsed_ms,
                status: Some(status),
                input_tokens: None,
                output_tokens: None,
            });
            return Err(Failure {
                class: FailureClass::Http,
                detail: format!(
                    "POST {url} returned HTTP {status}: {}",
                    truncate(&text, ERROR_BODY_CHARS)
                ),
            });
        }

        let parsed: SystemOneResponse = match serde_json::from_str(&text) {
            Ok(parsed) => parsed,
            Err(e) => {
                self.log(CallRecord {
                    pages,
                    elapsed_ms,
                    status: Some(status),
                    input_tokens: None,
                    output_tokens: None,
                });
                return Err(Failure {
                    class: FailureClass::Protocol,
                    detail: format!(
                        "POST {url} returned HTTP {status} with unparseable JSON ({e}): {}",
                        truncate(&text, ERROR_BODY_CHARS)
                    ),
                });
            }
        };
        self.log(CallRecord {
            pages,
            elapsed_ms,
            status: Some(status),
            input_tokens: parsed.usage.map(|usage| usage.input_tokens),
            output_tokens: parsed.usage.map(|usage| usage.output_tokens),
        });
        Ok(parsed)
    }

    fn log(&self, record: CallRecord) {
        self.calls.lock().expect("call log mutex").push(record);
    }

    /// Ask about every chunk and collect the verdicts, stopping at the first
    /// failure.
    ///
    /// Stopping is deliberate. A half-answered document cannot be completed from
    /// the remaining chunks, and mixing Jev's verdicts for pages 0..49 with the
    /// heuristic's for 50..99 would produce a document whose confidence column
    /// means two different things in two halves — unscoreable, and worse than
    /// either answer alone.
    fn ask(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Failure> {
        let mut verdicts = Vec::with_capacity(evidence.len());
        for chunk in wire::chunks(evidence) {
            let response = match self.post(chunk) {
                Ok(response) => response,
                Err(failure) => {
                    self.record_failure();
                    return Err(failure);
                }
            };
            self.record_success();
            for page in chunk {
                match response.noul_for(page.page) {
                    Ok(p) => verdicts.push(self.verdict(page.page, p)),
                    Err(e) => {
                        // A response that parsed but does not answer is still a
                        // failure of the vendor's, so it counts against the
                        // breaker even though the call itself "succeeded".
                        self.record_failure();
                        return Err(protocol_failure(&e));
                    }
                }
            }
        }
        Ok(verdicts)
    }
}

impl PageJudge for JevJudge {
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        if evidence.is_empty() {
            return Ok(Vec::new());
        }

        // Computed up front rather than in each branch: it is the fallback for
        // every failure path and the answer itself for a gated no-fire, it
        // cannot fail, and it costs a `Vec` allocation.
        let heuristic = HeuristicJudge.judge(evidence)?;

        if self.mode == JevMode::Gated && !is_ambiguous(evidence) {
            return Ok(Self::relabel(heuristic, REASON_NOT_ESCALATED));
        }

        if self.breaker_is_open() {
            let failure = Failure {
                class: FailureClass::BreakerOpen,
                detail: format!(
                    "{} consecutive failures; not calling {} until the cool-off elapses",
                    self.failure_threshold,
                    self.endpoint()
                ),
            };
            // Note this failure is *not* recorded against the breaker: a
            // refusal the breaker itself produced would otherwise keep it open
            // forever, and nothing was learned about the vendor by not calling.
            return self.fall_back(&failure, heuristic);
        }

        match self.ask(evidence) {
            Ok(verdicts) => Ok(verdicts),
            Err(failure) => self.fall_back(&failure, heuristic),
        }
    }

    fn name(&self) -> &'static str {
        self.mode.judge_name()
    }

    /// Always true, and this is the judge's main running cost.
    ///
    /// It forces the extra pdf-inspector extraction pass
    /// [`PageJudge::needs_text`] exists to let a judge avoid, on every document,
    /// before any network call is made — including, in [`JevMode::Gated`], the
    /// documents the gate then declines to escalate.
    ///
    /// It is not optional. The pages this judge exists to catch are the ones
    /// whose text layer is present but lying, and there is no way to notice that
    /// without reading the text. A judge that asked a model about
    /// `flagged_by_inspector` and nothing else would be an expensive way to
    /// reproduce [`HeuristicJudge`].
    fn needs_text(&self) -> bool {
        true
    }
}

/// The agent every call goes through: one global timeout, and 4xx/5xx delivered
/// as ordinary responses so the body can go into the error message.
fn build_agent(timeout_seconds: f64) -> ureq::Agent {
    let timeout = (timeout_seconds.is_finite() && timeout_seconds > 0.0)
        .then(|| Duration::from_secs_f64(timeout_seconds));
    ureq::Agent::config_builder()
        .timeout_global(timeout)
        .http_status_as_error(false)
        .build()
        .new_agent()
}

fn protocol_failure(e: &ProtocolError) -> Failure {
    Failure {
        class: FailureClass::Protocol,
        detail: format!("System One response is unusable: {e}"),
    }
}

fn env_non_blank(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

/// Clip `text` to `max` characters (not bytes), marking the cut.
fn truncate(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    let head: String = text.chars().take(max).collect();
    format!("{head}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A judge that will never be called: every test in this module exercises a
    /// path that returns before any HTTP happens, or sets the breaker by hand.
    fn judge() -> JevJudge {
        JevJudge::new("not-a-real-key").with_base_url("http://127.0.0.1:1")
    }

    fn evidence(page: u32, flagged: bool, reasons: &[&str]) -> PageEvidence {
        PageEvidence {
            page,
            text: Some("some text".to_string()),
            reasons: reasons.iter().map(|r| (*r).to_string()).collect(),
            flagged_by_inspector: flagged,
            has_tables: false,
            has_columns: false,
            has_encoding_issues: false,
        }
    }

    #[test]
    fn the_mode_decides_the_name_and_the_names_are_the_registry_keys() {
        assert_eq!(judge().name(), JUDGE_NAME_ALWAYS);
        assert_eq!(judge().with_mode(JevMode::Gated).name(), JUDGE_NAME_GATED);
        assert_eq!(JUDGE_NAME_ALWAYS, "jev");
        assert_eq!(JUDGE_NAME_GATED, "jev_gated");
    }

    #[test]
    fn the_judge_asks_for_text() {
        assert!(judge().needs_text());
    }

    #[test]
    fn debug_redacts_the_api_key() {
        // Not a credential, and deliberately not shaped like one: no file in
        // this repository may contain anything a reader could mistake for a key.
        let rendered = format!("{:?}", JevJudge::new("the-key-goes-here"));
        assert!(!rendered.contains("the-key-goes-here"));
        assert!(rendered.contains("<redacted>"));
        // The non-secret configuration is still there, which is the point of
        // implementing `Debug` at all.
        assert!(rendered.contains("jev-latest"));
    }

    #[test]
    fn the_endpoint_never_doubles_a_slash() {
        assert_eq!(
            judge().with_base_url("https://example.test/").endpoint(),
            "https://example.test/v1/systemone"
        );
    }

    // --- the confidence mapping ---------------------------------------------

    #[test]
    fn a_high_probability_routes_to_ocr_and_is_confident_about_it() {
        let verdict = judge().verdict(3, 0.9);
        assert_eq!(verdict.page, 3);
        assert!(verdict.needs_ocr);
        assert_eq!(verdict.confidence, 0.9);
        assert_eq!(verdict.reason, REASON_NEEDS_OCR);
    }

    #[test]
    fn a_low_probability_is_a_confident_no_not_an_unconfident_yes() {
        let verdict = judge().verdict(0, 0.02);
        assert!(!verdict.needs_ocr);
        // The whole point: 0.02 -> 0.98, not 0.02.
        assert!((verdict.confidence - 0.98).abs() < 1e-6);
        assert_eq!(verdict.reason, REASON_CLEAR);
    }

    #[test]
    fn the_threshold_is_inclusive_and_a_coin_flip_is_half_confident() {
        let judge = judge();
        let at = judge.verdict(0, DEFAULT_THRESHOLD);
        assert!(at.needs_ocr, "p == threshold routes to OCR");
        assert_eq!(at.confidence, DEFAULT_THRESHOLD);

        let below = judge.verdict(0, DEFAULT_THRESHOLD - 0.01);
        assert!(!below.needs_ocr);
        assert!((below.confidence - (1.0 - (DEFAULT_THRESHOLD - 0.01))).abs() < 1e-6);
    }

    #[test]
    fn a_moved_threshold_moves_the_verdict_but_not_the_probability() {
        let cautious = judge().with_threshold(0.2);
        let verdict = cautious.verdict(0, 0.3);
        assert!(verdict.needs_ocr, "0.3 >= 0.2");
        assert!((verdict.confidence - 0.3).abs() < 1e-6);
        // The same probability under the default threshold goes the other way.
        assert!(!judge().verdict(0, 0.3).needs_ocr);
    }

    #[test]
    fn probabilities_outside_the_unit_interval_are_clamped() {
        assert_eq!(judge().verdict(0, 1.5).confidence, 1.0);
        assert_eq!(judge().verdict(0, -0.5).confidence, 1.0);
        assert!(!judge().verdict(0, -0.5).needs_ocr);
    }

    #[test]
    fn a_nonsense_threshold_is_ignored_rather_than_poisoning_the_judge() {
        assert_eq!(
            judge().with_threshold(f32::NAN).threshold(),
            DEFAULT_THRESHOLD
        );
        assert_eq!(judge().with_threshold(4.0).threshold(), 1.0);
        assert_eq!(judge().with_threshold(-1.0).threshold(), 0.0);
    }

    // --- the gate -----------------------------------------------------------

    #[test]
    fn encoding_issues_anywhere_make_the_document_ambiguous() {
        let mut pages = vec![evidence(0, false, &[]), evidence(1, false, &[])];
        assert!(!is_ambiguous(&pages));
        pages[1].has_encoding_issues = true;
        assert!(is_ambiguous(&pages));
    }

    #[test]
    fn a_partly_flagged_document_is_ambiguous() {
        let pages = vec![
            evidence(0, true, &["scanned_page"]),
            evidence(1, false, &[]),
            evidence(2, false, &[]),
        ];
        assert!(is_ambiguous(&pages));
    }

    #[test]
    fn a_flagged_page_with_no_reason_is_ambiguous_even_when_every_page_is_flagged() {
        let pages = vec![evidence(0, true, &["scanned_page"]), evidence(1, true, &[])];
        assert!(is_ambiguous(&pages));
    }

    #[test]
    fn a_consistently_flagged_or_consistently_clear_document_is_not_ambiguous() {
        let all_flagged = vec![
            evidence(0, true, &["scanned_page"]),
            evidence(1, true, &["no_text_layer"]),
        ];
        assert!(!is_ambiguous(&all_flagged));

        let all_clear = vec![evidence(0, false, &[]), evidence(1, false, &[])];
        assert!(!is_ambiguous(&all_clear));
    }

    #[test]
    fn layout_complexity_alone_is_not_ambiguity() {
        let mut pages = vec![evidence(0, false, &[]), evidence(1, false, &[])];
        pages[0].has_tables = true;
        pages[1].has_columns = true;
        assert!(!is_ambiguous(&pages));
    }

    #[test]
    fn an_empty_document_is_not_ambiguous() {
        assert!(!is_ambiguous(&[]));
    }

    // --- what the modes actually do -----------------------------------------

    #[test]
    fn a_gated_judge_that_does_not_escalate_makes_no_call() {
        let judge = judge().with_mode(JevMode::Gated);
        let pages = vec![evidence(0, false, &[]), evidence(1, false, &[])];
        let verdicts = judge.judge(&pages).expect("no call, no failure");

        assert_eq!(verdicts.len(), 2);
        assert!(verdicts.iter().all(|v| !v.needs_ocr));
        assert!(verdicts.iter().all(|v| v.reason == REASON_NOT_ESCALATED));
        // The heuristic's verdicts, unchanged apart from the reason.
        assert!(verdicts.iter().all(|v| v.confidence == 1.0));
        assert!(judge.calls().is_empty(), "the gate did not fire");
    }

    #[test]
    fn a_strict_gated_judge_that_does_not_escalate_still_succeeds() {
        // Not escalating is not a failure, so strictness has nothing to say
        // about it. Only a *failed* call errors under strict.
        let judge = judge().with_mode(JevMode::Gated).with_strict(true);
        let pages = vec![
            evidence(0, true, &["scanned_page"]),
            evidence(1, true, &["scanned_page"]),
        ];
        let verdicts = judge.judge(&pages).expect("no call, no failure");
        assert!(verdicts.iter().all(|v| v.needs_ocr));
        assert!(verdicts.iter().all(|v| v.reason == REASON_NOT_ESCALATED));
    }

    #[test]
    fn no_pages_means_no_verdicts_and_no_call() {
        let judge = judge();
        assert_eq!(judge.judge(&[]).expect("empty"), Vec::new());
        assert!(judge.calls().is_empty());
    }

    // --- the circuit breaker ------------------------------------------------

    #[test]
    fn the_breaker_opens_after_the_configured_run_of_failures() {
        let judge = judge().with_failure_threshold(3);
        assert!(!judge.breaker_is_open());
        judge.record_failure();
        judge.record_failure();
        assert!(!judge.breaker_is_open(), "two is not three");
        judge.record_failure();
        assert!(judge.breaker_is_open());
    }

    #[test]
    fn a_success_resets_the_run() {
        let judge = judge().with_failure_threshold(3);
        judge.record_failure();
        judge.record_failure();
        judge.record_success();
        judge.record_failure();
        judge.record_failure();
        assert!(!judge.breaker_is_open(), "the run restarted at the success");
    }

    #[test]
    fn the_breaker_closes_again_once_the_cool_off_elapses() {
        let judge = judge()
            .with_failure_threshold(1)
            .with_cool_off_seconds(3600.0);
        judge.record_failure();
        assert!(judge.breaker_is_open());

        // A cool-off of zero has always already elapsed, which is how this gets
        // tested without sleeping: the next read finds the breaker expired.
        let judge = judge.with_cool_off_seconds(0.0);
        assert!(!judge.breaker_is_open());
        // Reading it also reset the run, so the next call is a real probe rather
        // than the last straw of a run that already ended.
        assert_eq!(
            judge
                .breaker
                .lock()
                .expect("breaker mutex")
                .consecutive_failures,
            0
        );
    }

    #[test]
    fn a_zero_failure_threshold_disables_the_breaker() {
        let judge = judge().with_failure_threshold(0);
        for _ in 0..100 {
            judge.record_failure();
        }
        assert!(!judge.breaker_is_open());
    }

    #[test]
    fn an_open_breaker_short_circuits_without_calling() {
        let judge = judge()
            .with_failure_threshold(1)
            .with_cool_off_seconds(3600.0);
        judge.record_failure();
        let pages = vec![
            evidence(0, true, &["scanned_page"]),
            evidence(1, false, &[]),
        ];
        let verdicts = judge.judge(&pages).expect("non-strict falls back");
        assert!(
            judge.calls().is_empty(),
            "no HTTP while the breaker is open"
        );
        assert!(verdicts
            .iter()
            .all(|v| v.reason == REASON_FALLBACK_BREAKER_OPEN));
        assert_eq!(
            verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
            vec![true, false],
            "the heuristic's verdicts"
        );
    }

    #[test]
    fn an_open_breaker_errors_rather_than_falling_back_under_strict() {
        let judge = judge()
            .with_strict(true)
            .with_failure_threshold(1)
            .with_cool_off_seconds(3600.0);
        judge.record_failure();
        let err = judge
            .judge(&[evidence(0, false, &[])])
            .expect_err("strict propagates");
        assert!(matches!(err, Error::Judge(_)), "got {err:?}");
        assert!(judge.calls().is_empty());
    }

    // --- the failure policy -------------------------------------------------

    #[test]
    fn every_failure_class_has_its_own_reason_string() {
        assert_eq!(FailureClass::Timeout.reason(), "jev_fallback_timeout");
        assert_eq!(FailureClass::Http.reason(), "jev_fallback_http");
        assert_eq!(FailureClass::Protocol.reason(), "jev_fallback_protocol");
        assert_eq!(
            FailureClass::BreakerOpen.reason(),
            "jev_fallback_breaker_open"
        );
    }

    #[test]
    fn a_non_strict_fallback_keeps_the_heuristics_answer_and_names_the_failure() {
        let judge = judge();
        let heuristic = HeuristicJudge
            .judge(&[
                evidence(0, true, &["scanned_page"]),
                evidence(1, false, &[]),
            ])
            .expect("heuristic");
        let failure = Failure {
            class: FailureClass::Timeout,
            detail: "POST https://example.test/v1/systemone failed: timeout".to_string(),
        };
        let verdicts = judge.fall_back(&failure, heuristic).expect("non-strict");
        assert_eq!(
            verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
            vec![true, false]
        );
        assert!(verdicts.iter().all(|v| v.reason == REASON_FALLBACK_TIMEOUT));
    }

    #[test]
    fn a_strict_fallback_is_an_error_carrying_the_detail() {
        let judge = judge().with_strict(true);
        assert!(judge.is_strict());
        let failure = Failure {
            class: FailureClass::Protocol,
            detail: "System One response is unusable: no answer for page_4".to_string(),
        };
        let err = judge
            .fall_back(&failure, Vec::new())
            .expect_err("strict propagates");
        match err {
            Error::Judge(message) => assert!(message.contains("page_4"), "got {message}"),
            other => panic!("expected Error::Judge, got {other:?}"),
        }
    }

    #[test]
    fn a_protocol_error_becomes_a_protocol_failure() {
        let failure = protocol_failure(&ProtocolError::MissingAnswer {
            key: "page_7".to_string(),
        });
        assert_eq!(failure.class, FailureClass::Protocol);
        assert!(failure.detail.contains("page_7"));
    }

    #[test]
    fn truncate_marks_the_cut() {
        assert_eq!(truncate("abc", 10), "abc");
        assert_eq!(truncate("abcdef", 3), "abc…");
    }
}
