//! `JevJudge` end to end: real HTTP against a mock System One endpoint.
//!
//! The unit tests in `src/` cover everything that can be decided without a
//! socket. What is left, and what lives here, is the half that only a real
//! request answers: that the body which goes on the wire is the body the API
//! documents, and that each way a vendor can let you down — a status, a body, a
//! silence — lands in the failure class whose reason string says so.
//!
//! No test here needs a credential or reaches the network. The key below is a
//! literal string asserted against an `Authorization` header and is not a
//! credential for anything.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use doc_router::{PageEvidence, PageJudge};
use doc_router_jev::{
    JevJudge, JevMode, MAX_PAGES_PER_REQUEST, REASON_CLEAR, REASON_FALLBACK_BREAKER_OPEN,
    REASON_FALLBACK_HTTP, REASON_FALLBACK_PROTOCOL, REASON_FALLBACK_TIMEOUT, REASON_NEEDS_OCR,
};
use httpmock::prelude::*;
use serde_json::{json, Value};

/// Not a credential: an arbitrary string the mock asserts arrives verbatim.
const TEST_KEY: &str = "test-key-not-a-credential";

/// One request as the mock endpoint received it.
#[derive(Debug, Clone)]
struct Captured {
    body: Value,
    headers: Vec<(String, String)>,
}

impl Captured {
    /// The question names, sorted by the page number they encode.
    fn question_pages(&self) -> Vec<u32> {
        let mut pages: Vec<u32> = self.body["questions"]
            .as_object()
            .expect("questions is an object")
            .keys()
            .map(|key| {
                key.strip_prefix("page_")
                    .expect("every question key is page_<n>")
                    .parse()
                    .expect("every question key ends in a number")
            })
            .collect();
        pages.sort_unstable();
        pages
    }

    /// The first value of `name`, case-insensitively.
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

type Log = Arc<Mutex<Vec<Captured>>>;

fn log() -> Log {
    Arc::new(Mutex::new(Vec::new()))
}

/// A `when.is_true` matcher that always matches and records what it saw.
fn record_into(log: &Log) -> impl Fn(&HttpMockRequest) -> bool + Send + Sync + 'static {
    let log = Arc::clone(log);
    move |req: &HttpMockRequest| {
        log.lock().expect("request log").push(Captured {
            body: serde_json::from_slice(req.body_ref()).unwrap_or(Value::Null),
            headers: req.headers_vec().clone(),
        });
        true
    }
}

fn captured(log: &Log) -> Vec<Captured> {
    log.lock().expect("request log").clone()
}

fn evidence(page: u32, flagged: bool) -> PageEvidence {
    PageEvidence {
        page,
        text: Some(format!("the extracted text of page {page}")),
        reasons: if flagged {
            vec!["scanned_page".to_string()]
        } else {
            Vec::new()
        },
        flagged_by_inspector: flagged,
        has_tables: false,
        has_columns: false,
        has_encoding_issues: false,
    }
}

/// A System One response answering `(page, probability)` pairs.
fn answers(pages: &[(u32, f64)]) -> Value {
    let answers: serde_json::Map<String, Value> = pages
        .iter()
        .map(|(page, p)| (format!("page_{page}"), json!({"type": "noul", "noul": p})))
        .collect();
    json!({
        "model": "jev-2026-04",
        "answers": answers,
        "usage": {"input_tokens": 431, "output_tokens": 12},
    })
}

fn judge(server: &MockServer) -> JevJudge {
    JevJudge::new(TEST_KEY).with_base_url(server.base_url())
}

#[test]
fn a_document_is_one_call_whose_body_is_what_the_api_documents() {
    let server = MockServer::start();
    let log = log();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/systemone")
            .is_true(record_into(&log));
        then.status(200)
            .json_body(answers(&[(0, 0.03), (1, 0.97), (2, 0.5)]));
    });

    let pages = vec![evidence(0, false), evidence(1, true), evidence(2, false)];
    let verdicts = judge(&server)
        .judge(&pages)
        .expect("a 200 is not a failure");

    // One call for the whole document, not one per page.
    mock.assert_calls(1);
    let calls = captured(&log);
    let call = calls.last().expect("one recorded call");
    assert_eq!(
        call.header("authorization"),
        Some("Bearer test-key-not-a-credential")
    );
    assert_eq!(call.header("content-type"), Some("application/json"));
    assert_eq!(call.body["model"], "jev-latest");
    assert_eq!(call.question_pages(), vec![0, 1, 2]);
    assert_eq!(call.body["questions"]["page_1"]["type"], "noul");
    assert_eq!(call.body["state"].as_array().expect("state").len(), 3);
    assert_eq!(call.body["state"][1]["page"], 1);
    assert_eq!(
        call.body["state"][1]["text"],
        "the extracted text of page 1"
    );
    assert_eq!(call.body["state"][1]["inspector_flagged"], true);

    // And the answers come back as verdicts, in page order, with the confidence
    // mapping applied: 0.03 is a confident "no", not an unconfident "yes".
    assert_eq!(
        verdicts.iter().map(|v| v.page).collect::<Vec<_>>(),
        vec![0, 1, 2]
    );
    assert_eq!(
        verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
        vec![false, true, true]
    );
    assert!((verdicts[0].confidence - 0.97).abs() < 1e-6);
    assert!((verdicts[1].confidence - 0.97).abs() < 1e-6);
    assert!((verdicts[2].confidence - 0.5).abs() < 1e-6);
    assert_eq!(verdicts[0].reason, REASON_CLEAR);
    assert_eq!(verdicts[1].reason, REASON_NEEDS_OCR);
}

#[test]
fn the_verdicts_follow_the_keys_even_when_the_answers_arrive_shuffled() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        // Deliberately reversed relative to the request.
        then.status(200)
            .json_body(answers(&[(2, 0.9), (1, 0.1), (0, 0.8)]));
    });

    let pages = vec![evidence(0, false), evidence(1, false), evidence(2, false)];
    let verdicts = judge(&server).judge(&pages).expect("a 200");
    assert_eq!(
        verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
        vec![true, false, true],
        "page 1 is the one with the low probability, wherever it sat in the map"
    );
}

#[test]
fn a_long_document_is_chunked_and_every_page_is_still_asked_about_by_name() {
    let server = MockServer::start();
    let log = log();
    let total = MAX_PAGES_PER_REQUEST * 2 + 7;
    // Answer every page in every response; each chunk reads only its own keys.
    let every: Vec<(u32, f64)> = (0..total as u32).map(|page| (page, 0.8)).collect();
    let mock = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/systemone")
            .is_true(record_into(&log));
        then.status(200).json_body(answers(&every));
    });

    let pages: Vec<PageEvidence> = (0..total as u32)
        .map(|page| evidence(page, false))
        .collect();
    let verdicts = judge(&server).judge(&pages).expect("a 200 per chunk");

    mock.assert_calls(3);
    let calls = captured(&log);
    assert_eq!(calls[0].question_pages().len(), MAX_PAGES_PER_REQUEST);
    assert_eq!(calls[1].question_pages().len(), MAX_PAGES_PER_REQUEST);
    assert_eq!(calls[2].question_pages().len(), 7);
    // Chunking does not renumber: the second request asks about page 50, not 0.
    assert_eq!(calls[1].question_pages()[0], MAX_PAGES_PER_REQUEST as u32);
    let asked: Vec<u32> = calls.iter().flat_map(Captured::question_pages).collect();
    assert_eq!(asked, (0..total as u32).collect::<Vec<_>>());

    assert_eq!(verdicts.len(), total);
    assert!(verdicts.iter().all(|v| v.needs_ocr));
    assert_eq!(
        verdicts.iter().map(|v| v.page).collect::<Vec<_>>(),
        (0..total as u32).collect::<Vec<_>>()
    );
}

/// Every status that is not a 2xx, and what a non-strict judge does with it.
#[test]
fn a_rejected_or_overloaded_call_falls_back_to_the_heuristic() {
    for (status, body) in [
        (401, json!({"error": "missing or invalid API key"})),
        (422, json!({"error": "questions must not be empty"})),
        (429, json!({"error": "rate limit exceeded"})),
        (529, json!({"error": "overloaded"})),
    ] {
        let server = MockServer::start();
        let mock = server.mock(|when, then| {
            when.method(POST).path("/v1/systemone");
            then.status(status).json_body(body.clone());
        });

        let pages = vec![evidence(0, true), evidence(1, false)];
        let verdicts = judge(&server)
            .judge(&pages)
            .unwrap_or_else(|e| panic!("HTTP {status} must not break routing: {e}"));

        mock.assert_calls(1);
        assert_eq!(
            verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
            vec![true, false],
            "HTTP {status} falls back to the inspector's own answer"
        );
        assert!(
            verdicts.iter().all(|v| v.reason == REASON_FALLBACK_HTTP),
            "HTTP {status} is labelled as a fallback, never as a judgement"
        );
        assert!(verdicts.iter().all(|v| v.confidence == 1.0));
    }
}

#[test]
fn a_strict_judge_propagates_the_status_and_the_body_without_the_key() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        then.status(401)
            .json_body(json!({"error": "missing or invalid API key"}));
    });

    let err = judge(&server)
        .with_strict(true)
        .judge(&[evidence(0, true)])
        .expect_err("strict propagates");
    let message = err.to_string();
    assert!(message.contains("401"), "got {message}");
    assert!(
        message.contains("missing or invalid API key"),
        "got {message}"
    );
    assert!(
        !message.contains(TEST_KEY),
        "the key must never reach an error message: {message}"
    );
}

#[test]
fn a_body_that_is_not_json_is_a_protocol_failure_not_an_http_one() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        then.status(200)
            .header("content-type", "application/json")
            .body("<html>upstream proxy says hello</html>");
    });

    let pages = vec![evidence(0, true), evidence(1, false)];
    let verdicts = judge(&server).judge(&pages).expect("non-strict falls back");
    assert!(verdicts
        .iter()
        .all(|v| v.reason == REASON_FALLBACK_PROTOCOL));
    assert_eq!(
        verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
        vec![true, false]
    );
}

#[test]
fn a_response_missing_one_page_fails_rather_than_calling_that_page_clear() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        // Page 1 is simply absent. Defaulting it to `false` would route a scan
        // to the local text extractor and silently lose its content.
        then.status(200).json_body(answers(&[(0, 0.9), (2, 0.9)]));
    });

    let pages = vec![evidence(0, true), evidence(1, true), evidence(2, true)];
    let verdicts = judge(&server).judge(&pages).expect("non-strict falls back");
    assert!(verdicts
        .iter()
        .all(|v| v.reason == REASON_FALLBACK_PROTOCOL));
    assert!(
        verdicts.iter().all(|v| v.needs_ocr),
        "the heuristic's verdicts, not a partial mix of the two"
    );

    let err = judge(&server)
        .with_strict(true)
        .judge(&pages)
        .expect_err("strict propagates");
    assert!(err.to_string().contains("page_1"), "got {err}");
}

#[test]
fn a_call_that_never_answers_is_a_timeout_not_a_transport_error() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        then.status(200)
            .delay(Duration::from_secs(5))
            .json_body(answers(&[(0, 0.9)]));
    });

    let pages = vec![evidence(0, true), evidence(1, false)];
    let verdicts = judge(&server)
        .with_timeout(0.25)
        .judge(&pages)
        .expect("non-strict falls back");
    assert!(
        verdicts.iter().all(|v| v.reason == REASON_FALLBACK_TIMEOUT),
        "a slow vendor and an unreachable one want different operational responses"
    );
    assert_eq!(
        verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
        vec![true, false]
    );
}

#[test]
fn a_run_of_failures_opens_the_breaker_and_stops_the_calls() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        then.status(529).json_body(json!({"error": "overloaded"}));
    });

    let judge = judge(&server)
        .with_failure_threshold(2)
        .with_cool_off_seconds(3600.0);
    let pages = vec![evidence(0, true)];

    for _ in 0..2 {
        let verdicts = judge.judge(&pages).expect("non-strict");
        assert!(verdicts.iter().all(|v| v.reason == REASON_FALLBACK_HTTP));
    }
    mock.assert_calls(2);

    // The third document does not reach the network at all.
    let verdicts = judge.judge(&pages).expect("non-strict");
    assert!(verdicts
        .iter()
        .all(|v| v.reason == REASON_FALLBACK_BREAKER_OPEN));
    mock.assert_calls(2);
    assert_eq!(judge.calls().len(), 2, "only the two calls that happened");
}

#[test]
fn a_gated_judge_escalates_an_ambiguous_document_and_skips_a_clear_one() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        then.status(200).json_body(answers(&[(0, 0.2), (1, 0.8)]));
    });

    let judge = judge(&server).with_mode(JevMode::Gated);
    assert_eq!(judge.name(), "jev_gated");

    // Flagged-and-clear in the same document: the gate fires.
    let mixed = vec![evidence(0, true), evidence(1, false)];
    let verdicts = judge.judge(&mixed).expect("a 200");
    mock.assert_calls(1);
    assert_eq!(
        verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
        vec![false, true],
        "Jev disagreed with the inspector on both pages, which is the point"
    );

    // Every page flagged, with a reason: the inspector was consistent, so no call.
    let consistent = vec![evidence(0, true), evidence(1, true)];
    let verdicts = judge.judge(&consistent).expect("no call");
    mock.assert_calls(1);
    assert!(verdicts.iter().all(|v| v.reason == "jev_not_escalated"));
}

#[test]
fn the_call_log_records_the_pages_the_status_and_the_reported_usage() {
    let server = MockServer::start();
    server.mock(|when, then| {
        when.method(POST).path("/v1/systemone");
        then.status(200).json_body(answers(&[(0, 0.9), (1, 0.1)]));
    });

    let judge = judge(&server);
    judge
        .judge(&[evidence(0, true), evidence(1, false)])
        .expect("a 200");
    let calls = judge.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].pages, vec![0, 1]);
    assert_eq!(calls[0].status, Some(200));
    assert_eq!(calls[0].input_tokens, Some(431));
    assert_eq!(calls[0].output_tokens, Some(12));
}
