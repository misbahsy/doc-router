//! `LiteLlmHost` end to end: real HTTP against a mock LiteLLM proxy, driven by
//! `doc_router::run` so the wire format and the routing decisions are checked together.

mod common;

use std::sync::{Arc, Mutex};

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use common::fixture;
use doc_router::{classify, run, Config, Tiers, LOCAL_MODEL};
use doc_router_cli::LiteLlmHost;
use httpmock::prelude::*;
use serde_json::{json, Value};

const STANDARD_MODEL: &str = "mistral-ocr";
const FALLBACK_MODEL: &str = "gpt-4o-mini";
const API_KEY: &str = "sk-test-1234";

fn config() -> Config {
    Config::new(Tiers::new(LOCAL_MODEL, STANDARD_MODEL))
}

/// One request as the mock proxy received it.
#[derive(Debug, Clone)]
struct Captured {
    body: Value,
    headers: Vec<(String, String)>,
}

impl Captured {
    /// The `pages` field, or `None` when the key is absent.
    fn pages(&self) -> Option<&Value> {
        self.body.get("pages")
    }

    /// The base64 payload of the `document_url` data URI, decoded.
    fn document(&self) -> Vec<u8> {
        let url = self.body["document"]["document_url"]
            .as_str()
            .expect("document_url is a string");
        let b64 = url
            .strip_prefix("data:application/pdf;base64,")
            .expect("document_url is a base64 PDF data URI");
        BASE64.decode(b64).expect("document_url decodes")
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

/// A LiteLLM `OCRResponse` carrying `pages` as `(index, markdown)`.
fn ocr_response(model: &str, pages: &[(u32, &str)]) -> Value {
    json!({
        "pages": pages
            .iter()
            .map(|(index, markdown)| json!({
                "index": index,
                "markdown": markdown,
                "images": [],
                "dimensions": {"dpi": 200, "height": 1100, "width": 850},
            }))
            .collect::<Vec<_>>(),
        "model": model,
        "usage_info": {"pages_processed": pages.len(), "doc_size_bytes": 4096},
    })
}

fn host(server: &MockServer) -> LiteLlmHost {
    LiteLlmHost::new(server.base_url()).with_api_key(Some(API_KEY))
}

#[test]
fn a_mixed_pdf_sends_exactly_one_ocr_call_for_the_ocr_pages() {
    let server = MockServer::start();
    let log = log();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/ocr").is_true(record_into(&log));
        then.status(200).json_body(ocr_response(
            STANDARD_MODEL,
            &[(1, "scan of page two"), (3, "scan of page four")],
        ));
    });

    let bytes = fixture("mixed.pdf");
    let outcome = run(&bytes, &config(), None, &host(&server)).expect("run succeeds");

    // Exactly one HTTP call: the local leg never leaves the process.
    mock.assert_calls(1);
    let calls = captured(&log);
    let call = calls.last().expect("one recorded call");
    assert_eq!(call.body["model"], STANDARD_MODEL);
    assert_eq!(call.pages(), Some(&json!([1, 3])));
    assert_eq!(call.header("authorization"), Some("Bearer sk-test-1234"));
    assert_eq!(
        call.document(),
        bytes,
        "the whole document goes on the wire"
    );

    // The merged result is the whole document, in page order.
    let pages = &outcome.result.pages;
    assert_eq!(
        pages.iter().map(|p| p.index).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        pages.iter().map(|p| p.model.as_str()).collect::<Vec<_>>(),
        vec![LOCAL_MODEL, STANDARD_MODEL, LOCAL_MODEL, STANDARD_MODEL]
    );
    assert_eq!(pages[1].markdown, "scan of page two");
    assert_eq!(pages[3].markdown, "scan of page four");
    assert!(
        !pages[0].markdown.is_empty(),
        "page 0 came from the text layer"
    );
    assert!(
        !pages[2].markdown.is_empty(),
        "page 2 came from the text layer"
    );

    assert_eq!(
        outcome.result.model,
        format!("{LOCAL_MODEL},{STANDARD_MODEL}")
    );
    assert_eq!(outcome.metadata.reason, "mixed_split");
    assert_eq!(outcome.metadata.split, Some(true));
    assert_eq!(outcome.metadata.fallback_reason, None);
}

#[test]
fn a_scanned_pdf_sends_one_whole_document_call_with_no_pages_field() {
    let server = MockServer::start();
    let log = log();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/ocr").is_true(record_into(&log));
        then.status(200).json_body(ocr_response(
            STANDARD_MODEL,
            &[(0, "scan of page one"), (1, "scan of page two")],
        ));
    });

    let bytes = fixture("scanned.pdf");
    let outcome = run(&bytes, &config(), None, &host(&server)).expect("run succeeds");

    mock.assert_calls(1);
    let calls = captured(&log);
    let call = calls.last().expect("one recorded call");
    assert_eq!(call.body["model"], STANDARD_MODEL);
    assert_eq!(
        call.pages(),
        None,
        "a whole-document leg must not send a `pages` key"
    );

    assert_eq!(outcome.metadata.reason, "scanned");
    assert_eq!(outcome.result.model, STANDARD_MODEL);
    assert_eq!(
        outcome
            .result
            .pages
            .iter()
            .map(|p| p.markdown.as_str())
            .collect::<Vec<_>>(),
        vec!["scan of page one", "scan of page two"]
    );
}

#[test]
fn a_text_pdf_never_touches_the_network() {
    let server = MockServer::start();
    let log = log();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/ocr").is_true(record_into(&log));
        then.status(200)
            .json_body(ocr_response(STANDARD_MODEL, &[]));
    });

    let bytes = fixture("text.pdf");
    let outcome = run(&bytes, &config(), None, &host(&server)).expect("run succeeds");

    mock.assert_calls(0);
    assert!(captured(&log).is_empty());
    assert_eq!(outcome.metadata.reason, "text_layer");
    assert_eq!(outcome.result.model, LOCAL_MODEL);
    assert_eq!(outcome.result.pages.len(), 2);
}

#[test]
fn a_failing_leg_reruns_the_whole_document_on_the_default_model() {
    let server = MockServer::start();

    // The split leg (it is the only call that carries `pages`) fails...
    let leg = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/ocr")
            .is_true(|req: &HttpMockRequest| {
                let body: Value = serde_json::from_slice(req.body_ref()).unwrap_or(Value::Null);
                body.get("pages").is_some()
            });
        then.status(500)
            .json_body(json!({"error": {"message": "ocr backend exploded"}}));
    });
    // ...so the whole document is retried once, on the default model, with no pages.
    let fallback = server.mock(|when, then| {
        when.method(POST)
            .path("/v1/ocr")
            .is_true(|req: &HttpMockRequest| {
                let body: Value = serde_json::from_slice(req.body_ref()).unwrap_or(Value::Null);
                body.get("pages").is_none() && body["model"] == json!(FALLBACK_MODEL)
            });
        then.status(200).json_body(ocr_response(
            FALLBACK_MODEL,
            &[(0, "a"), (1, "b"), (2, "c"), (3, "d")],
        ));
    });

    let bytes = fixture("mixed.pdf");
    let outcome =
        run(&bytes, &config(), Some(FALLBACK_MODEL), &host(&server)).expect("run succeeds");

    leg.assert_calls(1);
    fallback.assert_calls(1);
    assert_eq!(
        outcome.metadata.fallback_reason.as_deref(),
        Some("leg_failed")
    );
    assert_eq!(outcome.metadata.routed_model, FALLBACK_MODEL);
    assert_eq!(outcome.result.model, FALLBACK_MODEL);
    assert_eq!(outcome.result.pages.len(), 4);
}

#[test]
fn a_failing_fallback_surfaces_the_http_status() {
    let server = MockServer::start();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/ocr");
        then.status(500).body("ocr backend exploded");
    });

    let bytes = fixture("mixed.pdf");
    let error = run(&bytes, &config(), Some(FALLBACK_MODEL), &host(&server))
        .expect_err("both calls fail, so run fails");

    // The split leg, then the whole-document retry.
    mock.assert_calls(2);
    let message = error.to_string();
    assert!(message.contains("500"), "{message}");
    assert!(message.contains("ocr backend exploded"), "{message}");
}

#[test]
fn split_subset_uploads_a_subset_pdf_and_remaps_the_pages_back() {
    let server = MockServer::start();
    let log = log();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/ocr").is_true(record_into(&log));
        // A provider that ignores `pages` numbers the subset it was given from 0.
        then.status(200).json_body(ocr_response(
            STANDARD_MODEL,
            &[(0, "subset page one"), (1, "subset page two")],
        ));
    });

    let bytes = fixture("mixed.pdf");
    let host = LiteLlmHost::new(server.base_url())
        .with_api_key(Some(API_KEY))
        .with_split_subset(true);
    let outcome = run(&bytes, &config(), None, &host).expect("run succeeds");

    mock.assert_calls(1);
    let calls = captured(&log);
    let call = calls.last().expect("one recorded call");
    assert_eq!(
        call.pages(),
        None,
        "a split subset replaces the `pages` field, it does not accompany it"
    );

    let uploaded = call.document();
    assert_ne!(uploaded, bytes, "the subset is not the original document");
    let uploaded_class = classify(&uploaded).expect("the uploaded subset is a readable PDF");
    assert_eq!(uploaded_class.page_count, 2);

    // The provider's 0/1 became the original 1/3 again.
    let pages = &outcome.result.pages;
    assert_eq!(
        pages.iter().map(|p| p.index).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(pages[1].markdown, "subset page one");
    assert_eq!(pages[3].markdown, "subset page two");
    assert_eq!(
        pages.iter().map(|p| p.model.as_str()).collect::<Vec<_>>(),
        vec![LOCAL_MODEL, STANDARD_MODEL, LOCAL_MODEL, STANDARD_MODEL]
    );
}

#[test]
fn a_host_without_a_key_sends_no_authorization_header() {
    let server = MockServer::start();
    let log = log();
    let mock = server.mock(|when, then| {
        when.method(POST).path("/v1/ocr").is_true(record_into(&log));
        then.status(200)
            .json_body(ocr_response(STANDARD_MODEL, &[(0, "a"), (1, "b")]));
    });

    let bytes = fixture("scanned.pdf");
    let host = LiteLlmHost::new(server.base_url()).with_api_key(None::<String>);
    run(&bytes, &config(), None, &host).expect("run succeeds");

    mock.assert_calls(1);
    let calls = captured(&log);
    assert_eq!(
        calls.last().expect("one call").header("authorization"),
        None
    );

    // The host's own call log mirrors what went out.
    let recorded = host.calls();
    assert_eq!(recorded.len(), 1);
    assert_eq!(recorded[0].model, STANDARD_MODEL);
    assert_eq!(recorded[0].pages, None);
    assert_eq!(recorded[0].status, Some(200));
}
