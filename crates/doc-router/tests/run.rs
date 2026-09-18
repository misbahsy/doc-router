//! End-to-end execution against a recording mock host.

mod common;

use std::sync::Mutex;

use common::fixture;
use doc_router::{run, Config, Decision, Error, OcrHost, OcrResult, Page, Tiers, LOCAL_MODEL};

const LOCAL: &str = "local_pdf/extract";
const STANDARD: &str = "mistral-ocr";

/// One `ocr` call as the mock saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Call {
    model: String,
    pages: Option<Vec<u32>>,
    doc_len: usize,
}

/// A host that records what it was asked for and fabricates pages in return.
struct MockHost {
    calls: Mutex<Vec<Call>>,
    /// Models whose calls fail instead of returning pages.
    fail_for: Vec<String>,
}

impl MockHost {
    fn new() -> Self {
        MockHost {
            calls: Mutex::new(Vec::new()),
            fail_for: Vec::new(),
        }
    }

    fn failing_for(model: &str) -> Self {
        MockHost {
            calls: Mutex::new(Vec::new()),
            fail_for: vec![model.to_string()],
        }
    }

    fn calls(&self) -> Vec<Call> {
        self.calls.lock().expect("mock host lock").clone()
    }
}

impl OcrHost for MockHost {
    fn ocr(&self, model: &str, document: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error> {
        self.calls.lock().expect("mock host lock").push(Call {
            model: model.to_string(),
            pages: pages.map(|p| p.to_vec()),
            doc_len: document.len(),
        });
        if self.fail_for.iter().any(|m| m == model) {
            return Err(Error::Host(format!("{model} is down")));
        }
        // Whole-document calls cover every page of the fixtures used here.
        let indices: Vec<u32> = match pages {
            Some(pages) => pages.to_vec(),
            None => (0..doc_router::classify(document)
                .map(|c| c.page_count)
                .unwrap_or(1))
                .collect(),
        };
        Ok(OcrResult {
            pages: indices
                .iter()
                .map(|index| Page::new(*index, format!("OCR page {index}"), model))
                .collect(),
            model: model.to_string(),
            pages_processed: indices.len() as u32,
            doc_size_bytes: Some(document.len() as u64),
        })
    }
}

fn config() -> Config {
    Config::new(Tiers::new(LOCAL, STANDARD))
}

#[test]
fn mixed_split_runs_the_local_leg_in_process_and_ocrs_the_rest() {
    let bytes = fixture("mixed.pdf");
    let host = MockHost::new();
    let outcome = run(&bytes, &config(), None, &host).expect("run");

    // Only the OCR leg reaches the host, once, with the OCR pages.
    assert_eq!(
        host.calls(),
        vec![Call {
            model: STANDARD.to_string(),
            pages: Some(vec![1, 3]),
            doc_len: bytes.len(),
        }]
    );

    assert!(matches!(outcome.decision, Decision::Route { .. }));
    assert_eq!(
        outcome
            .result
            .pages
            .iter()
            .map(|p| p.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(
        outcome
            .result
            .pages
            .iter()
            .map(|p| p.model.as_str())
            .collect::<Vec<_>>(),
        vec![LOCAL_MODEL, STANDARD, LOCAL_MODEL, STANDARD]
    );
    assert_eq!(outcome.result.model, "local_pdf/extract,mistral-ocr");
    assert_eq!(outcome.result.pages_processed, 4);
    assert_eq!(outcome.result.pages[1].markdown, "OCR page 1");
    assert!(!outcome.result.pages[0].markdown.is_empty());

    assert_eq!(outcome.metadata.tier, "standard");
    assert_eq!(outcome.metadata.reason, "mixed_split");
    assert_eq!(outcome.metadata.split, Some(true));
    assert_eq!(outcome.metadata.fallback_reason, None);
}

#[test]
fn scanned_documents_go_to_the_host_whole() {
    let bytes = fixture("scanned.pdf");
    let host = MockHost::new();
    let outcome = run(&bytes, &config(), None, &host).expect("run");

    assert_eq!(
        host.calls(),
        vec![Call {
            model: STANDARD.to_string(),
            pages: None,
            doc_len: bytes.len(),
        }]
    );
    assert_eq!(outcome.result.pages.len(), 2);
    assert_eq!(outcome.result.model, STANDARD);
    assert_eq!(outcome.metadata.reason, "scanned");
}

#[test]
fn text_documents_never_reach_the_host() {
    let bytes = fixture("text.pdf");
    let host = MockHost::new();
    let outcome = run(&bytes, &config(), None, &host).expect("run");

    assert!(host.calls().is_empty());
    assert_eq!(outcome.result.model, LOCAL_MODEL);
    assert_eq!(outcome.result.pages.len(), 2);
    assert_eq!(outcome.metadata.reason, "text_layer");
}

#[test]
fn a_failed_leg_reruns_the_whole_document_on_the_default_model() {
    let bytes = fixture("mixed.pdf");
    let host = MockHost::failing_for(STANDARD);
    let outcome = run(&bytes, &config(), Some("fallback-ocr"), &host).expect("run");

    assert_eq!(
        host.calls(),
        vec![
            Call {
                model: STANDARD.to_string(),
                pages: Some(vec![1, 3]),
                doc_len: bytes.len(),
            },
            Call {
                model: "fallback-ocr".to_string(),
                pages: None,
                doc_len: bytes.len(),
            },
        ]
    );
    assert_eq!(
        outcome.metadata.fallback_reason.as_deref(),
        Some("leg_failed")
    );
    assert_eq!(outcome.metadata.routed_model, "fallback-ocr");
    assert_eq!(outcome.result.model, "fallback-ocr");
    assert_eq!(outcome.result.pages.len(), 4);
}

#[test]
fn the_fallback_defaults_to_the_standard_tier() {
    let bytes = fixture("mixed.pdf");
    let host = MockHost::failing_for(STANDARD);
    // Both the OCR leg and the fallback are the standard model, so the fallback fails too.
    let err = run(&bytes, &config(), None, &host).expect_err("both attempts fail");
    assert!(matches!(err, Error::LegFailed { .. }), "{err}");
    let calls = host.calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[1].pages, None);
    assert_eq!(calls[1].model, STANDARD);
}

#[test]
fn bypassed_documents_go_straight_to_the_bypass_model() {
    let host = MockHost::new();
    let bytes = b"this is plainly not a pdf".to_vec();
    let outcome = run(&bytes, &config(), Some("bypass-model"), &host).expect("run");

    assert_eq!(
        host.calls(),
        vec![Call {
            model: "bypass-model".to_string(),
            pages: None,
            doc_len: bytes.len(),
        }]
    );
    assert!(matches!(outcome.decision, Decision::Bypass { .. }));
    assert_eq!(outcome.metadata.tier, "bypass");
    assert_eq!(outcome.metadata.reason, "not_pdf");
    assert_eq!(outcome.metadata.routed_model, "bypass-model");
}

#[test]
fn a_host_that_splits_pages_itself_still_reports_original_indices() {
    // What a provider without page-list support does inside `ocr`.
    struct SplittingHost;
    impl OcrHost for SplittingHost {
        fn ocr(
            &self,
            model: &str,
            document: &[u8],
            pages: Option<&[u32]>,
        ) -> Result<OcrResult, Error> {
            let pages = pages.expect("this test only sends subsets");
            let subset = doc_router::split_pdf(document, pages)?;
            let mut result = OcrResult {
                pages: (0..pages.len() as u32)
                    .map(|i| Page::new(i, format!("subset page {i}"), model))
                    .collect(),
                model: model.to_string(),
                pages_processed: pages.len() as u32,
                doc_size_bytes: Some(subset.len() as u64),
            };
            doc_router::remap_pages(&mut result, pages);
            Ok(result)
        }
    }

    let outcome = run(&fixture("mixed.pdf"), &config(), None, &SplittingHost).expect("run");
    assert_eq!(
        outcome
            .result
            .pages
            .iter()
            .map(|p| p.index)
            .collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(outcome.result.pages[1].markdown, "subset page 0");
    assert_eq!(outcome.result.pages[3].markdown, "subset page 1");
}
