//! The [`PageJudge`] seam, exercised through `classify_with` over real fixtures.
//!
//! [`HeuristicJudge`]'s own unit tests live next to it in `src/judge.rs`. What is
//! tested here is the wiring: that a judge's answer really is what
//! `pages_needing_ocr` and `pdf_type` are built from, that the evidence handed to
//! it describes the document in front of it, and that a judge which breaks the
//! contract is reported rather than believed.

mod common;

use std::sync::Mutex;

use common::fixture;
use doc_router::{
    classify, classify_with, Error, HeuristicJudge, PageEvidence, PageJudge, PageVerdict, PdfType,
};

/// The complement of `pages` within a `page_count`-page document.
fn complement(pages: &[u32], page_count: u32) -> Vec<u32> {
    (0..page_count).filter(|p| !pages.contains(p)).collect()
}

fn verdict(page: u32, needs_ocr: bool) -> PageVerdict {
    PageVerdict {
        page,
        needs_ocr,
        confidence: 0.5,
        reason: "test".to_string(),
    }
}

/// Says the opposite of pdf-inspector about every page, and keeps what it saw.
///
/// Inverting is the cheapest decision that cannot be confused with the default:
/// if `classify_with` were still reading pdf-inspector's list behind the judge's
/// back, no assertion below would hold.
#[derive(Default)]
struct InvertingJudge {
    wants_text: bool,
    seen: Mutex<Vec<PageEvidence>>,
}

impl InvertingJudge {
    fn with_text() -> Self {
        InvertingJudge {
            wants_text: true,
            seen: Mutex::new(Vec::new()),
        }
    }

    fn seen(&self) -> Vec<PageEvidence> {
        self.seen.lock().expect("evidence mutex").clone()
    }
}

impl PageJudge for InvertingJudge {
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        self.seen
            .lock()
            .expect("evidence mutex")
            .extend_from_slice(evidence);
        Ok(evidence
            .iter()
            .map(|page| verdict(page.page, !page.flagged_by_inspector))
            .collect())
    }

    fn name(&self) -> &'static str {
        "inverting"
    }

    fn needs_text(&self) -> bool {
        self.wants_text
    }
}

/// Returns whatever it is told to, contract or no contract.
struct BrokenJudge(fn(&[PageEvidence]) -> Vec<PageVerdict>);

impl PageJudge for BrokenJudge {
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        Ok((self.0)(evidence))
    }

    fn name(&self) -> &'static str {
        "broken"
    }
}

/// Fails outright, the way a hosted judge would when the network is down.
struct FailingJudge;

impl PageJudge for FailingJudge {
    fn judge(&self, _evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        Err(Error::Judge("no answer from the classifier".to_string()))
    }

    fn name(&self) -> &'static str {
        "failing"
    }
}

#[test]
fn the_default_judge_is_the_heuristic_one() {
    for name in ["text.pdf", "mixed.pdf", "scanned.pdf", "single_image.pdf"] {
        let bytes = fixture(name);
        let default = classify(&bytes).expect(name);
        let explicit = classify_with(&bytes, &HeuristicJudge).expect(name);
        assert_eq!(default.pdf_type, explicit.pdf_type, "{name}");
        assert_eq!(
            default.pages_needing_ocr, explicit.pages_needing_ocr,
            "{name}"
        );
        assert_eq!(default.page_count, explicit.page_count, "{name}");
        assert_eq!(
            default.is_complex_layout, explicit.is_complex_layout,
            "{name}"
        );
        assert_eq!(
            default.has_encoding_issues, explicit.has_encoding_issues,
            "{name}"
        );
        assert_eq!(default.ocr_reasons, explicit.ocr_reasons, "{name}");
    }
}

#[test]
fn an_inverting_judge_produces_the_complement_page_set() {
    // mixed.pdf is 4 pages with the two image pages flagged, so the complement
    // is the other two: a real split either way, not an empty set.
    let bytes = fixture("mixed.pdf");
    let default = classify(&bytes).expect("classify mixed.pdf");
    assert_eq!(default.pages_needing_ocr, vec![1, 3]);

    let inverted = classify_with(&bytes, &InvertingJudge::default()).expect("inverted mixed.pdf");
    assert_eq!(
        inverted.pages_needing_ocr,
        complement(&default.pages_needing_ocr, default.page_count)
    );
    assert_eq!(inverted.pages_needing_ocr, vec![0, 2]);
    assert_eq!(inverted.page_count, default.page_count);
}

#[test]
fn an_inverting_judge_moves_the_document_label_with_the_pages() {
    // scanned.pdf has every page flagged, so inverting empties the list — and a
    // document with nothing left to OCR is text-based, whatever detection said.
    let bytes = fixture("scanned.pdf");
    let default = classify(&bytes).expect("classify scanned.pdf");
    assert_eq!(default.pdf_type, PdfType::Scanned);
    assert_eq!(default.pages_needing_ocr, vec![0, 1]);

    let inverted = classify_with(&bytes, &InvertingJudge::default()).expect("inverted scanned.pdf");
    assert_eq!(inverted.pages_needing_ocr, Vec::<u32>::new());
    assert_eq!(inverted.pdf_type, PdfType::TextBased);
    assert_ne!(inverted.pdf_type, default.pdf_type);
}

#[test]
fn every_page_is_judged_not_only_the_flagged_ones() {
    let judge = InvertingJudge::default();
    let c = classify_with(&fixture("mixed.pdf"), &judge).expect("classify mixed.pdf");
    let seen = judge.seen();
    assert_eq!(seen.len() as u32, c.page_count);
    assert_eq!(
        seen.iter().map(|page| page.page).collect::<Vec<_>>(),
        (0..c.page_count).collect::<Vec<_>>()
    );
    assert_eq!(
        seen.iter()
            .map(|page| page.flagged_by_inspector)
            .collect::<Vec<_>>(),
        vec![false, true, false, true]
    );
    // pdf-inspector says why it flagged a page; unflagged pages carry no reasons.
    assert!(!seen[1].reasons.is_empty(), "{:?}", seen[1]);
    assert!(seen[0].reasons.is_empty(), "{:?}", seen[0]);
}

#[test]
fn text_is_populated_only_when_the_judge_asks_for_it() {
    let bytes = fixture("mixed.pdf");

    let silent = InvertingJudge::default();
    classify_with(&bytes, &silent).expect("classify without text");
    assert!(
        silent.seen().iter().all(|page| page.text.is_none()),
        "a judge that does not want text should not be given any"
    );

    let hungry = InvertingJudge::with_text();
    classify_with(&bytes, &hungry).expect("classify with text");
    let seen = hungry.seen();
    assert!(
        seen.iter().all(|page| page.text.is_some()),
        "every page should carry its extracted text: {:?}",
        seen.iter()
            .map(|page| page.text.is_some())
            .collect::<Vec<_>>()
    );
    let first = seen[0].text.as_deref().expect("page 0 text");
    assert!(
        first.contains("MIXED DOC PAGE ONE"),
        "page 0 text should be page 0's: {first:?}"
    );
}

#[test]
fn a_judge_that_returns_too_few_verdicts_is_an_error() {
    let judge = BrokenJudge(|evidence| {
        evidence
            .iter()
            .skip(1)
            .map(|page| verdict(page.page, true))
            .collect()
    });
    let err = classify_with(&fixture("mixed.pdf"), &judge).expect_err("short verdict list");
    assert!(matches!(err, Error::Judge(_)), "{err}");
    let message = err.to_string();
    assert!(message.contains("broken"), "{message}");
    assert!(message.contains("3 verdict(s) for 4 page(s)"), "{message}");
}

#[test]
fn a_judge_that_returns_too_many_verdicts_is_an_error() {
    let judge = BrokenJudge(|evidence| {
        evidence
            .iter()
            .map(|page| verdict(page.page, true))
            .chain(std::iter::once(verdict(0, true)))
            .collect()
    });
    let err = classify_with(&fixture("mixed.pdf"), &judge).expect_err("long verdict list");
    assert!(matches!(err, Error::Judge(_)), "{err}");
    assert!(
        err.to_string().contains("5 verdict(s) for 4 page(s)"),
        "{err}"
    );
}

#[test]
fn a_judge_that_names_a_page_out_of_range_is_an_error() {
    let judge = BrokenJudge(|evidence| {
        evidence
            .iter()
            .enumerate()
            .map(|(index, page)| verdict(if index == 2 { 99 } else { page.page }, true))
            .collect()
    });
    let err = classify_with(&fixture("mixed.pdf"), &judge).expect_err("page out of range");
    assert!(matches!(err, Error::Judge(_)), "{err}");
    let message = err.to_string();
    assert!(message.contains("page 99"), "{message}");
    assert!(message.contains("4-page document"), "{message}");
}

#[test]
fn a_judge_that_answers_out_of_order_is_an_error() {
    let judge = BrokenJudge(|evidence| {
        evidence
            .iter()
            .rev()
            .map(|page| verdict(page.page, true))
            .collect()
    });
    let err = classify_with(&fixture("mixed.pdf"), &judge).expect_err("out of order");
    assert!(matches!(err, Error::Judge(_)), "{err}");
    assert!(
        err.to_string().contains("page 3 where page 0 was expected"),
        "{err}"
    );
}

#[test]
fn a_judges_own_failure_surfaces_unchanged() {
    let err = classify_with(&fixture("mixed.pdf"), &FailingJudge).expect_err("failing judge");
    assert!(matches!(err, Error::Judge(_)), "{err}");
    assert!(
        err.to_string().contains("no answer from the classifier"),
        "{err}"
    );
}

#[test]
fn a_judge_never_sees_bytes_that_are_not_a_pdf() {
    let judge = InvertingJudge::default();
    let err = classify_with(b"hello world", &judge).expect_err("not a pdf");
    assert!(matches!(err, Error::NotPdf), "{err}");
    assert!(judge.seen().is_empty());
}
