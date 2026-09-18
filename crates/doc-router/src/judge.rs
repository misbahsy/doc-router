//! Who decides which pages need OCR.
//!
//! [`classify`](crate::classify) used to take pdf-inspector's `pages_needing_ocr`
//! and route on it directly. That list is a good default and a bad ceiling: it is
//! derived from the document's structure alone, so a page whose text layer is
//! present but useless (a scan with a bad OCR layer already baked in, a form
//! whose values live in an image) reads as fine. Making the decision a named
//! component lets a better-informed one be swapped in without touching the
//! classifier around it.
//!
//! The shape is deliberately the same as [`OcrHost`](crate::run::OcrHost): a
//! `Sync` trait in this crate, one in-process implementation here, and any
//! implementation that needs the network living in the CLI crate. This module
//! adds no dependencies; the core still makes no network calls.
//!
//! ```no_run
//! use doc_router::{classify_with, HeuristicJudge};
//!
//! let bytes = std::fs::read("scan.pdf")?;
//! // `classify(&bytes)` is exactly this call.
//! let c = classify_with(&bytes, &HeuristicJudge)?;
//! println!("{:?}", c.pages_needing_ocr);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Everything known about one page before deciding whether it needs OCR.
///
/// Every page of the document gets one of these, in page order, whether or not
/// pdf-inspector flagged it. Page numbers are 0-indexed like everywhere else in
/// this crate.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageEvidence {
    /// 0-indexed page number.
    pub page: u32,
    /// The page's extracted text, when the judge asked for it. `None` when the
    /// judge does not need text (see [`PageJudge::needs_text`]) or extraction
    /// failed.
    pub text: Option<String>,
    /// pdf-inspector's machine-readable reasons for flagging this page. Empty
    /// when it did not flag the page, and empty for every page when the analysis
    /// pass was skipped (`Scanned`/`ImageBased`) or failed.
    pub reasons: Vec<String>,
    /// True when pdf-inspector's own page list contains this page.
    pub flagged_by_inspector: bool,
    /// True when this page has detected table borders.
    pub has_tables: bool,
    /// True when this page has 2+ detected text columns.
    pub has_columns: bool,
    /// Document-level: broken font encodings were detected somewhere. Not
    /// per-page, because pdf-inspector does not report it per page.
    pub has_encoding_issues: bool,
}

/// One judge's decision about one page.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageVerdict {
    /// 0-indexed page number. Must match the [`PageEvidence::page`] it answers.
    pub page: u32,
    /// True when this page should be routed to OCR rather than local extraction.
    pub needs_ocr: bool,
    /// 0.0–1.0. [`HeuristicJudge`] has no real confidence to report and always
    /// returns 1.0 — see its doc comment.
    pub confidence: f32,
    /// Why, in machine-readable form.
    pub reason: String,
}

/// The boundary between the classifier and whatever decides a page needs OCR.
///
/// Implementations must be `Sync` for the same reason [`OcrHost`](crate::run::OcrHost)
/// is: nothing here promises to stay on one thread.
pub trait PageJudge: Sync {
    /// Decide, for each page, whether it needs OCR. Verdicts must come back one
    /// per input page, in the same order; [`classify_with`](crate::classify_with)
    /// rejects anything else rather than guessing which page a verdict meant.
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error>;

    /// A stable identifier for provenance and benchmarking, e.g. `"heuristic"`.
    fn name(&self) -> &'static str;

    /// True when this judge needs [`PageEvidence::text`] populated. Filling it in
    /// costs a whole extra pdf-inspector extraction pass, so the default is
    /// false and the default path costs exactly what it costs today.
    fn needs_text(&self) -> bool {
        false
    }
}

/// The reason recorded for a page pdf-inspector flagged but gave no reason for.
pub const REASON_FLAGGED: &str = "inspector_flagged";

/// The reason recorded for a page pdf-inspector did not flag.
pub const REASON_CLEAR: &str = "inspector_clear";

/// The judge that reproduces pdf-inspector's own answer, unchanged.
///
/// `needs_ocr` is [`PageEvidence::flagged_by_inspector`] and nothing else: this
/// is what the router did before judges existed, and it is the baseline any
/// other judge is measured against.
///
/// # Why `confidence` is always 1.0
///
/// pdf-inspector does report a `confidence`, but it is not a calibrated
/// probability — it is a lookup on whichever branch of the classifier fired.
/// `detector.rs:310-333` hands back a literal 0.95 for `Scanned`, 0.8 for
/// `ImageBased`, 0.7 for `Mixed`, and the document's text-page ratio for
/// `TextBased`. It is also a single document-level number, so even taken at face
/// value there is nothing in it that describes one page. Forwarding it would
/// mean presenting a constant chosen by a branch as a per-page belief, and
/// synthesising one from the flags would be worse still. So this judge says 1.0:
/// it has no uncertainty of its own to report, because it is copying an answer
/// rather than forming one. A hosted judge that computes a real posterior can
/// report it, and the two will then be comparable in a way they would not be if
/// this one had invented a number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HeuristicJudge;

impl PageJudge for HeuristicJudge {
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        Ok(evidence
            .iter()
            .map(|page| PageVerdict {
                page: page.page,
                needs_ocr: page.flagged_by_inspector,
                confidence: 1.0,
                reason: if page.flagged_by_inspector {
                    page.reasons
                        .first()
                        .cloned()
                        .unwrap_or_else(|| REASON_FLAGGED.to_string())
                } else {
                    REASON_CLEAR.to_string()
                },
            })
            .collect())
    }

    fn name(&self) -> &'static str {
        "heuristic"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn evidence(page: u32, flagged: bool, reasons: &[&str]) -> PageEvidence {
        PageEvidence {
            page,
            text: None,
            reasons: reasons.iter().map(|r| (*r).to_string()).collect(),
            flagged_by_inspector: flagged,
            has_tables: false,
            has_columns: false,
            has_encoding_issues: false,
        }
    }

    fn judged(pages: &[PageEvidence]) -> Vec<PageVerdict> {
        HeuristicJudge.judge(pages).expect("heuristic never fails")
    }

    #[test]
    fn heuristic_names_itself() {
        assert_eq!(HeuristicJudge.name(), "heuristic");
        assert!(!HeuristicJudge.needs_text());
    }

    #[test]
    fn heuristic_flags_every_page_when_the_inspector_did() {
        let pages = vec![
            evidence(0, true, &["scanned_page"]),
            evidence(1, true, &["no_text_layer"]),
        ];
        let verdicts = judged(&pages);
        assert_eq!(verdicts.len(), 2);
        assert_eq!(
            verdicts.iter().map(|v| v.page).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(verdicts.iter().all(|v| v.needs_ocr));
        assert_eq!(verdicts[0].reason, "scanned_page");
        assert_eq!(verdicts[1].reason, "no_text_layer");
        assert!(verdicts.iter().all(|v| v.confidence == 1.0));
    }

    #[test]
    fn heuristic_flags_no_page_when_the_inspector_did_not() {
        let pages = vec![evidence(0, false, &[]), evidence(1, false, &[])];
        let verdicts = judged(&pages);
        assert_eq!(verdicts.len(), 2);
        assert!(verdicts.iter().all(|v| !v.needs_ocr));
        assert!(verdicts.iter().all(|v| v.reason == REASON_CLEAR));
        assert!(verdicts.iter().all(|v| v.confidence == 1.0));
    }

    #[test]
    fn heuristic_returns_one_verdict_per_page_in_order_for_a_partial_flagging() {
        let pages = vec![
            evidence(0, false, &[]),
            evidence(1, true, &["image_only"]),
            evidence(2, false, &[]),
            evidence(3, true, &[]),
        ];
        let verdicts = judged(&pages);
        assert_eq!(
            verdicts.iter().map(|v| v.page).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            verdicts.iter().map(|v| v.needs_ocr).collect::<Vec<_>>(),
            vec![false, true, false, true]
        );
        // A flagged page with no reason still gets one, so `reason` is never empty.
        assert_eq!(verdicts[1].reason, "image_only");
        assert_eq!(verdicts[3].reason, REASON_FLAGGED);
        assert_eq!(verdicts[0].reason, REASON_CLEAR);
    }

    #[test]
    fn heuristic_returns_nothing_for_no_pages() {
        assert_eq!(judged(&[]), Vec::<PageVerdict>::new());
    }

    #[test]
    fn evidence_and_verdicts_round_trip_through_json() {
        let page = PageEvidence {
            text: Some("hello".to_string()),
            ..evidence(2, true, &["garbled_text"])
        };
        let wire = serde_json::to_string(&page).expect("serialise evidence");
        assert_eq!(
            serde_json::from_str::<PageEvidence>(&wire).expect("deserialise evidence"),
            page
        );

        let verdict = judged(std::slice::from_ref(&page)).remove(0);
        let wire = serde_json::to_string(&verdict).expect("serialise verdict");
        assert_eq!(
            serde_json::from_str::<PageVerdict>(&wire).expect("deserialise verdict"),
            verdict
        );
    }
}
