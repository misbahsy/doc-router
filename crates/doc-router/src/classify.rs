//! Structural classification of a PDF via pdf-inspector.
//!
//! No model is called and nothing leaves the process: pdf-inspector reads the
//! document's own structure (fonts, content streams, image coverage) and reports
//! which pages carry a usable text layer.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::error::Error;
use crate::judge::{HeuristicJudge, PageEvidence, PageJudge, PageVerdict};

/// The PDF magic number every PDF file starts with.
pub const PDF_MAGIC: &[u8] = b"%PDF";

/// What pdf-inspector thinks the document is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PdfType {
    /// Extractable text on every page.
    TextBased,
    /// Scanned images, no text layer.
    Scanned,
    /// Mostly images with minimal text.
    ImageBased,
    /// A mix of text pages and image pages.
    Mixed,
    /// Anything the classifier does not report as one of the above.
    Unknown,
}

impl PdfType {
    /// The wire string for this type.
    pub fn as_str(self) -> &'static str {
        match self {
            PdfType::TextBased => "text_based",
            PdfType::Scanned => "scanned",
            PdfType::ImageBased => "image_based",
            PdfType::Mixed => "mixed",
            PdfType::Unknown => "unknown",
        }
    }

    /// Parse a pdf-inspector type name, ignoring case and underscores, so both
    /// `"TextBased"` (the Python binding's spelling) and `"text_based"` parse.
    pub fn parse_loose(s: &str) -> PdfType {
        let normalized: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect();
        match normalized.as_str() {
            "textbased" => PdfType::TextBased,
            "scanned" => PdfType::Scanned,
            "imagebased" => PdfType::ImageBased,
            "mixed" => PdfType::Mixed,
            _ => PdfType::Unknown,
        }
    }
}

impl std::fmt::Display for PdfType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<pdf_inspector::PdfType> for PdfType {
    // pdf_inspector::PdfType is a closed enum of exactly these four variants, so
    // there is no wildcard arm to write here; `Unknown` exists for values that
    // reach us as strings (golden files, host metadata) rather than as this enum.
    fn from(value: pdf_inspector::PdfType) -> Self {
        match value {
            pdf_inspector::PdfType::TextBased => PdfType::TextBased,
            pdf_inspector::PdfType::Scanned => PdfType::Scanned,
            pdf_inspector::PdfType::ImageBased => PdfType::ImageBased,
            pdf_inspector::PdfType::Mixed => PdfType::Mixed,
        }
    }
}

/// Why pdf-inspector flagged one page for OCR.
///
/// This is doc-router's own 0-indexed mirror of `pdf_inspector::PageOcrReasons`
/// (which is 1-indexed and not serialisable), normalised by
/// [`normalize_ocr_reasons`] exactly like [`normalize_pages`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PageOcrReasons {
    /// 0-indexed page number, `< page_count`.
    pub page: u32,
    /// Machine-readable reason identifiers from pdf-inspector.
    pub reasons: Vec<String>,
}

/// What the structural classifier learned about one PDF.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Classification {
    /// The detected document type.
    pub pdf_type: PdfType,
    /// Detection confidence, 0.0–1.0.
    pub confidence: f32,
    /// Number of pages in the document.
    pub page_count: u32,
    /// 0-indexed pages with no usable text layer: sorted, de-duplicated, `< page_count`.
    pub pages_needing_ocr: Vec<u32>,
    /// True when any page carries tables or multi-column text.
    pub is_complex_layout: bool,
    /// True when pdf-inspector found broken font encodings (garbled text,
    /// replacement characters). Such pages extract as noise and should go to OCR
    /// even when they carry a text layer. False when the analysis pass was
    /// skipped (Scanned/ImageBased) or failed.
    pub has_encoding_issues: bool,
    /// Why each page was flagged, keyed by 0-indexed page. Empty when the
    /// analysis pass was skipped or failed.
    pub ocr_reasons: Vec<PageOcrReasons>,
    /// Wall-clock time spent classifying, in milliseconds.
    pub classify_ms: f64,
}

impl Classification {
    /// True when every page needs OCR (and there is at least one page).
    pub fn needs_ocr_everywhere(&self) -> bool {
        self.page_count > 0 && self.pages_needing_ocr.len() as u32 >= self.page_count
    }

    /// The 0-indexed pages the local extractor can read.
    pub fn text_layer_pages(&self) -> Vec<u32> {
        (0..self.page_count)
            .filter(|page| !self.pages_needing_ocr.contains(page))
            .collect()
    }
}

/// True when `bytes` starts with the PDF magic number.
pub fn is_pdf(bytes: &[u8]) -> bool {
    bytes.starts_with(PDF_MAGIC)
}

/// Coerce pdf-inspector's 1-indexed page list into sorted, de-duplicated,
/// in-range 0-indexed pages. Mirrors `classifier.py::_normalize_pages`.
pub(crate) fn normalize_pages(raw: &[u32], page_count: u32) -> Vec<u32> {
    let mut pages: Vec<u32> = raw
        .iter()
        .filter_map(|page| page.checked_sub(1))
        .filter(|page| *page < page_count)
        .collect();
    pages.sort_unstable();
    pages.dedup();
    pages
}

/// Coerce pdf-inspector's 1-indexed per-page OCR reasons into sorted, in-range
/// 0-indexed [`PageOcrReasons`]. Same guards as [`normalize_pages`]: page 0
/// would underflow and anything at or past `page_count` is dropped.
pub(crate) fn normalize_ocr_reasons(
    raw: &[pdf_inspector::PageOcrReasons],
    page_count: u32,
) -> Vec<PageOcrReasons> {
    let mut reasons: Vec<PageOcrReasons> = raw
        .iter()
        .filter_map(|entry| {
            entry.page.checked_sub(1).map(|page| PageOcrReasons {
                page,
                reasons: entry.reasons.clone(),
            })
        })
        .filter(|entry| entry.page < page_count)
        .collect();
    reasons.sort_by_key(|entry| entry.page);
    reasons
}

/// Derive the document-level label from the per-page OCR list.
///
/// pdf-inspector decides `pdf_type` from a document-level text-page *ratio* and
/// then, for `TextBased`, throws the per-page list away unconditionally
/// (`detector.rs`: `PdfType::TextBased => Vec::new()`). That erases facts it had
/// already measured: a 24-page document with an image on every third page is 67%
/// text pages, clears the ratio threshold, and reports an empty
/// `pages_needing_ocr` — so every image page routes to the local extractor and
/// comes back blank.
///
/// [`classify`] therefore asks pdf-inspector for the page list (with the erasure
/// disabled, see the `text_page_ratio_threshold` note there) and derives the
/// label from it here. A label is a summary of the pages; the pages are not a
/// consequence of the label.
///
/// * no page needs OCR -> `TextBased`.
/// * every page needs OCR -> keep `reported`. `Scanned` vs `ImageBased` turns on
///   image coverage, not on page counts, so it cannot be re-derived here.
/// * some pages need OCR -> `Mixed`.
/// * no pages at all -> keep `reported`; there is nothing to derive from.
pub(crate) fn label_from_pages(
    reported: PdfType,
    pages_needing_ocr: &[u32],
    page_count: u32,
) -> PdfType {
    if page_count == 0 {
        reported
    } else if pages_needing_ocr.is_empty() {
        PdfType::TextBased
    } else if pages_needing_ocr.len() as u32 >= page_count {
        reported
    } else {
        PdfType::Mixed
    }
}

/// Classify raw PDF bytes with the default page judge.
///
/// This is [`classify_with`] with [`HeuristicJudge`], i.e. with pdf-inspector's
/// own `pages_needing_ocr` taken as the answer. It is the behaviour the router
/// has always had and the baseline every other judge is compared against.
///
/// Errors only when the bytes cannot be parsed as a PDF.
pub fn classify(bytes: &[u8]) -> Result<Classification, Error> {
    classify_with(bytes, &HeuristicJudge)
}

/// Classify raw PDF bytes.
///
/// Errors only when the bytes cannot be parsed as a PDF.
///
/// Both calls override pdf-inspector's detection config, in two ways.
///
/// `ScanStrategy::Full` replaces the default `ScanStrategy::Sample(8)`, which
/// looks at up to eight evenly spread pages and extrapolates the whole document
/// from them. On a 24-page document with an image on every third page the
/// sampler lands on seven text pages and one image page, and `pages_needing_ocr`
/// then describes a document that does not exist. Full scans every page, which
/// is the only way that list can be trusted as a routing input, and it is
/// practically free here: the second call re-uses the same per-page analysis, so
/// the Analyze pass dominates either way.
///
/// `text_page_ratio_threshold: 1.0` replaces the default `0.6`. This is not a
/// tuned value, it is the value that disables a step. pdf-inspector labels a
/// document `TextBased` once the share of pages carrying text clears the
/// threshold, and then discards `pages_needing_ocr` for exactly that label
/// (`detector.rs`: `PdfType::TextBased => Vec::new()`) -- a document-level
/// summary erasing per-page facts the same pass had already measured. At `1.0`
/// the erasing branch is reachable only when literally every page has text, i.e.
/// only when the list it clears would have been empty anyway, so pdf-inspector
/// always hands back the real list. doc-router then derives its own label from
/// that list in [`label_from_pages`]. `min_text_ops_per_page` stays at its
/// default: what counts as a page with text is pdf-inspector's judgement and is
/// not being overridden here, only what is done with the answer.
///
/// Two pdf-inspector calls, deliberately:
///
/// * `ProcessMode::DetectOnly` supplies `confidence`, `page_count` and
///   `pages_needing_ocr`, plus the reported type that [`label_from_pages`]
///   refines. This is the same pass the Python reference's
///   `pdf_inspector.detect_pdf_bytes` runs (modulo the detection config above),
///   and the fuller modes can *change* those fields (extraction can re-flag
///   pages as garbled and downgrade `Mixed` to `Scanned`), so they are taken
///   from detect.
/// * `is_complex`, `has_encoding_issues` and `ocr_reasons_by_page` are **not**
///   populated by detect mode: `process_document` returns
///   `LayoutComplexity::default()` and exits before layout analysis. The cheapest
///   mode that does compute them is `ProcessMode::Analyze`, which runs detection
///   plus extraction plus layout analysis but skips markdown rendering
///   (`extract_pages_markdown_mem` would additionally render every page's
///   markdown). That second call is skipped for `Scanned`/`ImageBased`
///   documents, where pdf-inspector itself short-circuits to the default layout;
///   those get `has_encoding_issues: false` and no reasons. It carries the same
///   `DetectionConfig`, or its own detection pass would go back to sampling
///   eight pages and the layout verdict would be drawn from a different document
///   than the one detect looked at.
///
/// The per-page decision itself is `judge`'s, not pdf-inspector's: detection and
/// analysis gather the evidence, [`PageJudge::judge`] rules on it, and
/// [`label_from_pages`] then summarises whatever page list came back. A third
/// pdf-inspector call, `extract_pages_markdown_mem`, is made only when the judge
/// reports [`PageJudge::needs_text`]; [`HeuristicJudge`] does not, so the default
/// path is the two calls described above and nothing more.
pub fn classify_with(bytes: &[u8], judge: &dyn PageJudge) -> Result<Classification, Error> {
    let start = Instant::now();
    if !is_pdf(bytes) {
        return Err(Error::NotPdf);
    }
    let detection = pdf_inspector::DetectionConfig {
        strategy: pdf_inspector::ScanStrategy::Full,
        text_page_ratio_threshold: 1.0,
        ..Default::default()
    };
    let detected = pdf_inspector::process_pdf_mem_with_options(
        bytes,
        pdf_inspector::PdfOptions::detect_only().detection(detection.clone()),
    )
    .map_err(|e| Error::Pdf(e.to_string()))?;
    let page_count = detected.page_count;
    let reported = PdfType::from(detected.pdf_type);
    let inspector_pages = normalize_pages(&detected.pages_needing_ocr, page_count);

    // Whether the Analyze pass runs has to be settled before the judge speaks,
    // so it turns on the label the page list *arrives* with, not the one it
    // leaves with. For `HeuristicJudge` the two are the same value; for a judge
    // that moves pages, this keeps the second pdf-inspector call exactly where
    // it is today rather than making it depend on the judge's answer.
    let analysis = match label_from_pages(reported, &inspector_pages, page_count) {
        PdfType::Scanned | PdfType::ImageBased => Analysis::default(),
        _ => analyze(bytes, detection, page_count),
    };

    let texts = if judge.needs_text() {
        page_texts(bytes, page_count)
    } else {
        vec![None; page_count as usize]
    };

    let evidence: Vec<PageEvidence> = texts
        .into_iter()
        .enumerate()
        .map(|(index, text)| {
            let page = index as u32;
            PageEvidence {
                page,
                text,
                reasons: analysis
                    .ocr_reasons
                    .binary_search_by_key(&page, |entry| entry.page)
                    .map(|found| analysis.ocr_reasons[found].reasons.clone())
                    .unwrap_or_default(),
                flagged_by_inspector: inspector_pages.binary_search(&page).is_ok(),
                has_tables: analysis.pages_with_tables.binary_search(&page).is_ok(),
                has_columns: analysis.pages_with_columns.binary_search(&page).is_ok(),
                has_encoding_issues: analysis.has_encoding_issues,
            }
        })
        .collect();

    let verdicts = judge.judge(&evidence)?;
    let pages_needing_ocr = pages_from_verdicts(judge.name(), &evidence, &verdicts, page_count)?;
    // The label is a summary of the post-judge page list, so a judge that moves
    // pages moves the label with them.
    let pdf_type = label_from_pages(reported, &pages_needing_ocr, page_count);

    Ok(Classification {
        pdf_type,
        confidence: if detected.confidence.is_finite() {
            detected.confidence
        } else {
            0.0
        },
        page_count,
        pages_needing_ocr,
        is_complex_layout: analysis.is_complex,
        has_encoding_issues: analysis.has_encoding_issues,
        ocr_reasons: analysis.ocr_reasons,
        classify_ms: start.elapsed().as_secs_f64() * 1000.0,
    })
}

/// What the Analyze pass adds on top of detection, all of it already converted
/// to this crate's 0-indexed, sorted, in-range form.
///
/// The default is what a skipped or failed Analyze pass yields: no complexity,
/// no encoding issues, no reasons, no per-page layout. It is a struct rather
/// than a tuple because the judge needs five things out of that pass and a
/// five-tuple stops being readable.
#[derive(Debug, Default)]
struct Analysis {
    is_complex: bool,
    has_encoding_issues: bool,
    ocr_reasons: Vec<PageOcrReasons>,
    pages_with_tables: Vec<u32>,
    pages_with_columns: Vec<u32>,
}

/// Run the Analyze pass, or return [`Analysis::default`] if it fails.
///
/// A failure here is not fatal: detection already answered the questions the
/// router cannot do without, and everything this pass adds has a defined
/// "not known" value.
fn analyze(bytes: &[u8], detection: pdf_inspector::DetectionConfig, page_count: u32) -> Analysis {
    pdf_inspector::process_pdf_mem_with_options(
        bytes,
        pdf_inspector::PdfOptions::new()
            .mode(pdf_inspector::ProcessMode::Analyze)
            .detection(detection),
    )
    .map(|analyzed| Analysis {
        is_complex: analyzed.layout.is_complex,
        has_encoding_issues: analyzed.has_encoding_issues,
        ocr_reasons: normalize_ocr_reasons(&analyzed.ocr_reasons_by_page, page_count),
        pages_with_tables: normalize_pages(&analyzed.layout.pages_with_tables, page_count),
        pages_with_columns: normalize_pages(&analyzed.layout.pages_with_columns, page_count),
    })
    .unwrap_or_default()
}

/// Extract every page's text for a judge that asked for it.
///
/// Returns exactly `page_count` entries so it can be zipped with the page
/// numbers. A page pdf-inspector did not return, and a whole extraction that
/// failed, both come back as `None`: text is an input a judge may or may not
/// get, never one it can assume.
fn page_texts(bytes: &[u8], page_count: u32) -> Vec<Option<String>> {
    let mut texts = vec![None; page_count as usize];
    let Ok(extracted) = pdf_inspector::extract_pages_markdown_mem(bytes, None) else {
        return texts;
    };
    for page in extracted.pages {
        // `PageMarkdown::page` is 0-indexed, unlike the layout page lists.
        if let Some(slot) = texts.get_mut(page.page as usize) {
            *slot = Some(page.markdown);
        }
    }
    texts
}

/// Turn a judge's verdicts into a `pages_needing_ocr` list, or say why they are
/// not usable.
///
/// The contract is one verdict per page in page order, and it is checked rather
/// than assumed: a short list would silently drop pages from the document and a
/// stray page number would route bytes that do not exist. Both are the judge's
/// bug, and both should read as such.
fn pages_from_verdicts(
    judge: &str,
    evidence: &[PageEvidence],
    verdicts: &[PageVerdict],
    page_count: u32,
) -> Result<Vec<u32>, Error> {
    if verdicts.len() != evidence.len() {
        return Err(Error::Judge(format!(
            "judge `{judge}` returned {} verdict(s) for {} page(s)",
            verdicts.len(),
            evidence.len()
        )));
    }
    for (verdict, page) in verdicts.iter().zip(evidence) {
        if verdict.page >= page_count {
            return Err(Error::Judge(format!(
                "judge `{judge}` returned a verdict for page {} of a {page_count}-page document",
                verdict.page
            )));
        }
        if verdict.page != page.page {
            return Err(Error::Judge(format!(
                "judge `{judge}` returned a verdict for page {} where page {} was expected",
                verdict.page, page.page
            )));
        }
    }
    let mut pages: Vec<u32> = verdicts
        .iter()
        .filter(|verdict| verdict.needs_ocr)
        .map(|verdict| verdict.page)
        .collect();
    pages.sort_unstable();
    pages.dedup();
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classification(page_count: u32, pages_needing_ocr: Vec<u32>) -> Classification {
        Classification {
            pdf_type: PdfType::Mixed,
            confidence: 0.9,
            page_count,
            pages_needing_ocr,
            is_complex_layout: false,
            has_encoding_issues: false,
            ocr_reasons: Vec::new(),
            classify_ms: 0.0,
        }
    }

    #[test]
    fn is_pdf_checks_the_magic_number() {
        assert!(is_pdf(b"%PDF-1.7\n..."));
        assert!(!is_pdf(b"PDF-1.7"));
        assert!(!is_pdf(b""));
        assert!(!is_pdf(b"%PD"));
        assert!(!is_pdf(b"not a pdf at all"));
    }

    #[test]
    fn normalize_shifts_sorts_and_dedupes() {
        assert_eq!(normalize_pages(&[3, 1, 2, 1], 5), vec![0, 1, 2]);
    }

    #[test]
    fn normalize_drops_out_of_range_pages() {
        // 0 would underflow to -1 in the reference; 9 is past the end.
        assert_eq!(normalize_pages(&[0, 1, 9], 3), vec![0]);
        assert_eq!(normalize_pages(&[1, 2, 3], 0), Vec::<u32>::new());
        assert_eq!(normalize_pages(&[], 4), Vec::<u32>::new());
    }

    fn raw_reasons(pages: &[(u32, &str)]) -> Vec<pdf_inspector::PageOcrReasons> {
        pages
            .iter()
            .map(|(page, reason)| pdf_inspector::PageOcrReasons {
                page: *page,
                reasons: vec![(*reason).to_string()],
            })
            .collect()
    }

    fn ours(page: u32, reason: &str) -> PageOcrReasons {
        PageOcrReasons {
            page,
            reasons: vec![reason.to_string()],
        }
    }

    #[test]
    fn normalize_reasons_shifts_and_sorts() {
        assert_eq!(
            normalize_ocr_reasons(&raw_reasons(&[(3, "scanned"), (1, "no_text")]), 5),
            vec![ours(0, "no_text"), ours(2, "scanned")]
        );
    }

    #[test]
    fn normalize_reasons_drops_out_of_range_pages() {
        // 0 would underflow to -1 in the reference; 9 is past the end.
        assert_eq!(
            normalize_ocr_reasons(
                &raw_reasons(&[(0, "scanned"), (1, "no_text"), (9, "scanned")]),
                3
            ),
            vec![ours(0, "no_text")]
        );
        assert_eq!(
            normalize_ocr_reasons(&raw_reasons(&[(1, "scanned")]), 0),
            Vec::<PageOcrReasons>::new()
        );
        assert_eq!(normalize_ocr_reasons(&[], 4), Vec::<PageOcrReasons>::new());
    }

    #[test]
    fn label_from_pages_calls_an_empty_ocr_list_text_based() {
        // Whatever pdf-inspector reported, nothing needs OCR, so nothing is scanned.
        assert_eq!(label_from_pages(PdfType::Mixed, &[], 5), PdfType::TextBased);
        assert_eq!(
            label_from_pages(PdfType::TextBased, &[], 5),
            PdfType::TextBased
        );
    }

    #[test]
    fn label_from_pages_keeps_the_reported_label_when_every_page_needs_ocr() {
        // Scanned vs ImageBased is a distinction about image coverage that the
        // page list cannot reproduce, so it is passed through untouched.
        assert_eq!(
            label_from_pages(PdfType::Scanned, &[0, 1, 2], 3),
            PdfType::Scanned
        );
        assert_eq!(
            label_from_pages(PdfType::ImageBased, &[0, 1, 2], 3),
            PdfType::ImageBased
        );
    }

    #[test]
    fn label_from_pages_calls_a_partial_ocr_list_mixed() {
        // The case pdf-inspector gets wrong: a majority-text document that still
        // has image pages is Mixed, not TextBased.
        assert_eq!(
            label_from_pages(PdfType::TextBased, &[2, 5, 8], 24),
            PdfType::Mixed
        );
        assert_eq!(label_from_pages(PdfType::Scanned, &[0], 2), PdfType::Mixed);
    }

    #[test]
    fn label_from_pages_keeps_the_reported_label_for_an_empty_document() {
        // No pages, nothing to derive from; an empty PDF is not text-based.
        assert_eq!(label_from_pages(PdfType::Scanned, &[], 0), PdfType::Scanned);
    }

    #[test]
    fn needs_ocr_everywhere_requires_pages() {
        assert!(classification(2, vec![0, 1]).needs_ocr_everywhere());
        assert!(!classification(2, vec![1]).needs_ocr_everywhere());
        assert!(!classification(0, vec![]).needs_ocr_everywhere());
    }

    #[test]
    fn text_layer_pages_is_the_complement() {
        assert_eq!(classification(4, vec![1, 3]).text_layer_pages(), vec![0, 2]);
        assert_eq!(classification(3, vec![]).text_layer_pages(), vec![0, 1, 2]);
        assert_eq!(
            classification(2, vec![0, 1]).text_layer_pages(),
            Vec::<u32>::new()
        );
    }

    #[test]
    fn pdf_type_serialises_snake_case() {
        for (value, wire) in [
            (PdfType::TextBased, "\"text_based\""),
            (PdfType::Scanned, "\"scanned\""),
            (PdfType::ImageBased, "\"image_based\""),
            (PdfType::Mixed, "\"mixed\""),
            (PdfType::Unknown, "\"unknown\""),
        ] {
            assert_eq!(serde_json::to_string(&value).unwrap(), wire);
            assert_eq!(
                serde_json::from_str::<PdfType>(wire).unwrap(),
                value,
                "{wire}"
            );
        }
    }

    #[test]
    fn parse_loose_ignores_case_and_underscores() {
        assert_eq!(PdfType::parse_loose("TextBased"), PdfType::TextBased);
        assert_eq!(PdfType::parse_loose("text_based"), PdfType::TextBased);
        assert_eq!(PdfType::parse_loose("IMAGE_BASED"), PdfType::ImageBased);
        assert_eq!(PdfType::parse_loose("something else"), PdfType::Unknown);
    }

    #[test]
    fn classify_rejects_non_pdf_bytes() {
        assert!(matches!(classify(b"hello world"), Err(Error::NotPdf)));
    }

    #[test]
    fn classify_reports_a_pdf_error_for_broken_pdfs() {
        let err = classify(b"%PDF-1.7\nbut not really").expect_err("broken pdf");
        assert!(matches!(err, Error::Pdf(_)), "{err}");
    }
}
