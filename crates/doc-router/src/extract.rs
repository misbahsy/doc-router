//! Local text-layer extraction and the OCR-response shape everything merges into.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// The pseudo-model name reported for pages read from the PDF's own text layer.
pub const LOCAL_MODEL: &str = "local_pdf/extract";

/// One page of extracted content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Page {
    /// The page's ORIGINAL 0-indexed position in the source document.
    pub index: u32,
    /// The page's content as markdown.
    pub markdown: String,
    /// The model that produced this page.
    pub model: String,
}

impl Page {
    /// A page produced by `model`.
    pub fn new(index: u32, markdown: impl Into<String>, model: impl Into<String>) -> Self {
        Page {
            index,
            markdown: markdown.into(),
            model: model.into(),
        }
    }
}

/// The subset of an OCR provider's response the router cares about.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OcrResult {
    /// Extracted pages, in original page order once merged.
    pub pages: Vec<Page>,
    /// The model, or comma-joined models, that produced this result.
    pub model: String,
    /// How many pages were processed (summed across legs).
    pub pages_processed: u32,
    /// Size of the source document in bytes, when known.
    pub doc_size_bytes: Option<u64>,
}

impl OcrResult {
    /// An empty result attributed to `model`.
    pub fn empty(model: impl Into<String>) -> Self {
        OcrResult {
            pages: Vec::new(),
            model: model.into(),
            pages_processed: 0,
            doc_size_bytes: None,
        }
    }
}

/// Read `pages` (0-indexed; `None` = every page) out of the PDF's text layer.
///
/// Pages the extractor flags `needs_ocr` are still returned, with whatever markdown it
/// produced (often empty), so downstream page counts stay honest. `Page.index` is the
/// original page number: pdf-inspector's `PageMarkdown.page` is already 0-indexed and
/// already refers to the source document even when a subset was requested.
pub fn extract_local(bytes: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error> {
    if !crate::classify::is_pdf(bytes) {
        return Err(Error::NotPdf);
    }
    let extraction = pdf_inspector::extract_pages_markdown_mem(bytes, pages)
        .map_err(|e| Error::Pdf(e.to_string()))?;
    let out: Vec<Page> = extraction
        .pages
        .into_iter()
        .map(|page| Page::new(page.page, page.markdown, LOCAL_MODEL))
        .collect();
    Ok(OcrResult {
        pages_processed: out.len() as u32,
        pages: out,
        model: LOCAL_MODEL.to_string(),
        doc_size_bytes: Some(bytes.len() as u64),
    })
}
