//! The ground-truth corpus manifest: what each document really is, page by page.
//!
//! Ground truth is **per page, not per document**. A document-level label
//! (`scanned`, `mixed`) cannot say whether page 7 in particular needs OCR, and
//! per-page routing is the thing being scored, so the manifest records the page
//! list and the harness derives everything else from it.
//!
//! # Page numbering
//!
//! `needs_ocr` is **0-indexed**, exactly like
//! [`Classification::pages_needing_ocr`](doc_router::Classification::pages_needing_ocr)
//! and everything else in `doc-router`. No conversion happens anywhere in this
//! crate: an off-by-one here would produce plausible-looking scores that mean
//! nothing, which is the single most dangerous failure mode in the harness.
//!
//! # Paths
//!
//! `file` is resolved **relative to the manifest's own directory**, so the corpus
//! directory can be moved without editing it and a second manifest can point at
//! real documents that live outside the repository. An absolute `file` is used
//! as-is.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

/// The manifest file's on-disk shape.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    /// Every document in the corpus, in report order.
    pub documents: Vec<DocumentSpec>,
}

/// One row of the manifest: one document and its hand-labelled truth.
///
/// Unknown fields are rejected rather than ignored. A typo in `needs_ocr` would
/// otherwise default to "no page needs OCR" and score every judge against a
/// truth nobody wrote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentSpec {
    /// Path to the PDF, relative to the manifest's directory (or absolute).
    pub file: String,
    /// Free text describing where the document came from, e.g. `"synthetic"` or
    /// `"real"`. Used to group the report, because the two kinds mean very
    /// different things: see the caveat the harness prints for synthetic groups.
    pub source: String,
    /// How many pages the document has. Checked against what the classifier
    /// reports; a mismatch fails the document rather than scoring it against the
    /// wrong page numbering.
    pub page_count: u32,
    /// The **0-indexed** pages that truly need OCR. Sorted, de-duplicated, every
    /// entry `< page_count`.
    pub needs_ocr: Vec<u32>,
    /// Why this document is in the corpus / what it is labelled the way it is.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// One manifest row with its path resolved and its truth validated.
#[derive(Debug, Clone)]
pub struct CorpusDocument {
    /// The row as written.
    pub spec: DocumentSpec,
    /// `spec.file` resolved against the manifest's directory.
    pub path: PathBuf,
    /// `spec.needs_ocr`, validated: sorted, de-duplicated, in range.
    pub truth: Vec<u32>,
}

impl CorpusDocument {
    /// A short name for the report: the file's base name, or the raw `file`
    /// string when it has none.
    pub fn label(&self) -> String {
        self.path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.spec.file.clone())
    }
}

/// A loaded, validated corpus.
#[derive(Debug, Clone)]
pub struct Corpus {
    /// Where the manifest was read from.
    pub manifest_path: PathBuf,
    /// The documents, in manifest order.
    pub documents: Vec<CorpusDocument>,
}

impl Corpus {
    /// Read and validate a manifest.
    pub fn load(path: &Path) -> Result<Corpus> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("could not read corpus manifest {}", path.display()))?;
        Corpus::parse(path, &text)
    }

    /// Validate an already-read manifest. `path` is only used to resolve the
    /// documents' relative paths and to name the file in errors.
    pub fn parse(path: &Path, text: &str) -> Result<Corpus> {
        let manifest: Manifest = serde_json::from_str(text)
            .with_context(|| format!("could not parse corpus manifest {}", path.display()))?;
        let dir = path.parent().unwrap_or_else(|| Path::new("."));
        let documents = manifest
            .documents
            .into_iter()
            .map(|spec| {
                let truth = validated_truth(&spec)
                    .with_context(|| format!("in manifest entry `{}`", spec.file))?;
                Ok(CorpusDocument {
                    path: resolve(dir, &spec.file),
                    spec,
                    truth,
                })
            })
            .collect::<Result<Vec<_>>>()
            .with_context(|| format!("in corpus manifest {}", path.display()))?;
        if documents.is_empty() {
            bail!("corpus manifest {} lists no documents", path.display());
        }
        Ok(Corpus {
            manifest_path: path.to_path_buf(),
            documents,
        })
    }

    /// The distinct `source` values, in first-seen order.
    pub fn sources(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for doc in &self.documents {
            if !seen.contains(&doc.spec.source) {
                seen.push(doc.spec.source.clone());
            }
        }
        seen
    }
}

/// Resolve one `file` entry against the manifest's directory.
fn resolve(dir: &Path, file: &str) -> PathBuf {
    let candidate = Path::new(file);
    if candidate.is_absolute() {
        candidate.to_path_buf()
    } else {
        dir.join(candidate)
    }
}

/// Check a row's `needs_ocr` and return it sorted and de-duplicated.
///
/// Out-of-range and duplicate pages are manifest bugs, not data to be cleaned up
/// quietly: both mean the person who wrote the row believed something about the
/// document's page numbering that is not true.
fn validated_truth(spec: &DocumentSpec) -> Result<Vec<u32>> {
    let mut pages = spec.needs_ocr.clone();
    pages.sort_unstable();
    let before = pages.len();
    pages.dedup();
    if pages.len() != before {
        bail!("needs_ocr lists the same page twice: {:?}", spec.needs_ocr);
    }
    if let Some(page) = pages.iter().find(|page| **page >= spec.page_count) {
        bail!(
            "needs_ocr contains page {page} but page_count is {} \
             (pages are 0-indexed, so the last page is {})",
            spec.page_count,
            spec.page_count.saturating_sub(1)
        );
    }
    Ok(pages)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest_text(rows: &str) -> String {
        format!("{{\"documents\": [{rows}]}}")
    }

    const ROW: &str = r#"{"file": "../fixtures/mixed.pdf", "source": "synthetic",
        "page_count": 4, "needs_ocr": [3, 1], "note": "t,i,t,i"}"#;

    #[test]
    fn parses_a_row_and_resolves_the_path_against_the_manifest_directory() {
        let corpus = Corpus::parse(
            Path::new("/repo/tests/corpus/manifest.json"),
            &manifest_text(ROW),
        )
        .expect("valid manifest");
        assert_eq!(corpus.documents.len(), 1);
        let doc = &corpus.documents[0];
        assert_eq!(
            doc.path,
            Path::new("/repo/tests/corpus/../fixtures/mixed.pdf")
        );
        assert_eq!(doc.spec.page_count, 4);
        // Sorted, and left 0-indexed exactly as written.
        assert_eq!(doc.truth, vec![1, 3]);
        assert_eq!(doc.spec.source, "synthetic");
        assert_eq!(doc.label(), "mixed.pdf");
    }

    #[test]
    fn an_absolute_file_is_used_as_written() {
        let row = r#"{"file": "/elsewhere/real/invoice.pdf", "source": "real",
            "page_count": 1, "needs_ocr": [0]}"#;
        let corpus = Corpus::parse(Path::new("corpus/manifest.json"), &manifest_text(row))
            .expect("valid manifest");
        assert_eq!(
            corpus.documents[0].path,
            Path::new("/elsewhere/real/invoice.pdf")
        );
    }

    #[test]
    fn a_zero_page_document_with_no_ocr_pages_is_valid() {
        let row = r#"{"file": "../fixtures/empty.pdf", "source": "synthetic",
            "page_count": 0, "needs_ocr": []}"#;
        let corpus = Corpus::parse(Path::new("m.json"), &manifest_text(row)).expect("valid");
        assert!(corpus.documents[0].truth.is_empty());
    }

    #[test]
    fn a_page_past_the_end_is_a_manifest_error() {
        let row = r#"{"file": "a.pdf", "source": "real", "page_count": 3, "needs_ocr": [3]}"#;
        let err =
            Corpus::parse(Path::new("m.json"), &manifest_text(row)).expect_err("out of range");
        let text = format!("{err:#}");
        assert!(text.contains("page 3"), "{text}");
        assert!(text.contains("0-indexed"), "{text}");
    }

    #[test]
    fn a_repeated_page_is_a_manifest_error() {
        let row = r#"{"file": "a.pdf", "source": "real", "page_count": 3, "needs_ocr": [1, 1]}"#;
        let err = Corpus::parse(Path::new("m.json"), &manifest_text(row)).expect_err("duplicate");
        assert!(format!("{err:#}").contains("twice"));
    }

    #[test]
    fn an_unknown_field_is_rejected_rather_than_ignored() {
        // The failure this guards: `need_ocr` would deserialise to an empty
        // `needs_ocr` and score every judge against a truth nobody wrote.
        let row = r#"{"file": "a.pdf", "source": "real", "page_count": 3, "needs_ocr": [],
            "need_ocr": [1]}"#;
        let err = Corpus::parse(Path::new("m.json"), &manifest_text(row)).expect_err("typo");
        assert!(format!("{err:#}").contains("need_ocr"));
    }

    #[test]
    fn an_empty_manifest_is_rejected() {
        let err = Corpus::parse(Path::new("m.json"), &manifest_text("")).expect_err("no documents");
        assert!(format!("{err:#}").contains("no documents"));
    }

    #[test]
    fn sources_are_listed_once_in_first_seen_order() {
        let rows = format!(
            "{ROW}, {}, {}",
            r#"{"file": "b.pdf", "source": "real", "page_count": 1, "needs_ocr": []}"#,
            r#"{"file": "c.pdf", "source": "synthetic", "page_count": 1, "needs_ocr": []}"#
        );
        let corpus = Corpus::parse(Path::new("m.json"), &manifest_text(&rows)).expect("valid");
        assert_eq!(corpus.sources(), vec!["synthetic", "real"]);
    }
}
