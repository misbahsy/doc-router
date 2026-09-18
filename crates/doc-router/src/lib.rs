//! Per-page routing between local PDF text extraction and paid OCR models.
//!
//! `doc-router` looks at a PDF's structure (never its meaning: no model is called, no
//! bytes leave the process), decides which pages have a usable text layer and which need
//! OCR, extracts the former in-process and hands the latter to a host, then merges both
//! back into one page-ordered result.
//!
//! ```no_run
//! use doc_router::{decide, Config, Decision, Tiers};
//!
//! let cfg = Config::new(Tiers::new("local_pdf/extract", "mistral-ocr"));
//! let bytes = std::fs::read("scan.pdf")?;
//! match decide(&bytes, &cfg, None) {
//!     Decision::Route { plan, .. } => println!("{} leg(s), {}", plan.legs.len(), plan.reason),
//!     Decision::Bypass { reason, model, .. } => println!("bypass {reason} -> {model}"),
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Conventions:
//!
//! * **Every page number in this crate is 0-indexed.** pdf-inspector reports some page
//!   lists 1-indexed; that is normalised exactly once, in [`classify`].
//! * **The core makes no network calls and has no async.** Fetching documents and calling
//!   OCR providers is the host's job, behind [`OcrHost`].
//! * **Deterministic**: the same bytes and config always produce the same [`Plan`].

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod classify;
pub mod config;
pub mod error;
pub mod extract;
pub mod judge;
pub mod merge;
pub mod metadata;
pub mod policy;
pub mod run;
pub mod split;

pub use classify::{
    classify, classify_with, is_pdf, Classification, PageOcrReasons, PdfType, PDF_MAGIC,
};
pub use config::{
    Config, OcrTier, Tier, Tiers, DEFAULT_FETCH_TIMEOUT_SECONDS, DEFAULT_MAX_DOCUMENT_BYTES,
    DEFAULT_MIN_CONFIDENCE,
};
pub use error::Error;
pub use extract::{extract_local, OcrResult, Page, LOCAL_MODEL};
pub use judge::{HeuristicJudge, PageEvidence, PageJudge, PageVerdict};
pub use merge::merge;
pub use metadata::{LegSummary, RouteMetadata, BYPASS_TIER, FALLBACK_LEG_FAILED};
pub use policy::{decide, decide_with, plan_route, Bypass, Decision, Leg, Plan, Reason};
pub use run::{run, run_with, OcrHost, Outcome};
pub use split::{remap_pages, split_pdf};
