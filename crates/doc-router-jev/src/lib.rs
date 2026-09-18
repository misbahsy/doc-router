//! [`JevJudge`]: a [`PageJudge`] backed by TypeSafe's hosted "System One" model.
//!
//! [`HeuristicJudge`] copies pdf-inspector's own answer, which is derived from
//! the document's structure alone. That is exactly the blind spot
//! [`doc_router::judge`] describes: a page whose text layer is *present but
//! lying* — a scan carrying a bad pre-existing OCR layer, a CID font with a
//! broken `ToUnicode` map, a page whose only text is a watermark — reads as fine
//! to any structural check, because structurally it is fine. Deciding that needs
//! someone to look at the text and say whether it means anything, which is what
//! this crate sends the text off to do.
//!
//! # Why this is its own crate
//!
//! For the same reason `LiteLlmHost` lives in the CLI crate: the core
//! `doc-router` crate makes no network calls and has no HTTP dependency, and
//! adding one for a judge would undo that for every consumer, including the ones
//! that never call a hosted judge at all. The trait is the seam; the network
//! lives on this side of it.
//!
//! ```no_run
//! use doc_router::classify_with;
//! use doc_router_jev::{JevJudge, JevMode};
//!
//! // `None` when neither TYPESAFE_API_KEY nor JEV_API_KEY is set.
//! let judge = JevJudge::from_env().expect("an API key").with_mode(JevMode::Gated);
//! let bytes = std::fs::read("scan.pdf")?;
//! let c = classify_with(&bytes, &judge)?;
//! println!("{:?}", c.pages_needing_ocr);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Conventions inherited from the core crate, and not negotiable here:
//!
//! * **Every page number is 0-indexed**, including the `page_<n>` question keys
//!   that go on the wire. Nothing in this crate converts page numbers.
//! * **Blocking HTTP, on purpose.** [`PageJudge::judge`] is a synchronous call
//!   made from wherever the classifier runs; a blocking client is the honest
//!   shape for it, as it is for [`OcrHost`](doc_router::OcrHost).
//!
//! [`PageJudge`]: doc_router::PageJudge
//! [`HeuristicJudge`]: doc_router::HeuristicJudge
//! [`PageJudge::judge`]: doc_router::PageJudge::judge

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod judge;
pub mod wire;

pub use judge::{
    capturing, is_ambiguous, CallLog, CallRecord, JevJudge, JevMode, API_KEY_ENV_VARS,
    BASE_URL_ENV_VAR, DEFAULT_COOL_OFF_SECONDS, DEFAULT_FAILURE_THRESHOLD, DEFAULT_THRESHOLD,
    DEFAULT_TIMEOUT_SECONDS, JUDGE_NAME_ALWAYS, JUDGE_NAME_GATED, MODEL_ENV_VAR, REASON_CLEAR,
    REASON_FALLBACK_BREAKER_OPEN, REASON_FALLBACK_HTTP, REASON_FALLBACK_PROTOCOL,
    REASON_FALLBACK_TIMEOUT, REASON_NEEDS_OCR, REASON_NOT_ESCALATED,
};
pub use wire::{
    DEFAULT_BASE_URL, DEFAULT_MODEL, ENDPOINT_PATH, MAX_PAGES_PER_REQUEST, MAX_REQUEST_BYTES,
    MAX_TEXT_CHARS, QUESTION_KEY_PREFIX,
};
