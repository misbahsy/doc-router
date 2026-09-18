//! Scoring a [`PageJudge`] against a corpus whose per-page truth is known.
//!
//! The harness exists to answer one question about any judge, including ones
//! that do not exist yet: *if this judge decided routing, which pages would come
//! out wrong, and which kind of wrong?* The two kinds never get added together:
//!
//! - **missed OCR** -- a page that needed OCR did not get it. The document
//!   routes as text, the text is not there, and nothing downstream notices. A
//!   silent quality failure.
//! - **wasted OCR** -- a page that did not need OCR got it. The answer is still
//!   right; it cost money and latency.
//!
//! Everything else in here (precision, recall, F1, cost, reliability) is derived
//! from those counts and is reported after them.
//!
//! Adding a judge is one line in [`registry::judge_by_name`], which this crate
//! shares with the `doc-router` CLI.
//!
//! [`PageJudge`]: doc_router::PageJudge

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bench;
pub mod corpus;
pub mod dotenv;
pub mod meter;
pub mod registry;
pub mod report;
pub mod score;
