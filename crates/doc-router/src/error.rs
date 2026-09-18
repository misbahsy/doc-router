//! The one error type the library returns.

/// Everything that can go wrong inside the core library.
///
/// The library never performs I/O of its own, so every variant is either a
/// configuration mistake, a malformed document, or a failure reported by the
/// host that executes the OCR legs.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The `doc_router_config` block is not usable as written.
    #[error("invalid doc_router config: {0}")]
    InvalidConfig(String),
    /// The bytes handed in do not start with the `%PDF` magic number.
    #[error("document is not a PDF")]
    NotPdf,
    /// pdf-inspector could not read the document.
    #[error("PDF error: {0}")]
    Pdf(String),
    /// The document could not be split into a page subset.
    #[error("PDF split failed: {0}")]
    Split(String),
    /// The host reported a failure while running a leg.
    #[error("OCR host error: {0}")]
    Host(String),
    /// A [`PageJudge`](crate::judge::PageJudge) failed, or broke the
    /// one-verdict-per-page-in-order contract
    /// [`classify_with`](crate::classify_with) checks.
    #[error("page judge error: {0}")]
    Judge(String),
    /// One leg of a multi-leg plan failed.
    #[error("leg for model {model} failed: {source}")]
    LegFailed {
        /// The model the failed leg was routed to.
        model: String,
        /// The underlying failure.
        #[source]
        source: Box<Error>,
    },
}
