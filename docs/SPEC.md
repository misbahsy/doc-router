# doc-router — standalone Rust document router for OCR

Status: v0 spec. This file is the contract every crate in the workspace codes against.

## What it is

A Rust library (plus CLI and Python bindings) that decides, per page, whether a PDF
needs a paid OCR model or can be read locally from its text layer, executes the local
part in-process, hands the OCR part to a *host* (LiteLLM today, anything with an
`/ocr`-shaped API), and merges the results back into one response in page order.

It is the Rust port of the Python reference implementation in
`litellm/router_strategy/doc_router/` and `litellm/llms/local_pdf/` in a LiteLLM
checkout. Behaviour must match the reference exactly (same decision table, same metadata, same page conventions).
The Python files are the spec of record for anything this document leaves open.

Design rules:
- **Core makes no network calls.** Fetching documents and calling OCR providers is the
  host's job (CLI, LiteLLM). Core takes bytes and returns plans/results.
- **Everything user-facing is 0-indexed.** pdf-inspector reports
  `PdfProcessResult.pages_needing_ocr` **1-indexed** while
  `extract_pages_markdown_mem(bytes, Some(&pages))` and `PageMarkdown.page` are
  **0-indexed**. Normalise at the boundary, exactly once, in `classify`.
- **Deterministic.** Same bytes + same config = same plan. No LLM anywhere in core.
- **Failure to classify is a bypass, never an error** at the `decide` level: the whole
  document goes to the default model, with a reason.
- Config JSON is byte-for-byte the same shape as the LiteLLM `doc_router_config` block,
  so one YAML serves both.

## Workspace layout

```
doc-router/
  Cargo.toml                 # workspace; members = crates/*
  LICENSE                    # MIT
  README.md                  # the short pitch
  docs/GUIDE.md              # full reference manual
  docs/SPEC.md               # this file
  crates/doc-router/         # core library. package "doc-router", lib name doc_router
  crates/doc-router-cli/     # binary "doc-router" (wave 2)
  crates/doc-router-py/      # PyO3 + maturin, python module "doc_router" (wave 2)
  tests/fixtures/*.pdf       # shared fixtures (text.pdf, scanned.pdf, mixed.pdf, + generated)
  tests/golden/*.json        # parity expectations generated from the Python reference
```

Toolchain: cargo 1.92 at /opt/homebrew/bin/cargo. pdf-inspector 1.17.0 on crates.io (MIT),
`rust-version = 1.88`. Use `pdf-inspector = "1.17"` with **default features** (no OCR
engine, no pdfium). Page splitting uses `lopdf = { version = "0.44", features = ["rayon"] }`
(the same version pdf-inspector pins, so it dedups). Serde + serde_json for every public
type. `thiserror` for errors. No `anyhow` in the library crate.

## Core public API (crate `doc-router`, `use doc_router::*`)

All public structs/enums derive `Debug, Clone, PartialEq, Serialize, Deserialize`
(f32 fields make `Eq` impossible; that is fine). Enums serialise as lowercase /
snake_case strings exactly as listed.

```rust
/// Where a page or a document can go.
pub enum Tier { Local, Standard, Premium }            // "local" | "standard" | "premium"
/// Tiers an OCR leg may use. `Local` is excluded on purpose.
pub enum OcrTier { Standard, Premium }                // "standard" | "premium"

pub struct Tiers {
    pub local: String,                // e.g. "local_pdf/extract"
    pub standard: String,             // e.g. "mistral-ocr"
    pub premium: Option<String>,
}

#[serde(deny_unknown_fields)]
pub struct Config {
    pub tiers: Tiers,
    #[serde(default = 0.6)]     pub min_confidence: f32,          // 0.0..=1.0
    #[serde(default = Premium)] pub complex_layout_tier: OcrTier,
    #[serde(default = 50 MiB)]  pub max_document_bytes: u64,      // > 0
    #[serde(default = 20.0)]    pub fetch_timeout_seconds: f32,   // > 0, host-only knob, kept for config parity
    #[serde(default = true)]    pub split_pages: bool,
}
impl Config {
    pub fn from_json(s: &str) -> Result<Config, Error>;   // parses then validates
    pub fn validate(&self) -> Result<(), Error>;          // ranges above; empty model names rejected
    pub fn resolved_complex_layout_tier(&self) -> OcrTier; // Premium -> Standard when tiers.premium is None
    pub fn model_for(&self, tier: Tier) -> Option<&str>;
}

pub enum PdfType { TextBased, Scanned, ImageBased, Mixed, Unknown } // "text_based" | "scanned" | "image_based" | "mixed" | "unknown"

pub struct Classification {
    pub pdf_type: PdfType,
    pub confidence: f32,
    pub page_count: u32,
    pub pages_needing_ocr: Vec<u32>,   // 0-indexed, sorted, de-duplicated, < page_count
    pub is_complex_layout: bool,
    pub classify_ms: f64,
}
impl Classification {
    pub fn needs_ocr_everywhere(&self) -> bool;          // page_count > 0 && len >= page_count
    pub fn text_layer_pages(&self) -> Vec<u32>;          // complement, 0-indexed
}

/// Structural classification via pdf-inspector. Errors only when the bytes cannot be
/// parsed as a PDF. `is_complex_layout` comes from pdf-inspector's layout analysis;
/// the implementer must check which call populates it (detect-only mode leaves
/// `layout` at its default) and use the cheapest call that does.
pub fn classify(bytes: &[u8]) -> Result<Classification, Error>;
pub fn is_pdf(bytes: &[u8]) -> bool;                     // starts with b"%PDF"

pub struct Leg {
    pub model: String,
    pub tier: Tier,
    pub pages: Option<Vec<u32>>,      // None = whole document; Some = 0-indexed subset
}

pub enum Reason { LowConfidence, TextLayer, Scanned, MixedSplit, MixedUnsplit }
// "low_confidence" | "text_layer" | "scanned" | "mixed_split" | "mixed_unsplit"

pub struct Plan {
    pub legs: Vec<Leg>,
    pub reason: Reason,
    pub tier: Tier,                   // the OCR leg's tier, or Local when local is the only leg
    pub routed_model: String,
}
impl Plan { pub fn is_split(&self) -> bool { self.legs.len() > 1 } }

/// The decision table. Pure. Order (identical to policy.py):
/// 1. confidence < min_confidence            -> one Standard leg, LowConfidence
/// 2. ocr_tier = resolved_complex_layout_tier() if is_complex_layout else Standard
/// 3. no page needs OCR                       -> one Local leg, TextLayer
/// 4. every page needs OCR                    -> one ocr_tier leg, pages=None, Scanned
/// 5. mixed, split_pages=false                -> one ocr_tier leg, pages=None, MixedUnsplit
/// 6. mixed, split_pages=true                 -> [Local leg pages=text pages, ocr leg pages=ocr pages], MixedSplit
///    plan.tier = ocr_tier, plan.routed_model = ocr model.
pub fn plan_route(c: &Classification, cfg: &Config) -> Result<Plan, Error>;

pub enum Bypass { NotPdf, Oversize, Image, ClassifierFailed }
// "not_pdf" | "oversize" | "image" | "classifier_unavailable"   <- note the last string

pub enum Decision {
    Bypass { reason: Bypass, model: String, detail: Option<String> },
    Route  { plan: Plan, classification: Classification },
}

/// Top-level entry. Never errors. `default_model` falls back to `cfg.tiers.standard`.
/// Bypass when: !is_pdf -> NotPdf; len > max_document_bytes -> Oversize; classify() errs -> ClassifierFailed.
/// (`Image` is for hosts that know the input was an image_url; core exposes the variant, the
/// host produces it via `Decision::bypass_image(cfg, default_model)`.)
pub fn decide(bytes: &[u8], cfg: &Config, default_model: Option<&str>) -> Decision;

/// Metadata stamped on the request, same keys/values as the Python `doc_route` dict.
pub struct RouteMetadata {
    pub tier: String,                 // plan.tier or "bypass"
    pub reason: String,
    pub routed_model: String,
    pub pdf_type: Option<String>, pub confidence: Option<f32>, pub page_count: Option<u32>,
    pub pages_needing_ocr: Option<Vec<u32>>, pub classify_ms: Option<f64>,
    pub legs: Option<Vec<LegSummary>>, // {model, tier, pages}
    pub split: Option<bool>,
    pub fallback_reason: Option<String>, // "leg_failed" when the fallback path ran
}
impl Decision { pub fn metadata(&self) -> RouteMetadata; }

/// OCR-response shape (LiteLLM/Mistral OCRResponse subset). `index` is the ORIGINAL 0-indexed page.
pub struct Page { pub index: u32, pub markdown: String, pub model: String }
pub struct OcrResult {
    pub pages: Vec<Page>,
    pub model: String,
    pub pages_processed: u32,
    pub doc_size_bytes: Option<u64>,
}

/// Local text-layer extraction via pdf_inspector::extract_pages_markdown_mem. `pages` None = all.
/// model = "local_pdf/extract". Pages flagged needs_ocr by the extractor are STILL returned
/// (whatever markdown the extractor produced, possibly empty) so page counts stay honest;
/// `Page.index` is the original page number.
pub const LOCAL_MODEL: &str = "local_pdf/extract";
pub fn extract_local(bytes: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error>;

/// Subset PDF (lopdf) containing exactly `pages` (0-indexed, in the given order), for OCR
/// providers that cannot take a page list. Preserves resources/fonts/images per page.
pub fn split_pdf(bytes: &[u8], pages: &[u32]) -> Result<Vec<u8>, Error>;
/// After OCR of a split PDF the provider numbers pages 0..n; map them back to the originals.
pub fn remap_pages(result: &mut OcrResult, pages: &[u32]);

/// Reassemble one result from per-leg results in original page order (stable sort by index,
/// ties keep leg order). model = leg models joined by ",". pages_processed summed,
/// doc_size_bytes = max. Every page's `model` is overwritten with its leg's model.
pub fn merge(legs: &[(Leg, OcrResult)]) -> OcrResult;

/// Host boundary. Implemented by the CLI (reqwest to LiteLLM /v1/ocr) and by tests (mock).
pub trait OcrHost: Sync {
    /// Run `model` over `document` (full bytes). `pages` = 0-indexed subset or None.
    /// If the host's provider cannot take a page list it should call split_pdf/remap_pages itself.
    fn ocr(&self, model: &str, document: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error>;
}

pub struct Outcome { pub result: OcrResult, pub metadata: RouteMetadata, pub decision: Decision }

/// Execute a decision: local legs in-process via extract_local, OCR legs via `host`, legs run
/// concurrently (std::thread::scope is fine). Any leg failure -> rerun the WHOLE document once on
/// the default model via host (metadata.fallback_reason = "leg_failed"); if that fails too, Err.
/// Bypass decisions go straight to host with the bypass model.
pub fn run(bytes: &[u8], cfg: &Config, default_model: Option<&str>, host: &dyn OcrHost) -> Result<Outcome, Error>;

#[derive(thiserror::Error)]
pub enum Error { InvalidConfig(String), NotPdf, Pdf(String), Split(String), Host(String), LegFailed{model, source}, ... }
```

Anything not listed may be added as `pub` if useful, but nothing listed may be renamed.

## Golden files (`tests/golden/`)

Generated from the Python reference by a script kept at `tests/golden/generate.py`
(run with `PYTHONPATH=/path/to/litellm python3`).
One file per fixture: `tests/golden/<fixture-stem>.json`:

```json
{
  "fixture": "mixed.pdf",
  "sha256": "...",
  "classification": {"pdf_type": "...", "confidence": 0.9, "page_count": 3,
                     "pages_needing_ocr": [1], "is_complex_layout": false},
  "local_pages": [{"index": 0, "markdown": "..."}, ...],      // from pdf_inspector.extract_pages_markdown_bytes(data), all pages, 0-indexed
  "plans": [
    {"config": {...doc_router_config JSON...}, "default_model": null,
     "plan": {"legs":[{"model":"..","tier":"..","pages":[..]|null}], "reason":"..", "tier":"..", "routed_model":".."}}
  ]
}
```
`pdf_type` in golden files is pdf-inspector's Python string (e.g. "TextBased"); the Rust
test maps it to `PdfType` case-insensitively ignoring underscores. `confidence` compared
with tolerance 1e-3; `classify_ms` is never compared. The plan configs to cover, for every
fixture: (a) defaults with premium, (b) defaults without premium, (c) split_pages=false,
(d) min_confidence=1.0 (forces low_confidence), (e) complex_layout_tier="standard".

## Non-goals for v0
- No HTTP in core. No async in core (hosts may be async; bindings may spawn).
- No anydoc (office formats) yet; reserved as an optional feature `anydoc` later.
- No pdf-inspector local OCR engine (`ocr` feature) yet; reserved as a future Local-OCR tier.
