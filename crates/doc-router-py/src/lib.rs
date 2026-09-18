//! Python bindings for [`doc_router`].
//!
//! The strategy is deliberately boring: every public core type derives
//! `Serialize`/`Deserialize`, so the boundary is "plain Python dicts in, plain Python
//! dicts out", converted with `pythonize` through `serde_json::Value`. Nothing in this
//! crate re-implements policy; it only translates.
//!
//! Every entry point releases the GIL (`Python::detach`) around the Rust work, and the
//! [`PyHost`] adapter re-acquires it (`Python::attach`) inside `OcrHost::ocr`, which the
//! core calls from scoped worker threads.

use doc_router::{
    classify as core_classify, decide as core_decide, extract_local as core_extract_local,
    is_pdf as core_is_pdf, merge as core_merge, plan_route as core_plan_route,
    remap_pages as core_remap_pages, run as core_run, split_pdf as core_split_pdf, Classification,
    Config, Decision, Error, Leg, OcrHost, OcrResult, PdfType, RouteMetadata,
    DEFAULT_FETCH_TIMEOUT_SECONDS, DEFAULT_MAX_DOCUMENT_BYTES, DEFAULT_MIN_CONFIDENCE, LOCAL_MODEL,
};
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pythonize::{depythonize, pythonize};
use serde_json::{json, Map, Value};

create_exception!(
    _native,
    DocRouterError,
    PyException,
    "Base class for every error raised by doc_router."
);
create_exception!(
    _native,
    PdfError,
    DocRouterError,
    "The bytes are not a readable PDF, or a PDF operation failed."
);
create_exception!(
    _native,
    ConfigError,
    DocRouterError,
    "The doc_router_config block is not usable as written."
);
create_exception!(
    _native,
    HostError,
    DocRouterError,
    "The OCR host callable failed, or returned something unusable."
);

// ---------------------------------------------------------------------------
// error mapping
// ---------------------------------------------------------------------------

/// Which Python exception class a core error maps to.
enum Kind {
    Config,
    Pdf,
    Host,
}

fn kind_of(err: &Error) -> Kind {
    match err {
        Error::InvalidConfig(_) => Kind::Config,
        Error::NotPdf | Error::Pdf(_) | Error::Split(_) => Kind::Pdf,
        // A judge is an external decision-maker like a host, and its failures
        // arrive the same way: something outside the router did not cooperate.
        Error::Host(_) | Error::Judge(_) => Kind::Host,
        // A leg failure is reported as whatever actually went wrong underneath it, so a
        // local extraction failure stays a PdfError and a provider failure stays a HostError.
        Error::LegFailed { source, .. } => kind_of(source),
    }
}

fn error_to_py(err: &Error) -> PyErr {
    // `Error`'s Display already folds the `LegFailed` source into its own message.
    let message = err.to_string();
    match kind_of(err) {
        Kind::Config => ConfigError::new_err(message),
        Kind::Pdf => PdfError::new_err(message),
        Kind::Host => HostError::new_err(message),
    }
}

fn format_py_exception(py: Python<'_>, err: &PyErr) -> String {
    match err.value(py).str() {
        Ok(text) => {
            let text = text.to_string_lossy().into_owned();
            if text.is_empty() {
                err.to_string()
            } else {
                text
            }
        }
        Err(_) => err.to_string(),
    }
}

// ---------------------------------------------------------------------------
// JSON <-> Python plumbing
// ---------------------------------------------------------------------------

fn json_to_py(py: Python<'_>, value: &Value) -> PyResult<Py<PyAny>> {
    pythonize(py, value)
        .map(|bound| bound.unbind())
        .map_err(|e| DocRouterError::new_err(format!("could not build the Python result: {e}")))
}

fn py_to_json(obj: &Bound<'_, PyAny>, what: &str) -> PyResult<Value> {
    depythonize(obj)
        .map_err(|e| DocRouterError::new_err(format!("{what} is not a JSON-compatible value: {e}")))
}

fn to_json<T: serde::Serialize>(value: &T, what: &str) -> PyResult<Value> {
    serde_json::to_value(value)
        .map_err(|e| DocRouterError::new_err(format!("could not serialise {what}: {e}")))
}

/// A `f32` as the shortest `f64` that round-trips its decimal form.
///
/// `serde_json` widens `0.6f32` to `0.6000000238418579`; Python users expect `0.6`, and
/// the golden parity comparison is tolerance-based, so the shorter value is strictly
/// nicer on both sides.
fn f32_json(value: f32) -> Value {
    let short: f64 = format!("{value}").parse().unwrap_or(value as f64);
    serde_json::Number::from_f64(short)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn set_f32(map: &mut Map<String, Value>, key: &str, value: f32) {
    map.insert(key.to_string(), f32_json(value));
}

fn object_mut<'a>(value: &'a mut Value, what: &str) -> PyResult<&'a mut Map<String, Value>> {
    value
        .as_object_mut()
        .ok_or_else(|| DocRouterError::new_err(format!("{what} must be a dict")))
}

// ---------------------------------------------------------------------------
// core types -> JSON
// ---------------------------------------------------------------------------

fn classification_to_json(classification: &Classification) -> PyResult<Value> {
    let mut value = to_json(classification, "classification")?;
    let map = object_mut(&mut value, "classification")?;
    set_f32(map, "confidence", classification.confidence);
    Ok(value)
}

fn config_to_json(config: &Config) -> PyResult<Value> {
    let mut value = to_json(config, "config")?;
    let map = object_mut(&mut value, "config")?;
    set_f32(map, "min_confidence", config.min_confidence);
    set_f32(map, "fetch_timeout_seconds", config.fetch_timeout_seconds);
    Ok(value)
}

fn metadata_to_json(metadata: &RouteMetadata) -> PyResult<Value> {
    let mut value = to_json(metadata, "metadata")?;
    let map = object_mut(&mut value, "metadata")?;
    if let Some(confidence) = metadata.confidence {
        set_f32(map, "confidence", confidence);
    }
    Ok(value)
}

/// A [`Decision`] flattened into the documented Python shape.
///
/// The Rust enum is externally tagged (`{"route": {...}}`), which is awkward to consume;
/// Python instead gets a `"kind"` discriminant with the variant's fields alongside it.
fn decision_to_json(decision: &Decision) -> PyResult<Value> {
    let mut map = Map::new();
    match decision {
        Decision::Bypass {
            reason,
            model,
            detail,
        } => {
            map.insert("kind".to_string(), json!("bypass"));
            map.insert("reason".to_string(), json!(reason.as_str()));
            map.insert("model".to_string(), json!(model));
            map.insert(
                "detail".to_string(),
                detail.clone().map(Value::String).unwrap_or(Value::Null),
            );
        }
        Decision::Route {
            plan,
            classification,
        } => {
            map.insert("kind".to_string(), json!("route"));
            map.insert("plan".to_string(), to_json(plan, "plan")?);
            map.insert(
                "classification".to_string(),
                classification_to_json(classification)?,
            );
        }
    }
    Ok(Value::Object(map))
}

fn ocr_result_to_json(result: &OcrResult) -> PyResult<Value> {
    to_json(result, "OCR result")
}

// ---------------------------------------------------------------------------
// Python -> core types
// ---------------------------------------------------------------------------

fn config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Config> {
    let value: Value = depythonize(obj)
        .map_err(|e| ConfigError::new_err(format!("config is not a JSON-compatible dict: {e}")))?;
    let config: Config = serde_json::from_value(value)
        .map_err(|e| ConfigError::new_err(format!("invalid doc_router config: {e}")))?;
    config.validate().map_err(|e| error_to_py(&e))?;
    Ok(config)
}

fn classification_from_py(obj: &Bound<'_, PyAny>) -> PyResult<Classification> {
    let mut value = py_to_json(obj, "classification")?;
    {
        let map = object_mut(&mut value, "classification")?;
        // Accept pdf-inspector's Python spelling ("TextBased") as well as the wire form.
        if let Some(Value::String(raw)) = map.get("pdf_type") {
            let normalised = PdfType::parse_loose(raw).as_str();
            map.insert("pdf_type".to_string(), json!(normalised));
        }
        // Everything the caller can reasonably omit. A classification recorded by
        // LiteLLM's Python implementation, or hand-written to drive `plan_route`,
        // carries the four fields the decision table actually reads; the rest are
        // provenance. Each field the core gains has to be added here too, or a
        // previously valid dict stops deserialising.
        map.entry("classify_ms").or_insert_with(|| json!(0.0));
        map.entry("is_complex_layout")
            .or_insert_with(|| json!(false));
        map.entry("has_encoding_issues")
            .or_insert_with(|| json!(false));
        map.entry("ocr_reasons").or_insert_with(|| json!([]));
    }
    serde_json::from_value(value)
        .map_err(|e| DocRouterError::new_err(format!("invalid classification: {e}")))
}

/// Build an [`OcrResult`] from a dict, filling in everything a caller can reasonably omit.
///
/// `model` defaults to `default_model`, `pages_processed` to the number of pages, each
/// page's `model` to the result's model and `index` to its position in the list.
fn ocr_result_from_value(
    mut value: Value,
    default_model: &str,
    what: &str,
) -> Result<OcrResult, String> {
    let map = value
        .as_object_mut()
        .ok_or_else(|| format!("{what} must be a dict"))?;
    let page_count = map
        .get("pages")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    map.entry("pages").or_insert_with(|| json!([]));
    map.entry("model").or_insert_with(|| json!(default_model));
    map.entry("pages_processed")
        .or_insert_with(|| json!(page_count));
    map.entry("doc_size_bytes").or_insert(Value::Null);
    let model = map
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or(default_model)
        .to_string();
    if let Some(Value::Array(pages)) = map.get_mut("pages") {
        for (position, page) in pages.iter_mut().enumerate() {
            let Some(page) = page.as_object_mut() else {
                return Err(format!("{what}: every entry of `pages` must be a dict"));
            };
            page.entry("index").or_insert_with(|| json!(position));
            page.entry("markdown").or_insert_with(|| json!(""));
            page.entry("model").or_insert_with(|| json!(model));
        }
    }
    serde_json::from_value(value).map_err(|e| format!("{what}: {e}"))
}

fn ocr_result_from_py(
    obj: &Bound<'_, PyAny>,
    default_model: &str,
    what: &str,
) -> PyResult<OcrResult> {
    let value = py_to_json(obj, what)?;
    ocr_result_from_value(value, default_model, what).map_err(DocRouterError::new_err)
}

fn leg_from_value(mut value: Value) -> Result<Leg, String> {
    let map = value
        .as_object_mut()
        .ok_or_else(|| "every leg must be a dict".to_string())?;
    map.entry("pages").or_insert(Value::Null);
    if !map.contains_key("tier") {
        let local = map.get("model").and_then(Value::as_str) == Some(LOCAL_MODEL);
        map.insert(
            "tier".to_string(),
            json!(if local { "local" } else { "standard" }),
        );
    }
    serde_json::from_value(value).map_err(|e| format!("invalid leg: {e}"))
}

// ---------------------------------------------------------------------------
// the host adapter
// ---------------------------------------------------------------------------

/// Wraps a Python callable `(model, document, pages) -> dict` as an [`OcrHost`].
///
/// `Py<PyAny>` is `Send + Sync`, so this is safe to share across the scoped threads the
/// core spawns. The GIL is *not* held when `ocr` is entered (the caller released it
/// before entering the core), so `ocr` re-acquires it with `Python::attach`.
struct PyHost {
    callable: Py<PyAny>,
}

impl OcrHost for PyHost {
    fn ocr(&self, model: &str, document: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error> {
        Python::attach(|py| {
            let document = PyBytes::new(py, document);
            let pages: Py<PyAny> = match pages {
                Some(pages) => pythonize(py, &pages.to_vec())
                    .map_err(|e| Error::Host(format!("could not pass `pages` to the host: {e}")))?
                    .unbind(),
                None => py.None(),
            };
            let returned = self
                .callable
                .call1(py, (model, document, pages))
                .map_err(|e| Error::Host(format_py_exception(py, &e)))?;
            let value: Value = depythonize(returned.bind(py)).map_err(|e| {
                Error::Host(format!(
                    "host returned a value that is not a JSON-compatible dict: {e}"
                ))
            })?;
            ocr_result_from_value(value, model, "host result").map_err(Error::Host)
        })
    }
}

// ---------------------------------------------------------------------------
// module functions
// ---------------------------------------------------------------------------

/// classify(data)
///
/// Classify PDF bytes structurally: no model is called and nothing leaves the process.
///
/// Returns a dict with ``pdf_type`` ("text_based" | "scanned" | "image_based" | "mixed"
/// | "unknown"), ``confidence``, ``page_count``, ``pages_needing_ocr`` (0-indexed,
/// sorted, de-duplicated), ``is_complex_layout``, ``has_encoding_issues``,
/// ``ocr_reasons`` (per-page ``{"page", "reasons"}``) and ``classify_ms``.
///
/// Raises ``PdfError`` when the bytes cannot be parsed as a PDF.
#[pyfunction]
fn classify(py: Python<'_>, data: &[u8]) -> PyResult<Py<PyAny>> {
    let bytes = data.to_vec();
    let classification = py
        .detach(|| core_classify(&bytes))
        .map_err(|e| error_to_py(&e))?;
    json_to_py(py, &classification_to_json(&classification)?)
}

/// is_pdf(data)
///
/// True when ``data`` starts with the ``%PDF`` magic number.
#[pyfunction]
fn is_pdf(py: Python<'_>, data: &[u8]) -> bool {
    // Only the magic number matters, so copy the head rather than the whole document.
    let head: Vec<u8> = data.iter().copied().take(8).collect();
    py.detach(|| core_is_pdf(&head))
}

/// plan_route(classification, config)
///
/// Apply the decision table to an already-computed classification.
///
/// ``classification`` is a dict shaped like :func:`classify`'s return value
/// (``classify_ms``, ``is_complex_layout``, ``has_encoding_issues`` and ``ocr_reasons``
/// may be omitted, and ``pdf_type`` may use pdf-inspector's Python spelling, e.g.
/// ``"TextBased"``). Returns a Plan dict:
/// ``{"legs": [{"model", "tier", "pages"}], "reason", "tier", "routed_model"}``.
///
/// Raises ``ConfigError`` for an unusable config.
#[pyfunction]
fn plan_route(
    py: Python<'_>,
    classification: &Bound<'_, PyAny>,
    config: &Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    let classification = classification_from_py(classification)?;
    let config = config_from_py(config)?;
    let plan = py
        .detach(|| core_plan_route(&classification, &config))
        .map_err(|e| error_to_py(&e))?;
    json_to_py(py, &to_json(&plan, "plan")?)
}

/// decide(data, config, default_model=None)
///
/// Classify and plan in one call. Never raises for unroutable input: a non-PDF, an
/// oversized document or a classifier failure comes back as a bypass.
///
/// The Rust ``Decision`` enum is externally tagged; Python gets it flattened, with a
/// ``"kind"`` discriminant:
///
/// * route  -> ``{"kind": "route", "plan": {...}, "classification": {...},
///   "metadata": {...}}``
/// * bypass -> ``{"kind": "bypass", "reason": "not_pdf" | "oversize" | "image" |
///   "classifier_unavailable", "model": str, "detail": str | None, "metadata": {...}}``
///
/// ``metadata`` is the ``doc_route`` provenance dict (same keys as the LiteLLM Python
/// router): ``tier``, ``reason``, ``routed_model`` always, plus ``pdf_type``,
/// ``confidence``, ``page_count``, ``pages_needing_ocr``, ``classify_ms``, ``legs`` and
/// ``split`` for a route.
///
/// Raises ``ConfigError`` for an unusable config.
#[pyfunction]
#[pyo3(signature = (data, config, default_model=None))]
fn decide(
    py: Python<'_>,
    data: &[u8],
    config: &Bound<'_, PyAny>,
    default_model: Option<String>,
) -> PyResult<Py<PyAny>> {
    let config = config_from_py(config)?;
    let bytes = data.to_vec();
    let decision = py.detach(|| core_decide(&bytes, &config, default_model.as_deref()));
    let mut value = decision_to_json(&decision)?;
    object_mut(&mut value, "decision")?.insert(
        "metadata".to_string(),
        metadata_to_json(&decision.metadata())?,
    );
    json_to_py(py, &value)
}

/// extract_local(data, pages=None)
///
/// Read the PDF's own text layer. ``pages`` is a 0-indexed subset, or None for every
/// page. Returns an OcrResult dict:
/// ``{"pages": [{"index", "markdown", "model"}], "model", "pages_processed",
/// "doc_size_bytes"}``. ``model`` is ``"local_pdf/extract"``.
///
/// Raises ``PdfError`` when the bytes cannot be read.
#[pyfunction]
#[pyo3(signature = (data, pages=None))]
fn extract_local(py: Python<'_>, data: &[u8], pages: Option<Vec<u32>>) -> PyResult<Py<PyAny>> {
    let bytes = data.to_vec();
    let result = py
        .detach(|| core_extract_local(&bytes, pages.as_deref()))
        .map_err(|e| error_to_py(&e))?;
    json_to_py(py, &ocr_result_to_json(&result)?)
}

/// split_pdf(data, pages)
///
/// Build a new PDF holding exactly ``pages`` (0-indexed, in the order given), for OCR
/// providers that cannot take a page list. Returns bytes.
///
/// Raises ``PdfError`` when the document cannot be read or a page is out of range.
#[pyfunction]
fn split_pdf<'py>(py: Python<'py>, data: &[u8], pages: Vec<u32>) -> PyResult<Bound<'py, PyBytes>> {
    let bytes = data.to_vec();
    let split = py
        .detach(|| core_split_pdf(&bytes, &pages))
        .map_err(|e| error_to_py(&e))?;
    Ok(PyBytes::new(py, &split))
}

/// remap_pages(result, pages)
///
/// Map the 0..n page indices a provider returned for a split PDF back onto the original
/// page numbers in ``pages``. Returns a new OcrResult dict; the input is not modified.
#[pyfunction]
fn remap_pages(py: Python<'_>, result: &Bound<'_, PyAny>, pages: Vec<u32>) -> PyResult<Py<PyAny>> {
    let mut result = ocr_result_from_py(result, "", "result")?;
    py.detach(|| core_remap_pages(&mut result, &pages));
    json_to_py(py, &ocr_result_to_json(&result)?)
}

/// merge(legs)
///
/// Reassemble per-leg results into one page-ordered result. ``legs`` is a list of
/// ``(leg, result)`` pairs, where ``leg`` is a Plan leg dict and ``result`` an OcrResult
/// dict. Pages are stably sorted by ``index``, each page's ``model`` is overwritten with
/// its leg's model, ``model`` is the leg models joined by ``","``, ``pages_processed`` is
/// summed and ``doc_size_bytes`` is the largest any leg reported.
#[pyfunction]
fn merge(py: Python<'_>, legs: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let raw: Vec<(Value, Value)> = depythonize(legs).map_err(|e| {
        DocRouterError::new_err(format!(
            "legs must be a sequence of (leg, result) pairs: {e}"
        ))
    })?;
    let mut pairs: Vec<(Leg, OcrResult)> = Vec::with_capacity(raw.len());
    for (leg, result) in raw {
        let leg = leg_from_value(leg).map_err(DocRouterError::new_err)?;
        let result = ocr_result_from_value(result, &leg.model, "leg result")
            .map_err(DocRouterError::new_err)?;
        pairs.push((leg, result));
    }
    let merged = py.detach(|| core_merge(&pairs));
    json_to_py(py, &ocr_result_to_json(&merged)?)
}

/// run(data, config, host, default_model=None)
///
/// Decide and execute in one call. Local legs run in-process; OCR legs are handed to
/// ``host``, a callable ``(model: str, document: bytes, pages: list[int] | None) ->
/// dict`` returning an OcrResult-shaped dict (``model``, ``pages_processed``,
/// ``doc_size_bytes`` and each page's ``model``/``index`` may be omitted).
///
/// Legs run concurrently on scoped threads, so ``host`` may be called from a thread other
/// than the caller's; the GIL is re-acquired around each call. An exception raised by
/// ``host`` fails that leg, which triggers one whole-document retry on ``default_model``
/// (metadata gains ``fallback_reason = "leg_failed"``); if that retry also fails,
/// ``HostError`` is raised.
///
/// Returns ``{"result": {...OcrResult...}, "metadata": {...}, "decision": {...}}``, where
/// ``decision`` has the same flattened shape :func:`decide` returns, without the nested
/// ``metadata`` key (the top-level ``metadata`` is authoritative and may carry
/// ``fallback_reason``).
#[pyfunction]
#[pyo3(signature = (data, config, host, default_model=None))]
fn run(
    py: Python<'_>,
    data: &[u8],
    config: &Bound<'_, PyAny>,
    host: &Bound<'_, PyAny>,
    default_model: Option<String>,
) -> PyResult<Py<PyAny>> {
    let config = config_from_py(config)?;
    if !host.is_callable() {
        return Err(HostError::new_err(
            "host must be a callable (model, document, pages) -> dict",
        ));
    }
    let host = PyHost {
        callable: host.clone().unbind(),
    };
    let bytes = data.to_vec();
    // The GIL must be released here: the core runs legs on scoped threads that call back
    // into `PyHost::ocr`, which re-acquires it. Holding it across `core_run` deadlocks.
    let outcome = py
        .detach(|| core_run(&bytes, &config, default_model.as_deref(), &host))
        .map_err(|e| error_to_py(&e))?;

    let mut map = Map::new();
    map.insert("result".to_string(), ocr_result_to_json(&outcome.result)?);
    map.insert("metadata".to_string(), metadata_to_json(&outcome.metadata)?);
    map.insert("decision".to_string(), decision_to_json(&outcome.decision)?);
    json_to_py(py, &Value::Object(map))
}

/// validate_config(config)
///
/// Parse and validate a ``doc_router_config`` block, returning it normalised with every
/// default filled in. Unknown keys are rejected.
///
/// Raises ``ConfigError`` when the block is unusable.
#[pyfunction]
fn validate_config(py: Python<'_>, config: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let config = config_from_py(config)?;
    json_to_py(py, &config_to_json(&config)?)
}

fn defaults() -> Value {
    json!({
        "min_confidence": f32_json(DEFAULT_MIN_CONFIDENCE),
        "complex_layout_tier": "premium",
        "max_document_bytes": DEFAULT_MAX_DOCUMENT_BYTES,
        "fetch_timeout_seconds": f32_json(DEFAULT_FETCH_TIMEOUT_SECONDS),
        "split_pages": true,
        "local_model": LOCAL_MODEL,
    })
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("LOCAL_MODEL", LOCAL_MODEL)?;
    m.add("DEFAULTS", json_to_py(py, &defaults())?)?;

    m.add("DocRouterError", py.get_type::<DocRouterError>())?;
    m.add("PdfError", py.get_type::<PdfError>())?;
    m.add("ConfigError", py.get_type::<ConfigError>())?;
    m.add("HostError", py.get_type::<HostError>())?;

    m.add_function(wrap_pyfunction!(classify, m)?)?;
    m.add_function(wrap_pyfunction!(is_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(plan_route, m)?)?;
    m.add_function(wrap_pyfunction!(decide, m)?)?;
    m.add_function(wrap_pyfunction!(extract_local, m)?)?;
    m.add_function(wrap_pyfunction!(split_pdf, m)?)?;
    m.add_function(wrap_pyfunction!(remap_pages, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(validate_config, m)?)?;
    Ok(())
}
