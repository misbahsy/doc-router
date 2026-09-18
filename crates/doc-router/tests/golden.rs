//! Regression corpus, replayed from the recorded goldens.
//!
//! Each `tests/golden/*.json` file started as a recording of the Python
//! implementation (see `tests/golden/generate.py`). This test re-runs the Rust
//! classifier, local extractor and policy over the same fixture bytes and
//! asserts the results still agree. The goldens are optional: when the
//! directory is missing or empty the test prints a note and passes, so a fresh
//! checkout without a generated corpus is not a failure.
//!
//! These are **no longer pure Python parity**. They are a Rust-owned corpus,
//! and a golden may carry a `"_divergence"` key: a prose note saying that this
//! file was deliberately edited away from the recorded Python answer because
//! the Rust answer is better, and why. `"_divergence"` is documentation for
//! whoever next sees a diff here; nothing below reads it, and re-running
//! `generate.py` would silently revert those files to the Python answer.
//!
//! So a failure here is a question, not a verdict. Read the recorded value
//! against what the Rust side now produces and decide which one is right. If
//! the new answer is right, edit the golden and add or extend its
//! `"_divergence"`. If it is not, the change under test is a regression.

mod common;

use std::path::Path;

use common::{fixture, golden_dir};
use doc_router::{classify, extract_local, is_pdf, plan_route, Config, PdfType};
use serde_json::Value;

/// Confidence is a float that crossed a JSON round trip on both sides.
const CONFIDENCE_TOLERANCE: f64 = 1e-3;

fn golden_files() -> Vec<std::path::PathBuf> {
    let Some(dir) = golden_dir() else {
        return Vec::new();
    };
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut files: Vec<_> = entries
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    files.sort();
    files
}

fn load(path: &Path) -> Value {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_slice(&bytes).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

fn str_field<'a>(value: &'a Value, key: &str, what: &str) -> &'a str {
    value[key]
        .as_str()
        .unwrap_or_else(|| panic!("{what}: `{key}` must be a string, got {}", value[key]))
}

fn u32_list(value: &Value, key: &str, what: &str) -> Vec<u32> {
    value[key]
        .as_array()
        .unwrap_or_else(|| panic!("{what}: `{key}` must be an array, got {}", value[key]))
        .iter()
        .map(|v| {
            v.as_u64()
                .unwrap_or_else(|| panic!("{what}: `{key}` must hold page numbers, got {v}"))
                as u32
        })
        .collect()
}

/// A golden recorded as a bypass: the document never reaches the classifier.
fn check_bypass(name: &str, golden: &Value, bytes: &[u8]) {
    let expected = str_field(golden, "bypass", name);
    assert_eq!(
        expected, "not_pdf",
        "{name}: the only recorded bypass reason is `not_pdf`"
    );
    assert!(
        !is_pdf(bytes),
        "{name}: golden says bypass `{expected}` but `is_pdf` accepted the bytes"
    );
    assert!(
        classify(bytes).is_err(),
        "{name}: golden says bypass `{expected}` but classification succeeded"
    );
    println!("{name}: bypass {expected} (as recorded)");
}

fn check_classification(name: &str, golden: &Value, bytes: &[u8]) -> doc_router::Classification {
    let expected = &golden["classification"];
    let actual = classify(bytes).unwrap_or_else(|e| panic!("{name}: classify: {e}"));

    let expected_type = str_field(expected, "pdf_type", name);
    let parsed = PdfType::parse_loose(expected_type);
    assert_eq!(
        actual.pdf_type, parsed,
        "{name}: pdf_type: golden `{expected_type}` parsed as {parsed}, got {}",
        actual.pdf_type
    );

    let expected_pages = expected["page_count"]
        .as_u64()
        .unwrap_or_else(|| panic!("{name}: page_count must be a number"))
        as u32;
    assert_eq!(actual.page_count, expected_pages, "{name}: page_count");

    assert_eq!(
        actual.pages_needing_ocr,
        u32_list(expected, "pages_needing_ocr", name),
        "{name}: pages_needing_ocr (0-indexed)"
    );

    let expected_complex = expected["is_complex_layout"]
        .as_bool()
        .unwrap_or_else(|| panic!("{name}: is_complex_layout must be a bool"));
    assert_eq!(
        actual.is_complex_layout, expected_complex,
        "{name}: is_complex_layout"
    );

    let expected_confidence = expected["confidence"]
        .as_f64()
        .unwrap_or_else(|| panic!("{name}: confidence must be a number"));
    let delta = (actual.confidence as f64 - expected_confidence).abs();
    assert!(
        delta <= CONFIDENCE_TOLERANCE,
        "{name}: confidence {} differs from golden {expected_confidence} by {delta}",
        actual.confidence
    );

    println!(
        "{name}: {} conf={:.4} pages={} ocr={:?} complex={}",
        actual.pdf_type,
        actual.confidence,
        actual.page_count,
        actual.pages_needing_ocr,
        actual.is_complex_layout
    );
    actual
}

fn check_local_pages(name: &str, golden: &Value, bytes: &[u8]) {
    let Some(expected_pages) = golden["local_pages"].as_array() else {
        return;
    };
    if expected_pages.is_empty() {
        return;
    }
    let wanted: Vec<u32> = expected_pages
        .iter()
        .map(|page| {
            page["index"]
                .as_u64()
                .unwrap_or_else(|| panic!("{name}: local_pages[].index must be a number"))
                as u32
        })
        .collect();

    let actual = extract_local(bytes, Some(&wanted))
        .unwrap_or_else(|e| panic!("{name}: extract_local({wanted:?}): {e}"));
    assert_eq!(
        actual.pages.len(),
        expected_pages.len(),
        "{name}: local page count"
    );
    for (expected, actual) in expected_pages.iter().zip(&actual.pages) {
        let index = expected["index"].as_u64().unwrap_or_default() as u32;
        assert_eq!(actual.index, index, "{name}: local page order");
        assert_eq!(
            actual.markdown,
            str_field(expected, "markdown", name),
            "{name}: markdown for page {index}"
        );
    }
}

fn check_plans(name: &str, golden: &Value, classification: &doc_router::Classification) {
    let Some(plans) = golden["plans"].as_array() else {
        panic!("{name}: golden has no `plans` array");
    };
    for (i, entry) in plans.iter().enumerate() {
        let label = format!("{name}: plans[{i}]");
        let config_json = serde_json::to_string(&entry["config"])
            .unwrap_or_else(|e| panic!("{label}: re-serialize config: {e}"));
        let config = Config::from_json(&config_json)
            .unwrap_or_else(|e| panic!("{label}: Config::from_json({config_json}): {e}"));

        let plan = plan_route(classification, &config)
            .unwrap_or_else(|e| panic!("{label}: plan_route: {e}"));
        let actual = serde_json::to_value(&plan).unwrap_or_else(|e| panic!("{label}: {e}"));
        let expected = &entry["plan"];

        // Compare the parts individually first: the messages are far more
        // useful than a whole-object diff when something drifts.
        assert_eq!(
            actual["reason"], expected["reason"],
            "{label}: reason (config {config_json})"
        );
        assert_eq!(
            actual["tier"], expected["tier"],
            "{label}: tier (config {config_json})"
        );
        assert_eq!(
            actual["routed_model"], expected["routed_model"],
            "{label}: routed_model (config {config_json})"
        );
        assert_eq!(
            actual["legs"], expected["legs"],
            "{label}: legs (config {config_json})"
        );
        assert_eq!(actual, *expected, "{label}: plan (config {config_json})");
    }
}

#[test]
fn goldens_match_the_python_reference() {
    let files = golden_files();
    if files.is_empty() {
        println!(
            "note: no goldens found under tests/golden/*.json - skipping parity checks. \
             Regenerate them with `python3 tests/golden/generate.py`."
        );
        return;
    }

    for path in &files {
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("<golden>")
            .to_string();
        let golden = load(path);
        let fixture_name = str_field(&golden, "fixture", &name);
        let bytes = fixture(fixture_name);

        if golden.get("bypass").is_some() {
            check_bypass(&name, &golden, &bytes);
            continue;
        }

        let classification = check_classification(&name, &golden, &bytes);
        check_local_pages(&name, &golden, &bytes);
        check_plans(&name, &golden, &classification);
    }

    println!("{} goldens checked", files.len());
}
