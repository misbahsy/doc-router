//! The manifest checks, exercised against manifests written for the test.
//!
//! The checked-in `tests/corpus/manifest.json` is never modified: a test that
//! corrupts the real corpus to prove the harness notices would leave the repo
//! one failed assertion away from a wrong manifest. Every manifest here is
//! written into `CARGO_TARGET_TMPDIR` instead.

use std::path::{Path, PathBuf};

use doc_router_bench::bench::{self, BenchOptions};
use doc_router_bench::corpus::Corpus;

/// A scratch directory of our own under the target dir.
fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    dir
}

/// The real fixtures, which these manifests point at without touching them.
fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures")
}

fn write_manifest(dir: &Path, body: &str) -> PathBuf {
    let path = dir.join("manifest.json");
    std::fs::write(&path, body).expect("write manifest");
    path
}

fn options(corpus: PathBuf) -> BenchOptions {
    BenchOptions {
        corpus,
        judges: Vec::new(),
        repeat: 1,
        cost_per_page: 1.0,
        ..BenchOptions::default()
    }
}

#[test]
fn a_wrong_page_count_excludes_the_document_and_fails_the_run() {
    let dir = scratch("wrong_page_count");
    // mixed.pdf really has 4 pages. Claiming 3 means the needs_ocr list below
    // describes some other document.
    let manifest = write_manifest(
        &dir,
        &format!(
            r#"{{"documents":[
                {{"file":{fixture:?},"source":"synthetic","page_count":3,"needs_ocr":[1]}}
            ]}}"#,
            fixture = fixtures().join("mixed.pdf").display().to_string()
        ),
    );

    let report = bench::run(&options(manifest)).expect("the run itself succeeds");

    assert_eq!(report.documents_scored, 0);
    assert_eq!(report.failures.len(), 1);
    assert!(!report.ok(), "the process must exit non-zero");
    let error = &report.failures[0].error;
    assert!(error.contains("page_count mismatch"), "{error}");
    assert!(error.contains("declares 3"), "{error}");
    assert!(error.contains("reports 4"), "{error}");
    // The judge is still reported, with nothing scored: a harness failure must
    // not be silently turned into a perfect score.
    assert_eq!(report.judges.len(), 1);
    assert_eq!(report.judges[0].overall.documents, 0);
    assert!(report.caveats.is_empty());
}

#[test]
fn one_bad_document_does_not_stop_the_good_ones() {
    let dir = scratch("one_bad_document");
    let manifest = write_manifest(
        &dir,
        &format!(
            r#"{{"documents":[
                {{"file":{bad:?},"source":"synthetic","page_count":99,"needs_ocr":[0]}},
                {{"file":{good:?},"source":"synthetic","page_count":4,"needs_ocr":[1,3]}}
            ]}}"#,
            bad = fixtures().join("scanned.pdf").display().to_string(),
            good = fixtures().join("mixed.pdf").display().to_string(),
        ),
    );

    let report = bench::run(&options(manifest)).expect("run");

    assert_eq!(report.documents_scored, 1);
    assert_eq!(report.documents_in_corpus, 2);
    assert_eq!(report.failures.len(), 1);
    assert!(!report.ok());
    let judge = &report.judges[0];
    assert_eq!(judge.documents.len(), 1);
    assert_eq!(judge.documents[0].file, "mixed.pdf");
    assert!(judge.documents[0].route_exact);
}

#[test]
fn a_missing_file_is_a_harness_failure_not_a_score() {
    let dir = scratch("missing_file");
    let manifest = write_manifest(
        &dir,
        r#"{"documents":[{"file":"nope.pdf","source":"synthetic","page_count":1,"needs_ocr":[]}]}"#,
    );

    let report = bench::run(&options(manifest)).expect("run");

    assert!(!report.ok());
    assert_eq!(report.failures.len(), 1);
    assert!(
        report.failures[0].error.contains("could not read"),
        "{}",
        report.failures[0].error
    );
    // Resolved relative to the manifest, not the working directory.
    assert!(report.failures[0].path.ends_with("missing_file/nope.pdf"));
}

#[test]
fn the_checked_in_corpus_loads_and_agrees_with_itself() {
    // Reads the real manifest; writes nothing.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/corpus/manifest.json");
    let corpus = Corpus::load(&path).expect("the checked-in corpus must load");
    assert!(!corpus.documents.is_empty());
    for document in &corpus.documents {
        assert!(
            document.path.exists(),
            "{} does not exist",
            document.path.display()
        );
        for page in &document.truth {
            assert!(
                *page < document.spec.page_count,
                "{}: page {page} is outside a {}-page document",
                document.label(),
                document.spec.page_count
            );
        }
    }
    // Source order is manifest order: the ten fixtures first, then the adversarial
    // documents whose structure and ground truth disagree on purpose.
    assert_eq!(
        corpus.sources(),
        vec!["synthetic".to_string(), "adversarial".to_string()]
    );
}
