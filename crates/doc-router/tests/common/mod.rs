//! Shared helpers for the integration tests.
//!
//! Each integration test binary uses a different subset of these.
#![allow(dead_code)]

use std::path::PathBuf;

/// The shared fixture directory at the repository root.
pub fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures")
        .canonicalize()
        .expect("tests/fixtures exists at the repository root")
}

/// The golden-file directory at the repository root, if it exists.
pub fn golden_dir() -> Option<PathBuf> {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/golden")
        .canonicalize()
        .ok()
}

/// Read one fixture by file name.
pub fn fixture(name: &str) -> Vec<u8> {
    let path = fixture_dir().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}
