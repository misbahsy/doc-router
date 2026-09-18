//! Shared helpers for the CLI crate's integration tests.
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

/// The path of one fixture.
pub fn fixture_path(name: &str) -> PathBuf {
    fixture_dir().join(name)
}

/// Read one fixture by file name.
pub fn fixture(name: &str) -> Vec<u8> {
    let path = fixture_path(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
}

/// A private scratch directory under `target/`, unique to `name`.
pub fn scratch_dir(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("creating {}: {e}", dir.display()));
    dir
}
