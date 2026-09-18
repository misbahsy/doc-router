//! Support library for the `doc-router` command-line tool.
//!
//! The binary is a thin shell over [`doc_router`]; everything worth testing on its own —
//! the LiteLLM host and the config loading rules — lives here.
//!
//! ```no_run
//! use doc_router::{run, Config, Tiers};
//! use doc_router_cli::LiteLlmHost;
//!
//! let cfg = Config::new(Tiers::new("local_pdf/extract", "mistral-ocr"));
//! let host = LiteLlmHost::new("http://localhost:4000").with_api_key(std::env::var("LITELLM_API_KEY").ok());
//! let outcome = run(&std::fs::read("mixed.pdf")?, &cfg, Some("mistral-ocr"), &host)?;
//! println!("{} pages", outcome.result.pages.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod host;
pub mod judge;

pub use host::{CallRecord, LiteLlmHost, API_KEY_ENV_VARS, DEFAULT_TIMEOUT_SECONDS};
pub use judge::{
    judge_by_name, judge_help, JudgeFallback, JudgeLookup, JudgeProvenance, OnApiFailure,
    RecordingJudge, BASELINE_JUDGE, JUDGE_NAMES,
};

use doc_router::{Config, Error, Tiers};

/// The tier table used when no `--config` is given.
pub fn default_config() -> Config {
    Config::new(Tiers::new(doc_router::LOCAL_MODEL, "mistral-ocr"))
}

/// Read and validate a `doc_router_config` JSON file, or fall back to [`default_config`].
pub fn load_config(path: Option<&std::path::Path>) -> Result<Config, ConfigLoadError> {
    let Some(path) = path else {
        return Ok(default_config());
    };
    let text = std::fs::read_to_string(path).map_err(|source| ConfigLoadError::Read {
        path: path.display().to_string(),
        source,
    })?;
    Config::from_json(&text).map_err(|source| ConfigLoadError::Parse {
        path: path.display().to_string(),
        source,
    })
}

/// Why a `--config` file could not be used.
#[derive(Debug, thiserror::Error)]
pub enum ConfigLoadError {
    /// The file could not be read.
    #[error("could not read config {path}: {source}")]
    Read {
        /// The path as given on the command line.
        path: String,
        /// The underlying I/O failure.
        #[source]
        source: std::io::Error,
    },
    /// The file is not a valid `doc_router_config` block.
    #[error("invalid config {path}: {source}")]
    Parse {
        /// The path as given on the command line.
        path: String,
        /// The underlying validation failure.
        #[source]
        source: Error,
    },
}

/// Resolve the proxy key: the flag first, then [`API_KEY_ENV_VARS`] in order.
pub fn resolve_api_key(flag: Option<&str>) -> Option<String> {
    if let Some(key) = flag.filter(|k| !k.trim().is_empty()) {
        return Some(key.to_string());
    }
    API_KEY_ENV_VARS
        .iter()
        .find_map(|name| std::env::var(name).ok())
        .filter(|key| !key.trim().is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_config_is_valid_and_local_first() {
        let cfg = default_config();
        cfg.validate().expect("default config validates");
        assert_eq!(cfg.tiers.local, doc_router::LOCAL_MODEL);
        assert!(cfg.split_pages);
    }

    #[test]
    fn no_path_means_the_default_config() {
        assert_eq!(load_config(None).unwrap(), default_config());
    }

    #[test]
    fn a_missing_config_file_is_a_read_error() {
        let err = load_config(Some(std::path::Path::new("/nope/nothing.json"))).unwrap_err();
        assert!(matches!(err, ConfigLoadError::Read { .. }), "{err}");
    }

    #[test]
    fn the_flag_beats_the_environment() {
        assert_eq!(resolve_api_key(Some("sk-flag")).as_deref(), Some("sk-flag"));
        assert_eq!(resolve_api_key(Some("   ")), resolve_api_key(None));
    }
}
