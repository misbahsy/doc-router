//! A `.env` file, read at startup so a hosted judge can be benchmarked without
//! exporting anything into the shell.
//!
//! This is thirty lines rather than a dependency on purpose. The harness needs
//! one thing from the format — `KEY=VALUE` pairs, with the comments and quoting
//! people actually type — and the semantics that matter are the ones a crate
//! would have to be configured into anyway:
//!
//! * **The real environment always wins.** A `.env` file fills in what is
//!   missing; it never overrides what the operator explicitly exported. A run
//!   that silently used a stale key from a file, in preference to the key in the
//!   shell it was launched from, is the kind of surprise that costs an afternoon.
//! * **A missing file is not an error.** Most runs have no `.env` at all.
//! * **A malformed line is skipped, not fatal.** The file is a convenience; it
//!   is not worth failing a benchmark over a stray line.
//!
//! The file is in `.gitignore`, and nothing here writes to it.

use std::path::Path;

/// Read `path` and export every pair it declares that is not already set.
///
/// Returns the names that were set, in file order, for the caller to report.
/// A missing or unreadable file sets nothing and returns an empty list.
pub fn load(path: &Path) -> Vec<String> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let pairs = unset_pairs(&text, |key| std::env::var_os(key).is_some());
    let names = pairs.iter().map(|(key, _)| key.clone()).collect();
    for (key, value) in pairs {
        // `set_var` mutates process-global state that other threads may be
        // reading, which is why edition 2024 makes it `unsafe`. This crate is on
        // 2021, where it is not — but the hazard is the same, so the only call
        // site is `main`, before any thread or any judge exists. That is also
        // why this is called from the binary and not from the registry.
        std::env::set_var(key, value);
    }
    names
}

/// The pairs in `text` whose key `is_set` says the environment does not have.
///
/// Split out from [`load`] so it can be tested without touching the process
/// environment, which is global, shared with every other test in the binary,
/// and would make these tests order-dependent.
pub fn unset_pairs(text: &str, is_set: impl Fn(&str) -> bool) -> Vec<(String, String)> {
    parse(text)
        .into_iter()
        .filter(|(key, _)| !is_set(key))
        .collect()
}

/// Every `KEY=VALUE` pair in `text`, in file order.
///
/// Comments (`#` at the start of a line), blank lines and lines without a `=`
/// are skipped. A leading `export ` is allowed, because people paste these files
/// out of shell scripts. A value may be wrapped in single or double quotes,
/// which are stripped; anything else, including a `#` and any further `=`, is
/// part of the value.
pub fn parse(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let line = line.strip_prefix("export ").unwrap_or(line).trim_start();
        // `split_once` and not `split`: the first `=` separates, every later one
        // belongs to the value. Base64 and URLs are full of them.
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.is_empty() || !key.chars().all(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        pairs.push((key.to_string(), unquote(value.trim()).to_string()));
    }
    pairs
}

/// Strip one matching pair of surrounding quotes, if there is one.
fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|v| v.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Nothing is set, so every pair in the file comes back.
    fn nothing_set(_key: &str) -> bool {
        false
    }

    #[test]
    fn a_plain_pair_parses() {
        assert_eq!(
            parse("KEY=value"),
            vec![("KEY".to_string(), "value".to_string())]
        );
    }

    #[test]
    fn comments_and_blank_lines_are_skipped() {
        let text = "# a comment\n\n   \nKEY=value\n#KEY=shadowed\n";
        assert_eq!(
            parse(text),
            vec![("KEY".to_string(), "value".to_string())],
            "the commented-out second assignment must not win"
        );
    }

    #[test]
    fn an_export_prefix_is_allowed() {
        assert_eq!(
            parse("export KEY=value"),
            vec![("KEY".to_string(), "value".to_string())]
        );
    }

    #[test]
    fn quotes_are_stripped_but_only_in_matching_pairs() {
        let text = "A=\"double\"\nB='single'\nC=\"mismatched'\nD=say \"hi\"\n";
        assert_eq!(
            parse(text),
            vec![
                ("A".to_string(), "double".to_string()),
                ("B".to_string(), "single".to_string()),
                ("C".to_string(), "\"mismatched'".to_string()),
                ("D".to_string(), "say \"hi\"".to_string()),
            ]
        );
    }

    #[test]
    fn an_equals_inside_the_value_stays_in_the_value() {
        assert_eq!(
            parse("URL=https://example.test/x?a=1&b=2"),
            vec![(
                "URL".to_string(),
                "https://example.test/x?a=1&b=2".to_string()
            )]
        );
    }

    #[test]
    fn surrounding_whitespace_is_trimmed_on_both_sides() {
        assert_eq!(
            parse("  KEY  =  value  "),
            vec![("KEY".to_string(), "value".to_string())]
        );
    }

    #[test]
    fn a_line_without_an_equals_or_with_a_bad_key_is_skipped_not_fatal() {
        let text = "this is not a pair\nKEY WITH SPACES=1\n=novalue\nGOOD=1\n";
        assert_eq!(
            parse(text),
            vec![("GOOD".to_string(), "1".to_string())],
            "a malformed line must not take the rest of the file with it"
        );
    }

    #[test]
    fn an_empty_value_is_a_pair_not_a_skip() {
        assert_eq!(parse("KEY="), vec![("KEY".to_string(), String::new())]);
    }

    #[test]
    fn the_real_environment_wins_over_the_file() {
        let text = "ALREADY=from_file\nMISSING=from_file\n";
        let pairs = unset_pairs(text, |key| key == "ALREADY");
        assert_eq!(
            pairs,
            vec![("MISSING".to_string(), "from_file".to_string())],
            "a key exported into the shell is never overwritten by the file"
        );
    }

    #[test]
    fn everything_is_offered_when_nothing_is_set() {
        let pairs = unset_pairs("A=1\nB=2\n", nothing_set);
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn a_missing_file_sets_nothing_and_says_so() {
        let missing = std::path::Path::new("/nonexistent/doc-router-bench/.env");
        assert!(load(missing).is_empty());
    }
}
