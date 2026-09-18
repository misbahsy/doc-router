//! `doc-router-bench` — score any `PageJudge` against a ground-truth corpus.

#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::{Context, Result};
use clap::Parser;
use doc_router_bench::bench::{self, BenchOptions, DEFAULT_COST_PER_PAGE, DEFAULT_REPEAT};
use doc_router_bench::dotenv;
use doc_router_bench::registry::{BASELINE_JUDGE, JUDGE_NAMES};
use doc_router_bench::report;
use doc_router_cli::judge_help;

/// The corpus shipped with the repo, resolved at compile time from this crate's
/// own location so the binary works from any working directory (`cargo run`,
/// `target/debug/doc-router-bench`, a copy on `$PATH` inside a checkout).
const DEFAULT_CORPUS: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/corpus/manifest.json"
);

/// The optional `.env` at the workspace root, resolved the same way and for the
/// same reason as [`DEFAULT_CORPUS`]: the file belongs to the checkout, not to
/// whatever directory the binary happens to be run from.
const DEFAULT_DOTENV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.env");

/// Score judges against a corpus of PDFs whose per-page truth is known.
#[derive(Debug, Parser)]
#[command(name = "doc-router-bench", version, about, long_about = None)]
struct Cli {
    /// Corpus manifest. Paths inside it are resolved relative to its own
    /// directory, so a manifest outside the repo can point at real documents.
    #[arg(long, value_name = "PATH")]
    corpus: Option<PathBuf>,
    /// Judge to score. Repeatable. The baseline is always included.
    #[arg(long, value_name = "NAME", help = judge_help(JUDGE_LEAD, "baseline"))]
    judge: Vec<String>,
    /// Runs per document; the median is reported.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_REPEAT, value_parser = clap::value_parser!(u32).range(1..))]
    repeat: u32,
    /// Cost of one OCR page. Default 1.0 = one abstract unit per page, so costs
    /// read as page counts. Pass your own price to read them as money.
    #[arg(long, value_name = "F", default_value_t = DEFAULT_COST_PER_PAGE)]
    cost_per_page: f64,
    /// Cost of a million input tokens to a hosted judge, in the same unit as
    /// --cost-per-page. Omitted, the judge's calls are counted and left
    /// unpriced: a million tokens and an OCR page have no common default.
    #[arg(long, value_name = "F")]
    cost_per_million_input_tokens: Option<f64>,
    /// Print the whole report as JSON instead of a table.
    #[arg(long)]
    json: bool,
    /// Write the report to this file instead of stdout.
    #[arg(long, value_name = "PATH")]
    out: Option<PathBuf>,
}

/// The half of `--judge`'s help text that is specific to a benchmark. The rest
/// of it — the registered names — comes from
/// [`judge_help`](doc_router_cli::judge_help), which reads them from the
/// registry so adding a judge there makes it discoverable from `--help` without
/// a second edit.
const JUDGE_LEAD: &str = "Judge to score. Repeatable. The baseline is always included.";

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        // Every document was attempted and at least one could not be scored.
        // The report was still printed: a bad score is a result, an unscorable
        // document is a failure.
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<bool> {
    let cli = Cli::parse();
    // Before anything reads the environment, and before any judge is built. A
    // key already exported into the shell always wins; see `dotenv`.
    let loaded = dotenv::load(&PathBuf::from(DEFAULT_DOTENV));
    if !loaded.is_empty() {
        // The names only. A value from this file is a credential by assumption.
        eprintln!("loaded {} from .env", loaded.join(", "));
    }
    let corpus = match cli.corpus {
        Some(path) => path,
        None => {
            let default = PathBuf::from(DEFAULT_CORPUS);
            let default = default.canonicalize().unwrap_or(default);
            anyhow::ensure!(
                default.exists(),
                "no corpus at {}; pass --corpus <PATH>",
                default.display()
            );
            default
        }
    };

    let options = BenchOptions {
        corpus,
        judges: cli.judge,
        repeat: cli.repeat,
        cost_per_page: cli.cost_per_page,
        cost_per_million_input_tokens: cli.cost_per_million_input_tokens,
    };
    let report = bench::run(&options)?;

    let text = if cli.json {
        let mut json = serde_json::to_string_pretty(&report).context("could not render JSON")?;
        json.push('\n');
        json
    } else {
        report::render(&report)
    };

    match &cli.out {
        Some(path) => std::fs::write(path, &text)
            .with_context(|| format!("could not write {}", path.display()))?,
        None => {
            let mut stdout = std::io::stdout().lock();
            stdout.write_all(text.as_bytes())?;
            stdout.flush()?;
        }
    }

    for skip in &report.skipped_judges {
        // stderr, so a redirected report stays a report, and worded as something
        // to do: an unset key is the expected state of a fresh checkout, not an
        // error. The exit code deliberately does not move.
        eprintln!("skipped judge `{}`: {}", skip.judge, skip.reason);
    }

    if !report.ok() {
        eprintln!(
            "{} document(s) could not be scored; see the report. Registered judges: {} \
             (baseline: {BASELINE_JUDGE})",
            report.failures.len(),
            JUDGE_NAMES.join(", ")
        );
    }
    Ok(report.ok())
}
