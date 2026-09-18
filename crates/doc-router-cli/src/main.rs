//! `doc-router` — classify, plan, extract, split and run PDFs through the document router.

#![forbid(unsafe_code)]

use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use clap::{Args, Parser, Subcommand};
use doc_router::{
    classify_with, decide_with, extract_local, run_with, split_pdf, Classification, Config,
    Decision, LegSummary, OcrResult, Outcome, PageJudge, RouteMetadata,
};
use doc_router_cli::{
    judge_by_name, judge_help, load_config, resolve_api_key, JudgeLookup, JudgeProvenance,
    LiteLlmHost, OnApiFailure, RecordingJudge, BASELINE_JUDGE, DEFAULT_TIMEOUT_SECONDS,
    JUDGE_NAMES,
};
use serde::Serialize;

/// Per-page routing between local PDF text extraction and OCR models.
#[derive(Debug, Parser)]
#[command(name = "doc-router", version, about, long_about = None)]
struct Cli {
    /// Path to a `doc_router_config` JSON file. Defaults to local_pdf/extract + mistral-ocr.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,
    /// Print machine-readable JSON instead of a human summary.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Classify a PDF: type, confidence, page count, which pages need OCR.
    Classify {
        /// The PDF to classify.
        file: PathBuf,
        #[command(flatten)]
        judge: JudgeArgs,
    },
    /// Show the routing decision (and its `doc_route` metadata) without executing it.
    Plan {
        /// The PDF to plan for.
        file: PathBuf,
        /// Model a bypassed document would go to. Defaults to `tiers.standard`.
        #[arg(long, value_name = "MODEL")]
        default_model: Option<String>,
        #[command(flatten)]
        judge: JudgeArgs,
    },
    /// Extract the PDF's own text layer locally. No network, no models.
    Extract {
        /// The PDF to read.
        file: PathBuf,
        /// 0-indexed pages to extract, comma separated. Omit for every page.
        #[arg(long, value_name = "N,N", value_delimiter = ',')]
        pages: Vec<u32>,
        /// Write `page-<index>.md` into this directory instead of printing.
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
    },
    /// Write a new PDF containing only the given pages.
    Split {
        /// The PDF to subset.
        file: PathBuf,
        /// 0-indexed pages to keep, comma separated, in the order given.
        #[arg(long, value_name = "N,N", value_delimiter = ',', required = true)]
        pages: Vec<u32>,
        /// Where to write the subset PDF.
        #[arg(long, value_name = "PATH")]
        out: PathBuf,
    },
    /// Route and execute: local legs in-process, OCR legs on a LiteLLM proxy.
    Run {
        /// The PDF to process.
        file: PathBuf,
        #[command(flatten)]
        proxy: ProxyArgs,
        /// Model for bypasses and for the whole-document retry after a failed leg.
        #[arg(long, value_name = "MODEL")]
        default_model: Option<String>,
        /// Write the JSON result here as well as printing to stdout.
        #[arg(long, value_name = "PATH")]
        out: Option<PathBuf>,
        #[command(flatten)]
        judge: JudgeArgs,
    },
}

/// The half of `--judge`'s help text that is specific to this binary. The names
/// themselves come from the registry, via
/// [`judge_help`](doc_router_cli::judge_help), so registering a judge makes it
/// discoverable from `--help` without a second edit here.
const JUDGE_LEAD: &str = "Judge deciding which pages need OCR.";

/// Which judge decides whether a page needs OCR.
///
/// Only on the subcommands where it changes the answer. `extract` and `split` do
/// exactly what they are told to and never ask a judge anything, so a `--judge`
/// there would be a flag that does nothing.
#[derive(Debug, Args)]
struct JudgeArgs {
    /// Registered judge name. Defaults to the built-in heuristic.
    // `hide_default_value`: the help text already ends with "(default:
    // heuristic)", built from the registry, and clap's own `[default: …]` would
    // print the same fact twice.
    #[arg(
        long,
        value_name = "NAME",
        default_value = BASELINE_JUDGE,
        hide_default_value = true,
        help = judge_help(JUDGE_LEAD, "default")
    )]
    judge: String,
}

/// How to reach the OCR endpoint.
#[derive(Debug, Args)]
struct ProxyArgs {
    /// LiteLLM proxy base URL; `/v1/ocr` is appended.
    #[arg(long, value_name = "URL")]
    base_url: String,
    /// Bearer token. Falls back to $LITELLM_API_KEY, then $LITELLM_PROXY_API_KEY.
    #[arg(long, value_name = "KEY")]
    api_key: Option<String>,
    /// Per-call timeout in seconds.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_TIMEOUT_SECONDS)]
    timeout_seconds: f64,
    /// Upload a split subset PDF instead of a `pages` list, for providers that ignore it.
    #[arg(long)]
    split_subset: bool,
}

/// `classify --json`: the classification, plus which judge produced it.
///
/// Flattened, so every field the classification has always had stays exactly
/// where it was and `judge` is added beside them.
#[derive(Debug, Serialize)]
struct ClassifyOutput<'a> {
    #[serde(flatten)]
    classification: &'a Classification,
    judge: &'a JudgeProvenance,
}

/// `plan --json`: the decision plus the `doc_route` metadata it produces.
#[derive(Debug, Serialize)]
struct PlanOutput<'a> {
    decision: &'a Decision,
    metadata: &'a RouteMetadata,
    judge: &'a JudgeProvenance,
}

/// `extract --json --out DIR`: what was written where.
#[derive(Debug, Serialize)]
struct ExtractOutput {
    out: String,
    files: Vec<String>,
    pages_processed: u32,
}

/// `split --json`: what was written where.
#[derive(Debug, Serialize)]
struct SplitOutput<'a> {
    out: String,
    pages: &'a [u32],
    bytes: usize,
}

/// `run --json` (and the `--out` file): the merged result plus its metadata.
#[derive(Debug, Serialize)]
struct RunOutput<'a> {
    result: &'a OcrResult,
    metadata: &'a RouteMetadata,
    judge: &'a JudgeProvenance,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cfg = load_config(cli.config.as_deref())?;

    match &cli.command {
        Command::Classify { file, judge } => cmd_classify(file, &judge.judge, cli.json),
        Command::Plan {
            file,
            default_model,
            judge,
        } => cmd_plan(file, &cfg, default_model.as_deref(), &judge.judge, cli.json),
        Command::Extract { file, pages, out } => cmd_extract(file, pages, out.as_deref(), cli.json),
        Command::Split { file, pages, out } => cmd_split(file, pages, out, cli.json),
        Command::Run {
            file,
            proxy,
            default_model,
            out,
            judge,
        } => cmd_run(
            file,
            &cfg,
            proxy,
            default_model.as_deref(),
            out.as_deref(),
            &judge.judge,
            cli.json,
        ),
    }
}

/// Build the judge named on the command line, or explain why it cannot be built.
///
/// Both failures are the operator's to fix and neither is silent:
///
/// * An unregistered name is a typo. The error lists the names that do work.
/// * A registered judge with no credentials is an unset variable. The error
///   names the variables. It deliberately does **not** fall back to the built-in
///   judge: a run that quietly used the heuristic after being asked for a hosted
///   one, and said so nowhere, is the reason this flag exists.
///
/// A failure *after* the judge is built — a timeout, an HTTP error, an open
/// circuit breaker — is a different question, and the judge is built
/// [non-strict](OnApiFailure::FallBack) for it: one document an operator is
/// waiting on should still route, on the heuristic's answer, rather than fail.
/// That fallback is not silent either: [`RecordingJudge`] counts the pages it
/// touched, `--json` carries them under `judge.fallbacks`, and a line goes to
/// stderr.
fn resolve_judge(name: &str) -> Result<Box<dyn PageJudge>> {
    let lookup = judge_by_name(name, OnApiFailure::FallBack).ok_or_else(|| {
        anyhow!(
            "unknown judge `{name}`; registered judges are: {}",
            JUDGE_NAMES.join(", ")
        )
    })?;
    match lookup {
        JudgeLookup::Ready(judge) => Ok(judge),
        JudgeLookup::Unavailable { name, reason } => {
            Err(anyhow!("judge `{name}` is unavailable: {reason}"))
        }
    }
}

/// Report a judge that answered for itself only part of the time.
///
/// stderr, so a redirected `--json` document stays parseable, and unconditional:
/// a fallback means some of these verdicts came from a judge nobody asked for,
/// which the operator has to be able to see. The exit code does not move — the
/// document routed, and on the same answer the CLI would have given by default.
fn report_fallbacks(provenance: &JudgeProvenance) {
    if let Some(note) = provenance.fallback_note() {
        eprintln!("{note}");
    }
}

/// Name the judge in the human-readable output, when it is not the default one.
///
/// Silent for the default judge on purpose: every line this binary has ever
/// printed without `--judge` still reads exactly the same.
fn print_judge_line(provenance: &JudgeProvenance) {
    if provenance.name != BASELINE_JUDGE {
        println!("  judge: {}", provenance.name);
    }
}

fn read_pdf(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("could not read {}", path.display()))
}

fn print_json<T: serde::Serialize>(value: &T) -> Result<()> {
    let text = serde_json::to_string_pretty(value).context("serialising output")?;
    println!("{text}");
    Ok(())
}

fn cmd_classify(file: &Path, judge_name: &str, json: bool) -> Result<()> {
    let bytes = read_pdf(file)?;
    let judge = resolve_judge(judge_name)?;
    let judge = RecordingJudge::new(judge.as_ref());
    let c = classify_with(&bytes, &judge)
        .with_context(|| format!("could not classify {}", file.display()))?;
    let provenance = judge.provenance();
    report_fallbacks(&provenance);

    if json {
        return print_json(&ClassifyOutput {
            classification: &c,
            judge: &provenance,
        });
    }
    println!("{}: {}", file.display(), describe_classification(&c));
    print_judge_line(&provenance);
    Ok(())
}

fn describe_classification(c: &Classification) -> String {
    let layout = if c.is_complex_layout {
        "complex layout"
    } else {
        "simple layout"
    };
    format!(
        "{} (confidence {:.2}), {} page(s), {} needing OCR {}, {layout}, classified in {:.1} ms",
        c.pdf_type.as_str(),
        c.confidence,
        c.page_count,
        c.pages_needing_ocr.len(),
        format_pages(Some(&c.pages_needing_ocr)),
        c.classify_ms,
    )
}

fn format_pages(pages: Option<&[u32]>) -> String {
    match pages {
        None => "all".to_string(),
        Some([]) => "none".to_string(),
        Some(pages) => format!(
            "[{}]",
            pages
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn cmd_plan(
    file: &Path,
    cfg: &Config,
    default_model: Option<&str>,
    judge_name: &str,
    json: bool,
) -> Result<()> {
    let bytes = read_pdf(file)?;
    let judge = resolve_judge(judge_name)?;
    let judge = RecordingJudge::new(judge.as_ref());
    let decision = decide_with(&bytes, cfg, default_model, &judge);
    let metadata = decision.metadata();
    let provenance = judge.provenance();
    report_fallbacks(&provenance);

    if json {
        return print_json(&PlanOutput {
            decision: &decision,
            metadata: &metadata,
            judge: &provenance,
        });
    }

    match &decision {
        Decision::Bypass {
            reason,
            model,
            detail,
        } => {
            println!("{}: bypass {reason} -> {model}", file.display());
            if let Some(detail) = detail {
                println!("  {detail}");
            }
        }
        Decision::Route {
            plan,
            classification,
        } => {
            println!(
                "{}: route {} -> {} (tier {})",
                file.display(),
                plan.reason,
                plan.routed_model,
                plan.tier
            );
            println!("  {}", describe_classification(classification));
            for leg in &plan.legs {
                println!(
                    "  leg {:<24} {:<9} pages {}",
                    leg.model,
                    leg.tier.as_str(),
                    format_pages(leg.pages.as_deref())
                );
            }
        }
    }
    print_judge_line(&provenance);
    Ok(())
}

fn cmd_extract(file: &Path, pages: &[u32], out: Option<&Path>, json: bool) -> Result<()> {
    let bytes = read_pdf(file)?;
    let subset = (!pages.is_empty()).then_some(pages);
    let result = extract_local(&bytes, subset)
        .with_context(|| format!("could not extract {}", file.display()))?;

    if let Some(dir) = out {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("could not create {}", dir.display()))?;
        let mut written = Vec::new();
        for page in &result.pages {
            let path = dir.join(format!("page-{}.md", page.index));
            std::fs::write(&path, &page.markdown)
                .with_context(|| format!("could not write {}", path.display()))?;
            written.push(path.display().to_string());
        }
        if json {
            return print_json(&ExtractOutput {
                out: dir.display().to_string(),
                files: written,
                pages_processed: result.pages_processed,
            });
        }
        println!("wrote {} page file(s) to {}", written.len(), dir.display());
        return Ok(());
    }

    if json {
        return print_json(&result);
    }
    for page in &result.pages {
        println!("## page {}\n", page.index);
        println!("{}\n", page.markdown);
    }
    Ok(())
}

fn cmd_split(file: &Path, pages: &[u32], out: &Path, json: bool) -> Result<()> {
    let bytes = read_pdf(file)?;
    let subset =
        split_pdf(&bytes, pages).with_context(|| format!("could not split {}", file.display()))?;
    std::fs::write(out, &subset).with_context(|| format!("could not write {}", out.display()))?;

    if json {
        return print_json(&SplitOutput {
            out: out.display().to_string(),
            pages,
            bytes: subset.len(),
        });
    }
    println!(
        "wrote {} page(s) {} to {} ({} bytes)",
        pages.len(),
        format_pages(Some(pages)),
        out.display(),
        subset.len()
    );
    Ok(())
}

fn cmd_run(
    file: &Path,
    cfg: &Config,
    proxy: &ProxyArgs,
    default_model: Option<&str>,
    out: Option<&Path>,
    judge_name: &str,
    json: bool,
) -> Result<()> {
    let bytes = read_pdf(file)?;
    // Before the first byte goes anywhere: an unusable judge is a mistake in the
    // command line, and finding it out after a paid OCR call would be an
    // expensive way to learn about a typo.
    let judge = resolve_judge(judge_name)?;
    let judge = RecordingJudge::new(judge.as_ref());
    let host = LiteLlmHost::with_timeout(&proxy.base_url, proxy.timeout_seconds)
        .with_api_key(resolve_api_key(proxy.api_key.as_deref()))
        .with_split_subset(proxy.split_subset);

    let started = Instant::now();
    let outcome = run_with(&bytes, cfg, default_model, &host, &judge)?;
    let elapsed_ms = started.elapsed().as_millis();

    let provenance = judge.provenance();
    report_fallbacks(&provenance);
    let payload = RunOutput {
        result: &outcome.result,
        metadata: &outcome.metadata,
        judge: &provenance,
    };
    if let Some(path) = out {
        let mut file = std::fs::File::create(path)
            .with_context(|| format!("could not create {}", path.display()))?;
        let text = serde_json::to_string_pretty(&payload).context("serialising result")?;
        writeln!(file, "{text}").with_context(|| format!("could not write {}", path.display()))?;
    }

    if json {
        return print_json(&payload);
    }
    print_run_summary(file, &outcome, &host, elapsed_ms);
    print_judge_line(&provenance);
    Ok(())
}

fn print_run_summary(file: &Path, outcome: &Outcome, host: &LiteLlmHost, elapsed_ms: u128) {
    let RouteMetadata {
        tier,
        reason,
        routed_model,
        fallback_reason,
        legs,
        ..
    } = &outcome.metadata;
    println!(
        "{}: {reason} -> {routed_model} (tier {tier}) in {elapsed_ms} ms",
        file.display()
    );
    if let Some(fallback) = fallback_reason {
        println!("  fallback: {fallback} (whole document rerun on {routed_model})");
    }

    let calls = host.calls();
    let empty: Vec<LegSummary> = Vec::new();
    for leg in legs.as_ref().unwrap_or(&empty) {
        let ms = calls
            .iter()
            .find(|call| call.model == leg.model && call.pages == leg.pages)
            .map(|call| format!("{} ms", call.elapsed_ms))
            .unwrap_or_else(|| "in-process".to_string());
        println!(
            "  leg {:<24} {:<9} pages {:<12} {ms}",
            leg.model,
            leg.tier.as_str(),
            format_pages(leg.pages.as_deref())
        );
    }
    print_pages(&outcome.result);
}

fn print_pages(result: &OcrResult) {
    println!("  {} page(s) via {}:", result.pages.len(), result.model);
    for page in &result.pages {
        let first_line = page
            .markdown
            .lines()
            .find(|line| !line.trim().is_empty())
            .unwrap_or("(empty)");
        println!(
            "    {:>4}  {:<24} {}",
            page.index,
            page.model,
            elide(first_line, 80)
        );
    }
}

fn elide(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_string();
    }
    format!("{}…", text.chars().take(max).collect::<String>())
}
