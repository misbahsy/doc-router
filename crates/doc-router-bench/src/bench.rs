//! Running every judge over every document and collecting the results.

use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::{anyhow, Context, Result};
use doc_router::{classify_with, Classification, Error, PageEvidence, PageJudge, PageVerdict};
use serde::Serialize;

use crate::corpus::{Corpus, CorpusDocument};
use crate::meter::JudgeMeter;
use crate::registry::{
    metered_judge_by_name, JudgeLookup, OnApiFailure, BASELINE_JUDGE, JUDGE_NAMES,
};
use crate::score::{
    confidence_report, cost, judge_spend, latency, median, score_pages, ConfidenceReport,
    Confusion, Cost, JudgeSpend, Latency, ScoredConfidence,
};

/// Default `--repeat`: three runs per document, median reported.
pub const DEFAULT_REPEAT: u32 = 3;

/// Default `--cost-per-page`.
///
/// One **abstract cost unit** per OCR page, so with the default every cost in the
/// report reads as a page count. It is deliberately not a vendor's price: OCR
/// pricing is per-provider, per-contract and changes, and baking one in would
/// present a guess as a fact. Pass your own per-page price to read the report in
/// currency.
pub const DEFAULT_COST_PER_PAGE: f64 = 1.0;

/// Default `--cost-per-million-input-tokens`: **no price at all**.
///
/// Deliberately `None` rather than a `1.0` matching [`DEFAULT_COST_PER_PAGE`].
/// That default works for pages because every cost in the report is then a page
/// count, and page counts are comparable with each other. A million tokens and
/// an OCR page are not: pricing both at "one unit" would let the report subtract
/// one from the other and print a net saving that is an artefact of the default.
/// With no price the tokens are still counted and printed — they are a
/// measurement of the run — and only the money is left unstated.
pub const DEFAULT_COST_PER_MILLION_INPUT_TOKENS: Option<f64> = None;

/// What the harness was asked to do.
#[derive(Debug, Clone)]
pub struct BenchOptions {
    /// The corpus manifest.
    pub corpus: PathBuf,
    /// Judges named on the command line. The baseline is added if absent.
    pub judges: Vec<String>,
    /// Runs per document per judge; the median is reported.
    pub repeat: u32,
    /// Cost of one OCR page, in whatever unit the caller means.
    pub cost_per_page: f64,
    /// Cost of a million input tokens to a hosted judge, in the same unit.
    /// `None` leaves the judge's own calls counted and unpriced; see
    /// [`DEFAULT_COST_PER_MILLION_INPUT_TOKENS`].
    pub cost_per_million_input_tokens: Option<f64>,
}

impl Default for BenchOptions {
    fn default() -> Self {
        BenchOptions {
            corpus: PathBuf::new(),
            judges: Vec::new(),
            repeat: DEFAULT_REPEAT,
            cost_per_page: DEFAULT_COST_PER_PAGE,
            cost_per_million_input_tokens: DEFAULT_COST_PER_MILLION_INPUT_TOKENS,
        }
    }
}

/// A document that could not be scored, and why.
///
/// These are harness-level failures — a missing file, a PDF that will not parse,
/// a `page_count` that disagrees with the manifest. They are **not** a judge
/// scoring badly, which is a result and not a failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DocumentFailure {
    /// The document's file name.
    pub file: String,
    /// Its resolved path.
    pub path: String,
    /// The judge whose run hit the failure, when it was judge-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<String>,
    /// What went wrong.
    pub error: String,
}

/// A registered judge that could not be built on this machine, and why.
///
/// Deliberately not a [`DocumentFailure`] and deliberately not an error: no
/// document failed, and nothing about the run is wrong. The machine simply has
/// no credentials for a hosted judge, which is the normal state of a laptop, of
/// CI, and of anyone reading the report for the first time. The run scores the
/// judges it can and says plainly which one it could not, so the exit code stays
/// 0 and a missing key never looks like a broken harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SkippedJudge {
    /// The registered name that was asked for.
    pub judge: String,
    /// What is missing, phrased as something to do about it.
    pub reason: String,
}

/// A judge that will run, and the handle the harness reads its own spending
/// through.
///
/// The two travel together because they cannot travel through each other: see
/// [`crate::registry::metered_judge_by_name`]. `meter` is `None` for a judge
/// that makes no calls, which the report prints as "no token data" rather than
/// as a bill of zero.
pub struct ReadyJudge {
    /// The judge itself.
    pub judge: Box<dyn PageJudge>,
    /// Its call log, when it has one.
    pub meter: Option<JudgeMeter>,
}

impl ReadyJudge {
    /// The name the judge answers to, which is the name its row is printed
    /// under.
    #[must_use]
    pub fn name(&self) -> &'static str {
        self.judge.name()
    }
}

/// The outcome of turning `--judge` names into judges.
///
/// Both halves matter to the caller: the judges to run, and the ones that were
/// asked for and are not in that list. Returning only the first would make a
/// skipped judge indistinguishable from one that was never requested.
pub struct ResolvedJudges {
    /// The judges that will run, baseline first.
    pub judges: Vec<ReadyJudge>,
    /// The requested judges that could not be built here.
    pub skipped: Vec<SkippedJudge>,
}

/// One judge's result for one document.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct DocumentReport {
    /// File name.
    pub file: String,
    /// The manifest's `source` for this document.
    pub source: String,
    /// Pages in the document (manifest and classifier agreed, or this row would
    /// not exist).
    pub page_count: u32,
    /// The document-level label this judge produced.
    pub pdf_type: String,
    /// True when the judge's OCR page set is exactly truth's.
    pub route_exact: bool,
    /// Confusion counts for this document.
    pub confusion: Confusion,
    /// 0-indexed pages that needed OCR and did not get it.
    pub missed_ocr_pages: Vec<u32>,
    /// 0-indexed pages that got OCR and did not need it.
    pub wasted_ocr_pages: Vec<u32>,
    /// The cost view of those counts.
    pub cost: Cost,
    /// Median `classify_ms` over the repeats.
    pub classify_ms: f64,
    /// `classify_ms` minus the baseline judge's, for the same document. `None`
    /// for the baseline itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta_vs_baseline_ms: Option<f64>,
}

/// Everything about one `source` group, for one judge.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct GroupReport {
    /// The `source` value.
    pub source: String,
    /// Documents in the group.
    pub documents: u32,
    /// How many of them routed exactly.
    pub route_exact: u32,
    /// Summed confusion counts.
    pub confusion: Confusion,
    /// Summed cost.
    pub cost: Cost,
}

/// One judge's whole result.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct JudgeReport {
    /// The judge's name, as registered.
    pub judge: String,
    /// True for the judge every other one is measured against.
    pub baseline: bool,
    /// Per-document rows, in corpus order.
    pub documents: Vec<DocumentReport>,
    /// The same rows grouped by `source`.
    pub by_source: Vec<GroupReport>,
    /// Every document together.
    pub overall: GroupReport,
    /// Latency across the corpus, from the per-document medians.
    pub latency: Latency,
    /// Median per-document latency delta against the baseline judge. `None` for
    /// the baseline itself.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub overhead_vs_baseline_ms: Option<f64>,
    /// What can honestly be said about this judge's confidence values.
    pub confidence: ConfidenceReport,
    /// What the judge's own calls used and cost. `None` when this judge reports
    /// no token usage to the harness at all, which is not the same statement as
    /// "used none" and is printed differently.
    ///
    /// Serialised as an explicit `null` rather than skipped, for the same reason
    /// the text report prints `n/a (why)` instead of leaving the line out: a
    /// consumer that finds no key can default it to zero and read "this judge
    /// was free". A `null` cannot be mistaken for a measurement.
    pub spend: Option<JudgeSpend>,
}

/// A caveat about what the run can and cannot measure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Caveat {
    /// The judge it is about, when it is about one. `None` for a caveat that
    /// applies to every row in the report — the cost arithmetic, say, which is
    /// the same arithmetic for every judge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub judge: Option<String>,
    /// The `source` group it is about, when it is about one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Documents in that group, or in the run when there is no group.
    pub documents: u32,
    /// The note, one line per entry.
    pub lines: Vec<String>,
}

/// The whole run.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct BenchReport {
    /// The manifest that was scored.
    pub corpus: String,
    /// Runs per document.
    pub repeat: u32,
    /// Cost of one OCR page, as passed.
    pub cost_per_page: f64,
    /// Cost of a million input tokens to a hosted judge, as passed. `None` when
    /// the caller gave no token price, which is not the same statement as a
    /// price of zero — so this too is serialised as an explicit `null` rather
    /// than skipped.
    pub cost_per_million_input_tokens: Option<f64>,
    /// Documents scored (excluding failures).
    pub documents_scored: u32,
    /// Documents in the manifest.
    pub documents_in_corpus: u32,
    /// Documents that could not be scored.
    pub failures: Vec<DocumentFailure>,
    /// Judges that were asked for and could not be built here. Never a failure:
    /// see [`SkippedJudge`].
    pub skipped_judges: Vec<SkippedJudge>,
    /// One entry per judge, baseline first.
    pub judges: Vec<JudgeReport>,
    /// What the corpus cannot tell you, stated rather than left to be inferred.
    pub caveats: Vec<Caveat>,
}

impl BenchReport {
    /// True when every document was scored. The process exit code follows this,
    /// so the harness is usable in CI; a judge scoring badly does not affect it.
    pub fn ok(&self) -> bool {
        self.failures.is_empty()
    }
}

/// A judge wrapper that keeps the verdicts on their way past.
///
/// [`classify_with`] returns a [`Classification`], which carries the *routing*
/// decision but not the per-page confidence behind it. Rather than change the
/// core crate to hand verdicts back, the harness wraps the judge it was given
/// and reads them off in passing. The wrapped judge cannot tell the difference.
struct RecordingJudge<'a> {
    inner: &'a dyn PageJudge,
    seen: Mutex<Vec<PageVerdict>>,
}

impl<'a> RecordingJudge<'a> {
    fn new(inner: &'a dyn PageJudge) -> Self {
        RecordingJudge {
            inner,
            seen: Mutex::new(Vec::new()),
        }
    }

    /// The verdicts from the last `judge` call.
    fn into_verdicts(self) -> Vec<PageVerdict> {
        self.seen.into_inner().unwrap_or_else(|e| e.into_inner())
    }
}

impl PageJudge for RecordingJudge<'_> {
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        let verdicts = self.inner.judge(evidence)?;
        *self.seen.lock().unwrap_or_else(|e| e.into_inner()) = verdicts.clone();
        Ok(verdicts)
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn needs_text(&self) -> bool {
        self.inner.needs_text()
    }
}

/// One judge's raw output for one document, before scoring.
struct DocumentRun {
    classification: Classification,
    verdicts: Vec<PageVerdict>,
    classify_ms: f64,
}

/// Resolve the judges to run: the baseline first, then whatever was asked for,
/// each at most once.
///
/// An unregistered name is an error and stops the run — it is a typo, and
/// scoring three judges when four were asked for would hide it. A *registered*
/// judge that cannot be built here is not an error: it lands in
/// [`ResolvedJudges::skipped`] and the rest of the run proceeds. See
/// [`crate::registry`] for why those two are different answers.
pub fn resolve_judges(requested: &[String]) -> Result<ResolvedJudges> {
    let mut names: Vec<String> = vec![BASELINE_JUDGE.to_string()];
    for name in requested {
        if !names.iter().any(|seen| seen == name) {
            names.push(name.clone());
        }
    }
    let mut judges: Vec<ReadyJudge> = Vec::with_capacity(names.len());
    let mut skipped: Vec<SkippedJudge> = Vec::new();
    for name in names {
        // Strict: a hosted judge that cannot reach its API fails the document
        // rather than answering with the heuristic's verdicts under its own
        // name. See `crate::registry` for why a benchmark and a CLI want
        // opposite answers here.
        let (lookup, meter) =
            metered_judge_by_name(&name, OnApiFailure::Fail).ok_or_else(|| {
                anyhow!(
                    "unknown judge `{name}`; registered judges are: {}",
                    JUDGE_NAMES.join(", ")
                )
            })?;
        match lookup {
            JudgeLookup::Ready(judge) => judges.push(ReadyJudge { judge, meter }),
            // The registry names the variables; the `.env` file is this
            // crate's own convenience, so this crate is what mentions it.
            JudgeLookup::Unavailable { name, reason } => skipped.push(SkippedJudge {
                judge: name.to_string(),
                reason: format!("{reason}, or put it in a .env file at the workspace root"),
            }),
        }
    }
    Ok(ResolvedJudges { judges, skipped })
}

/// Score every judge over every document in `options.corpus`.
///
/// Errors only when the run could not happen at all (an unreadable manifest, an
/// unknown judge name). A document that fails to classify or fails its
/// `page_count` check is recorded in [`BenchReport::failures`] and excluded from
/// the score; a judge that scores badly is simply a low number.
pub fn run(options: &BenchOptions) -> Result<BenchReport> {
    let corpus = Corpus::load(&options.corpus)?;
    let ResolvedJudges { judges, skipped } = resolve_judges(&options.judges)?;
    let repeat = options.repeat.max(1);

    let mut failures: Vec<DocumentFailure> = Vec::new();
    // rows[judge index] -> one row per successfully scored document.
    let mut rows: Vec<Vec<DocumentReport>> = vec![Vec::new(); judges.len()];
    let mut confidences: Vec<Vec<ScoredConfidence>> = vec![Vec::new(); judges.len()];
    let mut scored_documents = 0u32;

    for document in &corpus.documents {
        let bytes = match std::fs::read(&document.path) {
            Ok(bytes) => bytes,
            Err(e) => {
                failures.push(DocumentFailure {
                    file: document.label(),
                    path: document.path.display().to_string(),
                    judge: None,
                    error: format!("could not read {}: {e}", document.path.display()),
                });
                continue;
            }
        };

        // Every judge runs before anything is recorded: one judge failing the
        // page_count check means the document's truth cannot be trusted for any
        // of them, and judges must be compared on the same set of documents.
        let mut runs: Vec<DocumentRun> = Vec::with_capacity(judges.len());
        let mut failure = None;
        for judge in &judges {
            match run_document(&bytes, judge.judge.as_ref(), repeat, document) {
                Ok(run) => runs.push(run),
                Err(e) => {
                    failure = Some(DocumentFailure {
                        file: document.label(),
                        path: document.path.display().to_string(),
                        judge: Some(judge.name().to_string()),
                        error: format!("{e:#}"),
                    });
                    break;
                }
            }
        }
        if let Some(failure) = failure {
            failures.push(failure);
            continue;
        }

        scored_documents += 1;
        let baseline_ms = runs[0].classify_ms;
        for (index, run) in runs.into_iter().enumerate() {
            let scores = score_pages(&document.truth, &run.verdicts);
            confidences[index].extend(run.verdicts.iter().map(|verdict| ScoredConfidence {
                confidence: verdict.confidence,
                correct: verdict.needs_ocr == document.truth.contains(&verdict.page),
            }));
            rows[index].push(DocumentReport {
                file: document.label(),
                source: document.spec.source.clone(),
                page_count: run.classification.page_count,
                pdf_type: run.classification.pdf_type.as_str().to_string(),
                route_exact: scores.route_exact,
                cost: cost(&scores.confusion, options.cost_per_page),
                confusion: scores.confusion,
                missed_ocr_pages: scores.missed_ocr_pages,
                wasted_ocr_pages: scores.wasted_ocr_pages,
                classify_ms: run.classify_ms,
                delta_vs_baseline_ms: (index > 0).then_some(run.classify_ms - baseline_ms),
            });
        }
    }

    let judge_reports: Vec<JudgeReport> = judges
        .iter()
        .enumerate()
        .map(|(index, judge)| {
            judge_report(
                judge.name(),
                index == 0,
                std::mem::take(&mut rows[index]),
                &confidences[index],
                &corpus,
                options.cost_per_page,
                judge.meter.as_ref().map(|meter| {
                    judge_spend(&meter.usage(), options.cost_per_million_input_tokens)
                }),
            )
        })
        .collect();

    let mut caveats: Vec<Caveat> = judge_reports.iter().flat_map(caveats_for).collect();
    // Only when something was actually scored. On an empty run every cost in
    // the report is zero for want of input, and a paragraph about how to read
    // those zeros would be the loudest thing in the output.
    if scored_documents > 0 {
        caveats.push(cost_caveat(scored_documents));
    }

    Ok(BenchReport {
        corpus: corpus.manifest_path.display().to_string(),
        repeat,
        cost_per_page: options.cost_per_page,
        cost_per_million_input_tokens: options.cost_per_million_input_tokens,
        documents_scored: scored_documents,
        documents_in_corpus: corpus.documents.len() as u32,
        failures,
        skipped_judges: skipped,
        judges: judge_reports,
        caveats,
    })
}

/// Classify one document `repeat` times with one judge, and check the result
/// against the manifest before anyone scores it.
fn run_document(
    bytes: &[u8],
    judge: &dyn PageJudge,
    repeat: u32,
    document: &CorpusDocument,
) -> Result<DocumentRun> {
    let mut samples = Vec::with_capacity(repeat as usize);
    let mut last: Option<(Classification, Vec<PageVerdict>)> = None;
    for _ in 0..repeat {
        let recorder = RecordingJudge::new(judge);
        let classification = classify_with(bytes, &recorder)
            .with_context(|| format!("could not classify {}", document.path.display()))?;
        samples.push(classification.classify_ms);
        last = Some((classification, recorder.into_verdicts()));
    }
    let (classification, verdicts) = last.expect("repeat is at least 1");

    // The page_count check. An off-by-one or a stale manifest row would score a
    // judge against a page numbering that describes a different document, and
    // every number downstream would look reasonable and mean nothing. So it
    // fails the document rather than warning about it.
    if classification.page_count != document.spec.page_count {
        return Err(anyhow!(
            "page_count mismatch: the manifest declares {} page(s), the classifier reports {}. \
             The manifest's 0-indexed needs_ocr list ({:?}) therefore describes a different \
             document; excluded from the score rather than scored against the wrong pages",
            document.spec.page_count,
            classification.page_count,
            document.truth
        ));
    }
    if verdicts.len() as u32 != classification.page_count {
        return Err(anyhow!(
            "judge `{}` produced {} verdict(s) for {} page(s)",
            judge.name(),
            verdicts.len(),
            classification.page_count
        ));
    }
    let routed: Vec<u32> = verdicts
        .iter()
        .filter(|verdict| verdict.needs_ocr)
        .map(|verdict| verdict.page)
        .collect();
    if routed != classification.pages_needing_ocr {
        return Err(anyhow!(
            "judge `{}` verdicts route {routed:?} but the classification says {:?}",
            judge.name(),
            classification.pages_needing_ocr
        ));
    }

    Ok(DocumentRun {
        classification,
        verdicts,
        classify_ms: median(&samples),
    })
}

/// Assemble one judge's report from its per-document rows.
fn judge_report(
    name: &str,
    baseline: bool,
    documents: Vec<DocumentReport>,
    confidences: &[ScoredConfidence],
    corpus: &Corpus,
    cost_per_page: f64,
    spend: Option<JudgeSpend>,
) -> JudgeReport {
    let by_source: Vec<GroupReport> = corpus
        .sources()
        .into_iter()
        .map(|source| {
            let rows: Vec<&DocumentReport> = documents
                .iter()
                .filter(|row| row.source == source)
                .collect();
            group(source, &rows, cost_per_page)
        })
        .filter(|group| group.documents > 0)
        .collect();
    let overall = group(
        "all".to_string(),
        &documents.iter().collect::<Vec<_>>(),
        cost_per_page,
    );
    let timings: Vec<f64> = documents.iter().map(|row| row.classify_ms).collect();
    let deltas: Vec<f64> = documents
        .iter()
        .filter_map(|row| row.delta_vs_baseline_ms)
        .collect();

    JudgeReport {
        judge: name.to_string(),
        baseline,
        by_source,
        overall,
        latency: latency(&timings),
        overhead_vs_baseline_ms: (!baseline && !deltas.is_empty()).then(|| median(&deltas)),
        confidence: confidence_report(confidences),
        spend,
        documents,
    }
}

/// Sum a set of document rows into a group.
fn group(source: String, rows: &[&DocumentReport], cost_per_page: f64) -> GroupReport {
    let mut confusion = Confusion::default();
    for row in rows {
        confusion.add(&row.confusion);
    }
    GroupReport {
        source,
        documents: rows.len() as u32,
        route_exact: rows.iter().filter(|row| row.route_exact).count() as u32,
        cost: cost(&confusion, cost_per_page),
        confusion,
    }
}

/// The note a perfect score on synthetic fixtures earns.
///
/// Stated in the harness's own output rather than left for a reader to infer: a
/// perfect score here is a floor, not a ranking.
fn caveats_for(report: &JudgeReport) -> Vec<Caveat> {
    report
        .by_source
        .iter()
        .filter(|group| group.source.eq_ignore_ascii_case("synthetic"))
        .filter(|group| group.confusion.is_perfect() && group.confusion.pages() > 0)
        .map(|group| Caveat {
            judge: Some(report.judge.clone()),
            source: Some(group.source.clone()),
            documents: group.documents,
            lines: vec![
                format!(
                    "`{}` scores perfectly on all {} synthetic document(s): {} page(s), \
                     0 missed, 0 wasted.",
                    report.judge,
                    group.documents,
                    group.confusion.pages()
                ),
                "A perfect score on synthetic fixtures measures non-regression, and nothing else."
                    .to_string(),
                "It cannot separate judges: every page in these files either carries text"
                    .to_string(),
                "operators or carries none, which is exactly the signal the heuristic reads,"
                    .to_string(),
                "so no judge can score higher here and no two judges can be told apart."
                    .to_string(),
                String::new(),
                "Separating judges needs real documents where the text layer lies:".to_string(),
                "  - a scanned page carrying a bad pre-existing OCR layer;".to_string(),
                "  - a text page hidden under a full-page image watermark;".to_string(),
                "  - a CID-font document with a broken ToUnicode map.".to_string(),
                "Add them to the manifest with a source of their own; see tests/corpus/README.md."
                    .to_string(),
            ],
        })
        .collect()
}

/// What the COST block does not price, said in the harness's own output.
///
/// The block reports what a judge's routing would be billed against OCR-ing
/// every page. That subtraction is exact and its inputs are not, and a reader
/// who takes the saving to the finance team should know which parts of it this
/// corpus can stand behind.
fn cost_caveat(documents: u32) -> Caveat {
    Caveat {
        judge: None,
        source: None,
        documents,
        lines: vec![
            "The COST block's saving is arithmetic over this corpus, not a forecast.".to_string(),
            "  - It prices pages, linearly. Real OCR contracts have minimums, tiers and"
                .to_string(),
            "    per-document fees, so a saving of half the pages is not a saving of half"
                .to_string(),
            "    the bill.".to_string(),
            "  - The saving is measured against OCR-ing every page, which is the baseline"
                .to_string(),
            "    that needs no judge. It is not measured against whatever you route with"
                .to_string(),
            "    today, and this harness has never seen that.".to_string(),
            "  - It prices OCR and, if you passed a token price, the judge's own calls. It"
                .to_string(),
            "    does not price the judge's latency, its failures, or the engineering cost"
                .to_string(),
            "    of depending on a vendor.".to_string(),
            "  - `missed OCR` is priced as underspend and is never netted off. A judge can"
                .to_string(),
            "    still look cheap here while being wrong; the ceiling line is what makes"
                .to_string(),
            "    that visible, and the error counts above are what make it legible.".to_string(),
            "  - Token counts cover the whole run, every `--repeat` included, because that"
                .to_string(),
            "    is what the run was billed. One pass over the corpus costs proportionally"
                .to_string(),
            "    less.".to_string(),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use doc_router::HeuristicJudge;

    #[test]
    fn the_baseline_is_always_first_even_when_it_was_not_asked_for() {
        let resolved = resolve_judges(&[]).expect("the baseline always resolves");
        assert_eq!(resolved.judges.len(), 1);
        assert_eq!(resolved.judges[0].name(), BASELINE_JUDGE);
        assert!(resolved.skipped.is_empty());
    }

    #[test]
    fn naming_the_baseline_does_not_run_it_twice() {
        let resolved = resolve_judges(&[BASELINE_JUDGE.to_string(), BASELINE_JUDGE.to_string()])
            .expect("resolves");
        assert_eq!(resolved.judges.len(), 1);
    }

    /// A hosted judge with no key on this machine is skipped, not fatal, and the
    /// baseline still runs. On a machine that *does* have a key it simply runs,
    /// so the test asserts the invariant both branches share: the request was
    /// honoured somewhere, and the run was not stopped.
    #[test]
    fn a_judge_that_cannot_be_built_here_is_skipped_rather_than_fatal() {
        let resolved = resolve_judges(&["jev".to_string()]).expect("a known name never errors");
        assert_eq!(resolved.judges[0].name(), BASELINE_JUDGE);
        let ran = resolved.judges.iter().any(|judge| judge.name() == "jev");
        let skipped = resolved.skipped.iter().any(|skip| skip.judge == "jev");
        assert!(ran ^ skipped, "`jev` either runs or is reported as skipped");
        if skipped {
            assert_eq!(resolved.judges.len(), 1, "the baseline still runs");
            let reason = &resolved.skipped[0].reason;
            assert!(reason.contains("TYPESAFE_API_KEY"), "{reason}");
        }
    }

    #[test]
    fn an_unknown_judge_name_stops_the_run_and_lists_the_registered_ones() {
        let text = match resolve_judges(&["nope".to_string()]) {
            Ok(_) => panic!("an unknown judge must not silently fall back to the baseline"),
            Err(e) => format!("{e:#}"),
        };
        assert!(text.contains("unknown judge `nope`"), "{text}");
        assert!(text.contains(BASELINE_JUDGE), "{text}");
    }

    #[test]
    fn the_recording_judge_sees_the_verdicts_without_changing_them() {
        let evidence = vec![PageEvidence {
            page: 0,
            text: None,
            reasons: vec!["no_text_layer".to_string()],
            flagged_by_inspector: true,
            has_tables: false,
            has_columns: false,
            has_encoding_issues: false,
        }];
        let recorder = RecordingJudge::new(&HeuristicJudge);
        assert_eq!(recorder.name(), "heuristic");
        assert!(!recorder.needs_text());
        let passed_through = recorder.judge(&evidence).expect("heuristic never fails");
        assert_eq!(recorder.into_verdicts(), passed_through);
        assert!(passed_through[0].needs_ocr);
    }
}
