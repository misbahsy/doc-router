//! Rendering a [`BenchReport`] for a human.
//!
//! The order is deliberate. Raw error counts come first, split into the two
//! kinds, because they are the numbers that tell you whether the judge is safe
//! to route with. Precision, recall and F1 come after, as derived views of
//! those counts -- they compress the two failure modes into one number, which is
//! convenient for ranking and useless for deciding whether a regression is a
//! quality problem or a billing one.

use std::fmt::Write as _;

use crate::bench::{BenchReport, DocumentReport, GroupReport, JudgeReport};
use crate::score::{ConfidenceReport, Cost, Latency};

/// Render the whole run as text.
pub fn render(report: &BenchReport) -> String {
    let mut out = String::new();
    header(&mut out, report);
    skipped(&mut out, report);
    failures(&mut out, report);
    for judge in &report.judges {
        judge_section(&mut out, judge, report);
    }
    caveats(&mut out, report);
    out
}

fn header(out: &mut String, report: &BenchReport) {
    let _ = writeln!(out, "doc-router bench");
    let _ = writeln!(out, "corpus   {}", report.corpus);
    let _ = writeln!(
        out,
        "scored   {} of {} document(s){}",
        report.documents_scored,
        report.documents_in_corpus,
        if report.failures.is_empty() {
            String::new()
        } else {
            format!(", {} excluded (see below)", report.failures.len())
        }
    );
    let _ = writeln!(
        out,
        "repeat   {} run(s) per document; every latency below is that document's median",
        report.repeat
    );
    let _ = writeln!(
        out,
        "cost     {:.4} per OCR page -- abstract cost units unless you passed a price of your",
        report.cost_per_page
    );
    let _ = writeln!(
        out,
        "         own via --cost-per-page; no vendor's price is baked in"
    );
    if let Some(price) = report.cost_per_million_input_tokens {
        let _ = writeln!(
            out,
            "tokens   {price:.4} per million input tokens to a hosted judge; output tokens are"
        );
        let _ = writeln!(
            out,
            "         counted and not priced -- see THE JUDGE ITSELF under COST"
        );
    }
    let _ = writeln!(out, "judges   {}", judge_names(report));
}

fn judge_names(report: &BenchReport) -> String {
    report
        .judges
        .iter()
        .map(|judge| {
            if judge.baseline {
                format!("{} (baseline)", judge.judge)
            } else {
                judge.judge.clone()
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// The judges that were asked for and are not in the report below.
///
/// Printed before any score, and stated as a skip rather than an error: the run
/// is fine, this machine just cannot build that judge. Saying so up front is the
/// difference between a reader trusting the numbers that *are* there and a
/// reader wondering whether the missing row scored zero.
fn skipped(out: &mut String, report: &BenchReport) {
    if report.skipped_judges.is_empty() {
        return;
    }
    let _ = writeln!(out, "\nSKIPPED JUDGES ({})", report.skipped_judges.len());
    for line in wrap(
        "These were requested and could not be built on this machine. They are not scored and \
         not counted as failures; the exit code is unaffected. No row below was produced by a \
         judge other than the one it names.",
        86,
    ) {
        let _ = writeln!(out, "  {line}");
    }
    for skip in &report.skipped_judges {
        let _ = writeln!(out, "  - {}", skip.judge);
        for line in wrap(&skip.reason, 80) {
            let _ = writeln!(out, "      {line}");
        }
    }
}

fn failures(out: &mut String, report: &BenchReport) {
    if report.failures.is_empty() {
        return;
    }
    let _ = writeln!(out, "\nEXCLUDED DOCUMENTS ({})", report.failures.len());
    for line in wrap(
        "These could not be scored at all. They are harness failures, not judge scores, and \
         the exit code is non-zero because of them. A judge scoring badly is a result, not a \
         failure, and never lands here.",
        86,
    ) {
        let _ = writeln!(out, "  {line}");
    }
    for failure in &report.failures {
        let _ = writeln!(out, "  - {}", failure.file);
        if let Some(judge) = &failure.judge {
            let _ = writeln!(out, "      judge: {judge}");
        }
        for line in wrap(&failure.error, 82) {
            let _ = writeln!(out, "      {line}");
        }
    }
}

fn judge_section(out: &mut String, judge: &JudgeReport, report: &BenchReport) {
    let _ = writeln!(
        out,
        "\n================ judge: {}{} ================",
        judge.judge,
        if judge.baseline { " (baseline)" } else { "" }
    );

    errors_block(out, judge);
    documents_block(out, judge);
    sources_block(out, judge);
    rates_block(out, judge);
    cost_block(out, judge, report);
    latency_block(out, judge, report.repeat);
    confidence_block(out, &judge.confidence);
}

fn errors_block(out: &mut String, judge: &JudgeReport) {
    let confusion = &judge.overall.confusion;
    let _ = writeln!(
        out,
        "\nERRORS (raw counts; the two kinds are never added together)"
    );
    let _ = writeln!(
        out,
        "  missed OCR   {:>5}   page needed OCR and did not get it",
        confusion.missed_ocr
    );
    let _ = writeln!(
        out,
        "                       -> silent quality failure: the document routes as text and the"
    );
    let _ = writeln!(out, "                          text is not there");
    let _ = writeln!(
        out,
        "  wasted OCR   {:>5}   page did not need OCR and got it",
        confusion.wasted_ocr
    );
    let _ = writeln!(
        out,
        "                       -> costs money and latency; the output is still correct"
    );
    // Ground truth, not this judge's true positives and true negatives. The
    // two are easy to confuse and the line says "truly", so it has to be the
    // pair that is the same for every judge scored against these documents:
    // `correct_ocr` here would make the corpus appear to change with the judge.
    let _ = writeln!(
        out,
        "  pages        {:>5}   {} truly need OCR, {} truly do not",
        confusion.pages(),
        confusion.truly_needs_ocr(),
        confusion.truly_local()
    );
    let _ = writeln!(
        out,
        "  documents    {:>5}   {} routed exactly (every page on the correct side)",
        judge.overall.documents, judge.overall.route_exact
    );
}

fn documents_block(out: &mut String, judge: &JudgeReport) {
    if judge.documents.is_empty() {
        let _ = writeln!(out, "\nPER DOCUMENT\n  (nothing scored)");
        return;
    }
    let width = judge
        .documents
        .iter()
        .map(|row| row.file.len())
        .max()
        .unwrap_or(8)
        .max(8);
    let _ = writeln!(out, "\nPER DOCUMENT");
    let _ = writeln!(
        out,
        "  {:<width$}  {:>5}  {:<12}  {:>6}  {:>6}  {:>5}  {:>9}",
        "document", "pages", "pdf_type", "missed", "wasted", "route", "ms",
    );
    for row in &judge.documents {
        let _ = writeln!(
            out,
            "  {:<width$}  {:>5}  {:<12}  {:>6}  {:>6}  {:>5}  {:>9.3}{}",
            row.file,
            row.page_count,
            row.pdf_type,
            row.confusion.missed_ocr,
            row.confusion.wasted_ocr,
            if row.route_exact { "exact" } else { "OFF" },
            row.classify_ms,
            delta_suffix(row),
        );
        offending_pages(out, row, width);
    }
}

fn delta_suffix(row: &DocumentReport) -> String {
    match row.delta_vs_baseline_ms {
        Some(delta) => format!("  ({delta:+.3} vs baseline)"),
        None => String::new(),
    }
}

/// The page numbers behind a non-zero count, because "2 missed" is not
/// actionable and "missed pages [5, 11]" is.
fn offending_pages(out: &mut String, row: &DocumentReport, width: usize) {
    let pad = width + 2;
    if !row.missed_ocr_pages.is_empty() {
        let _ = writeln!(
            out,
            "  {:<pad$}missed OCR on 0-indexed page(s) {:?}",
            "", row.missed_ocr_pages
        );
    }
    if !row.wasted_ocr_pages.is_empty() {
        let _ = writeln!(
            out,
            "  {:<pad$}wasted OCR on 0-indexed page(s) {:?}",
            "", row.wasted_ocr_pages
        );
    }
}

fn sources_block(out: &mut String, judge: &JudgeReport) {
    let _ = writeln!(out, "\nBY SOURCE");
    let width = judge
        .by_source
        .iter()
        .map(|group| group.source.len())
        .max()
        .unwrap_or(6)
        .max(6);
    let _ = writeln!(
        out,
        "  {:<width$}  {:>4}  {:>5}  {:>6}  {:>6}  {:>11}",
        "source", "docs", "pages", "missed", "wasted", "route exact",
    );
    for group in &judge.by_source {
        let _ = writeln!(
            out,
            "  {:<width$}  {:>4}  {:>5}  {:>6}  {:>6}  {:>11}",
            group.source,
            group.documents,
            group.confusion.pages(),
            group.confusion.missed_ocr,
            group.confusion.wasted_ocr,
            format!("{}/{}", group.route_exact, group.documents),
        );
    }
}

fn rates_block(out: &mut String, judge: &JudgeReport) {
    let confusion = &judge.overall.confusion;
    let _ = writeln!(out, "\nDERIVED RATES (from the counts above)");
    let _ = writeln!(
        out,
        "  precision  {}   of the pages routed to OCR, the share that needed it",
        rate(confusion.precision(), "no pages routed to OCR")
    );
    let _ = writeln!(
        out,
        "  recall     {}   of the pages needing OCR, the share that got it",
        rate(confusion.recall(), "no page needed OCR")
    );
    let _ = writeln!(
        out,
        "  f1         {}   their harmonic mean; it hides which of the two failed,",
        rate(confusion.f1(), "undefined without both")
    );
    let _ = writeln!(out, "                      so read the counts above first");
}

/// `None` prints as a reason rather than as `0.000`: a rate with an empty
/// denominator is undefined, and printing a zero would read as a failure.
fn rate(value: Option<f64>, why: &str) -> String {
    match value {
        Some(value) => format!("{value:>6.3}"),
        None => format!("{:>6}", format!("n/a ({why})")),
    }
}

fn cost_block(out: &mut String, judge: &JudgeReport, report: &BenchReport) {
    let cost = &judge.overall.cost;
    let cost_per_page = report.cost_per_page;
    let _ = writeln!(out, "\nCOST ({cost_per_page:.4} per OCR page)");
    let _ = writeln!(
        out,
        "  OCR pages routed   {:>5}   ideal {:>5}",
        cost.ocr_pages, cost.ideal_ocr_pages
    );
    let _ = writeln!(
        out,
        "  overspend          {:>5} page(s)  {:>10.4}   paid for OCR that was not needed",
        cost.overspend_pages, cost.overspend
    );
    let _ = writeln!(
        out,
        "  underspend         {:>5} page(s)  {:>10.4}   the quality failure priced: OCR that",
        cost.underspend_pages, cost.underspend
    );
    let _ = writeln!(
        out,
        "                                                 was needed and not bought. Not a"
    );
    let _ = writeln!(
        out,
        "                                                 saving, and never netted off the"
    );
    let _ = writeln!(
        out,
        "                                                 overspend above."
    );
    savings_block(out, cost);
    judge_spend_block(out, judge, cost);
}

/// What the routing saved against not routing at all — reported only ever
/// alongside the ceiling.
///
/// "OCR everything" is the baseline worth measuring against because it is the
/// one that is always available: it needs no judge, no key and no network, and
/// it never misses a page. The saving against it is the number a reader came
/// for, and on its own it is dangerous, because **under-buying OCR looks exactly
/// like efficiency**. A judge that routes nothing saves everything and is
/// useless.
///
/// So the ceiling is printed on the next line and the two are compared out loud.
/// A correct router still has to buy every page that truly needs OCR, so the
/// ceiling is the most any honest saving can be; a judge past it has not been
/// clever, it has skipped work. The comparison is the point of the block, which
/// is why the verdict line is not optional and is phrased as a judgement.
fn savings_block(out: &mut String, cost: &Cost) {
    let _ = writeln!(
        out,
        "\n  VS OCR EVERYTHING (the baseline that needs no judge and never misses a page)"
    );
    if cost.all_pages == 0 {
        let _ = writeln!(
            out,
            "    n/a (nothing was scored, so there is no bill to compare against)"
        );
        return;
    }
    let saving_pages = cost.all_pages - cost.ocr_pages;
    let ceiling_pages = cost.all_pages - cost.ideal_ocr_pages;
    let _ = writeln!(
        out,
        "    OCR everything   {:>5} page(s)  {:>10.4}",
        cost.all_pages, cost.naive
    );
    let _ = writeln!(
        out,
        "    this judge       {:>5} page(s)  {:>10.4}   {} of the naive bill",
        cost.ocr_pages,
        cost.routed,
        rate(cost.routed_share, "nothing scored")
    );
    let _ = writeln!(
        out,
        "    saved            {:>5} page(s)  {:>10.4}",
        saving_pages, cost.saving
    );
    let _ = writeln!(
        out,
        "    ceiling          {:>5} page(s)  {:>10.4}   the most a router that agreed with",
        ceiling_pages, cost.ceiling_saving
    );
    let _ = writeln!(
        out,
        "                                                 truth on every page could save; it"
    );
    let _ = writeln!(
        out,
        "                                                 still has to buy the {} page(s)",
        cost.ideal_ocr_pages
    );
    let _ = writeln!(
        out,
        "                                                 that truly need OCR"
    );
    for line in wrap(&ceiling_verdict(cost), 82) {
        let _ = writeln!(out, "    {line}");
    }
}

/// The sentence that stops the saving being read on its own.
fn ceiling_verdict(cost: &Cost) -> String {
    let excess = cost.saving_beyond_ceiling_pages;
    // Both gaps are read off the priced figures rather than negated from each
    // other: at `--cost-per-page 0` a negation yields `-0.0000`, which reads as a
    // direction the number does not have. The page counts either side of it are
    // what carry the finding when the price is zero.
    let beyond_ceiling = cost.saving - cost.ceiling_saving;
    let short_of_ceiling = cost.ceiling_saving - cost.saving;
    match excess {
        // Saved more than a correct router could. The only way to do that is to
        // not buy OCR that was needed, so the excess is named as what it is.
        excess if excess > 0 => format!(
            "-> this judge saves {:.4} MORE than the ceiling, which no correct router can do.              The excess is {} page(s) of OCR it needed and did not buy, less {} page(s) it              bought and did not need. Those {} page(s) are silent quality failures, not              savings, and they are why this row looks frugal.",
            beyond_ceiling, cost.underspend_pages, cost.overspend_pages, cost.underspend_pages
        ),
        // Exactly on the ceiling. Perfect judges land here — and so do judges
        // whose two errors happen to be the same size, which is a coincidence in
        // the bill and not one in the output.
        0 if cost.underspend_pages == 0 && cost.overspend_pages == 0 => {
            "-> this judge is on the ceiling and made no error of either kind: the best a router              can do on this corpus."
                .to_string()
        }
        0 => format!(
            "-> this judge lands exactly on the ceiling and is NOT exact: {} missed page(s) and              {} wasted page(s) are the same size, so they cancel in the bill and do not cancel              in the output. Read the error counts above, not this line.",
            cost.underspend_pages, cost.overspend_pages
        ),
        _ => format!(
            "-> this judge saves {:.4} less than the ceiling. That shortfall is the overspend              above: {} page(s) of OCR bought and not needed. It costs money and the output is              still correct.",
            short_of_ceiling, cost.overspend_pages
        ),
    }
}

/// What the judge itself cost, and the saving net of it.
///
/// A hosted judge is not free, and a saving that ignores the judge's own bill is
/// half an answer. The three states below are deliberately printed differently:
/// a judge with no token data is not a judge that used no tokens, and neither is
/// a `0.0000`.
fn judge_spend_block(out: &mut String, judge: &JudgeReport, cost: &Cost) {
    let _ = writeln!(
        out,
        "\n  THE JUDGE ITSELF (its own calls, not the OCR it buys)"
    );
    let Some(spend) = &judge.spend else {
        for line in wrap(
            "tokens   n/a (this judge reports no token usage to the harness, so its own cost is              unknown here -- which is not the same as free. The baseline makes no calls at all;              a judge that does and cannot account for them would print this too.)",
            82,
        ) {
            let _ = writeln!(out, "    {line}");
        }
        let _ = writeln!(
            out,
            "    net      n/a (the saving above is not net of the judge)"
        );
        return;
    };

    if spend.calls == 0 {
        for line in wrap(
            "calls    0 -- this judge made no call on this run, so it used nothing and cost              nothing. This zero is a measurement.",
            82,
        ) {
            let _ = writeln!(out, "    {line}");
        }
    } else {
        let _ = writeln!(
            out,
            "    calls    {:>10}   over the whole run, every --repeat included",
            spend.calls
        );
        let _ = writeln!(
            out,
            "    input    {:>10} token(s){}",
            spend.input_tokens,
            if spend.usage_complete {
                ""
            } else {
                "  (a floor)"
            }
        );
        let _ = writeln!(
            out,
            "    output   {:>10} token(s)  counted, not priced",
            spend.output_tokens
        );
        if !spend.usage_complete {
            for line in wrap(
                &format!(
                    "-> {} of {} call(s) came back with no usage attached, so the token counts                      and every figure derived from them are a floor and not a total.",
                    spend.calls - spend.calls_reporting_usage,
                    spend.calls
                ),
                82,
            ) {
                let _ = writeln!(out, "    {line}");
            }
        }
    }

    match spend.cost {
        Some(judge_cost) => {
            let _ = writeln!(out, "    cost     {judge_cost:>10.4}");
            if cost.all_pages == 0 {
                let _ = writeln!(
                    out,
                    "    net      n/a (nothing was scored, so there is no saving to net it off)"
                );
            } else {
                let _ = writeln!(
                    out,
                    "    net      {:>10.4}   saved on OCR ({:.4}) less the judge's own cost",
                    cost.saving - judge_cost,
                    cost.saving
                );
                if cost.saving_beyond_ceiling_pages > 0 {
                    for line in wrap(
                        "-> this nets off a saving that is already above the ceiling, so it                          inherits that problem: paying less for a judge that under-buys OCR                          does not make the under-buying cheaper.",
                        82,
                    ) {
                        let _ = writeln!(out, "    {line}");
                    }
                }
            }
        }
        None => {
            for line in wrap(
                &format!(
                    "cost     n/a (no --cost-per-million-input-tokens was given, so the {}                      input token(s) above are counted and not priced; no vendor's rate is baked                      in here either)",
                    spend.input_tokens
                ),
                82,
            ) {
                let _ = writeln!(out, "    {line}");
            }
            let _ = writeln!(
                out,
                "    net      n/a (the saving above is not net of the judge)"
            );
        }
    }
}

fn latency_block(out: &mut String, judge: &JudgeReport, repeat: u32) {
    let Latency {
        p50_ms,
        p95_ms,
        max_ms,
        documents,
    } = judge.latency;
    let _ = writeln!(
        out,
        "\nLATENCY (classify_ms; each document's median of {repeat} run(s), then across \
         {documents} document(s))"
    );
    let _ = writeln!(
        out,
        "  p50 {p50_ms:.3} ms   p95 {p95_ms:.3} ms   max {max_ms:.3} ms"
    );
    match judge.overhead_vs_baseline_ms {
        Some(overhead) => {
            let _ = writeln!(
                out,
                "  judge overhead vs baseline: {overhead:+.3} ms (median of the per-document \
                 differences)"
            );
        }
        None if judge.baseline => {
            let _ = writeln!(
                out,
                "  judge overhead vs baseline: 0 by definition -- this is the baseline"
            );
        }
        None => {}
    }
}

fn confidence_block(out: &mut String, confidence: &ConfidenceReport) {
    let _ = writeln!(out, "\nCONFIDENCE");
    let _ = writeln!(
        out,
        "  {} distinct value(s) over {} verdict(s)",
        confidence.distinct.len(),
        confidence.verdicts
    );
    for value in &confidence.distinct {
        let _ = writeln!(out, "    {:.3}  x{}", f64::from(value.value), value.count);
    }
    if confidence.degenerate {
        if let Some(constant) = confidence.constant {
            let _ = writeln!(out, "  degenerate -- constant {:.3}", f64::from(constant));
        } else {
            let _ = writeln!(out, "  degenerate -- no verdicts");
        }
        if let Some(reason) = &confidence.skipped_reason {
            for line in wrap(reason, 88) {
                let _ = writeln!(out, "  {line}");
            }
        }
        return;
    }

    let _ = writeln!(
        out,
        "\n  RELIABILITY (does a confidence of c mean it is right about c of the time?)"
    );
    let _ = writeln!(
        out,
        "  {:<14}  {:>7}  {:>7}  {:>8}  {:>10}",
        "bucket", "pages", "correct", "accuracy", "mean conf",
    );
    for bucket in &confidence.buckets {
        let _ = writeln!(
            out,
            "  [{:.1}, {:.1}{}  {:>7}  {:>7}  {:>8.3}  {:>10.3}",
            bucket.lower,
            bucket.upper,
            if (bucket.upper - 1.0).abs() < f64::EPSILON {
                "]".to_string()
            } else {
                ")".to_string()
            },
            bucket.count,
            bucket.correct,
            bucket.accuracy,
            bucket.mean_confidence,
        );
    }
    if let Some(brier) = confidence.brier {
        let _ = writeln!(
            out,
            "  Brier {brier:.4} (mean squared error of the confidence against whether that \
             verdict was right; 0 is perfect, 0.25 is a coin flip stated at 0.5)"
        );
    }
}

fn caveats(out: &mut String, report: &BenchReport) {
    if report.caveats.is_empty() {
        return;
    }
    let _ = writeln!(out, "\nWHAT THIS RUN CANNOT TELL YOU");
    for caveat in &report.caveats {
        for line in &caveat.lines {
            if line.is_empty() {
                let _ = writeln!(out);
            } else {
                let _ = writeln!(out, "  {line}");
            }
        }
    }
}

/// Wrap `text` on word boundaries at `width` columns.
fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut line = String::new();
    for word in text.split_whitespace() {
        if !line.is_empty() && line.len() + 1 + word.len() > width {
            lines.push(std::mem::take(&mut line));
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        lines.push(line);
    }
    lines
}

/// Group header used by the JSON path's summary line.
pub fn one_line(group: &GroupReport) -> String {
    format!(
        "{}: {} doc(s), {} page(s), {} missed, {} wasted",
        group.source,
        group.documents,
        group.confusion.pages(),
        group.confusion.missed_ocr,
        group.confusion.wasted_ocr
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::{BenchReport, Caveat, GroupReport, JudgeReport};
    use crate::meter::TokenUsage;
    use crate::score::{
        confidence_report, cost, judge_spend, latency, Confusion, ScoredConfidence,
    };

    fn document(file: &str, missed: Vec<u32>, wasted: Vec<u32>) -> DocumentReport {
        // A 4-page document: 2 pages truly need OCR, 2 truly do not, and the
        // arguments say which of them the judge got wrong.
        let confusion = Confusion {
            correct_ocr: 2 - missed.len() as u32,
            correct_local: 2 - wasted.len() as u32,
            missed_ocr: missed.len() as u32,
            wasted_ocr: wasted.len() as u32,
        };
        DocumentReport {
            file: file.to_string(),
            source: "synthetic".to_string(),
            page_count: 4,
            pdf_type: "mixed".to_string(),
            route_exact: missed.is_empty() && wasted.is_empty(),
            cost: cost(&confusion, 1.0),
            confusion,
            missed_ocr_pages: missed,
            wasted_ocr_pages: wasted,
            classify_ms: 1.5,
            delta_vs_baseline_ms: None,
        }
    }

    fn report_with(documents: Vec<DocumentReport>, scored: &[ScoredConfidence]) -> BenchReport {
        let mut confusion = Confusion::default();
        for row in &documents {
            confusion.add(&row.confusion);
        }
        let overall = GroupReport {
            source: "all".to_string(),
            documents: documents.len() as u32,
            route_exact: documents.iter().filter(|row| row.route_exact).count() as u32,
            cost: cost(&confusion, 1.0),
            confusion,
        };
        let group = GroupReport {
            source: "synthetic".to_string(),
            ..overall.clone()
        };
        let timings: Vec<f64> = documents.iter().map(|row| row.classify_ms).collect();
        BenchReport {
            corpus: "/tmp/manifest.json".to_string(),
            repeat: 3,
            cost_per_page: 1.0,
            cost_per_million_input_tokens: None,
            documents_scored: documents.len() as u32,
            documents_in_corpus: documents.len() as u32,
            failures: Vec::new(),
            skipped_judges: Vec::new(),
            judges: vec![JudgeReport {
                judge: "heuristic".to_string(),
                baseline: true,
                by_source: vec![group],
                overall,
                latency: latency(&timings),
                overhead_vs_baseline_ms: None,
                confidence: confidence_report(scored),
                documents,
                spend: None,
            }],
            caveats: Vec::new(),
        }
    }

    #[test]
    fn a_clean_run_leads_with_the_two_error_counts_and_never_adds_them_together() {
        let report = report_with(
            vec![document("mixed.pdf", vec![], vec![])],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: true,
            }],
        );
        let text = render(&report);
        let errors = text.find("ERRORS").expect("errors block");
        let rates = text.find("DERIVED RATES").expect("rates block");
        assert!(errors < rates, "raw counts must come before derived rates");
        assert!(text.contains("missed OCR       0"), "{text}");
        assert!(text.contains("wasted OCR       0"), "{text}");
        assert!(
            !text.contains("total errors"),
            "the two kinds must never be summed: {text}"
        );
    }

    #[test]
    fn offending_page_numbers_are_printed_under_the_document_that_owns_them() {
        let report = report_with(
            vec![document("mixed.pdf", vec![1], vec![2, 3])],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: false,
            }],
        );
        let text = render(&report);
        assert!(
            text.contains("missed OCR on 0-indexed page(s) [1]"),
            "{text}"
        );
        assert!(
            text.contains("wasted OCR on 0-indexed page(s) [2, 3]"),
            "{text}"
        );
    }

    #[test]
    fn a_degenerate_confidence_prints_the_constant_and_no_reliability_table() {
        let scored = vec![
            ScoredConfidence {
                confidence: 1.0,
                correct: true,
            },
            ScoredConfidence {
                confidence: 1.0,
                correct: false,
            },
        ];
        let report = report_with(vec![document("mixed.pdf", vec![1], vec![])], &scored);
        let text = render(&report);
        assert!(text.contains("degenerate -- constant 1.000"), "{text}");
        assert!(!text.contains("RELIABILITY"), "{text}");
        assert!(!text.contains("Brier"), "{text}");
    }

    #[test]
    fn varying_confidence_prints_the_reliability_table_and_a_brier_score() {
        let scored = vec![
            ScoredConfidence {
                confidence: 0.9,
                correct: true,
            },
            ScoredConfidence {
                confidence: 0.2,
                correct: false,
            },
        ];
        let report = report_with(vec![document("mixed.pdf", vec![], vec![])], &scored);
        let text = render(&report);
        assert!(text.contains("RELIABILITY"), "{text}");
        assert!(text.contains("Brier"), "{text}");
    }

    #[test]
    fn an_undefined_rate_says_so_instead_of_printing_zero() {
        // A document with no OCR pages at all: precision has an empty
        // denominator, and printing 0.000 would read as "got everything wrong".
        let confusion = Confusion {
            correct_ocr: 0,
            correct_local: 2,
            missed_ocr: 0,
            wasted_ocr: 0,
        };
        let mut report = report_with(
            vec![document("text.pdf", vec![], vec![])],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: true,
            }],
        );
        report.judges[0].overall.confusion = confusion;
        let text = render(&report);
        assert!(text.contains("n/a (no pages routed to OCR)"), "{text}");
        assert!(text.contains("n/a (no page needed OCR)"), "{text}");
    }

    #[test]
    fn an_excluded_document_is_called_out_before_any_score() {
        let mut report = report_with(
            vec![document("mixed.pdf", vec![], vec![])],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: true,
            }],
        );
        report.failures.push(crate::bench::DocumentFailure {
            file: "wrong.pdf".to_string(),
            path: "/tmp/wrong.pdf".to_string(),
            judge: Some("heuristic".to_string()),
            error: "page_count mismatch: the manifest declares 3 page(s)".to_string(),
        });
        let text = render(&report);
        let excluded = text.find("EXCLUDED DOCUMENTS").expect("excluded block");
        let judge = text.find("judge: heuristic").expect("judge section");
        assert!(excluded < judge, "{text}");
        assert!(text.contains("page_count mismatch"), "{text}");
    }

    #[test]
    fn a_skipped_judge_is_named_before_any_score_and_is_not_a_failure() {
        let mut report = report_with(
            vec![document("mixed.pdf", vec![], vec![])],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: true,
            }],
        );
        report.skipped_judges.push(crate::bench::SkippedJudge {
            judge: "jev".to_string(),
            reason: "no API key in the environment: set TYPESAFE_API_KEY".to_string(),
        });
        let text = render(&report);
        let skipped = text.find("SKIPPED JUDGES").expect("skipped block");
        let judge = text.find("judge: heuristic").expect("judge section");
        assert!(skipped < judge, "{text}");
        assert!(text.contains("TYPESAFE_API_KEY"), "{text}");
        assert!(
            !text.contains("EXCLUDED DOCUMENTS"),
            "a skip is not a failure"
        );
        assert!(report.ok(), "a skip must not change the exit code");
    }

    #[test]
    fn the_synthetic_caveat_is_printed_when_there_is_one() {
        let mut report = report_with(
            vec![document("mixed.pdf", vec![], vec![])],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: true,
            }],
        );
        report.caveats.push(Caveat {
            judge: Some("heuristic".to_string()),
            source: Some("synthetic".to_string()),
            documents: 1,
            lines: vec!["measures non-regression only".to_string()],
        });
        let text = render(&report);
        assert!(text.contains("WHAT THIS RUN CANNOT TELL YOU"), "{text}");
        assert!(text.contains("measures non-regression only"), "{text}");
    }

    fn one(missed: Vec<u32>, wasted: Vec<u32>) -> BenchReport {
        report_with(
            vec![document("mixed.pdf", missed, wasted)],
            &[ScoredConfidence {
                confidence: 1.0,
                correct: true,
            }],
        )
    }

    fn ground_truth_line(text: &str) -> String {
        text.lines()
            .find(|line| line.contains("truly need OCR"))
            .expect("the headline ground-truth line")
            .to_string()
    }

    /// The headline line says "truly", so it has to be a fact about the corpus.
    /// Four judges making four different sets of mistakes over the same four
    /// pages must print the same sentence.
    ///
    /// Printing the true positives and true negatives there instead -- the bug
    /// this test exists for -- makes the corpus appear to shrink as the judge
    /// gets worse.
    #[test]
    fn the_ground_truth_line_does_not_change_with_the_judge() {
        let judges = [
            (vec![], vec![]),
            (vec![0], vec![]),
            (vec![0, 1], vec![2, 3]),
            (vec![], vec![2]),
            (vec![1], vec![3]),
        ];
        let lines: Vec<String> = judges
            .into_iter()
            .map(|(missed, wasted)| ground_truth_line(&render(&one(missed, wasted))))
            .collect();
        assert!(
            lines.windows(2).all(|pair| pair[0] == pair[1]),
            "the corpus does not change with the judge: {lines:#?}"
        );
        assert!(
            lines[0].contains("2 truly need OCR, 2 truly do not"),
            "{}",
            lines[0]
        );
        assert!(
            lines[0].contains("pages            4"),
            "and the two sides add up to the page count: {}",
            lines[0]
        );
    }

    /// `--cost-per-page 0` prices every row at nothing, so only the page counts
    /// carry the finding. The verdict sentence has to survive that -- and a gap
    /// of zero must not print as `-0.0000`, which reads as a direction the
    /// number does not have.
    #[test]
    fn a_free_page_price_never_prints_a_signed_zero() {
        let mut report = one(vec![0, 1], vec![]);
        report.cost_per_page = 0.0;
        for judge in &mut report.judges {
            judge.overall.cost = cost(&judge.overall.confusion, 0.0);
            for group in &mut judge.by_source {
                group.cost = cost(&group.confusion, 0.0);
            }
        }
        let text = render(&report);
        assert!(!text.contains("-0.0000"), "{text}");
        assert!(
            text.contains("MORE than the ceiling, which no correct router can do"),
            "the page counts still say it undercut the ceiling: {text}"
        );
    }

    #[test]
    fn a_judge_that_routes_nothing_is_not_reported_as_impressively_frugal() {
        // The cheapest row this corpus can produce, and the most broken one:
        // both pages that needed OCR were missed and nothing was wasted.
        let text = render(&one(vec![0, 1], vec![]));
        assert!(text.contains("VS OCR EVERYTHING"), "{text}");
        assert!(text.contains("this judge           0 page(s)"), "{text}");
        assert!(
            text.contains("MORE than the ceiling, which no correct router can do"),
            "{text}"
        );
        assert!(
            text.contains("silent quality failures, not"),
            "the excess has to be named as the quality failure it is: {text}"
        );
    }

    #[test]
    fn a_judge_on_the_ceiling_with_errors_is_told_apart_from_a_correct_one() {
        let cancelling = render(&one(vec![0], vec![2]));
        assert!(
            cancelling.contains("lands exactly on the ceiling and is NOT exact"),
            "{cancelling}"
        );
        let exact = render(&one(vec![], vec![]));
        assert!(exact.contains("made no error of either kind"), "{exact}");
        assert!(
            !exact.contains("NOT exact"),
            "a correct router must not be accused of coincidence: {exact}"
        );
    }

    #[test]
    fn overspending_is_reported_as_a_shortfall_against_the_ceiling() {
        let text = render(&one(vec![], vec![2]));
        assert!(text.contains("less than the ceiling"), "{text}");
        assert!(
            text.contains("still correct"),
            "wasted OCR costs money and nothing else: {text}"
        );
    }

    #[test]
    fn a_run_that_scored_nothing_has_no_bill_to_compare_against() {
        let text = render(&report_with(Vec::new(), &[]));
        assert!(
            text.contains("n/a (nothing was scored, so there is no bill to compare against)"),
            "{text}"
        );
        assert!(
            !text.contains("MORE than the ceiling"),
            "nothing was scored, so nothing beat anything: {text}"
        );
    }

    #[test]
    fn a_judge_with_no_token_data_says_so_rather_than_printing_a_zero() {
        let text = render(&one(vec![0], vec![]));
        assert!(text.contains("THE JUDGE ITSELF"), "{text}");
        assert!(
            text.contains("reports no token usage to the harness"),
            "{text}"
        );
        assert!(text.contains("which is not the same as free"), "{text}");
        assert!(
            text.contains("net      n/a (the saving above is not net of the judge)"),
            "{text}"
        );
    }

    #[test]
    fn a_judge_that_made_no_call_prints_a_zero_and_says_it_is_a_measurement() {
        let mut report = one(vec![0], vec![]);
        report.judges[0].spend = Some(judge_spend(&TokenUsage::default(), Some(3.0)));
        let text = render(&report);
        assert!(text.contains("This zero is a measurement"), "{text}");
        assert!(text.contains("cost         0.0000"), "{text}");
    }

    #[test]
    fn a_metered_judge_prints_its_tokens_and_nets_them_off_the_saving() {
        let mut report = one(vec![], vec![]);
        report.cost_per_million_input_tokens = Some(2.0);
        report.judges[0].spend = Some(judge_spend(
            &TokenUsage {
                calls: 2,
                calls_reporting_usage: 2,
                input_tokens: 500_000,
                output_tokens: 40,
            },
            Some(2.0),
        ));
        let text = render(&report);
        assert!(text.contains("calls             2"), "{text}");
        assert!(text.contains("input        500000 token(s)"), "{text}");
        assert!(
            text.contains("output           40 token(s)  counted, not priced"),
            "{text}"
        );
        assert!(text.contains("cost         1.0000"), "{text}");
        // A perfect judge on this corpus routes 2 of 4 pages, saving 2.0000.
        assert!(
            text.contains("net          1.0000   saved on OCR (2.0000)"),
            "{text}"
        );
        assert!(
            !text.contains("(a floor)"),
            "every call reported, so nothing here is a floor: {text}"
        );
    }

    #[test]
    fn an_unpriced_token_count_is_counted_and_not_guessed_at() {
        let mut report = one(vec![], vec![]);
        report.judges[0].spend = Some(judge_spend(
            &TokenUsage {
                calls: 1,
                calls_reporting_usage: 1,
                input_tokens: 1_234,
                output_tokens: 5,
            },
            None,
        ));
        let text = render(&report);
        assert!(text.contains("input          1234 token(s)"), "{text}");
        assert!(
            text.contains("no --cost-per-million-input-tokens was given"),
            "{text}"
        );
        assert!(text.contains("no vendor's rate is baked"), "{text}");
        assert!(
            text.contains("net      n/a"),
            "an unpriced judge cannot be netted off: {text}"
        );
    }

    #[test]
    fn a_call_that_reported_no_usage_makes_the_printed_totals_a_floor() {
        let mut report = one(vec![], vec![]);
        report.cost_per_million_input_tokens = Some(2.0);
        report.judges[0].spend = Some(judge_spend(
            &TokenUsage {
                calls: 3,
                calls_reporting_usage: 1,
                input_tokens: 1_000_000,
                output_tokens: 0,
            },
            Some(2.0),
        ));
        let text = render(&report);
        assert!(text.contains("(a floor)"), "{text}");
        assert!(
            text.contains("2 of 3 call(s) came back with no usage attached"),
            "{text}"
        );
    }

    #[test]
    fn netting_a_judge_off_a_saving_that_beats_the_ceiling_says_so() {
        let mut report = one(vec![0, 1], vec![]);
        report.cost_per_million_input_tokens = Some(1.0);
        report.judges[0].spend = Some(judge_spend(
            &TokenUsage {
                calls: 1,
                calls_reporting_usage: 1,
                input_tokens: 100_000,
                output_tokens: 1,
            },
            Some(1.0),
        ));
        let text = render(&report);
        assert!(text.contains("under-buys OCR does not make the"), "{text}");
    }

    #[test]
    fn the_token_price_is_echoed_in_the_header_only_when_one_was_given() {
        let mut report = one(vec![], vec![]);
        assert!(!render(&report).contains("per million input tokens"));
        report.cost_per_million_input_tokens = Some(2.5);
        assert!(render(&report).contains("per million input tokens"));
    }

    #[test]
    fn a_group_summarises_in_one_line() {
        let report = report_with(vec![document("mixed.pdf", vec![1], vec![])], &[]);
        let line = one_line(&report.judges[0].by_source[0]);
        assert_eq!(line, "synthetic: 1 doc(s), 4 page(s), 1 missed, 0 wasted");
    }
}
