//! The scoring maths: confusion counts, cost, latency, confidence.
//!
//! Everything here is a pure function over verdicts and truth. Nothing in this
//! module reads a file or runs a classifier, so the numbers can be tested
//! directly against hand-built [`PageVerdict`] lists.
//!
//! # The two error types are never summed
//!
//! A single "accuracy" number would average a silent quality failure against a
//! small bill, and they are not the same kind of thing:
//!
//! * [`Confusion::missed_ocr`] — truth says the page needs OCR, the judge said
//!   no. The page is extracted locally and comes back blank or garbled. Nothing
//!   downstream notices.
//! * [`Confusion::wasted_ocr`] — truth says it does not, the judge said yes. The
//!   page goes to the paid OCR API for nothing. Output quality is unaffected.
//!
//! Precision / recall / F1 are derived from these, and are reported after them.

use std::collections::BTreeSet;

use doc_router::PageVerdict;
use serde::Serialize;

use crate::meter::TokenUsage;

/// The per-page confusion matrix for one document, or for a group of them.
///
/// # No field here is ground truth
///
/// All four count what *this judge did* to a page. Two are agreements and two
/// are errors, and a worse judge moves pages between them: change the judge and
/// every field changes. The corpus does not.
///
/// So ground truth is never one field. It is an agreement plus the error on the
/// same side of truth — [`Confusion::truly_needs_ocr`] and
/// [`Confusion::truly_local`], which come out the same for every judge scored
/// against the same documents. Reading `correct_ocr` as "the pages that need
/// OCR" is the one mistake this matrix invites; it is why those two methods
/// exist, and why the fields are not named `true_ocr` and `true_local`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct Confusion {
    /// Truth says OCR, judge said OCR: a **true positive**, which is this judge
    /// agreeing and not a property of the corpus. **Not** the pages that need
    /// OCR — see [`Confusion::truly_needs_ocr`].
    pub correct_ocr: u32,
    /// Truth says local, judge said local: a **true negative**, again this judge
    /// agreeing. **Not** the pages that need no OCR — see
    /// [`Confusion::truly_local`].
    pub correct_local: u32,
    /// Truth says OCR, judge said local. **Silent quality failure.**
    pub missed_ocr: u32,
    /// Truth says local, judge said OCR. **Costs money only.**
    pub wasted_ocr: u32,
}

impl Confusion {
    /// Add another document's counts into this one.
    pub fn add(&mut self, other: &Confusion) {
        self.correct_ocr += other.correct_ocr;
        self.correct_local += other.correct_local;
        self.missed_ocr += other.missed_ocr;
        self.wasted_ocr += other.wasted_ocr;
    }

    /// Every page counted.
    pub fn pages(&self) -> u32 {
        self.correct_ocr + self.correct_local + self.missed_ocr + self.wasted_ocr
    }

    /// The pages the corpus says need OCR, whatever this judge did with them:
    /// the ones it got right plus the ones it missed. Judge-independent — every
    /// judge scored against the same documents returns the same number, which
    /// is exactly the property that makes this ground truth and
    /// [`Confusion::correct_ocr`] not.
    pub fn truly_needs_ocr(&self) -> u32 {
        self.correct_ocr + self.missed_ocr
    }

    /// The pages the corpus says need no OCR, whatever this judge did with
    /// them: the ones it left local plus the ones it paid to OCR anyway. Also
    /// judge-independent, and `truly_needs_ocr() + truly_local() == pages()`.
    pub fn truly_local(&self) -> u32 {
        self.correct_local + self.wasted_ocr
    }

    /// True when neither error occurred. An empty matrix is vacuously perfect.
    pub fn is_perfect(&self) -> bool {
        self.missed_ocr == 0 && self.wasted_ocr == 0
    }

    /// Of the pages the judge sent to OCR, the share that needed it. `None` when
    /// it sent none, which is a real state and not a 0.0.
    pub fn precision(&self) -> Option<f64> {
        ratio(self.correct_ocr, self.correct_ocr + self.wasted_ocr)
    }

    /// Of the pages that needed OCR, the share the judge sent. `None` when no
    /// page needed it — the common case on an all-text document.
    pub fn recall(&self) -> Option<f64> {
        ratio(self.correct_ocr, self.correct_ocr + self.missed_ocr)
    }

    /// Harmonic mean of [`Confusion::precision`] and [`Confusion::recall`], when
    /// both exist and are not both zero.
    pub fn f1(&self) -> Option<f64> {
        let (p, r) = (self.precision()?, self.recall()?);
        if p + r == 0.0 {
            return Some(0.0);
        }
        Some(2.0 * p * r / (p + r))
    }
}

/// `numerator / denominator`, or `None` when the denominator is zero.
fn ratio(numerator: u32, denominator: u32) -> Option<f64> {
    (denominator > 0).then(|| f64::from(numerator) / f64::from(denominator))
}

/// How one judge did on one document's pages.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PageScores {
    /// The confusion counts.
    pub confusion: Confusion,
    /// The 0-indexed pages the judge left local that needed OCR.
    pub missed_ocr_pages: Vec<u32>,
    /// The 0-indexed pages the judge sent to OCR that did not need it.
    pub wasted_ocr_pages: Vec<u32>,
    /// The pages the judge would send to OCR, in page order.
    pub ocr_pages: Vec<u32>,
    /// True when the judge's OCR page set is exactly truth's.
    pub route_exact: bool,
}

/// Score one judge's verdicts for one document against the 0-indexed truth.
///
/// `verdicts` is one per page in page order — the contract
/// [`classify_with`](doc_router::classify_with) already enforces. Both page
/// lists are 0-indexed and neither is shifted here.
pub fn score_pages(truth: &[u32], verdicts: &[PageVerdict]) -> PageScores {
    let truth: BTreeSet<u32> = truth.iter().copied().collect();
    let mut scores = PageScores {
        confusion: Confusion::default(),
        missed_ocr_pages: Vec::new(),
        wasted_ocr_pages: Vec::new(),
        ocr_pages: Vec::new(),
        route_exact: false,
    };
    for verdict in verdicts {
        let wanted = truth.contains(&verdict.page);
        if verdict.needs_ocr {
            scores.ocr_pages.push(verdict.page);
        }
        match (wanted, verdict.needs_ocr) {
            (true, true) => scores.confusion.correct_ocr += 1,
            (false, false) => scores.confusion.correct_local += 1,
            (true, false) => {
                scores.confusion.missed_ocr += 1;
                scores.missed_ocr_pages.push(verdict.page);
            }
            (false, true) => {
                scores.confusion.wasted_ocr += 1;
                scores.wasted_ocr_pages.push(verdict.page);
            }
        }
    }
    scores.ocr_pages.sort_unstable();
    scores.route_exact = scores.confusion.is_perfect();
    scores
}

/// What a judge's routing would be billed, against what truth says it should be
/// and against not routing at all.
///
/// `cost_per_page` is whatever unit the caller passed on the command line; this
/// module has no opinion about what an OCR page costs.
///
/// # Why a saving is never reported on its own
///
/// Routing exists to spend less than OCR-ing every page, so the question the
/// caller actually has is "how much did this save". The honest answer needs
/// three numbers, not one, because **a judge can lower the bill by being wrong**:
/// every [`Confusion::missed_ocr`] page is a page it did not buy and should
/// have, and it shows up as a saving.
///
/// So [`Cost::saving`] is always reported next to [`Cost::ceiling_saving`] — what
/// a router that agreed with truth on every page would save. That is the most any
/// *correct* router can save, and a judge beating it has not found an efficiency;
/// it has skipped OCR that the corpus says was needed. [`Cost::saving_beyond_ceiling_pages`]
/// is exactly how far past it the judge went, and is positive only when something
/// is wrong.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct Cost {
    /// Pages the judge would send to the OCR API.
    pub ocr_pages: u32,
    /// Pages truth says should have gone.
    pub ideal_ocr_pages: u32,
    /// Every page scored: what "OCR everything" would send.
    pub all_pages: u32,
    /// Pages billed for nothing. Same number as `wasted_ocr`, in money terms.
    pub overspend_pages: u32,
    /// Pages that should have been billed and were not. The money-shaped view of
    /// `missed_ocr` — the bill is smaller and the output is wrong.
    pub underspend_pages: u32,
    /// `overspend_pages * cost_per_page`.
    pub overspend: f64,
    /// `underspend_pages * cost_per_page`.
    pub underspend: f64,
    /// `all_pages * cost_per_page`: the bill for OCR-ing every page, which is
    /// what the router is there to avoid. It needs no judge, never misses a
    /// page, and is the only baseline that is always available.
    pub naive: f64,
    /// `ocr_pages * cost_per_page`: the bill this judge's routing produces.
    pub routed: f64,
    /// `ideal_ocr_pages * cost_per_page`: the bill a router that agreed with
    /// truth on every page would produce.
    pub ideal: f64,
    /// `ocr_pages / all_pages`: the share of the naive bill this judge's routing
    /// comes to. Taken from the page counts rather than the money so that
    /// `--cost-per-page 0` gives a ratio instead of a `NaN`. `None` when nothing
    /// was scored — an empty run has no ratio, and `0.000` would read as "free".
    pub routed_share: Option<f64>,
    /// `naive - routed`: what this judge's routing did not spend. **Not a
    /// virtue on its own** — see the type docs, and read it against
    /// [`Cost::ceiling_saving`].
    pub saving: f64,
    /// `naive - ideal`: the most a correct router could save, because it still
    /// has to buy every page that genuinely needs OCR. A judge saving more than
    /// this is under-buying, not out-performing.
    pub ceiling_saving: f64,
    /// `ideal_ocr_pages - ocr_pages`, i.e. `missed_ocr - wasted_ocr`, in pages.
    ///
    /// Positive: the judge "saved" past the ceiling, and the excess is OCR it
    /// was supposed to buy. Zero: it sits exactly on the ceiling, which a
    /// perfect judge does — and so does one whose two errors happen to cancel,
    /// which is why this number is never read without the error counts. Negative:
    /// it spent past the ideal, which costs money and nothing else.
    pub saving_beyond_ceiling_pages: i64,
}

/// Turn a confusion matrix into the cost view at `cost_per_page`.
pub fn cost(confusion: &Confusion, cost_per_page: f64) -> Cost {
    let overspend_pages = confusion.wasted_ocr;
    let underspend_pages = confusion.missed_ocr;
    let ocr_pages = confusion.correct_ocr + confusion.wasted_ocr;
    let ideal_ocr_pages = confusion.truly_needs_ocr();
    let all_pages = confusion.pages();
    let naive = f64::from(all_pages) * cost_per_page;
    let routed = f64::from(ocr_pages) * cost_per_page;
    let ideal = f64::from(ideal_ocr_pages) * cost_per_page;
    Cost {
        ocr_pages,
        ideal_ocr_pages,
        all_pages,
        overspend_pages,
        underspend_pages,
        overspend: f64::from(overspend_pages) * cost_per_page,
        underspend: f64::from(underspend_pages) * cost_per_page,
        naive,
        routed,
        ideal,
        routed_share: ratio(ocr_pages, all_pages),
        saving: naive - routed,
        ceiling_saving: naive - ideal,
        saving_beyond_ceiling_pages: i64::from(ideal_ocr_pages) - i64::from(ocr_pages),
    }
}

/// What a judge's own calls cost to make, priced per million input tokens.
///
/// # Why this is a separate number from [`Cost`]
///
/// [`Cost`] prices the *OCR* the judge's decisions buy. This prices the judge.
/// A hosted judge is not free: it is an extra API call per document, and a judge
/// whose token bill exceeds the OCR it avoids has saved nothing while adding a
/// network dependency. The two are only comparable once the caller supplies both
/// prices, which is why [`JudgeSpend::cost`] is an `Option` and not a `0.0`.
///
/// Output tokens are counted and not priced: the hosted judge answers with a
/// probability per page, so the output is a rounding error next to the page text
/// it is sent. The count is reported anyway, so a judge that starts writing
/// essays is visible rather than silently free.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct JudgeSpend {
    /// Calls the judge made over the whole run, including every `--repeat`.
    pub calls: u32,
    /// How many of those calls came back with a usage figure attached. Less than
    /// `calls` means the totals below are a floor, not a total.
    pub calls_reporting_usage: u32,
    /// Input tokens summed over the calls that reported any.
    pub input_tokens: u64,
    /// Output tokens summed over the calls that reported any.
    pub output_tokens: u64,
    /// `input_tokens / 1e6 * cost_per_million_input_tokens`.
    ///
    /// `None` when the caller passed no token price: the tokens are still a
    /// measurement, but nothing in this crate knows what a million of them
    /// costs, and printing `0.0000` would claim the judge was free.
    pub cost: Option<f64>,
    /// True when every call reported its usage, so the token counts above are
    /// the whole run and not the part of it the vendor chose to tell us about.
    pub usage_complete: bool,
}

/// Price a judge's token usage. `cost_per_million_input_tokens` is `None` when
/// the caller gave no price; see [`JudgeSpend::cost`].
pub fn judge_spend(usage: &TokenUsage, cost_per_million_input_tokens: Option<f64>) -> JudgeSpend {
    let usage_complete = usage.calls_reporting_usage == usage.calls;
    // Zero tokens cost zero at every price, so a judge that made no call has a
    // measured cost of nothing rather than an unpriced unknown. That is the only
    // case where a zero here is a measurement.
    let cost = if usage.input_tokens == 0 && usage_complete {
        Some(0.0)
    } else {
        cost_per_million_input_tokens.map(|price| usage.input_tokens as f64 / 1_000_000.0 * price)
    };
    JudgeSpend {
        calls: usage.calls,
        calls_reporting_usage: usage.calls_reporting_usage,
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cost,
        usage_complete,
    }
}

/// p50 / p95 / max over a set of per-document timings, in milliseconds.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize)]
pub struct Latency {
    /// Median.
    pub p50_ms: f64,
    /// 95th percentile, nearest-rank.
    pub p95_ms: f64,
    /// Slowest document.
    pub max_ms: f64,
    /// How many documents went into the summary.
    pub documents: u32,
}

/// Summarise per-document timings. An empty input summarises to all zeroes.
pub fn latency(per_document_ms: &[f64]) -> Latency {
    let mut sorted = per_document_ms.to_vec();
    sorted.sort_by(f64::total_cmp);
    Latency {
        p50_ms: percentile(&sorted, 0.50),
        p95_ms: percentile(&sorted, 0.95),
        max_ms: sorted.last().copied().unwrap_or(0.0),
        documents: sorted.len() as u32,
    }
}

/// Nearest-rank percentile of an already-sorted slice.
pub fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (fraction * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

/// The median of a set of samples. Used to damp run-to-run noise in the repeated
/// timings of one document; an even count averages the middle two.
pub fn median(samples: &[f64]) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let middle = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[middle - 1] + sorted[middle]) / 2.0
    } else {
        sorted[middle]
    }
}

/// One distinct confidence value a judge emitted, and how often.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ConfidenceValue {
    /// The value itself.
    pub value: f32,
    /// How many verdicts carried it.
    pub count: u32,
}

/// One row of the reliability table.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ReliabilityBucket {
    /// Inclusive lower edge.
    pub lower: f64,
    /// Upper edge, inclusive only for the last bucket.
    pub upper: f64,
    /// Verdicts falling in this bucket.
    pub count: u32,
    /// Of those, how many were right.
    pub correct: u32,
    /// `correct / count`.
    pub accuracy: f64,
    /// Mean confidence of the verdicts in this bucket, i.e. what the judge
    /// claimed the accuracy would be.
    pub mean_confidence: f64,
}

/// What can honestly be said about a judge's `confidence` values.
///
/// Whether the reliability table and the Brier score are computed is a property
/// of the data, not of which judge produced it: a judge emitting one distinct
/// value has nothing to calibrate, whoever it is. A judge emitting real
/// probabilities lights the table up with no change here.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ConfidenceReport {
    /// Every distinct value emitted, ascending, with counts.
    pub distinct: Vec<ConfidenceValue>,
    /// True when fewer than two distinct values were emitted.
    pub degenerate: bool,
    /// The single value, when degenerate and at least one verdict was seen.
    pub constant: Option<f32>,
    /// Why the table was skipped, when it was.
    pub skipped_reason: Option<String>,
    /// The reliability table. Empty when degenerate.
    pub buckets: Vec<ReliabilityBucket>,
    /// Mean squared error of `confidence` against "this verdict was right",
    /// lower is better. `None` when degenerate.
    pub brier: Option<f64>,
    /// Verdicts scored.
    pub verdicts: u32,
}

/// One scored verdict: the confidence the judge attached, and whether it was
/// right.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredConfidence {
    /// The judge's stated confidence in this verdict, 0.0–1.0.
    pub confidence: f32,
    /// True when the verdict matched truth.
    pub correct: bool,
}

/// Number of equal-width buckets in the reliability table.
pub const RELIABILITY_BUCKETS: usize = 10;

/// Summarise a judge's confidence values.
///
/// `confidence` is read as the judge's stated probability that **its own
/// verdict** is right, which is the only reading the trait supports: the value
/// sits on a `PageVerdict` that has already decided `needs_ocr`. So the Brier
/// score is the mean squared error of `confidence` against 1.0 for a correct
/// verdict and 0.0 for a wrong one, and a bucket's observed accuracy is the
/// share of verdicts in it that were right.
pub fn confidence_report(scored: &[ScoredConfidence]) -> ConfidenceReport {
    let mut distinct: Vec<ConfidenceValue> = Vec::new();
    for entry in scored {
        match distinct
            .iter_mut()
            .find(|seen| seen.value.to_bits() == entry.confidence.to_bits())
        {
            Some(seen) => seen.count += 1,
            None => distinct.push(ConfidenceValue {
                value: entry.confidence,
                count: 1,
            }),
        }
    }
    distinct.sort_by(|a, b| a.value.total_cmp(&b.value));

    let verdicts = scored.len() as u32;
    if distinct.len() < 2 {
        let constant = distinct.first().map(|only| only.value);
        let skipped_reason = Some(match constant {
            Some(value) => format!(
                "every verdict carried the same confidence ({value:.2}), so there is nothing to \
                 calibrate: one value cannot be split into buckets and scores only itself"
            ),
            None => "no verdicts were scored".to_string(),
        });
        return ConfidenceReport {
            distinct,
            degenerate: true,
            constant,
            skipped_reason,
            buckets: Vec::new(),
            brier: None,
            verdicts,
        };
    }

    let mut buckets: Vec<ReliabilityBucket> = (0..RELIABILITY_BUCKETS)
        .map(|index| {
            let width = 1.0 / RELIABILITY_BUCKETS as f64;
            ReliabilityBucket {
                lower: index as f64 * width,
                upper: (index + 1) as f64 * width,
                count: 0,
                correct: 0,
                accuracy: 0.0,
                mean_confidence: 0.0,
            }
        })
        .collect();
    let mut brier = 0.0;
    for entry in scored {
        let index = bucket_index(entry.confidence);
        let bucket = &mut buckets[index];
        bucket.count += 1;
        bucket.correct += u32::from(entry.correct);
        bucket.mean_confidence += f64::from(entry.confidence);
        let observed = if entry.correct { 1.0 } else { 0.0 };
        brier += (f64::from(entry.confidence) - observed).powi(2);
    }
    for bucket in &mut buckets {
        if bucket.count > 0 {
            bucket.accuracy = f64::from(bucket.correct) / f64::from(bucket.count);
            bucket.mean_confidence /= f64::from(bucket.count);
        }
    }
    buckets.retain(|bucket| bucket.count > 0);

    ConfidenceReport {
        distinct,
        degenerate: false,
        constant: None,
        skipped_reason: None,
        buckets,
        brier: Some(brier / f64::from(verdicts)),
        verdicts,
    }
}

/// Which reliability bucket a confidence falls in. Values at or above 1.0 land
/// in the last bucket; values below 0.0 in the first.
fn bucket_index(confidence: f32) -> usize {
    let scaled = (f64::from(confidence) * RELIABILITY_BUCKETS as f64).floor();
    if scaled < 0.0 {
        return 0;
    }
    (scaled as usize).min(RELIABILITY_BUCKETS - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A verdict with a page number, a decision and a confidence.
    fn verdict(page: u32, needs_ocr: bool, confidence: f32) -> PageVerdict {
        PageVerdict {
            page,
            needs_ocr,
            confidence,
            reason: "test".to_string(),
        }
    }

    /// Verdicts for `0..n`, sending exactly `ocr` to OCR, all at confidence 1.0.
    fn verdicts(n: u32, ocr: &[u32]) -> Vec<PageVerdict> {
        (0..n)
            .map(|page| verdict(page, ocr.contains(&page), 1.0))
            .collect()
    }

    #[test]
    fn a_perfect_judge_has_no_errors_of_either_kind() {
        let scores = score_pages(&[1, 3], &verdicts(4, &[1, 3]));
        assert_eq!(
            scores.confusion,
            Confusion {
                correct_ocr: 2,
                correct_local: 2,
                missed_ocr: 0,
                wasted_ocr: 0
            }
        );
        assert!(scores.missed_ocr_pages.is_empty());
        assert!(scores.wasted_ocr_pages.is_empty());
        assert_eq!(scores.ocr_pages, vec![1, 3]);
        assert!(scores.route_exact);
    }

    #[test]
    fn the_two_error_types_are_counted_separately_and_name_their_pages() {
        // Truth: 1 and 3 need OCR. Judge sends 2 and 3: it misses 1 (quality
        // failure) and wastes 2 (money). One of each, never one "error".
        let scores = score_pages(&[1, 3], &verdicts(4, &[2, 3]));
        assert_eq!(
            scores.confusion,
            Confusion {
                correct_ocr: 1,
                correct_local: 1,
                missed_ocr: 1,
                wasted_ocr: 1
            }
        );
        assert_eq!(scores.missed_ocr_pages, vec![1]);
        assert_eq!(scores.wasted_ocr_pages, vec![2]);
        assert!(!scores.route_exact);
        assert_eq!(scores.confusion.pages(), 4);
    }

    #[test]
    fn missing_every_ocr_page_is_recall_zero_and_no_precision() {
        let scores = score_pages(&[0, 1], &verdicts(2, &[]));
        assert_eq!(scores.confusion.missed_ocr, 2);
        assert_eq!(scores.confusion.wasted_ocr, 0);
        assert_eq!(scores.confusion.recall(), Some(0.0));
        // It sent nothing, so "of what it sent, how much was right" has no answer.
        assert_eq!(scores.confusion.precision(), None);
        assert_eq!(scores.confusion.f1(), None);
    }

    #[test]
    fn sending_every_page_of_an_all_text_document_is_precision_zero() {
        let scores = score_pages(&[], &verdicts(3, &[0, 1, 2]));
        assert_eq!(scores.confusion.wasted_ocr, 3);
        assert_eq!(scores.confusion.precision(), Some(0.0));
        // Nothing needed OCR, so recall has no answer.
        assert_eq!(scores.confusion.recall(), None);
    }

    #[test]
    fn precision_recall_and_f1_come_from_the_counts() {
        let confusion = Confusion {
            correct_ocr: 3,
            correct_local: 5,
            missed_ocr: 1,
            wasted_ocr: 1,
        };
        assert_eq!(confusion.precision(), Some(0.75));
        assert_eq!(confusion.recall(), Some(0.75));
        assert_eq!(confusion.f1(), Some(0.75));
    }

    #[test]
    fn a_zero_page_document_scores_nothing_and_divides_by_nothing() {
        let scores = score_pages(&[], &[]);
        assert_eq!(scores.confusion, Confusion::default());
        assert_eq!(scores.confusion.pages(), 0);
        assert_eq!(scores.confusion.precision(), None);
        assert_eq!(scores.confusion.recall(), None);
        // Vacuously exact: it routed the empty set truth asked for.
        assert!(scores.route_exact);
    }

    #[test]
    fn confusions_add_up_across_documents() {
        let mut total = Confusion::default();
        total.add(&score_pages(&[1], &verdicts(2, &[1])).confusion);
        total.add(&score_pages(&[0, 1], &verdicts(2, &[0])).confusion);
        assert_eq!(
            total,
            Confusion {
                correct_ocr: 2,
                correct_local: 1,
                missed_ocr: 1,
                wasted_ocr: 0
            }
        );
    }

    #[test]
    fn cost_splits_over_and_underspend_and_never_nets_them_off() {
        let confusion = Confusion {
            correct_ocr: 4,
            correct_local: 10,
            missed_ocr: 2,
            wasted_ocr: 3,
        };
        let cost = cost(&confusion, 0.25);
        assert_eq!(cost.ocr_pages, 7);
        assert_eq!(cost.ideal_ocr_pages, 6);
        assert_eq!(cost.overspend_pages, 3);
        assert_eq!(cost.underspend_pages, 2);
        assert!((cost.overspend - 0.75).abs() < 1e-12, "{cost:?}");
        assert!((cost.underspend - 0.50).abs() < 1e-12, "{cost:?}");
    }

    #[test]
    fn cost_of_a_perfect_judge_is_the_ideal_spend_and_nothing_else() {
        let confusion = Confusion {
            correct_ocr: 8,
            correct_local: 16,
            missed_ocr: 0,
            wasted_ocr: 0,
        };
        let cost = cost(&confusion, 2.0);
        assert_eq!((cost.ocr_pages, cost.ideal_ocr_pages), (8, 8));
        assert_eq!(cost.overspend, 0.0);
        assert_eq!(cost.underspend, 0.0);
    }

    /// Three judges over the same 155 pages, 87 of which truly need OCR. They
    /// disagree about every page; they cannot disagree about the corpus.
    ///
    /// This is the shape of the bug that made the headline line print
    /// `59 truly need OCR` for the heuristic and `76` for `jev` on one corpus:
    /// the true positives were being printed as the truth.
    #[test]
    fn ground_truth_is_the_same_for_every_judge_over_the_same_pages() {
        let judges = [
            // correct_ocr, correct_local, missed, wasted
            Confusion {
                correct_ocr: 59,
                correct_local: 59,
                missed_ocr: 28,
                wasted_ocr: 9,
            },
            Confusion {
                correct_ocr: 76,
                correct_local: 59,
                missed_ocr: 11,
                wasted_ocr: 9,
            },
            Confusion {
                correct_ocr: 87,
                correct_local: 68,
                missed_ocr: 0,
                wasted_ocr: 0,
            },
            Confusion {
                correct_ocr: 0,
                correct_local: 68,
                missed_ocr: 87,
                wasted_ocr: 0,
            },
        ];
        for judge in &judges {
            assert_eq!(judge.truly_needs_ocr(), 87, "{judge:?}");
            assert_eq!(judge.truly_local(), 68, "{judge:?}");
            assert_eq!(
                judge.truly_needs_ocr() + judge.truly_local(),
                judge.pages(),
                "the two sides of truth are every page, once each: {judge:?}"
            );
        }
    }

    #[test]
    fn a_judge_that_agrees_everywhere_still_has_two_sides_of_truth() {
        let all_text = Confusion {
            correct_ocr: 0,
            correct_local: 12,
            missed_ocr: 0,
            wasted_ocr: 0,
        };
        assert_eq!(all_text.truly_needs_ocr(), 0);
        assert_eq!(all_text.truly_local(), 12);
        assert_eq!(Confusion::default().truly_needs_ocr(), 0);
        assert_eq!(Confusion::default().truly_local(), 0);
    }

    #[test]
    fn the_saving_is_measured_against_ocring_every_page() {
        // 155 pages, 87 truly needing OCR; this judge routes 68 of them.
        let c = Confusion {
            correct_ocr: 59,
            correct_local: 59,
            missed_ocr: 28,
            wasted_ocr: 9,
        };
        let cost = cost(&c, 2.0);
        assert_eq!(cost.all_pages, 155);
        assert_eq!(cost.naive, 310.0);
        assert_eq!(cost.routed, 136.0, "68 pages routed to OCR");
        assert_eq!(cost.ideal, 174.0, "87 pages a correct router must buy");
        assert_eq!(cost.saving, 174.0);
        assert_eq!(cost.ceiling_saving, 136.0);
        assert_eq!(
            cost.saving_beyond_ceiling_pages, 19,
            "under the ceiling by 28 missed less 9 wasted"
        );
        assert_eq!(cost.routed_share, Some(68.0 / 155.0));
    }

    #[test]
    fn a_judge_that_routes_nothing_saves_everything_and_is_visibly_broken() {
        let nothing = Confusion {
            correct_ocr: 0,
            correct_local: 68,
            missed_ocr: 87,
            wasted_ocr: 0,
        };
        let cost = cost(&nothing, 1.0);
        assert_eq!(cost.routed, 0.0, "infinitely cheap");
        assert_eq!(cost.saving, 155.0);
        assert_eq!(cost.ceiling_saving, 68.0);
        assert_eq!(
            cost.saving_beyond_ceiling_pages, 87,
            "every page it should have bought is a page it 'saved'"
        );
        assert_eq!(cost.routed_share, Some(0.0));
    }

    #[test]
    fn a_judge_sitting_exactly_on_the_ceiling_can_still_be_wrong_everywhere() {
        // Equal numbers of each error cancel in money and in nothing else.
        let c = Confusion {
            correct_ocr: 10,
            correct_local: 10,
            missed_ocr: 5,
            wasted_ocr: 5,
        };
        let cost = cost(&c, 1.0);
        assert_eq!(cost.saving_beyond_ceiling_pages, 0);
        assert_eq!(cost.saving, cost.ceiling_saving);
        assert!(!c.is_perfect(), "and it is not a correct router");
    }

    #[test]
    fn a_zero_page_run_has_no_share_and_nothing_to_save() {
        let cost = cost(&Confusion::default(), 1.0);
        assert_eq!(cost.all_pages, 0);
        assert_eq!(cost.naive, 0.0);
        assert_eq!(cost.saving, 0.0);
        assert_eq!(cost.ceiling_saving, 0.0);
        assert_eq!(cost.saving_beyond_ceiling_pages, 0);
        assert_eq!(
            cost.routed_share, None,
            "a share of nothing is undefined, not zero"
        );
    }

    #[test]
    fn a_free_price_leaves_the_share_defined() {
        // `--cost-per-page 0` makes every money figure zero; the share is
        // counted from pages so it stays a real number rather than 0.0/0.0.
        let c = Confusion {
            correct_ocr: 1,
            correct_local: 3,
            missed_ocr: 0,
            wasted_ocr: 0,
        };
        assert_eq!(cost(&c, 0.0).routed_share, Some(0.25));
    }

    #[test]
    fn a_judge_that_made_no_calls_has_a_measured_cost_of_nothing() {
        let spend = judge_spend(&TokenUsage::default(), None);
        assert_eq!(spend.calls, 0);
        assert_eq!(
            spend.cost,
            Some(0.0),
            "zero tokens cost zero at every price, so this zero is measured"
        );
        assert!(spend.usage_complete);
    }

    #[test]
    fn tokens_without_a_price_are_counted_and_left_unpriced() {
        let usage = TokenUsage {
            calls: 4,
            calls_reporting_usage: 4,
            input_tokens: 2_500_000,
            output_tokens: 400,
        };
        let unpriced = judge_spend(&usage, None);
        assert_eq!(unpriced.input_tokens, 2_500_000);
        assert_eq!(
            unpriced.cost, None,
            "nothing here knows what a million tokens costs"
        );
        let priced = judge_spend(&usage, Some(3.0));
        assert_eq!(priced.cost, Some(7.5));
    }

    #[test]
    fn a_call_that_reported_no_usage_makes_the_totals_a_floor() {
        let usage = TokenUsage {
            calls: 3,
            calls_reporting_usage: 2,
            input_tokens: 1_000_000,
            output_tokens: 0,
        };
        let spend = judge_spend(&usage, Some(2.0));
        assert!(!spend.usage_complete);
        assert_eq!(spend.cost, Some(2.0), "priced on what was reported");

        // The same shape with nothing reported at all: still not free.
        let silent = judge_spend(
            &TokenUsage {
                calls: 3,
                calls_reporting_usage: 0,
                input_tokens: 0,
                output_tokens: 0,
            },
            None,
        );
        assert!(!silent.usage_complete);
        assert_eq!(
            silent.cost, None,
            "three calls that said nothing are unknown, not zero"
        );
    }

    #[test]
    fn median_damps_a_single_slow_run() {
        assert_eq!(median(&[1.0, 40.0, 1.2]), 1.2);
        assert_eq!(median(&[2.0, 4.0]), 3.0);
        assert_eq!(median(&[7.0]), 7.0);
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn latency_reports_nearest_rank_percentiles() {
        let summary = latency(&[5.0, 1.0, 4.0, 2.0, 3.0]);
        assert_eq!(summary.p50_ms, 3.0);
        assert_eq!(summary.p95_ms, 5.0);
        assert_eq!(summary.max_ms, 5.0);
        assert_eq!(summary.documents, 5);
        assert_eq!(latency(&[]), Latency::default());
    }

    /// Scored verdicts from `(confidence, correct)` pairs.
    fn scored(pairs: &[(f32, bool)]) -> Vec<ScoredConfidence> {
        pairs
            .iter()
            .map(|(confidence, correct)| ScoredConfidence {
                confidence: *confidence,
                correct: *correct,
            })
            .collect()
    }

    #[test]
    fn a_constant_confidence_judge_takes_the_degenerate_path() {
        // This is HeuristicJudge's shape: 1.0 on every verdict, right or wrong.
        let report = confidence_report(&scored(&[
            (1.0, true),
            (1.0, true),
            (1.0, false),
            (1.0, true),
        ]));
        assert!(report.degenerate);
        assert_eq!(report.constant, Some(1.0));
        assert_eq!(report.distinct.len(), 1);
        assert_eq!(report.distinct[0].count, 4);
        // No table, no score: both would be inventing information.
        assert!(report.buckets.is_empty());
        assert_eq!(report.brier, None);
        assert!(report.skipped_reason.is_some());
    }

    #[test]
    fn a_constant_confidence_below_one_is_equally_degenerate() {
        // Degeneracy is a property of the data, not a special case for 1.0 or
        // for any named judge.
        let report = confidence_report(&scored(&[(0.42, true), (0.42, false)]));
        assert!(report.degenerate);
        assert_eq!(report.constant, Some(0.42));
        assert_eq!(report.brier, None);
    }

    #[test]
    fn no_verdicts_at_all_is_degenerate_with_no_constant() {
        let report = confidence_report(&[]);
        assert!(report.degenerate);
        assert_eq!(report.constant, None);
        assert_eq!(report.verdicts, 0);
        assert!(report.skipped_reason.is_some());
    }

    #[test]
    fn a_varying_confidence_judge_gets_a_reliability_table_and_a_brier_score() {
        // Two distinct values is the threshold, and this is what step 4's judge
        // is expected to look like: real probabilities, bucketed.
        let report = confidence_report(&scored(&[
            (0.95, true),
            (0.95, true),
            (0.55, true),
            (0.55, false),
        ]));
        assert!(!report.degenerate);
        assert_eq!(report.constant, None);
        assert_eq!(report.distinct.len(), 2);
        assert_eq!(report.distinct[0].value, 0.55);
        assert_eq!(report.distinct[1].value, 0.95);

        // Only non-empty buckets are kept: [0.5,0.6) and [0.9,1.0].
        assert_eq!(report.buckets.len(), 2);
        assert_eq!(report.buckets[0].count, 2);
        assert_eq!(report.buckets[0].accuracy, 0.5);
        assert!((report.buckets[0].mean_confidence - 0.55).abs() < 1e-6);
        assert_eq!(report.buckets[1].count, 2);
        assert_eq!(report.buckets[1].accuracy, 1.0);

        // (0.05^2 + 0.05^2 + 0.45^2 + 0.55^2) / 4
        let brier = report.brier.expect("a varying judge is scored");
        let expected = (0.05f64.powi(2) * 2.0 + 0.45f64.powi(2) + 0.55f64.powi(2)) / 4.0;
        assert!((brier - expected).abs() < 1e-6, "{brier} vs {expected}");
    }

    #[test]
    fn confidences_land_in_the_bucket_they_belong_to() {
        assert_eq!(bucket_index(0.0), 0);
        assert_eq!(bucket_index(0.0999), 0);
        assert_eq!(bucket_index(0.1), 1);
        assert_eq!(bucket_index(0.999), 9);
        assert_eq!(bucket_index(1.0), 9);
        // Out of contract, but must not panic or index out of bounds.
        assert_eq!(bucket_index(1.5), 9);
        assert_eq!(bucket_index(-0.5), 0);
    }
}
