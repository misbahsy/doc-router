//! Reading what a judge's own calls cost, without the core crate learning about
//! tokens.
//!
//! # The problem this module solves
//!
//! The harness holds judges as `Box<dyn PageJudge>`, which is the whole point:
//! [`crate::registry`] is the only place that knows which judges exist, and the
//! scoring and the report work off the trait. But token usage is not a trait
//! concept. `doc-router` is network-free by design -- it has no calls, no vendor
//! and no tokens -- and widening [`PageJudge`] with a usage accessor would put a
//! billing concept into a crate that can never have one.
//!
//! So the meter is carried *alongside* the `dyn PageJudge` rather than through
//! it: [`doc_router_jev::capturing`] hands the harness a second handle on the
//! judge's own call log at the moment the registry builds it, and this module
//! folds that log into a [`TokenUsage`]. A judge that makes no calls has no log,
//! and the report says so in words instead of printing a zero.
//!
//! [`PageJudge`]: doc_router::PageJudge

use doc_router_jev::CallLog;
use serde::Serialize;

/// What a judge's calls have used over the whole run.
///
/// Every field counts the *run*, not one document: a `--repeat 3` run calls the
/// judge three times per document and is billed for all three, so that is what
/// this reports.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct TokenUsage {
    /// Calls made, whether or not they came back with a usage figure -- or came
    /// back at all.
    pub calls: u32,
    /// How many of those reported their token usage. A failed call, or a vendor
    /// that omits the field, counts in `calls` and not here, which is what makes
    /// the token totals a floor rather than a total.
    pub calls_reporting_usage: u32,
    /// Input tokens over the calls that reported any.
    pub input_tokens: u64,
    /// Output tokens over the calls that reported any.
    pub output_tokens: u64,
}

/// A handle on one judge's call log, read after the run.
///
/// Holding this is what distinguishes "this judge used no tokens" from "nobody
/// can say what this judge used": a judge the harness has no meter for is
/// reported as the second, in words. See [`crate::score::judge_spend`].
#[derive(Debug, Clone)]
pub struct JudgeMeter {
    log: CallLog,
}

impl JudgeMeter {
    /// Watch `log`, which is the judge's own and stays live as it calls.
    #[must_use]
    pub fn new(log: CallLog) -> Self {
        JudgeMeter { log }
    }

    /// Usage so far, over every call the judge has made.
    #[must_use]
    pub fn usage(&self) -> TokenUsage {
        let mut usage = TokenUsage::default();
        for call in self.log.lock().expect("call log mutex").iter() {
            usage.calls += 1;
            // `input_tokens` and `output_tokens` are set together, from the same
            // `usage` object in one response, so either one says the call
            // reported. A call that failed before a response reports neither.
            if call.input_tokens.is_some() || call.output_tokens.is_some() {
                usage.calls_reporting_usage += 1;
            }
            usage.input_tokens += call.input_tokens.unwrap_or(0);
            usage.output_tokens += call.output_tokens.unwrap_or(0);
        }
        usage
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use doc_router_jev::CallRecord;

    fn call(input: Option<u64>, output: Option<u64>) -> CallRecord {
        CallRecord {
            pages: vec![0],
            elapsed_ms: 1,
            status: Some(200),
            input_tokens: input,
            output_tokens: output,
        }
    }

    fn meter(calls: Vec<CallRecord>) -> JudgeMeter {
        let log = CallLog::default();
        log.lock().expect("call log mutex").extend(calls);
        JudgeMeter::new(log)
    }

    #[test]
    fn a_judge_that_has_not_been_called_reports_an_empty_usage() {
        assert_eq!(meter(Vec::new()).usage(), TokenUsage::default());
    }

    #[test]
    fn a_call_that_reported_nothing_still_counts_as_a_call() {
        let usage = meter(vec![call(Some(100), Some(7)), call(None, None)]).usage();
        assert_eq!(usage.calls, 2);
        assert_eq!(usage.calls_reporting_usage, 1);
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 7);
    }

    #[test]
    fn the_meter_reads_the_judges_own_log_and_not_a_snapshot_of_it() {
        let log = CallLog::default();
        let meter = JudgeMeter::new(CallLog::clone(&log));
        assert_eq!(meter.usage().calls, 0);
        log.lock()
            .expect("call log mutex")
            .push(call(Some(5), None));
        assert_eq!(meter.usage().calls, 1);
        assert_eq!(meter.usage().input_tokens, 5);
    }
}
