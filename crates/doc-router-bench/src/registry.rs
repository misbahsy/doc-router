//! Name -> judge. The registry itself lives in
//! [`doc_router_cli::judge`](doc_router_cli::judge) and is re-exported here.
//!
//! It moved there when the `doc-router` CLI grew its own `--judge` flag. Both
//! front ends have to answer the same question — "what does this name mean?" —
//! and two lookups that drifted apart would mean `--judge jev` selecting one
//! judge in the CLI and a different one here, which is exactly the mistake a
//! registry exists to prevent. The CLI crate is the one that can hold it: it
//! already assembles networked components out of the environment, and this crate
//! is `publish = false`, so the dependency can only point this way.
//!
//! Adding a judge is still a single arm in [`judge_by_name`] and its name in
//! [`JUDGE_NAMES`], in that module.
//!
//! What stays here is what is specific to a benchmark:
//!
//! * **[`OnApiFailure::Fail`].** The harness builds hosted judges strict, which
//!   is the opposite of the right setting for the CLI. A vendor outage falling
//!   back to the heuristic keeps an operator's document routing; here it would
//!   be a fabrication — the row labelled `jev` would carry the heuristic's
//!   verdicts, scored and printed under another judge's name, and nothing in the
//!   output would say so. A benchmark that quietly reports one judge's numbers
//!   under another's is worse than one that fails, so this one fails. See
//!   [`crate::bench::resolve_judges`], which passes it.
//! * **The `.env` hint** on an unavailable judge, added in the same place. Only
//!   this crate reads a `.env` (see [`crate::dotenv`]), so only this crate tells
//!   anyone to write one.

pub use doc_router_cli::judge::{
    judge_by_name, JudgeLookup, OnApiFailure, BASELINE_JUDGE, JUDGE_NAMES,
};

use crate::meter::JudgeMeter;

/// [`judge_by_name`], plus a way to read what the judge spends on its own calls.
///
/// # Why the meter cannot come through the judge
///
/// The registry hands back `Box<dyn PageJudge>`, and a trait object cannot be
/// asked what it cost: [`PageJudge`] is a routing decision, and the crate that
/// defines it makes no network calls and has no tokens to report. Widening that
/// trait would push a vendor's billing concept into the one crate deliberately
/// free of it, and widening [`JudgeLookup`] would push it into the CLI, which
/// does not report costs at all.
///
/// So the handle is taken at the moment of construction instead, from the crate
/// that actually makes the calls: [`doc_router_jev::capturing`] runs the lookup
/// and collects the call log of every hosted judge built inside it. The capture
/// is scoped to this one call, so a judge resolved here is metered here and
/// nowhere else.
///
/// The meter is `None` for a judge that makes no calls -- the baseline is local
/// and has nothing to bill — and for an unavailable one. The report is careful
/// to print that as "no token data" rather than as zero tokens; see
/// [`crate::score::judge_spend`].
///
/// [`PageJudge`]: doc_router::PageJudge
pub fn metered_judge_by_name(
    name: &str,
    on_api_failure: OnApiFailure,
) -> Option<(JudgeLookup, Option<JudgeMeter>)> {
    let (lookup, logs) = doc_router_jev::capturing(|| judge_by_name(name, on_api_failure));
    // One name builds at most one hosted judge. More would mean the registry
    // grew a composite judge, and summing the logs of its parts is still the
    // right answer for "what did this name cost".
    let meter = logs.into_iter().next().map(JudgeMeter::new);
    Some((lookup?, meter))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_baseline_resolves_with_no_meter_because_it_makes_no_calls() {
        let (lookup, meter) = metered_judge_by_name(BASELINE_JUDGE, OnApiFailure::Fail)
            .expect("the baseline always resolves");
        assert!(matches!(lookup, JudgeLookup::Ready(_)));
        assert!(meter.is_none());
    }

    #[test]
    fn an_unknown_name_is_none_here_too() {
        assert!(metered_judge_by_name("nope", OnApiFailure::Fail).is_none());
    }

    #[test]
    fn a_hosted_judge_is_metered_when_it_can_be_built_at_all() {
        // Without an API key the judge is `Unavailable` and there is nothing to
        // meter, which is the normal state of CI. Either way nothing calls out.
        let (lookup, meter) =
            metered_judge_by_name("jev", OnApiFailure::Fail).expect("a known name");
        match lookup {
            JudgeLookup::Ready(_) => {
                let usage = meter.expect("a built hosted judge is metered").usage();
                assert_eq!(usage.calls, 0, "nothing has been judged yet");
            }
            JudgeLookup::Unavailable { .. } => assert!(meter.is_none()),
        }
    }
}
