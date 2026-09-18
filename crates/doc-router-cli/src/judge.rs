//! Name -> judge. The one place a new judge is registered.
//!
//! Adding a judge is a single arm in [`judge_by_name`] and its name in
//! [`JUDGE_NAMES`]. Nothing else knows which judges exist: the CLI, the
//! benchmark harness, the scoring and the report all work off `&dyn PageJudge`.
//!
//! A judge that needs the network (an API-backed one, say) lives in the crate
//! that owns that dependency, exactly like [`OcrHost`](doc_router::OcrHost)
//! implementations do, and is registered here by adding this crate's dependency
//! on it. The core `doc-router` crate stays network-free.
//!
//! This module lives in the CLI crate because the CLI is where the workspace
//! already assembles networked components out of the environment — see
//! [`host`](crate::host), which does the same job for the OCR endpoint. The
//! benchmark harness re-exports it (`doc_router_bench::registry`) rather than
//! keeping a second copy: two lookups that drift apart would mean `--judge jev`
//! meaning one thing in the CLI and another in the bench, which is exactly the
//! bug a registry exists to prevent. The dependency only points this way because
//! `doc-router-bench` is `publish = false` and could never be a dependency of a
//! published crate.
//!
//! # Two ways a name can fail to produce a judge
//!
//! They are different questions and the caller has to be able to tell them
//! apart, so [`judge_by_name`] answers both:
//!
//! * **The name is not registered** (`None`) — a typo, or a judge from a branch
//!   that is not merged. The caller stops and lists [`JUDGE_NAMES`].
//! * **The name is registered but this machine cannot build it**
//!   ([`JudgeLookup::Unavailable`]) — an API-backed judge with no credentials in
//!   the environment. That is a property of the machine, not of the request, so
//!   the caller decides what to do about it: the CLI, which was asked for that
//!   judge and nothing else, fails; the bench, which was asked for several,
//!   scores the ones it can and reports the skip. Telling an operator "unknown
//!   judge `jev`" when the real problem is an unset variable sends them looking
//!   in the wrong place.

use std::sync::Mutex;

use doc_router::{Error, HeuristicJudge, PageEvidence, PageJudge, PageVerdict};
use doc_router_jev::{
    JevJudge, JevMode, API_KEY_ENV_VARS, JUDGE_NAME_ALWAYS, JUDGE_NAME_GATED,
    REASON_FALLBACK_BREAKER_OPEN, REASON_FALLBACK_HTTP, REASON_FALLBACK_PROTOCOL,
    REASON_FALLBACK_TIMEOUT,
};
use serde::Serialize;

/// The judge used when nothing asks for another one.
///
/// It is the baseline in both senses: the accuracy every other judge has to beat
/// and the latency every other judge's overhead is measured against.
pub const BASELINE_JUDGE: &str = "heuristic";

/// Every name [`judge_by_name`] answers to, for `--help` and for error messages.
pub const JUDGE_NAMES: &[&str] = &[BASELINE_JUDGE, JUDGE_NAME_ALWAYS, JUDGE_NAME_GATED];

/// What a hosted judge should do when the API does not answer.
///
/// This is not a preference, it is a property of the caller. A CLI run is one
/// document an operator is waiting on: falling back to the heuristic keeps that
/// document routing, and the page is labelled with the reason so the fallback is
/// visible rather than silent. A benchmark run is a measurement: a `jev` row
/// carrying the heuristic's verdicts, scored and printed under another judge's
/// name, would be a fabrication, so there the same failure has to stop the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnApiFailure {
    /// Fall back to the heuristic for that document, labelling every page with
    /// the reason the API did not answer.
    FallBack,
    /// Fail the document, so nothing is reported under a name that did not
    /// produce it.
    Fail,
}

/// What a registered name resolved to.
pub enum JudgeLookup {
    /// A judge that can run here and now.
    Ready(Box<dyn PageJudge>),
    /// A registered judge this machine cannot construct, and the one-line reason
    /// an operator needs in order to fix it.
    Unavailable {
        /// The registered name, so a failure can be reported by name.
        name: &'static str,
        /// What is missing, phrased as something to do.
        reason: String,
    },
}

/// Hand-written because [`PageJudge`] is not `Debug` and should not have to be:
/// a judge is behaviour, and the only part of it worth printing here is the name
/// it answers to.
impl std::fmt::Debug for JudgeLookup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JudgeLookup::Ready(judge) => f.debug_tuple("Ready").field(&judge.name()).finish(),
            JudgeLookup::Unavailable { name, reason } => f
                .debug_struct("Unavailable")
                .field("name", name)
                .field("reason", reason)
                .finish(),
        }
    }
}

impl JudgeLookup {
    /// The judge, or `None` when it is unavailable.
    pub fn ready(self) -> Option<Box<dyn PageJudge>> {
        match self {
            JudgeLookup::Ready(judge) => Some(judge),
            JudgeLookup::Unavailable { .. } => None,
        }
    }
}

/// Resolve a `--judge` name.
///
/// `None` means "no judge by that name", which a front end turns into an error
/// listing [`JUDGE_NAMES`]. A registered judge that cannot be built on this
/// machine is [`JudgeLookup::Unavailable`] and not `None`: see the module docs
/// for why those two are not the same answer.
pub fn judge_by_name(name: &str, on_api_failure: OnApiFailure) -> Option<JudgeLookup> {
    match name {
        BASELINE_JUDGE => Some(JudgeLookup::Ready(Box::new(HeuristicJudge))),
        JUDGE_NAME_ALWAYS => Some(jev(JevMode::Always, on_api_failure)),
        JUDGE_NAME_GATED => Some(jev(JevMode::Gated, on_api_failure)),
        _ => None,
    }
}

/// Build a [`JevJudge`] from the environment.
///
/// The reason on the unavailable branch names the variables and stops there. How
/// to get a key into the environment is the front end's business: the benchmark
/// harness reads a `.env` at the workspace root and says so, the CLI does not
/// read one — it is a binary that can be run from anywhere — and telling its
/// users to edit a file it never opens would be worse than saying nothing.
fn jev(mode: JevMode, on_api_failure: OnApiFailure) -> JudgeLookup {
    match JevJudge::from_env() {
        Some(judge) => JudgeLookup::Ready(Box::new(
            judge
                .with_mode(mode)
                .with_strict(on_api_failure == OnApiFailure::Fail),
        )),
        None => JudgeLookup::Unavailable {
            name: mode.judge_name(),
            reason: format!(
                "no API key in the environment: set {} (or {})",
                API_KEY_ENV_VARS[0], API_KEY_ENV_VARS[1]
            ),
        },
    }
}

/// A `--judge` help text: `lead`, then every registered name.
///
/// The names are read from [`JUDGE_NAMES`] rather than written out again at the
/// flag, so adding a judge to the registry makes it discoverable from `--help`
/// without a second edit. `baseline_label` is the one word the two front ends
/// disagree about: the bench always *includes* the baseline, the CLI *defaults*
/// to it.
///
/// Deliberately not a `PossibleValuesParser`: that would replace the
/// hand-written unknown-judge error, which distinguishes a name nobody
/// registered from one that is registered but unavailable (no API key), with
/// clap's generic one.
pub fn judge_help(lead: &str, baseline_label: &str) -> String {
    format!(
        "{lead} One of: {} ({baseline_label}: {BASELINE_JUDGE})",
        JUDGE_NAMES.join(", ")
    )
}

/// The fallback reasons a hosted judge records on a page it could not answer.
///
/// Listed from the constants rather than matched on a `jev_fallback_` prefix, so
/// a reason that is renamed there fails to compile here instead of quietly
/// ceasing to be reported.
const FALLBACK_REASONS: [&str; 4] = [
    REASON_FALLBACK_TIMEOUT,
    REASON_FALLBACK_HTTP,
    REASON_FALLBACK_PROTOCOL,
    REASON_FALLBACK_BREAKER_OPEN,
];

/// A judge that passes every call through and remembers the pages its inner
/// judge could not answer for itself.
///
/// A non-strict hosted judge answers a failed call with the heuristic's verdicts
/// relabelled — the right behaviour for one document an operator is waiting on,
/// and invisible by the time the verdicts reach the CLI: only
/// [`PageVerdict::reason`] carries the fallback, and
/// [`Classification`](doc_router::Classification) keeps pdf-inspector's reasons,
/// not the judge's. Wrapping the judge is the one place the reasons are all
/// visible, so this is where they are counted.
pub struct RecordingJudge<'a> {
    inner: &'a dyn PageJudge,
    /// `(reason, pages)` in first-seen order. A `Mutex` because [`PageJudge`] is
    /// `Sync` and takes `&self`, and because one document can be judged from
    /// more than one leg.
    fallbacks: Mutex<Vec<JudgeFallback>>,
}

impl std::fmt::Debug for RecordingJudge<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RecordingJudge")
            .field("inner", &self.inner.name())
            .field("fallbacks", &self.fallbacks.lock().ok())
            .finish()
    }
}

impl<'a> RecordingJudge<'a> {
    /// Wrap `inner`.
    pub fn new(inner: &'a dyn PageJudge) -> Self {
        RecordingJudge {
            inner,
            fallbacks: Mutex::new(Vec::new()),
        }
    }

    /// What ran, and what it could not answer.
    pub fn provenance(&self) -> JudgeProvenance {
        JudgeProvenance {
            name: self.inner.name().to_string(),
            fallbacks: self.fallbacks.lock().map(|f| f.clone()).unwrap_or_default(),
        }
    }
}

impl PageJudge for RecordingJudge<'_> {
    fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
        let verdicts = self.inner.judge(evidence)?;
        for verdict in &verdicts {
            if !FALLBACK_REASONS.contains(&verdict.reason.as_str()) {
                continue;
            }
            if let Ok(mut fallbacks) = self.fallbacks.lock() {
                match fallbacks.iter_mut().find(|f| f.reason == verdict.reason) {
                    Some(seen) => seen.pages += 1,
                    None => fallbacks.push(JudgeFallback {
                        reason: verdict.reason.clone(),
                        pages: 1,
                    }),
                }
            }
        }
        Ok(verdicts)
    }

    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn needs_text(&self) -> bool {
        self.inner.needs_text()
    }
}

/// Which judge produced a run's verdicts, for `--json` output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JudgeProvenance {
    /// The registered name of the judge that ran.
    pub name: String,
    /// Pages the judge could not answer for itself. Absent from the JSON when
    /// there were none, which is every run of the built-in judge.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub fallbacks: Vec<JudgeFallback>,
}

impl JudgeProvenance {
    /// One line for stderr when the judge fell back, or `None` when it did not.
    pub fn fallback_note(&self) -> Option<String> {
        if self.fallbacks.is_empty() {
            return None;
        }
        let detail: Vec<String> = self
            .fallbacks
            .iter()
            .map(|f| format!("{} page(s) {}", f.pages, f.reason))
            .collect();
        Some(format!(
            "judge `{}` fell back to the heuristic: {}",
            self.name,
            detail.join(", ")
        ))
    }
}

/// Pages a judge answered with a fallback, by reason.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JudgeFallback {
    /// The verdict reason, e.g. `jev_fallback_http`.
    pub reason: String,
    /// How many pages carried it.
    pub pages: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A judge that answers with whatever reasons the test asks for, so the
    /// recording can be tested without a key, a network or a mock server.
    struct FakeJudge(Vec<&'static str>);

    impl PageJudge for FakeJudge {
        fn judge(&self, evidence: &[PageEvidence]) -> Result<Vec<PageVerdict>, Error> {
            Ok(evidence
                .iter()
                .zip(&self.0)
                .map(|(page, reason)| PageVerdict {
                    page: page.page,
                    needs_ocr: true,
                    confidence: 1.0,
                    reason: (*reason).to_string(),
                })
                .collect())
        }

        fn name(&self) -> &'static str {
            "fake"
        }

        fn needs_text(&self) -> bool {
            true
        }
    }

    fn evidence(pages: u32) -> Vec<PageEvidence> {
        (0..pages)
            .map(|page| PageEvidence {
                page,
                text: None,
                reasons: Vec::new(),
                flagged_by_inspector: false,
                has_tables: false,
                has_columns: false,
                has_encoding_issues: false,
            })
            .collect()
    }

    #[test]
    fn the_baseline_resolves_and_names_itself() {
        let judge = judge_by_name(BASELINE_JUDGE, OnApiFailure::FallBack)
            .expect("the baseline is always registered")
            .ready()
            .expect("the baseline needs nothing from the environment");
        assert_eq!(judge.name(), BASELINE_JUDGE);
    }

    #[test]
    fn every_advertised_name_resolves() {
        for name in JUDGE_NAMES {
            match judge_by_name(name, OnApiFailure::FallBack)
                .unwrap_or_else(|| panic!("`{name}` is advertised"))
            {
                // The registry key and the judge's own name are the same string,
                // so a report column and a `--judge` flag never disagree.
                JudgeLookup::Ready(judge) => assert_eq!(judge.name(), *name),
                // Same promise on the other branch: a failure is reported under
                // the name that was asked for, with something actionable
                // attached.
                JudgeLookup::Unavailable {
                    name: reported,
                    reason,
                } => {
                    assert_eq!(reported, *name);
                    assert!(!reason.is_empty(), "`{name}` must say what is missing");
                }
            }
        }
    }

    #[test]
    fn an_unknown_name_is_none_rather_than_a_silent_fallback_to_the_baseline() {
        assert!(judge_by_name("heuristik", OnApiFailure::FallBack).is_none());
        assert!(judge_by_name("", OnApiFailure::FallBack).is_none());
        // Near-misses of the registered hosted names are unknown names too.
        assert!(judge_by_name("jev-gated", OnApiFailure::FallBack).is_none());
    }

    #[test]
    fn the_baseline_is_advertised() {
        assert!(JUDGE_NAMES.contains(&BASELINE_JUDGE));
    }

    #[test]
    fn the_hosted_judge_is_advertised_under_both_of_its_modes() {
        assert!(JUDGE_NAMES.contains(&JUDGE_NAME_ALWAYS));
        assert!(JUDGE_NAMES.contains(&JUDGE_NAME_GATED));
    }

    /// Whichever branch this machine takes, an unset key is never an unknown
    /// name — that distinction is the reason [`JudgeLookup`] exists.
    #[test]
    fn a_hosted_judge_is_a_known_name_with_or_without_a_key() {
        let lookup = judge_by_name(JUDGE_NAME_ALWAYS, OnApiFailure::FallBack)
            .expect("`jev` is a registered name");
        match lookup {
            JudgeLookup::Ready(judge) => {
                assert_eq!(judge.name(), JUDGE_NAME_ALWAYS);
                assert!(judge.needs_text(), "a hosted judge reads the text layer");
            }
            JudgeLookup::Unavailable { reason, .. } => {
                assert!(reason.contains(API_KEY_ENV_VARS[0]), "{reason}");
                assert!(reason.contains(API_KEY_ENV_VARS[1]), "{reason}");
            }
        }
    }

    #[test]
    fn the_help_text_lists_every_registered_name() {
        let help = judge_help("Judge to use.", "default");
        for name in JUDGE_NAMES {
            assert!(help.contains(name), "{help}");
        }
        assert!(help.starts_with("Judge to use. One of: "), "{help}");
        assert!(
            help.ends_with(&format!("(default: {BASELINE_JUDGE})")),
            "{help}"
        );
    }

    #[test]
    fn recording_passes_verdicts_through_unchanged() {
        let inner = FakeJudge(vec![doc_router::judge::REASON_CLEAR; 2]);
        let judge = RecordingJudge::new(&inner);
        let verdicts = judge.judge(&evidence(2)).expect("the fake always answers");
        assert_eq!(verdicts.len(), 2);
        assert_eq!(judge.name(), "fake");
        assert!(judge.needs_text(), "needs_text is the inner judge's answer");
        assert_eq!(judge.provenance().fallbacks, Vec::new());
        assert_eq!(judge.provenance().fallback_note(), None);
    }

    #[test]
    fn recording_counts_fallback_pages_by_reason() {
        let inner = FakeJudge(vec![
            REASON_FALLBACK_HTTP,
            doc_router::judge::REASON_CLEAR,
            REASON_FALLBACK_HTTP,
            REASON_FALLBACK_TIMEOUT,
        ]);
        let judge = RecordingJudge::new(&inner);
        judge.judge(&evidence(4)).expect("the fake always answers");

        let provenance = judge.provenance();
        assert_eq!(provenance.name, "fake");
        assert_eq!(
            provenance.fallbacks,
            vec![
                JudgeFallback {
                    reason: REASON_FALLBACK_HTTP.to_string(),
                    pages: 2,
                },
                JudgeFallback {
                    reason: REASON_FALLBACK_TIMEOUT.to_string(),
                    pages: 1,
                },
            ]
        );
        let note = provenance.fallback_note().expect("a fallback happened");
        assert!(note.contains("2 page(s) jev_fallback_http"), "{note}");
        assert!(note.contains("1 page(s) jev_fallback_timeout"), "{note}");
    }

    /// The provenance is what `--json` carries, so its shape is part of the
    /// output contract: a name always, fallbacks only when there were some.
    #[test]
    fn provenance_json_omits_an_empty_fallback_list() {
        let quiet = JudgeProvenance {
            name: BASELINE_JUDGE.to_string(),
            fallbacks: Vec::new(),
        };
        let json = serde_json::to_string(&quiet).expect("provenance serialises");
        assert_eq!(json, r#"{"name":"heuristic"}"#);

        let noisy = JudgeProvenance {
            name: JUDGE_NAME_ALWAYS.to_string(),
            fallbacks: vec![JudgeFallback {
                reason: REASON_FALLBACK_BREAKER_OPEN.to_string(),
                pages: 3,
            }],
        };
        let json = serde_json::to_string(&noisy).expect("provenance serialises");
        assert_eq!(
            json,
            r#"{"name":"jev","fallbacks":[{"reason":"jev_fallback_breaker_open","pages":3}]}"#
        );
    }
}
