//! Execution: run a decision's legs and merge them back into one result.

use crate::config::{Config, Tier};
use crate::error::Error;
use crate::extract::{extract_local, OcrResult};
use crate::judge::{HeuristicJudge, PageJudge};
use crate::metadata::{RouteMetadata, FALLBACK_LEG_FAILED};
use crate::policy::{decide_with, Decision, Leg};

/// The boundary between the router and whatever actually calls an OCR provider.
///
/// Implementations must be `Sync`: legs run concurrently on borrowed threads.
pub trait OcrHost: Sync {
    /// Run `model` over `document`. `pages` is a 0-indexed subset, or `None` for all pages.
    ///
    /// A provider that cannot take a page list should call [`crate::split_pdf`] and
    /// [`crate::remap_pages`] itself so the returned `Page.index` values stay original.
    fn ocr(&self, model: &str, document: &[u8], pages: Option<&[u32]>) -> Result<OcrResult, Error>;
}

/// The result of executing a decision, with the provenance that goes with it.
#[derive(Debug, Clone, PartialEq)]
pub struct Outcome {
    /// The merged, page-ordered result.
    pub result: OcrResult,
    /// The `doc_route` metadata for this run.
    pub metadata: RouteMetadata,
    /// The decision that produced it.
    pub decision: Decision,
}

fn run_leg(leg: &Leg, bytes: &[u8], host: &dyn OcrHost) -> Result<OcrResult, Error> {
    let pages = leg.pages.as_deref();
    let result = if leg.tier == Tier::Local {
        extract_local(bytes, pages)
    } else {
        host.ocr(&leg.model, bytes, pages)
    };
    result.map_err(|source| Error::LegFailed {
        model: leg.model.clone(),
        source: Box::new(source),
    })
}

/// Classify, plan and execute in one call.
///
/// Local legs run in-process; OCR legs go to `host`. Legs run concurrently on scoped
/// threads. If any leg fails the whole document is rerun once on the default model:
/// `metadata.fallback_reason` becomes `"leg_failed"` and `metadata.routed_model` becomes
/// the model that actually ran, so the metadata never claims bytes went somewhere they
/// did not. If that rerun also fails the error is returned.
pub fn run(
    bytes: &[u8],
    cfg: &Config,
    default_model: Option<&str>,
    host: &dyn OcrHost,
) -> Result<Outcome, Error> {
    run_with(bytes, cfg, default_model, host, &HeuristicJudge)
}

/// [`run`], with `judge` deciding which pages need OCR.
///
/// Only the decision changes; the legs, the concurrency and the one-shot
/// fallback are the same. A judge that talks to a network service is built
/// outside this crate and passed in here, exactly like an [`OcrHost`].
pub fn run_with(
    bytes: &[u8],
    cfg: &Config,
    default_model: Option<&str>,
    host: &dyn OcrHost,
    judge: &dyn PageJudge,
) -> Result<Outcome, Error> {
    let decision = decide_with(bytes, cfg, default_model, judge);
    let mut metadata = decision.metadata();

    let plan = match &decision {
        Decision::Bypass { model, .. } => {
            let result = host.ocr(model, bytes, None)?;
            return Ok(Outcome {
                result,
                metadata,
                decision,
            });
        }
        Decision::Route { plan, .. } => plan.clone(),
    };

    let leg_results: Vec<Result<OcrResult, Error>> = std::thread::scope(|scope| {
        let handles: Vec<_> = plan
            .legs
            .iter()
            .map(|leg| scope.spawn(move || run_leg(leg, bytes, host)))
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|_| Err(Error::Host("leg thread panicked".to_string())))
            })
            .collect()
    });

    let mut collected: Vec<(Leg, OcrResult)> = Vec::with_capacity(leg_results.len());
    let mut failure: Option<Error> = None;
    for (leg, result) in plan.legs.iter().zip(leg_results) {
        match result {
            Ok(ok) => collected.push((leg.clone(), ok)),
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }

    if let Some(error) = failure {
        // One retry of the whole document on the default model, exactly as the Python
        // reference's `legs_fallback_model` does.
        let fallback_model = default_model
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(cfg.tiers.standard.as_str())
            .to_string();
        let result = match host.ocr(&fallback_model, bytes, None) {
            Ok(ok) => ok,
            Err(fallback_error) => {
                // `Error::Host`'s Display already prefixes "OCR host error:", so
                // formatting a host error into the message of another `Error::Host`
                // printed that prefix twice. Unwrap exactly one level when the
                // failure is a host error; anything else keeps its own wording.
                let detail = match &fallback_error {
                    Error::Host(message) => message.clone(),
                    other => other.to_string(),
                };
                return Err(Error::LegFailed {
                    model: fallback_model,
                    source: Box::new(Error::Host(format!(
                        "{detail} (whole-document retry after {error})"
                    ))),
                });
            }
        };
        metadata.fallback_reason = Some(FALLBACK_LEG_FAILED.to_string());
        metadata.routed_model = fallback_model;
        return Ok(Outcome {
            result,
            metadata,
            decision,
        });
    }

    Ok(Outcome {
        result: crate::merge::merge(&collected),
        metadata,
        decision,
    })
}
