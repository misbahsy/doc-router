//! The routing decision table. Pure, deterministic, no I/O.

use serde::{Deserialize, Serialize};

use crate::classify::{classify_with, is_pdf, Classification};
use crate::config::{Config, OcrTier, Tier};
use crate::error::Error;
use crate::judge::{HeuristicJudge, PageJudge};

/// One unit of work: a model, its tier, and the pages it covers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Leg {
    /// The model that runs this leg.
    pub model: String,
    /// The tier the model belongs to.
    pub tier: Tier,
    /// 0-indexed pages, or `None` for the whole document.
    pub pages: Option<Vec<u32>>,
}

impl Leg {
    /// A leg covering the whole document.
    pub fn whole(model: impl Into<String>, tier: Tier) -> Self {
        Leg {
            model: model.into(),
            tier,
            pages: None,
        }
    }

    /// A leg covering a page subset.
    pub fn subset(model: impl Into<String>, tier: Tier, pages: Vec<u32>) -> Self {
        Leg {
            model: model.into(),
            tier,
            pages: Some(pages),
        }
    }

    /// True when this leg runs in-process rather than on a host.
    pub fn is_local(&self) -> bool {
        self.tier == Tier::Local
    }
}

/// Why the router chose this plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reason {
    /// Classification confidence was below `min_confidence`.
    LowConfidence,
    /// Every page has a usable text layer.
    TextLayer,
    /// Every page needs OCR.
    Scanned,
    /// Some pages need OCR and splitting is enabled.
    MixedSplit,
    /// Some pages need OCR but splitting is disabled.
    MixedUnsplit,
}

impl Reason {
    /// The wire string for this reason.
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::LowConfidence => "low_confidence",
            Reason::TextLayer => "text_layer",
            Reason::Scanned => "scanned",
            Reason::MixedSplit => "mixed_split",
            Reason::MixedUnsplit => "mixed_unsplit",
        }
    }
}

impl std::fmt::Display for Reason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The full routing plan for one document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Legs to execute, local first when the plan is split.
    pub legs: Vec<Leg>,
    /// Why this plan was chosen.
    pub reason: Reason,
    /// The OCR leg's tier, or `Local` when local is the only leg.
    pub tier: Tier,
    /// The model the request is billed/attributed to.
    pub routed_model: String,
}

impl Plan {
    /// True when the document is split across more than one leg.
    pub fn is_split(&self) -> bool {
        self.legs.len() > 1
    }
}

fn single(tier: Tier, model: &str, reason: Reason) -> Plan {
    Plan {
        legs: vec![Leg::whole(model, tier)],
        reason,
        tier,
        routed_model: model.to_string(),
    }
}

/// The decision table, in the same order as `policy.py::plan_route`.
///
/// Errors only when the config does not name the model a branch needs.
pub fn plan_route(c: &Classification, cfg: &Config) -> Result<Plan, Error> {
    let standard_model = cfg.model_for_tier(Tier::Standard)?.to_string();

    // 1. Not sure enough to route by page: send the whole document to standard.
    if c.confidence < cfg.min_confidence {
        return Ok(single(
            Tier::Standard,
            &standard_model,
            Reason::LowConfidence,
        ));
    }

    // 2. Complex layouts (tables, columns) may be worth the premium model.
    let ocr_tier: OcrTier = if c.is_complex_layout {
        cfg.resolved_complex_layout_tier()
    } else {
        OcrTier::Standard
    };
    let ocr_model = cfg.model_for_tier(ocr_tier.tier())?.to_string();

    // Pages out of range are already dropped in `classify`; re-filter defensively so a
    // hand-built Classification cannot produce a leg pointing past the end of the document.
    let ocr_pages: Vec<u32> = c
        .pages_needing_ocr
        .iter()
        .copied()
        .filter(|p| *p < c.page_count)
        .collect();

    // 3. Nothing needs OCR: read the text layer locally.
    if ocr_pages.is_empty() {
        let local_model = cfg.model_for_tier(Tier::Local)?.to_string();
        return Ok(single(Tier::Local, &local_model, Reason::TextLayer));
    }

    // 4. Everything needs OCR: one whole-document OCR call.
    if ocr_pages.len() as u32 >= c.page_count {
        return Ok(single(ocr_tier.tier(), &ocr_model, Reason::Scanned));
    }

    // 5. Mixed, but the host wants one call: whole document to OCR.
    if !cfg.split_pages {
        return Ok(single(ocr_tier.tier(), &ocr_model, Reason::MixedUnsplit));
    }

    // 6. Mixed and splitting is on: local for the text pages, OCR for the rest.
    let local_model = cfg.model_for_tier(Tier::Local)?.to_string();
    let text_pages = c.text_layer_pages();
    Ok(Plan {
        legs: vec![
            Leg::subset(local_model, Tier::Local, text_pages),
            Leg::subset(&ocr_model, ocr_tier.tier(), ocr_pages),
        ],
        reason: Reason::MixedSplit,
        tier: ocr_tier.tier(),
        routed_model: ocr_model,
    })
}

/// Why a document skipped routing entirely.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Bypass {
    /// The bytes are not a PDF.
    #[serde(rename = "not_pdf")]
    NotPdf,
    /// The document is larger than `max_document_bytes`.
    #[serde(rename = "oversize")]
    Oversize,
    /// The host knows the input was an image, not a document.
    #[serde(rename = "image")]
    Image,
    /// The classifier could not read the PDF.
    #[serde(rename = "classifier_unavailable")]
    ClassifierFailed,
}

impl Bypass {
    /// The wire string for this bypass reason.
    pub fn as_str(self) -> &'static str {
        match self {
            Bypass::NotPdf => "not_pdf",
            Bypass::Oversize => "oversize",
            Bypass::Image => "image",
            Bypass::ClassifierFailed => "classifier_unavailable",
        }
    }
}

impl std::fmt::Display for Bypass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What the router decided to do with a document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Decision {
    /// Send the whole document to one model without classifying it.
    Bypass {
        /// Why routing was skipped.
        reason: Bypass,
        /// The model that gets the whole document.
        model: String,
        /// Human-readable detail (e.g. the classifier's error).
        detail: Option<String>,
    },
    /// Execute this plan.
    Route {
        /// The plan to execute.
        plan: Plan,
        /// What the classifier found.
        classification: Classification,
    },
}

impl Decision {
    /// The model a bypass sends to: `default_model`, else `tiers.standard`.
    fn bypass_model(cfg: &Config, default_model: Option<&str>) -> String {
        default_model
            .filter(|m| !m.trim().is_empty())
            .unwrap_or(cfg.tiers.standard.as_str())
            .to_string()
    }

    /// A bypass for input the host already knows is an image.
    pub fn bypass_image(cfg: &Config, default_model: Option<&str>) -> Decision {
        Decision::Bypass {
            reason: Bypass::Image,
            model: Decision::bypass_model(cfg, default_model),
            detail: None,
        }
    }

    /// The plan, when this decision is a route.
    pub fn plan(&self) -> Option<&Plan> {
        match self {
            Decision::Route { plan, .. } => Some(plan),
            Decision::Bypass { .. } => None,
        }
    }

    /// The classification, when this decision is a route.
    pub fn classification(&self) -> Option<&Classification> {
        match self {
            Decision::Route { classification, .. } => Some(classification),
            Decision::Bypass { .. } => None,
        }
    }
}

/// Decide what to do with `bytes`. Never errors: anything unroutable becomes a bypass.
///
/// Uses the built-in [`HeuristicJudge`]. [`decide_with`] takes any other judge.
pub fn decide(bytes: &[u8], cfg: &Config, default_model: Option<&str>) -> Decision {
    decide_with(bytes, cfg, default_model, &HeuristicJudge)
}

/// [`decide`], with `judge` deciding which pages need OCR.
///
/// The judge is the only thing that changes. Everything downstream of it — the
/// size and type gates, the decision table, the bypass a classifier failure
/// produces — is the same code on the same inputs, so a run with a hosted judge
/// differs from a run with the built-in one only where the two disagree about a
/// page.
pub fn decide_with(
    bytes: &[u8],
    cfg: &Config,
    default_model: Option<&str>,
    judge: &dyn PageJudge,
) -> Decision {
    let bypass = |reason: Bypass, detail: Option<String>| Decision::Bypass {
        reason,
        model: Decision::bypass_model(cfg, default_model),
        detail,
    };

    if !is_pdf(bytes) {
        return bypass(Bypass::NotPdf, None);
    }
    if bytes.len() as u64 > cfg.max_document_bytes {
        return bypass(
            Bypass::Oversize,
            Some(format!(
                "{} bytes exceeds max_document_bytes {}",
                bytes.len(),
                cfg.max_document_bytes
            )),
        );
    }
    let classification = match classify_with(bytes, judge) {
        Ok(c) => c,
        Err(e) => return bypass(Bypass::ClassifierFailed, Some(e.to_string())),
    };
    match plan_route(&classification, cfg) {
        Ok(plan) => Decision::Route {
            plan,
            classification,
        },
        Err(e) => bypass(Bypass::ClassifierFailed, Some(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::PdfType;
    use crate::config::Tiers;

    const LOCAL: &str = "local_pdf/extract";
    const STANDARD: &str = "mistral-ocr";
    const PREMIUM: &str = "gpt-5-ocr";

    fn cfg_with_premium() -> Config {
        Config::new(Tiers::new(LOCAL, STANDARD).with_premium(PREMIUM))
    }

    fn cfg_without_premium() -> Config {
        Config::new(Tiers::new(LOCAL, STANDARD))
    }

    fn classification(
        page_count: u32,
        ocr: Vec<u32>,
        complex: bool,
        confidence: f32,
    ) -> Classification {
        Classification {
            pdf_type: PdfType::Mixed,
            confidence,
            page_count,
            pages_needing_ocr: ocr,
            is_complex_layout: complex,
            has_encoding_issues: false,
            ocr_reasons: Vec::new(),
            classify_ms: 1.0,
        }
    }

    #[test]
    fn row1_low_confidence_goes_to_standard() {
        for cfg in [cfg_with_premium(), cfg_without_premium()] {
            let c = classification(4, vec![1], true, 0.2);
            let plan = plan_route(&c, &cfg).expect("plan");
            assert_eq!(plan.reason, Reason::LowConfidence);
            assert_eq!(plan.tier, Tier::Standard);
            assert_eq!(plan.routed_model, STANDARD);
            assert_eq!(plan.legs, vec![Leg::whole(STANDARD, Tier::Standard)]);
            assert!(!plan.is_split());
        }
    }

    #[test]
    fn row1_fires_even_with_splitting_disabled() {
        let cfg = Config {
            split_pages: false,
            ..cfg_with_premium()
        };
        let plan = plan_route(&classification(4, vec![1], false, 0.1), &cfg).expect("plan");
        assert_eq!(plan.reason, Reason::LowConfidence);
    }

    #[test]
    fn confidence_exactly_at_the_threshold_is_not_low() {
        let cfg = cfg_with_premium();
        let plan =
            plan_route(&classification(2, vec![], false, cfg.min_confidence), &cfg).expect("plan");
        assert_eq!(plan.reason, Reason::TextLayer);
    }

    #[test]
    fn row3_text_layer_stays_local() {
        for cfg in [cfg_with_premium(), cfg_without_premium()] {
            for complex in [false, true] {
                for split in [false, true] {
                    let cfg = Config {
                        split_pages: split,
                        ..cfg.clone()
                    };
                    let plan =
                        plan_route(&classification(3, vec![], complex, 0.99), &cfg).expect("plan");
                    assert_eq!(plan.reason, Reason::TextLayer);
                    assert_eq!(plan.tier, Tier::Local);
                    assert_eq!(plan.routed_model, LOCAL);
                    assert_eq!(plan.legs, vec![Leg::whole(LOCAL, Tier::Local)]);
                }
            }
        }
    }

    #[test]
    fn row4_scanned_takes_the_whole_document() {
        let plan = plan_route(
            &classification(3, vec![0, 1, 2], false, 0.95),
            &cfg_with_premium(),
        )
        .expect("plan");
        assert_eq!(plan.reason, Reason::Scanned);
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.legs, vec![Leg::whole(STANDARD, Tier::Standard)]);

        // Complex layout promotes the tier, but only when premium is configured.
        let plan = plan_route(
            &classification(3, vec![0, 1, 2], true, 0.95),
            &cfg_with_premium(),
        )
        .expect("plan");
        assert_eq!(plan.tier, Tier::Premium);
        assert_eq!(plan.routed_model, PREMIUM);

        let plan = plan_route(
            &classification(3, vec![0, 1, 2], true, 0.95),
            &cfg_without_premium(),
        )
        .expect("plan");
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.routed_model, STANDARD);
    }

    #[test]
    fn row5_mixed_unsplit_takes_the_whole_document() {
        let cfg = Config {
            split_pages: false,
            ..cfg_with_premium()
        };
        let plan = plan_route(&classification(4, vec![1, 3], false, 0.7), &cfg).expect("plan");
        assert_eq!(plan.reason, Reason::MixedUnsplit);
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.legs, vec![Leg::whole(STANDARD, Tier::Standard)]);
        assert!(!plan.is_split());

        let cfg = Config {
            split_pages: false,
            ..cfg_with_premium()
        };
        let plan = plan_route(&classification(4, vec![1, 3], true, 0.7), &cfg).expect("plan");
        assert_eq!(plan.reason, Reason::MixedUnsplit);
        assert_eq!(plan.tier, Tier::Premium);
        assert_eq!(plan.routed_model, PREMIUM);
    }

    #[test]
    fn row6_mixed_split_has_a_local_leg_and_an_ocr_leg() {
        let plan = plan_route(
            &classification(4, vec![1, 3], false, 0.7),
            &cfg_with_premium(),
        )
        .expect("plan");
        assert_eq!(plan.reason, Reason::MixedSplit);
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.routed_model, STANDARD);
        assert!(plan.is_split());
        assert_eq!(
            plan.legs,
            vec![
                Leg::subset(LOCAL, Tier::Local, vec![0, 2]),
                Leg::subset(STANDARD, Tier::Standard, vec![1, 3]),
            ]
        );
    }

    #[test]
    fn row6_complex_layout_promotes_only_the_ocr_leg() {
        let plan = plan_route(
            &classification(4, vec![1, 3], true, 0.7),
            &cfg_with_premium(),
        )
        .expect("plan");
        assert_eq!(plan.tier, Tier::Premium);
        assert_eq!(plan.routed_model, PREMIUM);
        assert_eq!(
            plan.legs,
            vec![
                Leg::subset(LOCAL, Tier::Local, vec![0, 2]),
                Leg::subset(PREMIUM, Tier::Premium, vec![1, 3]),
            ]
        );

        let plan = plan_route(
            &classification(4, vec![1, 3], true, 0.7),
            &cfg_without_premium(),
        )
        .expect("plan");
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.legs[1].model, STANDARD);
    }

    #[test]
    fn complex_layout_tier_standard_never_promotes() {
        let cfg = Config {
            complex_layout_tier: OcrTier::Standard,
            ..cfg_with_premium()
        };
        let plan = plan_route(&classification(4, vec![1, 3], true, 0.7), &cfg).expect("plan");
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.routed_model, STANDARD);
    }

    #[test]
    fn empty_document_reads_as_text_layer() {
        // page_count 0 with no OCR pages: row 3 fires before row 4, as in policy.py.
        let plan =
            plan_route(&classification(0, vec![], false, 0.9), &cfg_with_premium()).expect("plan");
        assert_eq!(plan.reason, Reason::TextLayer);
    }

    #[test]
    fn out_of_range_ocr_pages_are_ignored() {
        let plan =
            plan_route(&classification(2, vec![7], false, 0.9), &cfg_with_premium()).expect("plan");
        assert_eq!(plan.reason, Reason::TextLayer);
    }

    #[test]
    fn reasons_and_bypasses_serialise_as_specified() {
        for (reason, wire) in [
            (Reason::LowConfidence, "\"low_confidence\""),
            (Reason::TextLayer, "\"text_layer\""),
            (Reason::Scanned, "\"scanned\""),
            (Reason::MixedSplit, "\"mixed_split\""),
            (Reason::MixedUnsplit, "\"mixed_unsplit\""),
        ] {
            assert_eq!(serde_json::to_string(&reason).unwrap(), wire);
        }
        for (bypass, wire) in [
            (Bypass::NotPdf, "\"not_pdf\""),
            (Bypass::Oversize, "\"oversize\""),
            (Bypass::Image, "\"image\""),
            (Bypass::ClassifierFailed, "\"classifier_unavailable\""),
        ] {
            assert_eq!(serde_json::to_string(&bypass).unwrap(), wire);
            assert_eq!(bypass.as_str(), wire.trim_matches('"'));
        }
    }

    #[test]
    fn decide_bypasses_non_pdf_bytes() {
        let cfg = cfg_with_premium();
        match decide(b"just some text", &cfg, None) {
            Decision::Bypass { reason, model, .. } => {
                assert_eq!(reason, Bypass::NotPdf);
                assert_eq!(model, STANDARD);
            }
            other => panic!("expected bypass, got {other:?}"),
        }
        match decide(b"just some text", &cfg, Some("fallback-model")) {
            Decision::Bypass { model, .. } => assert_eq!(model, "fallback-model"),
            other => panic!("expected bypass, got {other:?}"),
        }
    }

    #[test]
    fn decide_bypasses_oversize_documents() {
        let cfg = Config {
            max_document_bytes: 4,
            ..cfg_with_premium()
        };
        match decide(b"%PDF-1.7 and then some", &cfg, None) {
            Decision::Bypass { reason, detail, .. } => {
                assert_eq!(reason, Bypass::Oversize);
                assert!(detail.expect("detail").contains("max_document_bytes"));
            }
            other => panic!("expected bypass, got {other:?}"),
        }
    }

    #[test]
    fn decide_bypasses_unreadable_pdfs() {
        match decide(b"%PDF-1.7\nnot actually a pdf", &cfg_with_premium(), None) {
            Decision::Bypass { reason, detail, .. } => {
                assert_eq!(reason, Bypass::ClassifierFailed);
                assert!(detail.is_some());
            }
            other => panic!("expected bypass, got {other:?}"),
        }
    }

    #[test]
    fn bypass_image_is_available_to_hosts() {
        let cfg = cfg_with_premium();
        match Decision::bypass_image(&cfg, Some("vision-model")) {
            Decision::Bypass { reason, model, .. } => {
                assert_eq!(reason, Bypass::Image);
                assert_eq!(model, "vision-model");
            }
            other => panic!("expected bypass, got {other:?}"),
        }
    }
}
