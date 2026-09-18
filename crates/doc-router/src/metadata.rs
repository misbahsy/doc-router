//! The `doc_route` provenance blob stamped on a routed request.

use serde::{Deserialize, Serialize};

use crate::config::Tier;
use crate::policy::{Decision, Leg, Plan};

/// The tier string used for bypassed documents.
pub const BYPASS_TIER: &str = "bypass";

/// One leg as it appears in metadata.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LegSummary {
    /// The leg's model.
    pub model: String,
    /// The leg's tier.
    pub tier: Tier,
    /// The leg's 0-indexed pages, or `None` for the whole document.
    pub pages: Option<Vec<u32>>,
}

impl From<&Leg> for LegSummary {
    fn from(leg: &Leg) -> Self {
        LegSummary {
            model: leg.model.clone(),
            tier: leg.tier,
            pages: leg.pages.clone(),
        }
    }
}

/// Provenance for one routing decision. Serialises to the same keys and values as the
/// Python `doc_route` metadata dict.
///
/// Fields are declared in the Python dict's insertion order (tier, reason, pdf_type,
/// confidence, page_count, pages_needing_ocr, classify_ms, routed_model, legs, split) so
/// serialised key order matches too; SPEC.md lists the same fields in a different order,
/// which is cosmetic. `None` fields are omitted, so a bypass serialises to exactly the
/// Python bypass dict `{tier, reason, routed_model}` and `fallback_reason` only appears
/// when the fallback path ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RouteMetadata {
    /// The plan's tier, or `"bypass"`.
    pub tier: String,
    /// The plan reason, or the bypass reason.
    pub reason: String,
    /// The detected document type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pdf_type: Option<String>,
    /// Classification confidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// Page count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_count: Option<u32>,
    /// 0-indexed pages routed to OCR.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pages_needing_ocr: Option<Vec<u32>>,
    /// Time spent classifying, in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classify_ms: Option<f64>,
    /// The model the request is attributed to.
    pub routed_model: String,
    /// The legs that were planned.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legs: Option<Vec<LegSummary>>,
    /// Whether the document was split across legs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<bool>,
    /// `"leg_failed"` when the whole-document fallback ran.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fallback_reason: Option<String>,
}

/// The `fallback_reason` recorded when a leg failed and the whole document was rerun.
pub const FALLBACK_LEG_FAILED: &str = "leg_failed";

impl Decision {
    /// The `doc_route` metadata for this decision.
    pub fn metadata(&self) -> RouteMetadata {
        match self {
            Decision::Bypass { reason, model, .. } => RouteMetadata {
                tier: BYPASS_TIER.to_string(),
                reason: reason.as_str().to_string(),
                pdf_type: None,
                confidence: None,
                page_count: None,
                pages_needing_ocr: None,
                classify_ms: None,
                routed_model: model.clone(),
                legs: None,
                split: None,
                fallback_reason: None,
            },
            Decision::Route {
                plan,
                classification,
            } => {
                let Plan {
                    legs,
                    reason,
                    tier,
                    routed_model,
                } = plan;
                RouteMetadata {
                    tier: tier.as_str().to_string(),
                    reason: reason.as_str().to_string(),
                    pdf_type: Some(classification.pdf_type.as_str().to_string()),
                    confidence: Some(classification.confidence),
                    page_count: Some(classification.page_count),
                    pages_needing_ocr: Some(classification.pages_needing_ocr.clone()),
                    classify_ms: Some(classification.classify_ms),
                    routed_model: routed_model.clone(),
                    legs: Some(legs.iter().map(LegSummary::from).collect()),
                    split: Some(plan.is_split()),
                    fallback_reason: None,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::{Classification, PdfType};
    use crate::config::{Config, Tiers};
    use crate::policy::{plan_route, Bypass, Decision};

    fn classification() -> Classification {
        Classification {
            pdf_type: PdfType::Mixed,
            confidence: 0.7,
            page_count: 4,
            pages_needing_ocr: vec![1, 3],
            is_complex_layout: false,
            has_encoding_issues: false,
            ocr_reasons: Vec::new(),
            classify_ms: 2.5,
        }
    }

    fn route() -> Decision {
        let cfg = Config::new(Tiers::new("local_pdf/extract", "mistral-ocr"));
        let classification = classification();
        let plan = plan_route(&classification, &cfg).expect("plan");
        Decision::Route {
            plan,
            classification,
        }
    }

    #[test]
    fn route_metadata_has_the_python_keys_in_order() {
        let metadata = route().metadata();
        let value = serde_json::to_value(&metadata).expect("serialize");
        let object = value.as_object().expect("object");
        // Exactly the keys the Python `doc_route` dict writes -- no more, no fewer.
        // (serde_json::Value sorts keys, so the set is compared here and the emitted
        // order is checked against the raw JSON below.)
        let python_keys = [
            "tier",
            "reason",
            "pdf_type",
            "confidence",
            "page_count",
            "pages_needing_ocr",
            "classify_ms",
            "routed_model",
            "legs",
            "split",
        ];
        let mut sorted = python_keys;
        sorted.sort_unstable();
        assert_eq!(
            object.keys().map(String::as_str).collect::<Vec<_>>(),
            sorted
        );

        // ... and they are emitted in the Python dict's insertion order.
        let json = serde_json::to_string(&metadata).expect("serialize");
        let mut previous = 0usize;
        for key in python_keys {
            let at = json
                .find(&format!("\"{key}\":"))
                .unwrap_or_else(|| panic!("{key} missing from {json}"));
            assert!(at >= previous, "{key} is out of order in {json}");
            previous = at;
        }
        assert_eq!(object["tier"], "standard");
        assert_eq!(object["reason"], "mixed_split");
        assert_eq!(object["pdf_type"], "mixed");
        assert_eq!(object["page_count"], 4);
        assert_eq!(object["pages_needing_ocr"], serde_json::json!([1, 3]));
        assert_eq!(object["routed_model"], "mistral-ocr");
        assert_eq!(object["split"], true);
        assert_eq!(
            object["legs"],
            serde_json::json!([
                {"model": "local_pdf/extract", "tier": "local", "pages": [0, 2]},
                {"model": "mistral-ocr", "tier": "standard", "pages": [1, 3]},
            ])
        );
    }

    #[test]
    fn single_leg_metadata_reports_pages_null_and_split_false() {
        let cfg = Config::new(Tiers::new("local_pdf/extract", "mistral-ocr"));
        let classification = Classification {
            pages_needing_ocr: vec![],
            ..classification()
        };
        let plan = plan_route(&classification, &cfg).expect("plan");
        let metadata = Decision::Route {
            plan,
            classification,
        }
        .metadata();
        let value = serde_json::to_value(&metadata).expect("serialize");
        assert_eq!(value["split"], false);
        assert_eq!(value["tier"], "local");
        assert_eq!(value["legs"][0]["pages"], serde_json::Value::Null);
    }

    #[test]
    fn bypass_metadata_is_exactly_the_python_bypass_dict() {
        let decision = Decision::Bypass {
            reason: Bypass::NotPdf,
            model: "mistral-ocr".to_string(),
            detail: Some("ignored in metadata".to_string()),
        };
        let metadata = decision.metadata();
        assert_eq!(metadata.tier, BYPASS_TIER);
        let value = serde_json::to_value(&metadata).expect("serialize");
        assert_eq!(
            value,
            serde_json::json!({
                "tier": "bypass",
                "reason": "not_pdf",
                "routed_model": "mistral-ocr",
            })
        );
    }

    #[test]
    fn classifier_failure_uses_the_python_reason_string() {
        let metadata = Decision::Bypass {
            reason: Bypass::ClassifierFailed,
            model: "m".to_string(),
            detail: None,
        }
        .metadata();
        assert_eq!(metadata.reason, "classifier_unavailable");
    }

    #[test]
    fn metadata_round_trips_through_json() {
        let metadata = route().metadata();
        let json = serde_json::to_string(&metadata).expect("serialize");
        let back: RouteMetadata = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, metadata);
    }
}
