//! Operator configuration: the `doc_router_config` block, byte-for-byte the
//! same JSON shape LiteLLM accepts.

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// Default classifier confidence below which a document is not split.
pub const DEFAULT_MIN_CONFIDENCE: f32 = 0.6;
/// Default size ceiling above which a document bypasses classification (50 MiB).
pub const DEFAULT_MAX_DOCUMENT_BYTES: u64 = 50 * 1024 * 1024;
/// Default host-side document fetch timeout, in seconds.
pub const DEFAULT_FETCH_TIMEOUT_SECONDS: f32 = 20.0;

/// Where a page or a document can go.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// The in-process text-layer extractor.
    Local,
    /// The standard OCR model group.
    Standard,
    /// The premium OCR model group.
    Premium,
}

impl Tier {
    /// The wire string for this tier (`"local"`, `"standard"`, `"premium"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::Local => "local",
            Tier::Standard => "standard",
            Tier::Premium => "premium",
        }
    }
}

impl std::fmt::Display for Tier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Tiers an OCR leg may use. `Local` is excluded on purpose: a page that needs
/// OCR is exactly the page the local extractor cannot read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OcrTier {
    /// The standard OCR model group.
    Standard,
    /// The premium OCR model group.
    Premium,
}

impl OcrTier {
    /// The corresponding [`Tier`].
    pub fn tier(self) -> Tier {
        match self {
            OcrTier::Standard => Tier::Standard,
            OcrTier::Premium => Tier::Premium,
        }
    }

    /// The wire string for this tier.
    pub fn as_str(self) -> &'static str {
        self.tier().as_str()
    }
}

impl From<OcrTier> for Tier {
    fn from(value: OcrTier) -> Self {
        value.tier()
    }
}

/// Model groups backing each tier. `local` and `standard` are required.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Tiers {
    /// The in-process extractor deployment, e.g. `local_pdf/extract`.
    pub local: String,
    /// The standard OCR model group.
    pub standard: String,
    /// The premium OCR model group, when one is configured.
    #[serde(default)]
    pub premium: Option<String>,
}

impl Tiers {
    /// Build a tier table without a premium tier.
    pub fn new(local: impl Into<String>, standard: impl Into<String>) -> Self {
        Self {
            local: local.into(),
            standard: standard.into(),
            premium: None,
        }
    }

    /// The same table with a premium tier configured.
    #[must_use]
    pub fn with_premium(mut self, premium: impl Into<String>) -> Self {
        self.premium = Some(premium.into());
        self
    }

    /// The model group configured for `tier`, or `None` when it is not configured.
    pub fn model_for(&self, tier: Tier) -> Option<&str> {
        match tier {
            Tier::Local => Some(self.local.as_str()),
            Tier::Standard => Some(self.standard.as_str()),
            Tier::Premium => self.premium.as_deref(),
        }
    }
}

fn default_min_confidence() -> f32 {
    DEFAULT_MIN_CONFIDENCE
}
fn default_complex_layout_tier() -> OcrTier {
    OcrTier::Premium
}
fn default_max_document_bytes() -> u64 {
    DEFAULT_MAX_DOCUMENT_BYTES
}
fn default_fetch_timeout_seconds() -> f32 {
    DEFAULT_FETCH_TIMEOUT_SECONDS
}
fn default_split_pages() -> bool {
    true
}

/// The `doc_router_config` block of an `auto_router/doc_router` deployment.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// Model groups backing each tier.
    pub tiers: Tiers,
    /// Classifier confidence below which the whole document goes to `standard` untouched.
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
    /// Tier used for the OCR leg when the classifier flags a complex layout.
    #[serde(default = "default_complex_layout_tier")]
    pub complex_layout_tier: OcrTier,
    /// Documents larger than this bypass to the default model instead of being classified.
    #[serde(default = "default_max_document_bytes")]
    pub max_document_bytes: u64,
    /// Timeout for fetching a document from a URL, in seconds. Host-only knob,
    /// kept here so one config block serves both LiteLLM and this library.
    #[serde(default = "default_fetch_timeout_seconds")]
    pub fetch_timeout_seconds: f32,
    /// `false` = never split a mixed document; it goes to the OCR tier whole.
    #[serde(default = "default_split_pages")]
    pub split_pages: bool,
}

impl Config {
    /// A config with every knob at its default, backed by `tiers`.
    pub fn new(tiers: Tiers) -> Self {
        Self {
            tiers,
            min_confidence: DEFAULT_MIN_CONFIDENCE,
            complex_layout_tier: OcrTier::Premium,
            max_document_bytes: DEFAULT_MAX_DOCUMENT_BYTES,
            fetch_timeout_seconds: DEFAULT_FETCH_TIMEOUT_SECONDS,
            split_pages: true,
        }
    }

    /// Parse a `doc_router_config` JSON object, then validate it.
    pub fn from_json(s: &str) -> Result<Config, Error> {
        let config: Config =
            serde_json::from_str(s).map_err(|e| Error::InvalidConfig(e.to_string()))?;
        config.validate()?;
        Ok(config)
    }

    /// Check every range and reject empty model names.
    pub fn validate(&self) -> Result<(), Error> {
        if self.tiers.local.trim().is_empty() {
            return Err(Error::InvalidConfig("tiers.local must not be empty".into()));
        }
        if self.tiers.standard.trim().is_empty() {
            return Err(Error::InvalidConfig(
                "tiers.standard must not be empty".into(),
            ));
        }
        if self
            .tiers
            .premium
            .as_ref()
            .is_some_and(|p| p.trim().is_empty())
        {
            return Err(Error::InvalidConfig(
                "tiers.premium must not be empty when set".into(),
            ));
        }
        if !self.min_confidence.is_finite() || !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(Error::InvalidConfig(format!(
                "min_confidence must be between 0.0 and 1.0, got {}",
                self.min_confidence
            )));
        }
        if self.max_document_bytes == 0 {
            return Err(Error::InvalidConfig(
                "max_document_bytes must be greater than 0".into(),
            ));
        }
        if !self.fetch_timeout_seconds.is_finite() || self.fetch_timeout_seconds <= 0.0 {
            return Err(Error::InvalidConfig(format!(
                "fetch_timeout_seconds must be greater than 0, got {}",
                self.fetch_timeout_seconds
            )));
        }
        Ok(())
    }

    /// `complex_layout_tier`, downgraded to `standard` when premium is not configured.
    pub fn resolved_complex_layout_tier(&self) -> OcrTier {
        if self.complex_layout_tier == OcrTier::Premium && self.tiers.premium.is_none() {
            OcrTier::Standard
        } else {
            self.complex_layout_tier
        }
    }

    /// The model group for `tier`, or `None` when that tier is not configured.
    pub fn model_for(&self, tier: Tier) -> Option<&str> {
        self.tiers.model_for(tier)
    }

    /// The model group for `tier`, as an error when the tier is unconfigured.
    pub fn model_for_tier(&self, tier: Tier) -> Result<&str, Error> {
        self.model_for(tier).ok_or_else(|| {
            Error::InvalidConfig(format!("no model configured for tier {}", tier.as_str()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiers() -> Tiers {
        Tiers::new("local_pdf/extract", "mistral-ocr").with_premium("gpt-5-ocr")
    }

    #[test]
    fn defaults_match_the_reference() {
        let cfg = Config::new(tiers());
        assert_eq!(cfg.min_confidence, 0.6);
        assert_eq!(cfg.complex_layout_tier, OcrTier::Premium);
        assert_eq!(cfg.max_document_bytes, 50 * 1024 * 1024);
        assert_eq!(cfg.fetch_timeout_seconds, 20.0);
        assert!(cfg.split_pages);
    }

    #[test]
    fn minimal_json_fills_in_defaults() {
        let cfg = Config::from_json(
            r#"{"tiers": {"local": "local_pdf/extract", "standard": "mistral-ocr"}}"#,
        )
        .expect("valid config");
        assert_eq!(cfg.tiers.premium, None);
        assert_eq!(cfg.min_confidence, DEFAULT_MIN_CONFIDENCE);
        assert!(cfg.split_pages);
    }

    #[test]
    fn json_round_trips() {
        let cfg = Config {
            split_pages: false,
            min_confidence: 0.75,
            complex_layout_tier: OcrTier::Standard,
            ..Config::new(tiers())
        };
        let json = serde_json::to_string(&cfg).expect("serialize");
        assert_eq!(Config::from_json(&json).expect("reparse"), cfg);
    }

    #[test]
    fn enums_serialise_as_snake_case() {
        assert_eq!(serde_json::to_string(&Tier::Local).unwrap(), "\"local\"");
        assert_eq!(
            serde_json::to_string(&Tier::Standard).unwrap(),
            "\"standard\""
        );
        assert_eq!(
            serde_json::to_string(&Tier::Premium).unwrap(),
            "\"premium\""
        );
        assert_eq!(
            serde_json::to_string(&OcrTier::Premium).unwrap(),
            "\"premium\""
        );
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err =
            Config::from_json(r#"{"tiers": {"local": "l", "standard": "s"}, "split_page": true}"#)
                .expect_err("unknown key");
        assert!(matches!(err, Error::InvalidConfig(_)), "{err}");

        let err = Config::from_json(r#"{"tiers": {"local": "l", "standard": "s", "gold": "g"}}"#)
            .expect_err("unknown tier");
        assert!(matches!(err, Error::InvalidConfig(_)), "{err}");
    }

    #[test]
    fn premium_downgrades_when_unconfigured() {
        let cfg = Config::new(Tiers::new("local_pdf/extract", "mistral-ocr"));
        assert_eq!(cfg.complex_layout_tier, OcrTier::Premium);
        assert_eq!(cfg.resolved_complex_layout_tier(), OcrTier::Standard);
        assert_eq!(cfg.model_for(Tier::Premium), None);

        let cfg = Config::new(tiers());
        assert_eq!(cfg.resolved_complex_layout_tier(), OcrTier::Premium);
        assert_eq!(cfg.model_for(Tier::Premium), Some("gpt-5-ocr"));
    }

    #[test]
    fn model_for_returns_each_tier() {
        let cfg = Config::new(tiers());
        assert_eq!(cfg.model_for(Tier::Local), Some("local_pdf/extract"));
        assert_eq!(cfg.model_for(Tier::Standard), Some("mistral-ocr"));
    }

    #[test]
    fn validation_rejects_bad_values() {
        let bad = [
            r#"{"tiers": {"local": "", "standard": "s"}}"#,
            r#"{"tiers": {"local": "l", "standard": "   "}}"#,
            r#"{"tiers": {"local": "l", "standard": "s", "premium": ""}}"#,
            r#"{"tiers": {"local": "l", "standard": "s"}, "min_confidence": -0.1}"#,
            r#"{"tiers": {"local": "l", "standard": "s"}, "min_confidence": 1.5}"#,
            r#"{"tiers": {"local": "l", "standard": "s"}, "max_document_bytes": 0}"#,
            r#"{"tiers": {"local": "l", "standard": "s"}, "fetch_timeout_seconds": 0}"#,
            r#"{"tiers": {"local": "l", "standard": "s"}, "fetch_timeout_seconds": -1}"#,
        ];
        for json in bad {
            let err = Config::from_json(json).expect_err(json);
            assert!(matches!(err, Error::InvalidConfig(_)), "{json}: {err}");
        }
    }

    #[test]
    fn model_for_tier_errors_when_unconfigured() {
        let cfg = Config::new(Tiers::new("l", "s"));
        assert!(matches!(
            cfg.model_for_tier(Tier::Premium),
            Err(Error::InvalidConfig(_))
        ));
    }
}
