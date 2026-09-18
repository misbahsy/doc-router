//! Classification and planning against the real fixture PDFs.

mod common;

use common::fixture;
use doc_router::{
    classify, extract_local, plan_route, split_pdf, Classification, Config, Leg, PdfType, Reason,
    Tier, Tiers,
};

const LOCAL: &str = "local_pdf/extract";
const STANDARD: &str = "mistral-ocr";
const PREMIUM: &str = "gpt-5-ocr";

fn config() -> Config {
    Config::new(Tiers::new(LOCAL, STANDARD).with_premium(PREMIUM))
}

/// Print what pdf-inspector actually reported, so the numbers behind a failure are visible.
fn show(name: &str, c: &Classification) {
    println!(
        "{name}: pdf_type={} confidence={} page_count={} pages_needing_ocr={:?} is_complex_layout={}",
        c.pdf_type, c.confidence, c.page_count, c.pages_needing_ocr, c.is_complex_layout
    );
}

#[test]
fn text_pdf_stays_local() {
    let c = classify(&fixture("text.pdf")).expect("classify text.pdf");
    show("text.pdf", &c);
    assert_eq!(c.pdf_type, PdfType::TextBased);
    assert_eq!(c.page_count, 2);
    assert_eq!(c.pages_needing_ocr, Vec::<u32>::new());
    assert!(!c.needs_ocr_everywhere());
    assert_eq!(c.text_layer_pages(), vec![0, 1]);

    let plan = plan_route(&c, &config()).expect("plan");
    if c.confidence < config().min_confidence {
        assert_eq!(plan.reason, Reason::LowConfidence);
    } else {
        assert_eq!(plan.reason, Reason::TextLayer);
        assert_eq!(plan.tier, Tier::Local);
        assert_eq!(plan.legs, vec![Leg::whole(LOCAL, Tier::Local)]);
    }
}

#[test]
fn scanned_pdf_goes_to_ocr_whole() {
    let c = classify(&fixture("scanned.pdf")).expect("classify scanned.pdf");
    show("scanned.pdf", &c);
    assert_eq!(c.pdf_type, PdfType::Scanned);
    assert_eq!(c.page_count, 2);
    assert_eq!(c.pages_needing_ocr, vec![0, 1]);
    assert!(c.needs_ocr_everywhere());

    let plan = plan_route(&c, &config()).expect("plan");
    if c.confidence < config().min_confidence {
        assert_eq!(plan.reason, Reason::LowConfidence);
    } else {
        assert_eq!(plan.reason, Reason::Scanned);
        assert_eq!(plan.tier, Tier::Standard);
        assert_eq!(plan.legs, vec![Leg::whole(STANDARD, Tier::Standard)]);
    }
}

#[test]
fn mixed_pdf_splits_by_page() {
    let c = classify(&fixture("mixed.pdf")).expect("classify mixed.pdf");
    show("mixed.pdf", &c);
    assert_eq!(c.pdf_type, PdfType::Mixed);
    assert_eq!(c.page_count, 4);
    assert_eq!(c.pages_needing_ocr, vec![1, 3]);
    assert_eq!(c.text_layer_pages(), vec![0, 2]);

    let plan = plan_route(&c, &config()).expect("plan");
    if c.confidence < config().min_confidence {
        assert_eq!(plan.reason, Reason::LowConfidence);
    } else {
        assert_eq!(plan.reason, Reason::MixedSplit);
        assert!(plan.is_split());
        assert_eq!(
            plan.legs,
            vec![
                Leg::subset(LOCAL, Tier::Local, vec![0, 2]),
                Leg::subset(STANDARD, Tier::Standard, vec![1, 3]),
            ]
        );
    }
}

#[test]
fn local_extraction_returns_original_page_indices() {
    let bytes = fixture("mixed.pdf");
    let all = extract_local(&bytes, None).expect("extract all");
    assert_eq!(all.pages.len(), 4);
    assert_eq!(
        all.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert!(all.pages.iter().all(|p| p.model == LOCAL));
    assert_eq!(all.model, LOCAL);
    assert_eq!(all.pages_processed, 4);
    assert_eq!(all.doc_size_bytes, Some(bytes.len() as u64));

    let subset = extract_local(&bytes, Some(&[0, 2])).expect("extract subset");
    assert_eq!(
        subset.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
        vec![0, 2],
        "PageMarkdown.page must be the ORIGINAL 0-indexed page"
    );
    assert_eq!(subset.pages[0].markdown, all.pages[0].markdown);
    assert_eq!(subset.pages[1].markdown, all.pages[2].markdown);
}

#[test]
fn split_of_the_ocr_pages_round_trips_through_classification() {
    let split = split_pdf(&fixture("mixed.pdf"), &[1, 3]).expect("split");
    let c = classify(&split).expect("classify the split");
    show("mixed.pdf split [1, 3]", &c);
    assert_eq!(c.page_count, 2);
    assert_eq!(
        c.pages_needing_ocr,
        vec![0, 1],
        "both extracted pages still need OCR"
    );
}

#[test]
fn split_of_a_text_page_extracts_the_same_markdown() {
    let bytes = fixture("text.pdf");
    let original = extract_local(&bytes, Some(&[0])).expect("extract page 0");

    let split = split_pdf(&bytes, &[0]).expect("split");
    let extracted = extract_local(&split, None).expect("extract the split");
    assert_eq!(extracted.pages.len(), 1);
    assert_eq!(extracted.pages[0].index, 0);
    assert_eq!(extracted.pages[0].markdown, original.pages[0].markdown);
}

#[test]
fn split_preserves_the_requested_order() {
    let bytes = fixture("text.pdf");
    let whole = extract_local(&bytes, None).expect("extract");

    let reversed = split_pdf(&bytes, &[1, 0]).expect("split");
    let extracted = extract_local(&reversed, None).expect("extract the split");
    assert_eq!(extracted.pages.len(), 2);
    assert_eq!(extracted.pages[0].markdown, whole.pages[1].markdown);
    assert_eq!(extracted.pages[1].markdown, whole.pages[0].markdown);
}

#[test]
fn split_rejects_pages_past_the_end() {
    let err = split_pdf(&fixture("text.pdf"), &[0, 7]).expect_err("out of range");
    assert!(err.to_string().contains("out of range"), "{err}");
}
