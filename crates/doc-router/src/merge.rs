//! Reassembly of per-leg results into one document-ordered result.

use crate::extract::{OcrResult, Page};
use crate::policy::Leg;

/// Merge per-leg results back into a single result in original page order.
///
/// Pages are stably sorted by `index`, so pages with the same index keep leg order.
/// Each page's `model` is overwritten with its leg's model, `model` is the leg models
/// joined by `","` in leg order, `pages_processed` is summed and `doc_size_bytes` is
/// the largest reported by any leg.
pub fn merge(legs: &[(Leg, OcrResult)]) -> OcrResult {
    let mut pages: Vec<Page> = Vec::new();
    let mut models: Vec<&str> = Vec::new();
    let mut pages_processed: u32 = 0;
    let mut doc_size_bytes: Option<u64> = None;

    for (leg, result) in legs {
        models.push(leg.model.as_str());
        pages_processed = pages_processed.saturating_add(result.pages_processed);
        doc_size_bytes = match (doc_size_bytes, result.doc_size_bytes) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        for page in &result.pages {
            pages.push(Page {
                index: page.index,
                markdown: page.markdown.clone(),
                model: leg.model.clone(),
            });
        }
    }

    pages.sort_by_key(|page| page.index);

    OcrResult {
        pages,
        model: models.join(","),
        pages_processed,
        doc_size_bytes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Tier;
    use crate::extract::LOCAL_MODEL;

    fn result(pages: &[(u32, &str)], model: &str, processed: u32, size: Option<u64>) -> OcrResult {
        OcrResult {
            pages: pages
                .iter()
                .map(|(index, md)| Page::new(*index, *md, model))
                .collect(),
            model: model.to_string(),
            pages_processed: processed,
            doc_size_bytes: size,
        }
    }

    #[test]
    fn merge_orders_pages_and_joins_models() {
        let local = Leg::subset(LOCAL_MODEL, Tier::Local, vec![0, 2]);
        let ocr = Leg::subset("mistral-ocr", Tier::Standard, vec![1, 3]);
        let merged = merge(&[
            (
                local,
                result(
                    &[(0, "page zero"), (2, "page two")],
                    LOCAL_MODEL,
                    2,
                    Some(120),
                ),
            ),
            (
                ocr,
                result(
                    &[(1, "page one"), (3, "page three")],
                    "mistral-ocr",
                    2,
                    Some(500),
                ),
            ),
        ]);

        assert_eq!(
            merged.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );
        assert_eq!(
            merged
                .pages
                .iter()
                .map(|p| p.model.as_str())
                .collect::<Vec<_>>(),
            vec![LOCAL_MODEL, "mistral-ocr", LOCAL_MODEL, "mistral-ocr"]
        );
        assert_eq!(merged.pages[3].markdown, "page three");
        assert_eq!(merged.model, "local_pdf/extract,mistral-ocr");
        assert_eq!(merged.pages_processed, 4);
        assert_eq!(merged.doc_size_bytes, Some(500));
    }

    #[test]
    fn merge_overwrites_page_models_with_the_leg_model() {
        let leg = Leg::whole("mistral-ocr", Tier::Standard);
        let merged = merge(&[(
            leg,
            result(&[(0, "x")], "provider-said-something-else", 1, None),
        )]);
        assert_eq!(merged.pages[0].model, "mistral-ocr");
        assert_eq!(merged.model, "mistral-ocr");
        assert_eq!(merged.doc_size_bytes, None);
    }

    #[test]
    fn merge_is_stable_for_duplicate_indices() {
        let first = Leg::subset("a", Tier::Local, vec![0]);
        let second = Leg::subset("b", Tier::Standard, vec![0]);
        let merged = merge(&[
            (first, result(&[(0, "first")], "a", 1, None)),
            (second, result(&[(0, "second")], "b", 1, None)),
        ]);
        assert_eq!(
            merged
                .pages
                .iter()
                .map(|p| p.markdown.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "second"]
        );
    }

    #[test]
    fn merge_of_nothing_is_empty() {
        let merged = merge(&[]);
        assert!(merged.pages.is_empty());
        assert_eq!(merged.model, "");
        assert_eq!(merged.pages_processed, 0);
        assert_eq!(merged.doc_size_bytes, None);
    }

    #[test]
    fn merge_takes_the_max_doc_size_from_either_side() {
        let a = Leg::whole("a", Tier::Local);
        let b = Leg::whole("b", Tier::Standard);
        let merged = merge(&[
            (a, result(&[], "a", 0, None)),
            (b, result(&[], "b", 0, Some(42))),
        ]);
        assert_eq!(merged.doc_size_bytes, Some(42));
    }
}
