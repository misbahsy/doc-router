//! Page subsetting, for OCR providers that cannot take a page list.

use std::collections::HashMap;

use lopdf::{Dictionary, Document, Object, ObjectId};

use crate::error::Error;
use crate::extract::OcrResult;

/// Page attributes a page may inherit from its ancestors in the page tree.
const INHERITABLE: [&[u8]; 4] = [b"MediaBox", b"Resources", b"CropBox", b"Rotate"];

/// Keys dropped when a page is copied into the subset document.
///
/// `Parent` is replaced by the new page tree. `Annots` is dropped because annotations
/// routinely reference pages, destinations and structure elements outside the subset,
/// which would drag most of the source document along; OCR providers read page content,
/// not annotations.
const DROPPED: [&[u8]; 2] = [b"Parent", b"Annots"];

/// Deep-copy an object from `src` into `dst`, translating object ids.
///
/// `memo` maps source ids to destination ids. A new id is recorded in `memo` *before*
/// its target is copied, so reference cycles terminate. References that do not resolve
/// in the source document become `Null` rather than failing the whole split.
fn copy_object(
    src: &Document,
    dst: &mut Document,
    memo: &mut HashMap<ObjectId, ObjectId>,
    object: &Object,
    depth: u32,
) -> Object {
    if depth > 64 {
        return Object::Null;
    }
    match object {
        Object::Reference(id) => {
            if let Some(new_id) = memo.get(id) {
                return Object::Reference(*new_id);
            }
            let new_id = dst.new_object_id();
            memo.insert(*id, new_id);
            let copied = match src.get_object(*id) {
                Ok(target) => copy_object(src, dst, memo, target, depth + 1),
                Err(_) => Object::Null,
            };
            dst.set_object(new_id, copied);
            Object::Reference(new_id)
        }
        Object::Array(items) => Object::Array(
            items
                .iter()
                .map(|item| copy_object(src, dst, memo, item, depth + 1))
                .collect(),
        ),
        Object::Dictionary(dict) => {
            Object::Dictionary(copy_dictionary(src, dst, memo, dict, depth))
        }
        Object::Stream(stream) => {
            let mut copied = stream.clone();
            copied.dict = copy_dictionary(src, dst, memo, &stream.dict, depth);
            Object::Stream(copied)
        }
        other => other.clone(),
    }
}

fn copy_dictionary(
    src: &Document,
    dst: &mut Document,
    memo: &mut HashMap<ObjectId, ObjectId>,
    dict: &Dictionary,
    depth: u32,
) -> Dictionary {
    let mut out = Dictionary::new();
    for (key, value) in dict.iter() {
        out.set(key.clone(), copy_object(src, dst, memo, value, depth + 1));
    }
    out
}

/// Find an inheritable attribute on a page or, failing that, on its ancestors.
fn inherited<'a>(src: &'a Document, page_id: ObjectId, key: &[u8]) -> Option<&'a Object> {
    let mut current = page_id;
    for _ in 0..64 {
        let dict = src.get_dictionary(current).ok()?;
        if let Ok(value) = dict.get(key) {
            return Some(value);
        }
        current = dict.get(b"Parent").ok()?.as_reference().ok()?;
    }
    None
}

/// Build a PDF containing exactly `pages` (0-indexed) in the order given.
///
/// Per-page resources, fonts, images and content streams are carried over, along with
/// any `MediaBox`/`Resources`/`CropBox`/`Rotate` the page inherited from an ancestor.
pub fn split_pdf(bytes: &[u8], pages: &[u32]) -> Result<Vec<u8>, Error> {
    if !crate::classify::is_pdf(bytes) {
        return Err(Error::NotPdf);
    }
    if pages.is_empty() {
        return Err(Error::Split("no pages requested".to_string()));
    }
    let src = Document::load_mem(bytes).map_err(|e| Error::Split(e.to_string()))?;
    let source_pages = src.get_pages();

    let mut dst = Document::with_version(src.version.clone());
    let pages_id = dst.new_object_id();
    let mut memo: HashMap<ObjectId, ObjectId> = HashMap::new();
    let mut kids: Vec<Object> = Vec::with_capacity(pages.len());

    for &page in pages {
        // get_pages() is 1-indexed; the public API is 0-indexed everywhere.
        let one_indexed = page
            .checked_add(1)
            .ok_or_else(|| Error::Split(format!("page index {page} overflows")))?;
        let page_id = *source_pages.get(&one_indexed).ok_or_else(|| {
            Error::Split(format!(
                "page {page} is out of range for a {}-page document",
                source_pages.len()
            ))
        })?;
        let page_dict = src
            .get_dictionary(page_id)
            .map_err(|e| Error::Split(format!("page {page}: {e}")))?;

        let mut copied = Dictionary::new();
        for (key, value) in page_dict.iter() {
            if DROPPED.contains(&key.as_slice()) {
                continue;
            }
            copied.set(
                key.clone(),
                copy_object(&src, &mut dst, &mut memo, value, 0),
            );
        }
        for key in INHERITABLE {
            if copied.has(key) {
                continue;
            }
            if let Some(value) = inherited(&src, page_id, key) {
                let value = value.clone();
                copied.set(
                    key.to_vec(),
                    copy_object(&src, &mut dst, &mut memo, &value, 0),
                );
            }
        }
        copied.set("Type", Object::Name(b"Page".to_vec()));
        copied.set("Parent", Object::Reference(pages_id));
        kids.push(Object::Reference(dst.add_object(copied)));
    }

    let mut pages_dict = Dictionary::new();
    pages_dict.set("Type", Object::Name(b"Pages".to_vec()));
    pages_dict.set("Count", Object::Integer(kids.len() as i64));
    pages_dict.set("Kids", Object::Array(kids));
    dst.set_object(pages_id, Object::Dictionary(pages_dict));

    let mut catalog = Dictionary::new();
    catalog.set("Type", Object::Name(b"Catalog".to_vec()));
    catalog.set("Pages", Object::Reference(pages_id));
    let catalog_id = dst.add_object(catalog);
    dst.trailer.set("Root", Object::Reference(catalog_id));

    dst.prune_objects();
    dst.renumber_objects();
    dst.compress();

    let mut out = Vec::new();
    dst.save_to(&mut out)
        .map_err(|e| Error::Split(e.to_string()))?;
    Ok(out)
}

/// Map a split document's 0..n page numbering back onto the original page indices.
///
/// Page `i` of the split document becomes `pages[i]`. Indices past the end of `pages`
/// are left alone rather than guessed at.
pub fn remap_pages(result: &mut OcrResult, pages: &[u32]) {
    for page in &mut result.pages {
        if let Some(original) = pages.get(page.index as usize) {
            page.index = *original;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::Page;

    fn result(indices: &[u32]) -> OcrResult {
        OcrResult {
            pages: indices
                .iter()
                .map(|index| Page::new(*index, "", "mistral-ocr"))
                .collect(),
            model: "mistral-ocr".to_string(),
            pages_processed: indices.len() as u32,
            doc_size_bytes: None,
        }
    }

    #[test]
    fn remap_maps_split_positions_back_to_originals() {
        let mut ocr = result(&[0, 1]);
        remap_pages(&mut ocr, &[1, 3]);
        assert_eq!(
            ocr.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn remap_preserves_provider_order() {
        let mut ocr = result(&[1, 0]);
        remap_pages(&mut ocr, &[4, 9]);
        assert_eq!(
            ocr.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
            vec![9, 4]
        );
    }

    #[test]
    fn remap_leaves_unknown_indices_alone() {
        let mut ocr = result(&[5]);
        remap_pages(&mut ocr, &[1, 3]);
        assert_eq!(ocr.pages[0].index, 5);

        let mut ocr = result(&[0, 1]);
        remap_pages(&mut ocr, &[]);
        assert_eq!(
            ocr.pages.iter().map(|p| p.index).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn split_rejects_non_pdfs_and_empty_page_lists() {
        assert!(matches!(split_pdf(b"nope", &[0]), Err(Error::NotPdf)));
        assert!(matches!(split_pdf(b"%PDF-1.7", &[]), Err(Error::Split(_))));
    }
}
