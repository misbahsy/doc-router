# tests/fixtures

Synthetic PDFs used by the Rust parity tests in `crates/doc-router` and by the golden
generator in `tests/golden/generate.py`. Every file is hand-built by
[`make_fixtures.py`](make_fixtures.py) so the bytes are deterministic and no real
document or font is embedded.

Two page kinds are used:

* **text page** — a Helvetica (base-14, not embedded) content stream. pdf-inspector
  finds a usable text layer, so the page does *not* need OCR.
* **image page** — one full-bleed 50x50 `DeviceGray` image XObject and no text
  operators at all. pdf-inspector flags it as needing OCR.

## Regenerating

```sh
python3 tests/fixtures/make_fixtures.py
```

Only `pypdf` is needed. The script **never overwrites a file whose bytes would change**
— it prints `KEPT` and moves on — because `tests/golden/*.json` pins each fixture's
sha256. Pass `--force` to overwrite anyway, and then regenerate the goldens:

```sh
PYTHONPATH=/path/to/litellm \
  python3 tests/golden/generate.py
```

A run with no changes prints `unchanged` for every fixture. All eleven files below,
including the three originals (`text`, `scanned`, `mixed`), reproduce byte-for-byte with
pypdf 6.x; the exact page headings live in `SPECS` in `make_fixtures.py` and are
load-bearing for that.

## The fixtures

`classification` below is what the **Python reference** (`classify_pdf` ->
`pdf_inspector.detect_pdf_bytes`) reports; `pages_needing_ocr` is 0-indexed after the
reference's 1-indexed -> 0-indexed shift.

| file | pages | layout | pdf_type | conf | pages_needing_ocr | exercises |
|---|---|---|---|---|---|---|
| `text.pdf` | 2 | text, text | `text_based` | 1.00 | — | `text_layer`, single local leg |
| `scanned.pdf` | 2 | image x2 | `scanned` | 0.95 | 0,1 | `scanned`, single OCR leg |
| `mixed.pdf` | 4 | t,i,t,i | `mixed` | 0.70 | 1,3 | `mixed_split` / `mixed_unsplit` |
| `mixed_long.pdf` | 24 | (t,t,i) x8 | `text_based` | 0.875 | — | **detector quirk**, see below |
| `mixed_long_split.pdf` | 24 | (i,t,t) x8 | `mixed` | 0.70 | 0,3,…,21 | `mixed_split` with 8 OCR pages |
| `scanned_long.pdf` | 30 | image x30 | `scanned` | 0.95 | 0..29 | `scanned` at length; big page lists |
| `text_dense.pdf` | 11 | 10 dense + 1 two-column | `text_based` | 1.00 | — | layout analysis, see below |
| `single_text.pdf` | 1 | text | `text_based` | 1.00 | — | 1-page boundary, `text_layer` |
| `single_image.pdf` | 1 | image | `scanned` | 0.95 | 0 | 1-page boundary, `scanned` |
| `empty.pdf` | 0 | — | `scanned` | 0.90 | — | 0-page edge case |
| `not_a_pdf.bin` | — | plain text, no `%PDF` header | — | — | — | `Bypass::NotPdf` |

`empty.pdf` really is a 0-page PDF (pypdf writes one happily), so the `blank.pdf`
fallback in `make_fixtures.py` is currently unused. With `page_count == 0` the policy
falls through to a single **local** leg with reason `text_layer` — matching the
reference, which only reaches the `scanned` branch when `len(ocr_pages) >= page_count`
*and* `ocr_pages` is non-empty.

## Two things the Rust port must reproduce, not "fix"

**1. `mixed_long.pdf` is classified `text_based` with an empty `pages_needing_ocr`.**
Every third page is a pure image, and `extract_pages_markdown_bytes` correctly reports
`needs_ocr=True` for pages 2, 5, 8, … — but pdf-inspector's *detect* pass returns no OCR
pages at all, so the router plans a single local leg. The heuristic is position
sensitive, not a simple ratio: `(i,t,t) x8` (`mixed_long_split.pdf`, same 1:2 image
ratio) *is* reported as `mixed`. Both fixtures are kept so the Rust port pins the same
behaviour on both sides of that line.

**2. `is_complex_layout` is `false` for every fixture, `text_dense.pdf` included.**
`detect_pdf_bytes` does not run layout analysis: `PdfResult.is_complex_layout`,
`pages_with_tables` and `pages_with_columns` all come back empty in detect-only mode.
The two-column page in `text_dense.pdf` *is* detected — but only by
`extract_pages_markdown_bytes`, which returns `is_complex=True` and
`pages_with_columns=[11]` (1-indexed). The reference classifier never calls that, so no
golden has `is_complex_layout: true`, and the `complex_layout_tier` config knob (and
therefore the `premium` tier) is never reached by any golden plan. Those paths need
unit tests against a synthetic `Classification`, not a fixture.
