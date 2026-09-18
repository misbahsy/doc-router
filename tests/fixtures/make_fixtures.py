#!/usr/bin/env python3
"""Generate the PDF fixtures used by the doc-router parity tests.

Run with the pinned interpreter (pypdf must be importable):

    python3 tests/fixtures/make_fixtures.py

Existing files are only rewritten when the bytes are identical; anything that would
change on disk is reported and KEPT unless you pass `--force` (the goldens in
`tests/golden/` pin each fixture's sha256, so overwriting means re-running
`tests/golden/generate.py`). The three original fixtures -- text.pdf, scanned.pdf and
mixed.pdf -- do regenerate byte-for-byte with pypdf 6.x; the `SPECS` entries below
record their exact page headings so that stays true.

Every page is built by hand so no fonts are embedded and output is deterministic for
a given pypdf version:

* a "text" page carries a Helvetica (base-14, non-embedded) content stream, so
  pdf-inspector finds a usable text layer;
* an "image" page carries a single full-bleed DeviceGray image XObject and no text
  operators at all, so pdf-inspector flags it as needing OCR.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
from pathlib import Path

from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, NumberObject, StreamObject

FIXTURE_DIR = Path(__file__).resolve().parent

PAGE_WIDTH = 612
PAGE_HEIGHT = 792


def _helvetica_resources(writer: PdfWriter) -> DictionaryObject:
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    return DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})}
    )


def _text_block(lines: list[str], x: int = 72, y: int = 720, size: int = 12, leading: int = 14) -> str:
    body = " ".join(f"({line}) Tj T*" for line in lines)
    return f"BT /F1 {size} Tf {x} {y} Td {leading} TL {body} ET"


def text_page(writer: PdfWriter, text: str) -> None:
    """A single-column text page. `text` is newline separated."""
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _helvetica_resources(writer)
    stream = StreamObject()
    stream._data = _text_block(text.split("\n")).encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)


def two_column_page(writer: PdfWriter, left: list[str], right: list[str]) -> None:
    """Two text blocks side by side -- probes pdf-inspector's column detection."""
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _helvetica_resources(writer)
    body = (
        _text_block(left, x=60, y=730, size=10, leading=13)
        + " "
        + _text_block(right, x=330, y=730, size=10, leading=13)
    )
    stream = StreamObject()
    stream._data = body.encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)


def image_page(writer: PdfWriter) -> None:
    """A page whose only content is one full-bleed raster image (no text operators)."""
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    img = StreamObject()
    img._data = b"\x80" * (50 * 50)
    img.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(50),
            NameObject("/Height"): NumberObject(50),
            NameObject("/ColorSpace"): NameObject("/DeviceGray"),
            NameObject("/BitsPerComponent"): NumberObject(8),
        }
    )
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/XObject"): DictionaryObject({NameObject("/Im1"): writer._add_object(img)})}
    )
    cs = StreamObject()
    cs._data = b"q 612 0 0 792 0 0 cm /Im1 Do Q"
    page[NameObject("/Contents")] = writer._add_object(cs)


def blank_page(writer: PdfWriter) -> None:
    """A page with no /Contents at all."""
    writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)


#: 29 body lines. The count and wording are load-bearing: text.pdf / scanned.pdf /
#: mixed.pdf must regenerate byte-for-byte against the checked-in files.
PARAGRAPH = "\n".join(
    f"Line {i}: The quick brown fox jumps over the lazy dog near the river bank at dawn."
    for i in range(1, 30)
)


def build(spec: list[str | None]) -> bytes:
    """Build a PDF from a page spec: a str is a text page heading, None is an image page."""
    w = PdfWriter()
    for heading in spec:
        if heading is None:
            image_page(w)
        else:
            text_page(w, heading + "\n" + PARAGRAPH)
    return _render(w)


def repeating(kinds: str, heading: str) -> list[str | None]:
    """Page spec from a kind string: 't' = text page, 'i' = image page."""
    return [None if k == "i" else f"{heading} {i + 1}" for i, k in enumerate(kinds)]


def _dense_lines(page_no: int, count: int) -> list[str]:
    """A heading plus `count` long body lines -- 40+ lines/page probes layout analysis."""
    return [f"DENSE PAGE {page_no} HEADING"] + [
        f"Paragraph line {i} of page {page_no}: the quick brown fox jumps over the lazy dog "
        f"while the sleepy cat watches from the warm windowsill nearby."
        for i in range(1, count + 1)
    ]


def build_text_dense() -> bytes:
    """10 dense single-column text pages plus one two-column page."""
    w = PdfWriter()
    for page_no in range(1, 11):
        text_page(w, "\n".join(_dense_lines(page_no, 44)))
    left = [f"L{i}: column one carries the narrative body text here." for i in range(1, 45)]
    right = [f"R{i}: column two carries the sidebar commentary text." for i in range(1, 45)]
    two_column_page(w, left, right)
    return _render(w)


def build_empty() -> tuple[str, bytes]:
    """A 0-page PDF if pypdf will write one, else a 1-page blank named blank.pdf."""
    w = PdfWriter()
    try:
        data = _render(w)
    except Exception:  # pragma: no cover - depends on pypdf version
        data = None
    if data:
        return "empty.pdf", data
    w = PdfWriter()
    blank_page(w)
    return "blank.pdf", _render(w)


def _render(writer: PdfWriter) -> bytes:
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


#: The page layout of every generated fixture, as a spec for `build`.
SPECS: dict[str, list[str | None]] = {
    # --- the three originals; these strings reproduce the checked-in bytes exactly ---
    "text.pdf": ["TEXT ONLY DOCUMENT", "PAGE TWO OF TEXT"],
    "scanned.pdf": [None, None],
    "mixed.pdf": ["MIXED DOC PAGE ONE (text)", None, "MIXED DOC PAGE THREE (text)", None],
    # --- new fixtures ---
    # 24 pages, text/text/image repeating. NOTE: 16/24 = 0.667 of the pages carry text,
    # which clears pdf-inspector's default text_page_ratio_threshold of 0.6 -- at which
    # point it labels the document text_based and DISCARDS pages_needing_ocr, hiding the
    # 8 image pages. Kept precisely to pin that: the Rust classifier disables the
    # threshold and derives the label from the page list, so it reports mixed with
    # pages 2, 5, ... 23. See classify::label_from_pages and tests/golden/mixed_long.json.
    "mixed_long.pdf": repeating("tti" * 8, "MIXED LONG PAGE"),
    # Same 24 pages / same 1:2 image ratio, rotated so the image leads each group. Under
    # the old Sample(8) scan strategy this one landed on a different set of pages and was
    # reported mixed, which is how the discrepancy above was visible at all. Both are
    # mixed now; this one exercises mixed_split with a leading OCR page.
    "mixed_long_split.pdf": repeating("itt" * 8, "MIXED SPLIT PAGE"),
    "scanned_long.pdf": [None] * 30,
    "single_text.pdf": ["SINGLE TEXT PAGE"],
    "single_image.pdf": [None],
}

#: name -> zero-arg builder.
BUILDERS: dict[str, "callable"] = {name: (lambda s=spec: build(s)) for name, spec in SPECS.items()}
#: 10 dense single-column text pages plus one two-column page.
BUILDERS["text_dense.pdf"] = build_text_dense

#: Fixtures already checked in whose bytes the goldens pin; never rewritten without --force.
PROTECTED = ("text.pdf", "scanned.pdf", "mixed.pdf")

NOT_A_PDF = b"hello\n" * 60 + b"this file is deliberately not a PDF: it has no %PDF- header.\n"


def _write(path: Path, data: bytes, force: bool) -> str:
    digest = hashlib.sha256(data).hexdigest()
    if path.exists():
        existing = path.read_bytes()
        if existing == data:
            return f"unchanged  {path.name}  {len(data)}B  {digest[:16]}"
        if not force:
            got = hashlib.sha256(existing).hexdigest()
            return (
                f"KEPT       {path.name}  on-disk {len(existing)}B {got[:16]} != "
                f"generated {len(data)}B {digest[:16]} (use --force to overwrite)"
            )
    path.write_bytes(data)
    return f"wrote      {path.name}  {len(data)}B  {digest[:16]}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite fixtures that already exist")
    ap.add_argument("names", nargs="*", help="only build these fixture file names")
    args = ap.parse_args()

    wanted = set(args.names) if args.names else None
    for name, builder in BUILDERS.items():
        if wanted and name not in wanted:
            continue
        print(_write(FIXTURE_DIR / name, builder(), args.force))

    if not wanted or {"empty.pdf", "blank.pdf"} & wanted:
        name, data = build_empty()
        print(_write(FIXTURE_DIR / name, data, args.force))

    if not wanted or "not_a_pdf.bin" in wanted:
        print(_write(FIXTURE_DIR / "not_a_pdf.bin", NOT_A_PDF, args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
