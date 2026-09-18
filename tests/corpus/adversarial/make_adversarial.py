#!/usr/bin/env python3
"""Generate the adversarial corpus: PDFs where structure and truth disagree.

Run with the pinned interpreter (pypdf and Pillow must be importable):

    python3 tests/corpus/adversarial/make_adversarial.py

Every document here is a page whose *structural* signal and whose *real* content
point in opposite directions, so the judges in `crates/doc-router/src/judge.rs`
can be told apart. `tests/fixtures/` cannot do that: a page there either carries
text operators or carries none, which is exactly the signal `HeuristicJudge`
reads, so it scores 1.000 there by construction.

For each document this writes two files side by side:

* `<name>.pdf`       -- the document;
* `<name>.truth.json` -- the exact text a perfect extractor should return for
  every page, 0-indexed. See `README.md` in this directory for the schema.

Existing files are only rewritten when the bytes are identical; anything that
would change on disk is reported and KEPT unless you pass `--force`. Same rule,
and the same reason, as `tests/fixtures/make_fixtures.py`: the manifest's
`page_count` and these files' hashes are the only things standing between the
bench harness and a plausible-looking score against the wrong document.

Output is deterministic for a given pypdf/Pillow version: every random choice
goes through a seeded `random.Random`, and the seed is the page's own identity.

## The six page kinds, and which rule of pdf-inspector's each one hits

pdf-inspector flags a page inside a `Mixed` document when

    (has_template_image && looks_like_scan)                  # scan with no real text
    || has_vector_text                                       # glyphs drawn as paths
    || (has_template_image && text_ops < 10)                 # "sparse_text_over_scan"
    || (text_ops < 3 && has_images)

and, for any document, when a used font is Identity-H without `/ToUnicode` or is
Type3-only. `looks_like_scan` needs `unique_alphanum_chars < 10`, which no page
with real prose in a decodable font can reach. So the only per-page lever a page
with real text has is the **text-operator count**, and the traps below sit on
both sides of the `< 10` floor:

| kind            | text ops | image     | inspector | truth     | heuristic error |
|-----------------|----------|-----------|-----------|-----------|-----------------|
| `clean`         | 30-46    | none      | clear     | no OCR    | -- (correct)    |
| `scan`          | 0        | full page | flagged   | needs OCR | -- (correct)    |
| `bad_ocr`       | 60-130   | full page | clear     | needs OCR | missed OCR      |
| `bad_ocr_under` | 60-130   | full page | clear     | needs OCR | missed OCR      |
| `chrome`        | 10-20    | full page | clear     | needs OCR | missed OCR      |
| `mojibake`      | 30-46    | none      | clear     | needs OCR | missed OCR      |
| `watermark`     | 6-8      | full page | flagged   | no OCR    | wasted OCR      |

The two `bad_ocr` kinds differ only in how the layer is hidden -- invisible
render mode over the scan, or normal render mode under it. Both are what real
engines emit; they part company one layer down, because an extractor that skips
`3 Tr` returns nothing for the first and the garbled transcription for the
second. The corpus carries both so a judge is tested on "there is no text" and
on the harder "there is text and it is wrong".

`bad_ocr` and `chrome` are the same trap seen from two distances: enough text
operators to clear the floor, and not one usable word among them. `watermark` is
the floor read from the other side: a complete, correct text layer written in
eight operators, under a stamp, which the floor reads as chrome over a scan.
`mojibake` does not involve the floor at all -- the text layer is rich, and every
character of it is wrong, because the font's `/ToUnicode` CMap is.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
import zlib
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, NumberObject, StreamObject

OUT_DIR = Path(__file__).resolve().parent

#: US Letter, in points. Same as `tests/fixtures/make_fixtures.py`.
PAGE_WIDTH = 612
PAGE_HEIGHT = 792

#: The raster size of a "scanned" page: 100 dpi over US Letter. The pixel count
#: (935_000) is the load-bearing number -- pdf-inspector calls an image a
#: *template image* (a full-page background, i.e. a scan) only at 500_000 pixels
#: or more, and the 50x50 images in `tests/fixtures/` are three orders of
#: magnitude below that. Every trap that needs `has_template_image` needs this.
SCAN_WIDTH = 850
SCAN_HEIGHT = 1100

#: Where the wrong `/ToUnicode` CMap sends the Latin alphabet. Adding 0x0400 to
#: an ASCII letter lands in Cyrillic, which is what a real broken CMap produces:
#: text that renders correctly and extracts as a different script entirely.
MOJIBAKE_SHIFT = 0x0400


# --------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------


def _wrap(words: list[str], width: int) -> list[str]:
    """Greedy wrap to `width` characters. Deterministic, no hyphenation."""
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        if current and sum(len(w) + 1 for w in current) + len(word) > width:
            lines.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


#: Sentence stems the body text is assembled from. Deliberately plain and
#: business-shaped: the corpus is measuring routing, not language.
STEMS = [
    "The supplier shall deliver the goods described in Schedule {n} to the "
    "receiving dock at the address shown above.",
    "Payment falls due thirty days from the date of this invoice unless a "
    "different term is agreed in writing.",
    "Quantities recorded at goods-in take precedence over the quantities "
    "stated on the accompanying delivery note.",
    "Any discrepancy must be reported to the account manager within five "
    "working days of receipt.",
    "This page was produced by the records office and carries the file "
    "reference printed in the footer.",
    "Storage charges accrue from the first day after the free period ends "
    "and are billed monthly in arrears.",
    "The parties agree that the governing law of this agreement is that of "
    "the jurisdiction named in clause {n}.",
    "Serial numbers for every item in lot {n} are listed in the appendix and "
    "are not repeated here.",
]


def body_lines(title: str, page_no: int, count: int, width: int = 66) -> list[str]:
    """A page of body text: a title line, then `count` wrapped prose lines.

    Deterministic in `(title, page_no)`, so a page's text does not move when a
    neighbouring page changes.
    """
    rng = random.Random(f"{title}/{page_no}")
    words: list[str] = []
    while True:
        stem = STEMS[rng.randrange(len(STEMS))].format(n=rng.randrange(2, 90))
        words.extend(stem.split())
        if len(_wrap(words, width)) > count:
            break
    return [title] + _wrap(words, width)[:count]


#: Character confusions a cheap OCR engine makes: shapes it cannot tell apart.
CONFUSIONS = {
    "m": "rn",
    "rn": "m",
    "l": "1",
    "I": "l",
    "O": "0",
    "0": "O",
    "S": "5",
    "B": "8",
    "e": "c",
    "c": "e",
    "t": "f",
    "n": "ri",
    "h": "b",
    "u": "ii",
    "w": "vv",
    "g": "q",
    "y": "v",
    ".": ",",
    ",": ".",
}


def garble(lines: list[str], seed: str) -> list[str]:
    """Rewrite `lines` the way a cheap OCR pass would get them wrong.

    Three failure modes, all of them ones real pre-OCR'd scans show: characters
    confused for same-shaped ones, whole lines dropped where the engine lost the
    baseline, and words fused or split at the wrong place. The result is text
    that is *present*, so nothing structural objects to it, and unusable, so the
    page truly needs OCR.
    """
    rng = random.Random(seed)
    out: list[str] = []
    for index, line in enumerate(lines):
        # One line in seven is lost outright -- the engine skipped a baseline.
        if index and rng.random() < 0.14:
            continue
        chars: list[str] = []
        for char in line:
            if char in CONFUSIONS and rng.random() < 0.22:
                chars.append(CONFUSIONS[char])
            elif char == " " and rng.random() < 0.06:
                # Two words fused into one.
                continue
            else:
                chars.append(char)
        text = "".join(chars)
        if rng.random() < 0.18:
            # A stray mark read as punctuation.
            cut = rng.randrange(1, max(2, len(text)))
            text = text[:cut] + rng.choice([".", "'", "-", "|"]) + text[cut:]
        out.append(text)
    return out


def mojibake(text: str) -> str:
    """What the wrong `/ToUnicode` CMap below turns `text` into on extraction."""
    return "".join(
        chr(ord(ch) + MOJIBAKE_SHIFT) if ch.isascii() and ch.isalpha() else ch for ch in text
    )


# --------------------------------------------------------------------------
# raster
# --------------------------------------------------------------------------


def _font(size: int) -> ImageFont.FreeTypeFont:
    """Pillow's bundled scalable face. No system font is read, so the raster is
    the same on any machine with the same Pillow."""
    return ImageFont.load_default(size=size)


def render_scan(
    lines: list[str],
    seed: str,
    *,
    size: int = 21,
    leading: int = 30,
    top: int = 96,
    left: int = 96,
    stamp: str | None = None,
) -> bytes:
    """Rasterise `lines` as a 1-bit page image and return the FlateDecode stream.

    1 bit per pixel is what a real bitonal scan is, and it is also what keeps
    this corpus small enough to commit: a page of text packs to a few kilobytes.
    Pillow's `"1"` mode already uses 0 for black and 1 for white with rows padded
    to a byte boundary, which is exactly DeviceGray at `/BitsPerComponent 1`.

    `stamp` draws a large diagonal-looking overlay across the middle of the page
    -- the "RECEIVED" mark a records office puts on a page before filing it.
    """
    rng = random.Random(seed)
    page = Image.new("L", (SCAN_WIDTH, SCAN_HEIGHT), 255)
    draw = ImageDraw.Draw(page)
    face = _font(size)
    y = top
    for line in lines:
        draw.text((left, y), line, font=face, fill=0)
        y += leading
    if stamp is not None:
        big = _font(58)
        draw.text((120, SCAN_HEIGHT // 2), stamp, font=big, fill=110)
        draw.rectangle((100, SCAN_HEIGHT // 2 - 24, 760, SCAN_HEIGHT // 2 + 96), outline=110, width=4)
    # Scanner grime: a light pepper of stray pixels and a skewed platen edge.
    pixels = page.load()
    for _ in range(900):
        pixels[rng.randrange(SCAN_WIDTH), rng.randrange(SCAN_HEIGHT)] = 0
    for row in range(SCAN_HEIGHT):
        pixels[row * 3 // 200, row] = 0
    return zlib.compress(page.convert("1").tobytes(), 9)


# --------------------------------------------------------------------------
# PDF objects
# --------------------------------------------------------------------------


def _escape(text: str) -> str:
    """Escape a PDF literal string."""
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _tounicode_cmap() -> bytes:
    """A syntactically perfect `/ToUnicode` CMap that maps every letter wrong.

    This is the whole of the broken-CID trap. Rendering never consults
    `/ToUnicode`: the glyph comes from the font's encoding, so the page is
    letter-perfect on screen. Extraction consults nothing else, so every letter
    comes back as its Cyrillic neighbour. Nothing about the page is structurally
    unusual -- it is a Helvetica text page with an extra stream hanging off the
    font dictionary.
    """
    pairs = [(code, code + MOJIBAKE_SHIFT) for code in range(32, 127) if chr(code).isalpha()]
    out = [
        "/CIDInit /ProcSet findresource begin\n12 dict begin\nbegincmap\n"
        "/CIDSystemInfo << /Registry (Adobe) /Ordering (UCS) /Supplement 0 >> def\n"
        "/CMapName /Adobe-Identity-UCS def\n/CMapType 2 def\n"
        "1 begincodespacerange\n<00> <FF>\nendcodespacerange\n"
    ]
    for start in range(0, len(pairs), 100):
        chunk = pairs[start : start + 100]
        out.append(f"{len(chunk)} beginbfchar\n")
        out.extend(f"<{src:02X}> <{dst:04X}>\n" for src, dst in chunk)
        out.append("endbfchar\n")
    out.append("endcmap\nCMapName currentdict /CMap defineresource pop\nend\nend\n")
    return "".join(out).encode("latin-1")


def _helvetica(writer: PdfWriter, *, broken_tounicode: bool = False) -> DictionaryObject:
    """A base-14 Helvetica font dictionary; nothing is embedded."""
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    if broken_tounicode:
        cmap = StreamObject()
        cmap._data = _tounicode_cmap()
        font[NameObject("/ToUnicode")] = writer._add_object(cmap)
    return font


def _resources(
    writer: PdfWriter,
    *,
    font: DictionaryObject | None = None,
    image: StreamObject | None = None,
) -> DictionaryObject:
    res = DictionaryObject()
    if font is not None:
        res[NameObject("/Font")] = DictionaryObject(
            {NameObject("/F1"): writer._add_object(font)}
        )
    if image is not None:
        res[NameObject("/XObject")] = DictionaryObject(
            {NameObject("/Im1"): writer._add_object(image)}
        )
    return res


def _image_xobject(data: bytes) -> StreamObject:
    """A full-page 1-bit DeviceGray image XObject, already Flate-compressed."""
    img = StreamObject()
    img._data = data
    img.update(
        {
            NameObject("/Type"): NameObject("/XObject"),
            NameObject("/Subtype"): NameObject("/Image"),
            NameObject("/Width"): NumberObject(SCAN_WIDTH),
            NameObject("/Height"): NumberObject(SCAN_HEIGHT),
            NameObject("/ColorSpace"): NameObject("/DeviceGray"),
            NameObject("/BitsPerComponent"): NumberObject(1),
            NameObject("/Filter"): NameObject("/FlateDecode"),
        }
    )
    return img


#: Paint the page-sized image over the whole media box.
DRAW_IMAGE = f"q {PAGE_WIDTH} 0 0 {PAGE_HEIGHT} 0 0 cm /Im1 Do Q"


def _lines_block(lines: list[str], x: int, y: int, size: int, leading: int) -> str:
    """One text-showing operator per line -- the shape `make_fixtures.py` uses."""
    body = " ".join(f"({_escape(line)}) Tj T*" for line in lines)
    return f"BT /F1 {size} Tf {x} {y} Td {leading} TL {body} ET"


def _word_block(
    lines: list[str], x: int, y: int, size: int, leading: int, *, invisible: bool = True
) -> str:
    """One text-showing operator per *word*, positioned absolutely.

    `3 Tr` is the render mode most OCR layers use: the glyphs are laid over the
    scan for selection and search, and never painted. The other way engines hide
    a layer is to draw it in normal render mode and paint the scan on top of it,
    which `invisible=False` produces -- identical on screen, and reachable by an
    extractor that ignores render mode. Per-word `Tm` placement is what every
    engine emits, and it is also what pushes the page's text-operator count well
    past the floor pdf-inspector uses to spot a scan.
    """
    out = ["BT" if not invisible else "BT 3 Tr", f"/F1 {size} Tf"]
    cursor = y
    for line in lines:
        pen = x
        for word in line.split(" "):
            if word:
                out.append(f"1 0 0 1 {pen} {cursor} Tm ({_escape(word)}) Tj")
            pen += int((len(word) + 1) * size * 0.5)
        cursor -= leading
    out.append("ET")
    return " ".join(out)


# --------------------------------------------------------------------------
# page kinds
# --------------------------------------------------------------------------
#
# Every builder returns the page's truth: the text a perfect extractor should
# return, the text this PDF's own text layer actually holds, and whether the
# page needs OCR. The generator is the only thing that knows both, which is why
# the truth files are written here and not derived later from the PDF.


def page_clean(writer: PdfWriter, lines: list[str], _seed: str) -> dict:
    """An ordinary text page. No image, no trap: the baseline both ways."""
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _resources(writer, font=_helvetica(writer))
    stream = StreamObject()
    stream._data = _lines_block(lines, 72, 720, 12, 15).encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    text = "\n".join(lines)
    return {
        "needs_ocr": False,
        "trap": "none",
        "why": "Ordinary text page: the text layer is the page. Nothing to route to OCR.",
        "text": text,
        "text_layer": text,
    }


def page_scan(writer: PdfWriter, lines: list[str], seed: str) -> dict:
    """An honest scan: the page is an image of text and carries no text layer."""
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _resources(
        writer, image=_image_xobject(render_scan(lines, seed))
    )
    stream = StreamObject()
    stream._data = DRAW_IMAGE.encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    return {
        "needs_ocr": True,
        "trap": "none",
        "why": "Honest scan: an image of text with no text operators at all. "
        "The structural signal and the truth agree, and the heuristic gets it right.",
        "text": "\n".join(lines),
        "text_layer": None,
    }


def _bad_ocr(writer: PdfWriter, lines: list[str], seed: str, *, invisible: bool) -> dict:
    """A scan somebody already ran cheap OCR over, badly.

    The image holds the real text. The hidden text layer holds a transcription
    with characters confused, lines dropped and words fused -- present, and
    unusable. Structurally this page is indistinguishable from a *good* pre-OCR'd
    scan, so the inspector clears it; the page still needs OCR.

    `invisible` picks how the layer is hidden. Both ways are real, and they are
    not equivalent downstream: an extractor that skips `3 Tr` returns nothing at
    all for the invisible variant, and returns the garbled transcription for the
    under-image one. The second is the harder case -- there is text, it reads
    like text, and it is wrong.
    """
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _resources(
        writer,
        font=_helvetica(writer),
        image=_image_xobject(render_scan(lines, seed)),
    )
    layer = garble(lines, seed)
    block = _word_block(layer, 72, 716, 10, 21, invisible=invisible)
    # The invisible layer goes over the scan; the visible one goes under it.
    body = f"{DRAW_IMAGE} {block}" if invisible else f"{block} {DRAW_IMAGE}"
    stream = StreamObject()
    stream._data = body.encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    hidden = "invisible 3 Tr overlay" if invisible else "normal-render layer painted over by the scan"
    return {
        "needs_ocr": True,
        "trap": "bad_ocr_layer",
        "why": "Pre-OCR'd scan: a hidden text layer (" + hidden + ") covers the page, "
        "so nothing structural objects to it, but the transcription has confused "
        "characters, dropped lines and fused words. The page needs OCR precisely "
        "because it already has text.",
        "text": "\n".join(lines),
        "text_layer": "\n".join(layer),
    }


def page_bad_ocr(writer: PdfWriter, lines: list[str], seed: str) -> dict:
    """Pre-OCR'd scan, layer hidden the usual way: invisible render mode."""
    return _bad_ocr(writer, lines, seed, invisible=True)


def page_bad_ocr_under(writer: PdfWriter, lines: list[str], seed: str) -> dict:
    """Pre-OCR'd scan, layer hidden by painting the scan over it."""
    return _bad_ocr(writer, lines, seed, invisible=False)


def page_chrome(writer: PdfWriter, lines: list[str], seed: str) -> dict:
    """A scanned body page carrying only its furniture as text.

    A running head, a folio, a file reference, a received-stamp date: real
    characters, correctly transcribed, and not one word of the page's content.
    Between ten and twenty short operators, which is just enough to clear the
    floor below which pdf-inspector calls a page "chrome over a scan".
    """
    rng = random.Random(seed)
    furniture = [
        "CENTRAL RECORDS OFFICE",
        f"FILE {rng.randrange(1000, 9999)}/{rng.randrange(10, 99)}",
        "RETENTION: 7 YEARS",
        f"SHEET {rng.randrange(2, 40)}",
        "COPY",
        "NOT FOR CIRCULATION",
        f"BOX {rng.randrange(100, 999)}",
        "SCANNED",
        f"REF {rng.randrange(10000, 99999)}",
        "ARCHIVE",
        "PAGE CONTINUES",
        "SEE OVERLEAF",
    ]
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _resources(
        writer,
        font=_helvetica(writer),
        image=_image_xobject(render_scan(lines, seed, top=140)),
    )
    # One operator per furniture item, placed around the margins.
    placed = ["BT /F1 8 Tf"]
    for index, item in enumerate(furniture):
        x = 72 + (index % 3) * 170
        y = 760 - (index // 3) * 12
        placed.append(f"1 0 0 1 {x} {y} Tm ({_escape(item)}) Tj")
    placed.append("ET")
    stream = StreamObject()
    stream._data = (DRAW_IMAGE + " " + " ".join(placed)).encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    return {
        "needs_ocr": True,
        "trap": "sparse_decorative_text",
        "why": "Scanned body page whose only text layer is furniture -- running head, "
        "folio, file reference. The characters are real and correct, so the page "
        "reads as a text page structurally; every word of its content is in the "
        "image and reachable only by OCR.",
        "text": "\n".join(furniture + lines),
        "text_layer": "\n".join(furniture),
    }


def page_watermark(writer: PdfWriter, lines: list[str], seed: str) -> dict:
    """A short, complete text page under a full-page scanned stamp.

    The text layer is right and whole. The page is flagged anyway, because the
    stamp is a full-page image and the eight lines of body text are eight text
    operators, which is under the floor pdf-inspector reads as "chrome over a
    scan". Routing this page to OCR buys a worse copy of text already in hand.
    """
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    watermark = render_scan([], seed, stamp="RECEIVED")
    page[NameObject("/Resources")] = _resources(
        writer, font=_helvetica(writer), image=_image_xobject(watermark)
    )
    stream = StreamObject()
    stream._data = (DRAW_IMAGE + " " + _lines_block(lines, 72, 720, 12, 16)).encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    text = "\n".join(lines)
    return {
        "needs_ocr": False,
        "trap": "watermarked_text",
        "why": "Complete, correct text layer under a full-page RECEIVED stamp. The "
        "stamp makes the page image-dominant and the short body puts it under "
        "pdf-inspector's ten-operator floor, so it is flagged; every word of the "
        "page is already in the text layer, so OCR is pure waste.",
        "text": text,
        "text_layer": text,
    }


def page_mojibake(writer: PdfWriter, lines: list[str], _seed: str) -> dict:
    """A text page whose font carries a wrong `/ToUnicode` CMap.

    Renders letter-perfect -- the glyph is chosen by the encoding, which is
    untouched. Extracts as Cyrillic, because `/ToUnicode` is the only thing an
    extractor consults and every entry in it is wrong. No image, a full text
    layer, nothing structurally odd: the inspector clears it, and the page needs
    OCR.
    """
    page = writer.add_blank_page(width=PAGE_WIDTH, height=PAGE_HEIGHT)
    page[NameObject("/Resources")] = _resources(
        writer, font=_helvetica(writer, broken_tounicode=True)
    )
    stream = StreamObject()
    stream._data = _lines_block(lines, 72, 720, 12, 15).encode("latin-1")
    page[NameObject("/Contents")] = writer._add_object(stream)
    return {
        "needs_ocr": True,
        "trap": "broken_tounicode",
        "why": "The font's /ToUnicode CMap maps every letter to its Cyrillic "
        "neighbour. Rendering ignores /ToUnicode, so the page is correct on "
        "screen; extraction consults nothing else, so the text layer is mojibake "
        "and the page has to be re-read by OCR.",
        "text": "\n".join(lines),
        "text_layer": "\n".join(mojibake(line) for line in lines),
    }


KINDS = {
    "clean": (page_clean, 30),
    "scan": (page_scan, 26),
    "bad_ocr": (page_bad_ocr, 26),
    "bad_ocr_under": (page_bad_ocr_under, 26),
    "chrome": (page_chrome, 24),
    "watermark": (page_watermark, 8),
    "mojibake": (page_mojibake, 30),
}


# --------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------

#: name -> (title used for the body text, page kinds in order).
#:
#: The mix is deliberate. Four documents isolate one trap each so a failure has
#: one cause; `honest_mixed.pdf` contains no trap at all, so the corpus cannot
#: be read as rigged against the heuristic; the rest interleave traps with
#: ordinary pages so per-page routing is exercised rather than whole-document
#: classification.
DOCUMENTS: dict[str, tuple[str, list[str]]] = {
    "preocr_scan.pdf": (
        "GOODS RECEIVED LOG",
        ["bad_ocr", "bad_ocr_under", "bad_ocr", "bad_ocr_under", "bad_ocr", "bad_ocr_under"],
    ),
    "watermarked_memo.pdf": ("INTERNAL MEMORANDUM", ["watermark"] * 5),
    "broken_tounicode_report.pdf": ("QUARTERLY OPERATIONS REVIEW", ["mojibake"] * 5),
    "sparse_chrome_scan.pdf": ("CLAIMS CORRESPONDENCE", ["chrome"] * 5),
    "honest_mixed.pdf": (
        "SUPPLIER AGREEMENT",
        ["clean", "scan", "clean", "scan", "clean", "scan"],
    ),
    "stamped_invoices.pdf": (
        "INVOICE",
        ["watermark", "clean", "watermark", "clean"],
    ),
    "tounicode_mixed.pdf": (
        "SITE INSPECTION REPORT",
        ["clean", "mojibake", "clean", "mojibake", "scan"],
    ),
    "rescan_batch.pdf": (
        "ARCHIVE RESCAN BATCH",
        ["bad_ocr", "bad_ocr_under", "chrome", "scan", "bad_ocr_under", "chrome", "scan", "bad_ocr"],
    ),
    "mixed_adversarial.pdf": (
        "CASE FILE",
        [
            "clean",
            "bad_ocr",
            "clean",
            "watermark",
            "scan",
            "chrome",
            "mojibake",
            "clean",
            "bad_ocr_under",
            "watermark",
            "scan",
            "clean",
        ],
    ),
}


def build(name: str) -> tuple[bytes, dict]:
    """Build one document and its truth file."""
    title, kinds = DOCUMENTS[name]
    writer = PdfWriter()
    pages: dict[str, dict] = {}
    for index, kind in enumerate(kinds):
        builder, line_count = KINDS[kind]
        seed = f"{name}/{index}/{kind}"
        lines = body_lines(f"{title} -- PAGE {index + 1}", index, line_count)
        truth = builder(writer, lines, seed)
        truth["kind"] = kind
        pages[str(index)] = truth
    buf = io.BytesIO()
    writer.write(buf)
    manifest_truth = {
        "schema": "doc-router/corpus-truth@1",
        "document": name,
        "page_count": len(kinds),
        "needs_ocr": [int(i) for i, page in pages.items() if page["needs_ocr"]],
        "pages": pages,
    }
    return buf.getvalue(), manifest_truth


# --------------------------------------------------------------------------
# writing
# --------------------------------------------------------------------------


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
    ap = argparse.ArgumentParser(description="Generate the adversarial corpus.")
    ap.add_argument("--force", action="store_true", help="overwrite files that already exist")
    ap.add_argument("names", nargs="*", help="only build these document file names")
    args = ap.parse_args()

    wanted = set(args.names) if args.names else None
    for name in DOCUMENTS:
        if wanted and name not in wanted:
            continue
        pdf, truth = build(name)
        print(_write(OUT_DIR / name, pdf, args.force))
        truth_bytes = (json.dumps(truth, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        print(_write(OUT_DIR / f"{Path(name).stem}.truth.json", truth_bytes, args.force))
    return 0


if __name__ == "__main__":
    sys.exit(main())
