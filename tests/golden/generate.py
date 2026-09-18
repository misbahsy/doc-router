#!/usr/bin/env python3
"""Generate the parity goldens in tests/golden/ from the Python reference implementation.

The Rust port in crates/doc-router must reproduce every field these files record. The
source of truth is LiteLLM's document router:

    litellm/router_strategy/doc_router/{classifier,config,policy}.py

Run:

    PYTHONPATH=/path/to/litellm \
      python3 tests/golden/generate.py

One `<stem>.json` per file in tests/fixtures/ (every *.pdf plus not_a_pdf.bin). See
docs/SPEC.md, section "Golden files", for the schema. `classify_ms` is deliberately
omitted -- it is wall-clock noise and is never compared.

Under the hood:
  * `classify_pdf(data)` -> `pdf_inspector.detect_pdf_bytes(data)`, whose
    `pages_needing_ocr` is 1-INDEXED; the reference shifts it to 0-indexed and drops
    out-of-range entries. Goldens therefore carry 0-indexed pages.
  * `is_complex_layout` does NOT come from the detect call. Detect-only mode skips
    layout analysis, so `PdfResult.is_complex_layout` is False for every document --
    including genuinely multi-column ones (text_dense.pdf). The Python reference reads
    that always-False field; docs/SPEC.md requires the port to "use the cheapest call
    that does" populate layout, so these goldens record
    `extract_pages_markdown_bytes(...).is_complex` (identical to
    `process_pdf_bytes(...).is_complex_layout`, which the Rust side reaches through
    ProcessMode::Analyze). This is the one deliberate departure from the reference; it
    changes no plan in this corpus, since the only fixture affected is text-based and
    routes on its text layer.
  * `local_pages` come from `pdf_inspector.extract_pages_markdown_bytes(data)`
    (`PagesExtractionResult.pages`, `PageMarkdown.page` is 0-indexed).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pdf_inspector

from litellm.router_strategy.doc_router.classifier import DocClassification, classify_pdf
from litellm.router_strategy.doc_router.config import DocRouterConfig
from litellm.router_strategy.doc_router.policy import plan_route

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = ROOT / "tests" / "fixtures"
GOLDEN_DIR = ROOT / "tests" / "golden"

TIERS_WITH_PREMIUM: dict[str, Any] = {
    "local": "local_pdf/extract",
    "standard": "mistral-ocr",
    "premium": "gpt-5-ocr",
}
TIERS_NO_PREMIUM: dict[str, Any] = {"local": "local_pdf/extract", "standard": "mistral-ocr"}

#: The five config variants from docs/SPEC.md. Each dict is passed verbatim to
#: DocRouterConfig(**config) AND emitted verbatim into the golden, so only the keys set
#: here appear -- that is what exercises the Rust side's serde defaults.
CONFIG_VARIANTS: list[dict[str, Any]] = [
    {"tiers": TIERS_WITH_PREMIUM},                                        # (a) defaults, premium
    {"tiers": TIERS_NO_PREMIUM},                                          # (b) no premium tier
    {"tiers": TIERS_WITH_PREMIUM, "split_pages": False},                  # (c) never split
    {"tiers": TIERS_WITH_PREMIUM, "min_confidence": 1.0},                 # (d) force low_confidence
    {"tiers": TIERS_WITH_PREMIUM, "complex_layout_tier": "standard"},     # (e) no premium escalation
]


def layout_is_complex(data: bytes) -> bool:
    """True when layout analysis finds tables or multi-column text.

    Read from the extraction call, not from detect-only mode -- see the module
    docstring for why the reference's own value cannot be used here.
    """
    return bool(pdf_inspector.extract_pages_markdown_bytes(data).is_complex)


def classification_json(c: DocClassification, data: bytes) -> dict[str, Any]:
    """The classification block. `classify_ms` is intentionally left out."""
    return {
        "pdf_type": c.pdf_type,
        "confidence": c.confidence,
        "page_count": c.page_count,
        "pages_needing_ocr": list(c.pages_needing_ocr),
        "is_complex_layout": layout_is_complex(data),
    }


def local_pages_json(data: bytes) -> list[dict[str, Any]]:
    """Per-page markdown for every page, 0-indexed, in document order."""
    result = pdf_inspector.extract_pages_markdown_bytes(data)
    return [
        {"index": page.page, "markdown": page.markdown, "needs_ocr": bool(page.needs_ocr)}
        for page in result.pages
    ]


def plan_json(config: dict[str, Any], c: DocClassification) -> dict[str, Any]:
    """One `plans[]` entry: the raw config dict plus the plan the reference produced."""
    plan = plan_route(c, DocRouterConfig(**config))
    return {
        "config": config,
        "default_model": None,
        "plan": {
            "legs": [
                {"model": leg.model, "tier": leg.tier, "pages": leg.kwargs.get("pages")}
                for leg in plan.legs
            ],
            "reason": plan.reason,
            "tier": plan.tier,
            "routed_model": plan.routed_model,
        },
    }


def golden_for_pdf(path: Path, data: bytes) -> dict[str, Any]:
    c = classify_pdf(data)
    return {
        "fixture": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "classification": classification_json(c, data),
        "local_pages": local_pages_json(data),
        "plans": [plan_json(config, c) for config in CONFIG_VARIANTS],
    }


def golden_for_non_pdf(path: Path, data: bytes) -> dict[str, Any]:
    """Non-PDF bytes never reach the classifier: `decide` bypasses them as `not_pdf`."""
    return {
        "fixture": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bypass": "not_pdf",
    }


def write(golden: dict[str, Any], stem: str) -> Path:
    out = GOLDEN_DIR / f"{stem}.json"
    with out.open("w", encoding="utf-8") as fh:
        json.dump(golden, fh, indent=2, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    return out


def main() -> int:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    targets = sorted(FIXTURE_DIR.glob("*.pdf")) + sorted(FIXTURE_DIR.glob("*.bin"))
    if not targets:
        print(f"no fixtures found in {FIXTURE_DIR}", file=sys.stderr)
        return 1
    for path in targets:
        data = path.read_bytes()
        golden = (
            golden_for_pdf(path, data)
            if data.startswith(b"%PDF")
            else golden_for_non_pdf(path, data)
        )
        out = write(golden, path.stem)
        summary = golden.get("classification")
        if summary is None:
            print(f"{out.name:24} bypass={golden['bypass']}")
        else:
            print(
                f"{out.name:24} {summary['pdf_type']:11} conf={summary['confidence']:.4f} "
                f"pages={summary['page_count']:3} ocr={len(summary['pages_needing_ocr']):3} "
                f"complex={str(summary['is_complex_layout']):5} "
                f"reasons={[p['plan']['reason'] for p in golden['plans']]}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
