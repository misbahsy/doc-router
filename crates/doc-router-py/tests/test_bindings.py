"""End-to-end tests for the `doc_router` PyO3 bindings.

Run against the module built by `maturin develop --release` from
`crates/doc-router-py`, using the interpreter that module was built against.

The interesting parts are:

* `test_run_*` -- the host callable is invoked from the Rust core's scoped worker
  threads, so `run` must release the GIL before entering the core and the host adapter
  must re-acquire it. `test_run_releases_the_gil_for_the_host` proves that with a
  watchdog that kills the process instead of hanging forever.
* `test_golden_parity` -- every `tests/golden/*.json` recorded from LiteLLM's Python
  implementation is replayed through `classify` and `plan_route`.
"""

from __future__ import annotations

import faulthandler
import json
import threading
from pathlib import Path

import doc_router
import pytest

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"

TIERS = {"local": "local_pdf/extract", "standard": "mistral-ocr"}
CONFIG = {"tiers": TIERS}

#: Longest a `run` call may take before we assume the GIL deadlocked.
DEADLOCK_TIMEOUT_SECONDS = 60.0


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class RecordingHost:
    """A fake OCR host. Records its calls and returns one fake page per requested page."""

    def __init__(self, fail_for: str | None = None) -> None:
        self.calls: list[tuple[str, int, list[int] | None]] = []
        self.threads: set[int] = set()
        self.fail_for = fail_for

    def __call__(self, model: str, document: bytes, pages: list[int] | None) -> dict:
        assert isinstance(document, bytes)
        self.calls.append((model, len(document), pages))
        self.threads.add(threading.get_ident())
        if self.fail_for is not None and model == self.fail_for:
            raise RuntimeError(f"provider {model} is down")
        indices = list(pages) if pages is not None else [0]
        # `model`, `pages_processed`, `doc_size_bytes` and each page's `model` are
        # optional -- the binding fills them in.
        return {"pages": [{"index": i, "markdown": f"ocr page {i}"} for i in indices]}


# --------------------------------------------------------------------------- module


def test_module_surface() -> None:
    assert isinstance(doc_router.__version__, str)
    assert doc_router.LOCAL_MODEL == "local_pdf/extract"
    assert doc_router.DEFAULTS["min_confidence"] == pytest.approx(0.6)
    assert doc_router.DEFAULTS["split_pages"] is True
    assert doc_router.DEFAULTS["max_document_bytes"] == 50 * 1024 * 1024
    for name in ("PdfError", "ConfigError", "HostError"):
        assert issubclass(getattr(doc_router, name), doc_router.DocRouterError)


def test_is_pdf() -> None:
    assert doc_router.is_pdf(fixture("text.pdf"))
    assert not doc_router.is_pdf(fixture("not_a_pdf.bin"))
    assert not doc_router.is_pdf(b"")


# ------------------------------------------------------------------------- classify


def test_classify_text() -> None:
    c = doc_router.classify(fixture("text.pdf"))
    assert c["pdf_type"] == "text_based"
    assert c["page_count"] == 2
    assert c["pages_needing_ocr"] == []
    assert c["classify_ms"] >= 0.0


def test_classify_scanned() -> None:
    c = doc_router.classify(fixture("scanned.pdf"))
    assert c["pdf_type"] == "scanned"
    assert c["page_count"] == 2
    assert c["pages_needing_ocr"] == [0, 1]


def test_classify_mixed() -> None:
    c = doc_router.classify(fixture("mixed.pdf"))
    assert c["pdf_type"] == "mixed"
    assert c["page_count"] == 4
    assert c["pages_needing_ocr"] == [1, 3]
    assert c["is_complex_layout"] is False
    assert 0.0 <= c["confidence"] <= 1.0


def test_classify_rejects_non_pdf() -> None:
    with pytest.raises(doc_router.PdfError):
        doc_router.classify(fixture("not_a_pdf.bin"))


# ----------------------------------------------------------------------- plan/decide


def test_plan_route_accepts_a_partial_classification() -> None:
    # `classify_ms` and `is_complex_layout` are optional, and pdf-inspector's Python
    # spelling of `pdf_type` is accepted alongside the wire form.
    plan = doc_router.plan_route(
        {"pdf_type": "Mixed", "confidence": 0.9, "page_count": 4, "pages_needing_ocr": [1, 3]},
        CONFIG,
    )
    assert plan["reason"] == "mixed_split"
    assert [leg["pages"] for leg in plan["legs"]] == [[0, 2], [1, 3]]


def test_decide_mixed_routes_and_splits() -> None:
    decision = doc_router.decide(fixture("mixed.pdf"), CONFIG)

    assert decision["kind"] == "route"
    plan = decision["plan"]
    assert plan["reason"] == "mixed_split"
    assert plan["tier"] == "standard"
    assert plan["routed_model"] == "mistral-ocr"
    assert plan["legs"] == [
        {"model": "local_pdf/extract", "tier": "local", "pages": [0, 2]},
        {"model": "mistral-ocr", "tier": "standard", "pages": [1, 3]},
    ]
    assert decision["classification"]["pdf_type"] == "mixed"

    metadata = decision["metadata"]
    assert metadata["tier"] == "standard"
    assert metadata["reason"] == "mixed_split"
    assert metadata["routed_model"] == "mistral-ocr"
    assert metadata["pages_needing_ocr"] == [1, 3]
    assert metadata["split"] is True
    assert len(metadata["legs"]) == 2


def test_decide_bypasses_a_non_pdf() -> None:
    decision = doc_router.decide(fixture("not_a_pdf.bin"), CONFIG, "gpt-4o-mini")

    assert decision["kind"] == "bypass"
    assert decision["reason"] == "not_pdf"
    assert decision["model"] == "gpt-4o-mini"
    assert decision["detail"] is None
    assert decision["metadata"] == {
        "tier": "bypass",
        "reason": "not_pdf",
        "routed_model": "gpt-4o-mini",
    }


def test_decide_bypasses_an_oversized_document() -> None:
    config = {"tiers": TIERS, "max_document_bytes": 32}
    decision = doc_router.decide(fixture("mixed.pdf"), config)
    assert decision["kind"] == "bypass"
    assert decision["reason"] == "oversize"
    # No default model configured, so a bypass falls back to the standard tier.
    assert decision["model"] == "mistral-ocr"
    assert "max_document_bytes" in decision["detail"]


# ---------------------------------------------------------------------------- config


def test_validate_config_fills_in_defaults() -> None:
    normalised = doc_router.validate_config(CONFIG)
    assert normalised["tiers"] == {"local": "local_pdf/extract", "standard": "mistral-ocr",
                                   "premium": None}
    assert normalised["min_confidence"] == pytest.approx(0.6)
    assert normalised["complex_layout_tier"] == "premium"
    assert normalised["split_pages"] is True


@pytest.mark.parametrize(
    "config",
    [
        {},                                                     # no tiers at all
        {"tiers": {"local": "l"}},                              # standard missing
        {"tiers": {"local": "", "standard": "s"}},              # empty model name
        {"tiers": TIERS, "min_confidence": 1.5},                # out of range
        {"tiers": TIERS, "max_document_bytes": 0},              # must be > 0
        {"tiers": TIERS, "split_page": True},                   # typo: unknown key
        {"tiers": {"local": "l", "standard": "s", "gold": "g"}},  # unknown tier
    ],
)
def test_bad_config_raises_config_error(config: dict) -> None:
    with pytest.raises(doc_router.ConfigError):
        doc_router.validate_config(config)
    with pytest.raises(doc_router.ConfigError):
        doc_router.decide(fixture("mixed.pdf"), config)


# --------------------------------------------------------------------------- extract


def test_extract_local_reads_the_text_layer() -> None:
    result = doc_router.extract_local(fixture("text.pdf"))
    assert result["model"] == "local_pdf/extract"
    assert result["pages_processed"] == 2
    assert [page["index"] for page in result["pages"]] == [0, 1]
    assert all(page["model"] == "local_pdf/extract" for page in result["pages"])
    assert "TEXT ONLY DOCUMENT" in result["pages"][0]["markdown"]
    assert result["doc_size_bytes"] == len(fixture("text.pdf"))


def test_extract_local_takes_a_page_subset() -> None:
    result = doc_router.extract_local(fixture("mixed.pdf"), [0, 2])
    assert [page["index"] for page in result["pages"]] == [0, 2]
    assert result["pages_processed"] == 2


def test_extract_local_rejects_non_pdf() -> None:
    with pytest.raises(doc_router.PdfError):
        doc_router.extract_local(fixture("not_a_pdf.bin"))


# ----------------------------------------------------------------------------- split


def test_split_pdf_round_trips_two_pages() -> None:
    data = fixture("mixed.pdf")
    split = doc_router.split_pdf(data, [0, 2])

    assert isinstance(split, bytes)
    assert doc_router.is_pdf(split)
    assert doc_router.classify(split)["page_count"] == 2

    # The subset's page 0 and page 1 are the source's page 0 and page 2.
    extracted = doc_router.extract_local(split)
    original = doc_router.extract_local(data, [0, 2])
    assert extracted["pages"][0]["markdown"] == original["pages"][0]["markdown"]
    assert extracted["pages"][1]["markdown"] == original["pages"][1]["markdown"]


def test_split_pdf_rejects_an_out_of_range_page() -> None:
    with pytest.raises(doc_router.PdfError):
        doc_router.split_pdf(fixture("mixed.pdf"), [99])


def test_remap_pages_restores_original_indices() -> None:
    provider_result = {
        "pages": [
            {"index": 0, "markdown": "was page one"},
            {"index": 1, "markdown": "was page three"},
        ],
        "model": "mistral-ocr",
    }
    remapped = doc_router.remap_pages(provider_result, [1, 3])
    assert [page["index"] for page in remapped["pages"]] == [1, 3]
    # The input dict is not modified.
    assert [page["index"] for page in provider_result["pages"]] == [0, 1]


# ----------------------------------------------------------------------------- merge


def test_merge_orders_pages_and_joins_models() -> None:
    local_leg = {"model": "local_pdf/extract", "tier": "local", "pages": [0, 2]}
    ocr_leg = {"model": "mistral-ocr", "tier": "standard", "pages": [1, 3]}
    merged = doc_router.merge(
        [
            (
                local_leg,
                {
                    "pages": [
                        {"index": 2, "markdown": "two"},
                        {"index": 0, "markdown": "zero"},
                    ],
                    "doc_size_bytes": 120,
                },
            ),
            (
                ocr_leg,
                {
                    "pages": [
                        {"index": 1, "markdown": "one"},
                        {"index": 3, "markdown": "three"},
                    ],
                    "doc_size_bytes": 500,
                },
            ),
        ]
    )

    assert [page["index"] for page in merged["pages"]] == [0, 1, 2, 3]
    assert [page["markdown"] for page in merged["pages"]] == ["zero", "one", "two", "three"]
    assert [page["model"] for page in merged["pages"]] == [
        "local_pdf/extract",
        "mistral-ocr",
        "local_pdf/extract",
        "mistral-ocr",
    ]
    assert merged["model"] == "local_pdf/extract,mistral-ocr"
    assert merged["pages_processed"] == 4
    assert merged["doc_size_bytes"] == 500


# ------------------------------------------------------------------------------- run


def test_run_splits_locally_and_calls_the_host_once() -> None:
    host = RecordingHost()
    data = fixture("mixed.pdf")

    outcome = doc_router.run(data, CONFIG, host, "gpt-4o-mini")

    # Exactly one host call: the local leg never leaves the process.
    assert len(host.calls) == 1
    model, size, pages = host.calls[0]
    assert model == "mistral-ocr"
    assert size == len(data)
    assert pages == [1, 3]

    result = outcome["result"]
    assert [page["index"] for page in result["pages"]] == [0, 1, 2, 3]
    assert result["model"] == "local_pdf/extract,mistral-ocr"
    assert result["pages_processed"] == 4
    assert [page["model"] for page in result["pages"]] == [
        "local_pdf/extract",
        "mistral-ocr",
        "local_pdf/extract",
        "mistral-ocr",
    ]
    assert result["pages"][1]["markdown"] == "ocr page 1"
    assert "MIXED DOC PAGE ONE" in result["pages"][0]["markdown"]

    assert outcome["metadata"]["reason"] == "mixed_split"
    assert outcome["metadata"]["routed_model"] == "mistral-ocr"
    assert "fallback_reason" not in outcome["metadata"]
    assert outcome["decision"]["kind"] == "route"
    # The top-level metadata is authoritative; the decision does not repeat it.
    assert "metadata" not in outcome["decision"]


def test_run_falls_back_to_the_default_model_when_a_leg_fails() -> None:
    host = RecordingHost(fail_for="mistral-ocr")

    outcome = doc_router.run(fixture("mixed.pdf"), CONFIG, host, "gpt-4o-mini")

    assert [call[0] for call in host.calls] == ["mistral-ocr", "gpt-4o-mini"]
    # The retry is the whole document, not a page subset.
    assert host.calls[1][2] is None

    assert outcome["metadata"]["fallback_reason"] == "leg_failed"
    assert outcome["metadata"]["routed_model"] == "gpt-4o-mini"
    # ... while the decision still records what was originally planned.
    assert outcome["decision"]["plan"]["routed_model"] == "mistral-ocr"
    assert outcome["result"]["model"] == "gpt-4o-mini"


def test_run_raises_host_error_when_the_fallback_also_fails() -> None:
    def always_fails(model: str, document: bytes, pages: list[int] | None) -> dict:
        raise RuntimeError("everything is down")

    with pytest.raises(doc_router.HostError) as caught:
        doc_router.run(fixture("mixed.pdf"), CONFIG, always_fails, "gpt-4o-mini")
    assert "everything is down" in str(caught.value)


def test_run_bypasses_a_non_pdf_to_the_default_model() -> None:
    host = RecordingHost()
    outcome = doc_router.run(fixture("not_a_pdf.bin"), CONFIG, host, "gpt-4o-mini")

    assert [call[0] for call in host.calls] == ["gpt-4o-mini"]
    assert host.calls[0][2] is None
    assert outcome["decision"]["kind"] == "bypass"
    assert outcome["decision"]["reason"] == "not_pdf"
    assert outcome["metadata"] == {
        "tier": "bypass",
        "reason": "not_pdf",
        "routed_model": "gpt-4o-mini",
    }


def test_run_rejects_a_non_callable_host() -> None:
    with pytest.raises(doc_router.HostError):
        doc_router.run(fixture("mixed.pdf"), CONFIG, "not-callable", "gpt-4o-mini")


def test_run_releases_the_gil_for_the_host() -> None:
    """The core calls the host from its own worker threads.

    If `run` held the GIL across the call into Rust, the host adapter's
    `Python::attach` would block forever on a thread that can never make progress.
    The watchdog turns that hang into a hard failure with a traceback instead of a
    test run that never ends; it runs on a C thread, so it fires even while the GIL
    is held.
    """
    host = RecordingHost()
    main_thread = threading.get_ident()

    faulthandler.dump_traceback_later(DEADLOCK_TIMEOUT_SECONDS, exit=True)
    try:
        outcome = doc_router.run(fixture("mixed.pdf"), CONFIG, host, "gpt-4o-mini")
    finally:
        faulthandler.cancel_dump_traceback_later()

    assert len(outcome["result"]["pages"]) == 4
    # The host really did run on one of the core's scoped threads, holding a GIL it
    # had to re-acquire itself.
    assert host.threads and main_thread not in host.threads


def test_run_from_a_python_thread_while_the_main_thread_works() -> None:
    """Two concurrent `run` calls from Python threads must not serialise or deadlock."""
    outcomes: dict[int, dict] = {}
    errors: list[BaseException] = []

    def work(slot: int) -> None:
        try:
            outcomes[slot] = doc_router.run(
                fixture("mixed.pdf"), CONFIG, RecordingHost(), "gpt-4o-mini"
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised in the assertion below
            errors.append(exc)

    faulthandler.dump_traceback_later(DEADLOCK_TIMEOUT_SECONDS, exit=True)
    try:
        threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(DEADLOCK_TIMEOUT_SECONDS)
            assert not thread.is_alive(), "run() deadlocked in a worker thread"
    finally:
        faulthandler.cancel_dump_traceback_later()

    assert not errors, errors
    assert len(outcomes) == 4
    assert all(len(o["result"]["pages"]) == 4 for o in outcomes.values())


# ---------------------------------------------------------------------------- golden


def golden_files() -> list[Path]:
    return sorted(GOLDEN.glob("*.json")) if GOLDEN.is_dir() else []


def normalise_pdf_type(value: str) -> str:
    return value.replace("_", "").lower()


@pytest.mark.parametrize("path", golden_files(), ids=lambda p: p.stem)
def test_golden_parity(path: Path) -> None:
    """Replay a golden recorded from LiteLLM's Python implementation."""
    golden = json.loads(path.read_text())
    data = fixture(golden["fixture"])

    if "classification" not in golden:
        # The only recorded bypass is `not_pdf`; nothing to classify.
        assert golden["bypass"] == "not_pdf"
        assert not doc_router.is_pdf(data)
        assert doc_router.decide(data, CONFIG)["reason"] == golden["bypass"]
        return

    expected = golden["classification"]
    actual = doc_router.classify(data)

    assert normalise_pdf_type(actual["pdf_type"]) == normalise_pdf_type(expected["pdf_type"])
    assert actual["page_count"] == expected["page_count"]
    assert actual["pages_needing_ocr"] == expected["pages_needing_ocr"]
    assert actual["is_complex_layout"] == expected["is_complex_layout"]
    assert actual["confidence"] == pytest.approx(expected["confidence"], abs=1e-3)

    for entry in golden["plans"]:
        # Plan from the RECORDED classification, so this compares policy alone.
        plan = doc_router.plan_route(expected, entry["config"])
        assert plan == entry["plan"], f"{path.stem}: config {entry['config']}"
