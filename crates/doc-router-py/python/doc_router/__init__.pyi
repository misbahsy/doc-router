"""Type hints for the Rust extension module. See ``__init__.py`` for the docstrings."""

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypedDict

__version__: str
LOCAL_MODEL: str
DEFAULTS: dict[str, Any]

class DocRouterError(Exception): ...
class PdfError(DocRouterError): ...
class ConfigError(DocRouterError): ...
class HostError(DocRouterError): ...

PdfTypeStr = Literal["text_based", "scanned", "image_based", "mixed", "unknown"]
TierStr = Literal["local", "standard", "premium"]
ReasonStr = Literal[
    "low_confidence", "text_layer", "scanned", "mixed_split", "mixed_unsplit"
]
BypassStr = Literal["not_pdf", "oversize", "image", "classifier_unavailable"]

class Classification(TypedDict):
    pdf_type: PdfTypeStr
    confidence: float
    page_count: int
    pages_needing_ocr: list[int]
    is_complex_layout: bool
    classify_ms: float

class Leg(TypedDict):
    model: str
    tier: TierStr
    pages: list[int] | None

class Plan(TypedDict):
    legs: list[Leg]
    reason: ReasonStr
    tier: TierStr
    routed_model: str

class Page(TypedDict):
    index: int
    markdown: str
    model: str

class OcrResult(TypedDict):
    pages: list[Page]
    model: str
    pages_processed: int
    doc_size_bytes: int | None

class RouteMetadata(TypedDict, total=False):
    tier: str
    reason: str
    routed_model: str
    pdf_type: PdfTypeStr
    confidence: float
    page_count: int
    pages_needing_ocr: list[int]
    classify_ms: float
    legs: list[Leg]
    split: bool
    fallback_reason: Literal["leg_failed"]

class RouteDecision(TypedDict):
    kind: Literal["route"]
    plan: Plan
    classification: Classification
    metadata: RouteMetadata

class BypassDecision(TypedDict):
    kind: Literal["bypass"]
    reason: BypassStr
    model: str
    detail: str | None
    metadata: RouteMetadata

Decision = RouteDecision | BypassDecision

class Outcome(TypedDict):
    result: OcrResult
    metadata: RouteMetadata
    decision: Decision

#: ``(model, document, pages) -> OcrResult``-shaped dict. Called from a worker thread.
Host = Callable[[str, bytes, list[int] | None], dict[str, Any]]

def classify(data: bytes) -> Classification: ...
def is_pdf(data: bytes) -> bool: ...
def plan_route(classification: dict[str, Any], config: dict[str, Any]) -> Plan: ...
def decide(
    data: bytes, config: dict[str, Any], default_model: str | None = None
) -> Decision: ...
def extract_local(data: bytes, pages: Sequence[int] | None = None) -> OcrResult: ...
def split_pdf(data: bytes, pages: Sequence[int]) -> bytes: ...
def remap_pages(result: dict[str, Any], pages: Sequence[int]) -> OcrResult: ...
def merge(legs: Sequence[tuple[dict[str, Any], dict[str, Any]]]) -> OcrResult: ...
def run(
    data: bytes,
    config: dict[str, Any],
    host: Host,
    default_model: str | None = None,
) -> Outcome: ...
def validate_config(config: dict[str, Any]) -> dict[str, Any]: ...
