"""Per-page routing between local PDF text extraction and paid OCR models.

This is a thin re-export of the Rust extension module ``doc_router._native``; every
function below is implemented in Rust (crate ``doc-router``) and releases the GIL around
its work. The boundary is plain Python data: ``bytes`` in, ``dict`` out.

The five things worth knowing:

* **Pages are 0-indexed everywhere.** ``pages_needing_ocr``, ``Leg.pages``, the ``pages``
  argument of :func:`extract_local` / :func:`split_pdf` / the host callable, and each
  page's ``index`` all refer to the original document.
* **Nothing here calls a network.** :func:`run` hands OCR legs to a host callable you
  supply; local text-layer legs are read in-process.
* :func:`decide` never raises for unroutable input -- a non-PDF, an oversized document or
  a classifier failure comes back as ``{"kind": "bypass", ...}``.
* Config is the same JSON block LiteLLM's ``doc_router_config`` uses, and unknown keys
  are rejected.
* Errors are :class:`DocRouterError` subclasses: :class:`PdfError`, :class:`ConfigError`,
  :class:`HostError`.

>>> import doc_router
>>> config = {"tiers": {"local": "local_pdf/extract", "standard": "mistral-ocr"}}
>>> decision = doc_router.decide(pdf_bytes, config)          # doctest: +SKIP
>>> decision["kind"], decision["metadata"]["reason"]          # doctest: +SKIP
('route', 'mixed_split')
"""

from __future__ import annotations

from ._native import (
    DEFAULTS,
    LOCAL_MODEL,
    ConfigError,
    DocRouterError,
    HostError,
    PdfError,
    __version__,
    classify,
    decide,
    extract_local,
    is_pdf,
    merge,
    plan_route,
    remap_pages,
    run,
    split_pdf,
    validate_config,
)

__all__ = [
    "DEFAULTS",
    "LOCAL_MODEL",
    "ConfigError",
    "DocRouterError",
    "HostError",
    "PdfError",
    "__version__",
    "classify",
    "decide",
    "extract_local",
    "is_pdf",
    "merge",
    "plan_route",
    "remap_pages",
    "run",
    "split_pdf",
    "validate_config",
]
