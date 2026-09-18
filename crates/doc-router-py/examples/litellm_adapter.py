#!/usr/bin/env python3
"""LiteLLM's `DocumentRouter` pre-routing hook on top of `doc_router`.

Today the hook classifies, decides and merges in Python. With these bindings the hook
keeps only the parts that are genuinely LiteLLM's -- config lookup and the actual
provider call -- and delegates classification, policy and merging to the Rust core:

    classify + policy tables    ->  doc_router.decide()
    per-leg split/extract/merge ->  doc_router.run()

The provider call goes through `litellm.ocr()`, so every model alias, key and fallback
LiteLLM already knows about applies unchanged. There is no Rust SDK to call instead:
BerriAI's `litellm` crate is a 0.0.1 placeholder. LiteLLM's *own* Rust OCR path ships
inside this Python package as a native extension, so the transport can be Rust while
the call stays `litellm.ocr()`:

    from litellm.rust_bridge.ocr import use_litellm_rust, rust_ocr_enabled
    use_litellm_rust(True)      # or set the env var the bridge reads

Run it:

    export LITELLM_BASE_URL=http://0.0.0.0:4000
    export LITELLM_API_KEY=sk-...
    python examples/litellm_adapter.py [document.pdf]

With no key set it falls back to a printing stub, so the routing half still demos
offline. The key is read from the environment and never logged.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import doc_router

# What LiteLLM would read from `litellm_settings.document_router` in config.yaml.
CONFIG = {
    # Model ids as the *gateway* spells them. `mistral-ocr` is the name the Python
    # reference used and it resolves nowhere: this gateway registers OCR deployments as
    # `mistral/mistral-ocr-latest` and `mistral/mistral-ocr-2512`.
    "tiers": {
        "local": "local_pdf/extract",
        "standard": "mistral/mistral-ocr-latest",
        "premium": "gpt-4o",
    },
    "min_confidence": 0.6,
    "split_pages": True,
}

#: Same order the Rust CLI resolves in (`doc_router_cli::API_KEY_ENV_VARS`).
API_KEY_ENV_VARS = ("LITELLM_API_KEY", "LITELLM_PROXY_API_KEY")


def _api_key() -> str | None:
    for name in API_KEY_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _sdk_model(model: str, api_base: str | None) -> str:
    """Spell `model` the way `litellm.ocr()` must be given it to reach a proxy.

    The SDK resolves the provider *locally* from the model prefix and forwards only
    what is left. Against a gateway whose own model ids are provider-prefixed, that
    strips exactly the part the gateway needs, so the id has to be doubled:

        config says   mistral/mistral-ocr-latest      (what the gateway registers)
        SDK is given  mistral/mistral/mistral-ocr-latest
        SDK strips    mistral/    -> provider "mistral"
        gateway gets  mistral/mistral-ocr-latest      (resolves)

    `litellm_proxy/`, the passthrough prefix that avoids this for chat, is not wired
    for OCR -- `get_provider_ocr_config` has no entry for it and the call raises
    "OCR is not supported for provider: litellm_proxy".

    `LiteLlmHost`, the Rust host, sends the id verbatim to `/v1/ocr` and needs none of
    this. Without a proxy base URL the SDK talks to the provider directly and the id is
    already correct, so it is returned unchanged.
    """
    if not api_base or "/" not in model:
        return model
    return f"{model.split('/', 1)[0]}/{model}"


def _document_payload(document: bytes) -> dict[str, object]:
    """The `document` mapping `litellm.ocr()` takes.

    Byte-for-byte the body `LiteLlmHost` (the Rust host) PUTs on the wire, so both
    paths hit the provider with the same request.
    """
    encoded = base64.b64encode(document).decode("ascii")
    return {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{encoded}",
    }


def litellm_provider(model: str, document: bytes, pages: list[int] | None) -> dict:
    """Call LiteLLM for one leg and normalise the reply for the core.

    `document` is the *whole* PDF and `pages` names the 0-indexed pages this leg wants
    -- the core does not narrow the bytes for you (see `OcrHost` in run.rs). Passing
    `pages` through lets the provider do the narrowing; a provider that cannot should
    call `doc_router.split_pdf()` and `doc_router.remap_pages()` itself so the returned
    indices stay original.

    `OCRResponse.pages` carries `index` and `markdown`, which is already the core's
    contract, so this is a field copy and not a translation.
    """
    import litellm

    api_base = os.environ.get("LITELLM_BASE_URL") or None
    extra: dict[str, object] = {} if pages is None else {"pages": pages}
    response = litellm.ocr(
        model=_sdk_model(model, api_base),
        document=_document_payload(document),
        api_key=_api_key(),
        api_base=api_base,
        **extra,
    )
    return {
        "pages": [
            {"index": page.index, "markdown": page.markdown or ""} for page in response.pages
        ]
    }


def stub_provider(model: str, document: bytes, pages: list[int] | None) -> dict:
    """Offline stand-in: no network, no key, right shape."""
    print(f"  -> stub provider: model={model} bytes={len(document)} pages={pages}")
    wanted = pages if pages is not None else [0]
    return {"pages": [{"index": i, "markdown": f"<{model} output for page {i}>"} for i in wanted]}


def choose_provider():
    """`litellm.ocr()` when a key is in the environment, else the stub."""
    if _api_key() is None:
        print(f"no key in {' / '.join(API_KEY_ENV_VARS)} -- using the offline stub\n")
        return stub_provider
    base = os.environ.get("LITELLM_BASE_URL") or "litellm's own default"
    print(f"calling litellm.ocr() against {base}\n")
    return litellm_provider


def pre_routing_hook(document: bytes, requested_model: str, provider) -> dict:
    """Return the LiteLLM `doc_route` metadata plus the merged document text."""
    # Cheap path: what would we do, and why? No provider call, never raises on a
    # non-PDF -- it just bypasses.
    decision = doc_router.decide(document, CONFIG, requested_model)
    print("decision:", json.dumps(decision["metadata"], indent=2, sort_keys=True))
    if decision["kind"] == "bypass":
        print(f"bypassing to {decision['model']}: {decision['reason']}")

    # Full path: split, extract locally, call the host for the OCR legs, merge in
    # original page order. Falls back to `requested_model` for the whole document if a
    # leg fails, recording `fallback_reason` in the metadata.
    return doc_router.run(document, CONFIG, provider, requested_model)


def main() -> int:
    default = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "mixed.pdf"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default
    outcome = pre_routing_hook(path.read_bytes(), "gpt-4o-mini", choose_provider())

    print("\nmetadata:", json.dumps(outcome["metadata"], indent=2, sort_keys=True))
    print(f"\nmerged {outcome['result']['pages_processed']} pages "
          f"via {outcome['result']['model']}:")
    for page in outcome["result"]["pages"]:
        head = " ".join(page["markdown"].split())[:60]
        print(f"  page {page['index']} [{page['model']}]: {head}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
