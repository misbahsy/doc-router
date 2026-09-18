# doc-router (Python bindings)

A [PyO3](https://pyo3.rs) extension module exposing [`doc-router`](../../README.md) to
Python: classify a PDF page by page, decide which pages need paid OCR, read the rest
locally, and merge the two halves back into one page-ordered result.

The point of the bindings is that a Python proxy — LiteLLM's, for instance — can drop its
own classifier, policy table, splitter and merge code and keep only the part that is
genuinely its job: calling the OCR provider.

The boundary is plain Python data: `bytes` in, `dict` out. No custom classes to learn, no
objects to keep alive.

## Install

Requires Python **3.10+** (built `abi3-py3.10`, so one wheel serves every later version)
and [maturin](https://www.maturin.rs/).

```sh
cd crates/doc-router-py
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest
maturin develop --release
```

```sh
python -m pytest -q
```

```console
...........................................                              [100%]
43 passed in 0.09s
```

`python -m pytest` is a **separate suite from `cargo test`** — nothing in the Rust test
run exercises this boundary, so a change here needs both.

To build a wheel instead of installing into the active environment:

```sh
maturin build --release      # wheel lands in ../../target/wheels/
```

The compiled cdylib is `doc_router._native`; `python/doc_router/__init__.py` re-exports it
so callers only ever `import doc_router`. Type stubs (`__init__.pyi`) and a `py.typed`
marker ship with the package.

## Use it

```python
import doc_router

config = {"tiers": {"local": "local_pdf/extract", "standard": "mistral-ocr"}}
document = open("scan.pdf", "rb").read()

# What would happen, without doing it. Never raises for unroutable input.
decision = doc_router.decide(document, config, "gpt-4o-mini")
decision["kind"]                       # "route" or "bypass"
decision["metadata"]["reason"]         # e.g. "mixed_split", or "not_pdf" for a bypass

def host(model, document, pages):
    """Called once per non-local leg.

    `pages` is a 0-indexed subset of the ORIGINAL document, or None for all of it.
    Return an OcrResult-shaped dict: {"pages": [{"index": int, "markdown": str}, ...]}.
    `model`, `pages_processed`, `doc_size_bytes` and each page's `model` may be omitted.
    """
    return call_your_provider(model, document, pages)

outcome = doc_router.run(document, config, host, "gpt-4o-mini")
outcome["result"]["pages"]   # merged, in original page order
outcome["metadata"]          # the `doc_route` metadata dict to log
outcome["decision"]          # what was planned, before any fallback
```

`crates/doc-router-py/examples/litellm_adapter.py` is a runnable sketch of the whole
pre-routing hook, calling OCR through the LiteLLM SDK. With no key set it falls back to an
offline stub, so it runs on a fresh checkout:

```console
$ python examples/litellm_adapter.py
no key in LITELLM_API_KEY / LITELLM_PROXY_API_KEY -- using the offline stub
...
  -> stub provider: model=mistral/mistral-ocr-latest bytes=12161 pages=[1, 3]

merged 4 pages via local_pdf/extract,mistral/mistral-ocr-latest:
  page 0 [local_pdf/extract]: MIXED DOC PAGE ONE (text) Line 1: The quick brown fox jumps ...
  page 1 [mistral/mistral-ocr-latest]: <mistral/mistral-ocr-latest output for page 1>...
  page 2 [local_pdf/extract]: MIXED DOC PAGE THREE (text) Line 1: The quick brown fox jump...
  page 3 [mistral/mistral-ocr-latest]: <mistral/mistral-ocr-latest output for page 3>...
```

> Calling a real gateway through the SDK hits a model-id prefix gotcha that costs people
> an afternoon. It is written out in full in the guide under
> [Choosing an OCR backend](../../docs/GUIDE.md#the-prefix-gotcha-python-sdk-against-a-gateway),
> and in `_sdk_model`'s docstring in that example.

## API

| function | returns |
|----------|---------|
| `classify(data)` | classification dict: `pdf_type`, `confidence`, `page_count`, `pages_needing_ocr`, `is_complex_layout`, `has_encoding_issues`, `ocr_reasons`, `classify_ms` |
| `is_pdf(data)` | `bool`, cheap header check |
| `plan_route(classification, config)` | the plan for an already-classified document |
| `decide(data, config, default_model=None)` | `{"kind": "route"\|"bypass", …, "metadata": {…}}`. Raises `ConfigError` only |
| `extract_local(data, pages=None)` | OcrResult dict read from the PDF's own text layer. No network |
| `split_pdf(data, pages)` | `bytes` — a new PDF holding exactly those 0-indexed pages, in the order given |
| `remap_pages(result, pages)` | put original indices back on a result that came from a split subset |
| `merge(legs)` | merge `[(leg, result), …]` into one page-ordered OcrResult |
| `run(data, config, host, default_model=None)` | `{"result", "metadata", "decision"}` |
| `validate_config(config)` | the config normalised with every default filled in. Unknown keys rejected |

Also exported: `DEFAULTS` (the default config as a dict), `LOCAL_MODEL`
(`"local_pdf/extract"`), `__version__`.

### Errors

`DocRouterError` is the base class; catch it to catch everything.

| exception | when |
|-----------|------|
| `PdfError` | the bytes are not a readable PDF, or a PDF operation failed |
| `ConfigError` | the `doc_router_config` block is not usable as written |
| `HostError` | the host callable failed, or returned something unusable |

## Five things worth knowing

* **Pages are 0-indexed everywhere.** `pages_needing_ocr`, each leg's `pages`, the `pages`
  argument of `extract_local` / `split_pdf` / your host callable, and each returned page's
  `index` all refer to the **original** document.
* **Nothing here calls a network.** `run` hands OCR legs to your host callable; local
  text-layer legs are read in-process.
* **`decide` never raises for unroutable input.** A non-PDF, an oversized document or a
  classifier failure comes back as `{"kind": "bypass", …}`, so a caller can always fall
  back to its default model.
* **Config is LiteLLM's `doc_router_config` block verbatim**, and unknown keys are
  rejected, so a typo surfaces immediately instead of silently taking a default.
* **The GIL is released around the work.** `run` executes legs on scoped Rust threads, so
  your host callable may be invoked from a thread other than the caller's; the GIL is
  re-acquired around each call. (Holding it across the core call would deadlock, which is
  why `py.detach` wraps it.)

### Fallback semantics

An exception raised by your host fails that leg, which triggers **one** whole-document
retry on `default_model`. On success the metadata gains `fallback_reason = "leg_failed"`
and `routed_model` becomes the model that actually ran, so the metadata never claims bytes
went somewhere they did not. If the retry also fails, `HostError` is raised.

## A note for contributors

`classification_from_py` fills defaults for `classify_ms`, `is_complex_layout`,
`has_encoding_issues` and `ocr_reasons`, because the golden fixtures in `tests/golden/`
were recorded by an older Python implementation that had no such fields. **If you add a
field to the Rust `Classification`, add its default there too**, or every golden replay
stops deserialising — and `cargo test` will not tell you, because this boundary is only
covered by pytest.

`Error::LegFailed`'s `Display` deliberately folds in its source so the message is flat and
single-line; these bindings surface that message directly, so changing it would change the
Python-visible error text.
