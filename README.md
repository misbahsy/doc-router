# doc-router

**Don't pay to OCR a page that already has text on it.**

Most PDFs are not all-scan or all-text. A contract has two signature pages scanned in the
middle. A report's appendix was photocopied. An invoice batch has every third document off
a flatbed. Hand the whole file to a hosted OCR model and you pay for every page — including
the ones you could have read for free, instantly, with no network call.

`doc-router` looks at a PDF **page by page**, decides which pages have a usable text layer
and which genuinely need OCR, then acts on that: text pages are extracted locally in-process,
only the rest go to your OCR provider, and the two halves merge back into one page-ordered
result. It's Rust, and the core library never touches a network.

---

## What it bought us

Measured 2026-09-17 on 19 documents / 155 pages, `mistral-ocr-latest` through a live LiteLLM
gateway, 3 runs per document:

| | OCR every page | routed |
|---|---|---|
| pages billed | 155 | 87 |
| API requests | 19 | 13 |
| wall clock | 35,578 ms | 20,666 ms — **1.72x faster** |
| bill at $2.00/1k pages | $0.3100 | $0.1783 — **1.74x cheaper** |
| pages that needed OCR and didn't get it | 0 | 9 — vs **28** for a rules-based judge |

The judge costs **2.5% of the OCR bill it authorises**. The full write-up, including why a
cheaper bill is sometimes a *worse* result, is here: **[the story ↗](https://claude.ai/artifact/Udne2Jfubah1kCSQ7G8TdN)**.

## What's under the hood

- **[Jev](https://docs.typesafe.ai)** (TypeSafe System One) — the page judge. Given the
  evidence for a page, it answers one question: does this page need OCR? It catches what
  rules can't — a scan carrying a bad pre-existing OCR layer, a page whose only text is a
  watermark, a broken `ToUnicode` map. Swappable: it's one trait, and a zero-dependency
  local heuristic ships as the default.
- **[LiteLLM](https://docs.litellm.ai)** — the gateway. One endpoint, one key, any OCR
  provider behind it. Point `--base-url` at a proxy you already run, at a local gateway, or
  at a hosted one. Nothing in this repo is pinned to a vendor.
- **[pdf-inspector](https://crates.io/crates/pdf-inspector)** — per-page structural
  analysis: is there a text layer, how much, tables, columns, encoding damage.
- **[lopdf](https://crates.io/crates/lopdf)** — page splitting, so a subset of pages can be
  shipped to providers that ignore a `pages` parameter.
- **Rust** throughout, with **PyO3** bindings if you'd rather call it from Python.

It's a port of LiteLLM's Python document router, with the same classification, decision
table and metadata keys — a document routed by either lands on the same model.

---

## Run it

Needs Rust 1.88+. `cargo build --release`.

**One PDF, no keys, no network** — what *would* be routed where:

```bash
cargo run -q -p doc-router-cli -- classify ~/Documents/contract.pdf
```

**One PDF, for real** — local extraction plus OCR for the pages that need it:

```bash
export LITELLM_API_KEY=sk-...
cargo run -q -p doc-router-cli -- run \
  --base-url https://your-gateway --model mistral/mistral-ocr-latest \
  --judge jev ~/Documents/contract.pdf
```

**A whole directory**, one JSON result per document:

```bash
for f in ~/Documents/pdfs/*.pdf; do
  cargo run -q --release -p doc-router-cli -- run \
    --base-url https://your-gateway --model mistral/mistral-ocr-latest \
    --judge jev "$f" > "${f%.pdf}.json"
done
```

**Score it on your own corpus** — the numbers above, against your documents:

```bash
cargo run -q --release -p doc-router-bench -- \
  --corpus ~/my-corpus/manifest.json \
  --judge heuristic --judge jev \
  --cost-per-page 0.002 --cost-per-million-input-tokens 0.042
```

Keys go in a gitignored `.env` (`LITELLM_API_KEY`, `TYPESAFE_API_KEY`) or the environment.
With no Jev key it falls back to the local heuristic and says so.

## Hand it to an agent

Paste this into Claude Code or any coding agent, from the root of your own project:

```text
Use the doc-router repo at <path> to cut my OCR bill.

It's a Rust CLI that decides page by page which PDF pages actually need OCR, extracts the
rest locally, and merges the result. Read its docs/GUIDE.md first.

1. Build it: cargo build --release
2. Run `classify` over the PDFs in <my pdf directory>, with no keys, and tell me what
   fraction of pages would skip OCR entirely.
3. If that fraction is worth having, wire `doc-router-cli run` into <my pipeline> in place
   of my current whole-document OCR call. My OCR provider is behind LiteLLM at <base url>;
   the key is in $LITELLM_API_KEY. Use --judge jev if $TYPESAFE_API_KEY is set, otherwise
   the default heuristic.
4. Check the output is page-ordered and that no page is missing, then show me the
   before/after page count and cost.

Don't change the doc-router repo itself. If a page is misrouted, report it rather than
tuning thresholds to fit my documents.
```

---

## More

- **[docs/GUIDE.md](docs/GUIDE.md)** — the full manual: every flag, env var and judge, the
  benchmark report format, the architecture, troubleshooting.
- **[docs/SPEC.md](docs/SPEC.md)** — the routing spec and decision table.
- **[The story ↗](https://claude.ai/artifact/Udne2Jfubah1kCSQ7G8TdN)** — the measured
  results in full, and the methodology behind them.
- [LiteLLM docs](https://docs.litellm.ai) · [TypeSafe / Jev docs](https://docs.typesafe.ai)
  · [Mistral OCR](https://docs.mistral.ai/capabilities/OCR/basic_ocr/)

MIT.
