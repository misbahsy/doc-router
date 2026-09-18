# doc-router-jev

A [`PageJudge`](../doc-router/src/judge.rs) for [`doc-router`](../../README.md) backed by
TypeSafe's hosted **System One** ("Jev") model.

## Why it exists

The default judge is structural: it asks pdf-inspector whether a page carries text
operators, and believes the answer. That is correct almost all of the time, and wrong in
exactly the cases that cost the most money and the most silence:

* a scanned page that already carries a **bad pre-existing OCR layer** — structurally it
  is a text page, so it routes to the free local extractor and you get garbage;
* a page whose only text is a **watermark, header or footer** under a full-page image —
  structurally a text page, semantically a scan;
* a **CID font with a broken `ToUnicode` map** — the text extracts, and it is mojibake.

In all three the text is *there*. It just doesn't mean anything, and no amount of
structural inspection can tell. `JevJudge` sends the extracted text to a model and asks
the question directly: does this text faithfully represent what is printed on the page?

It is a separate crate so the core stays free of an HTTP client. `doc-router` has no
network dependency and should not grow one; `PageJudge` is the seam that keeps it that
way.

## Use it

```toml
[dependencies]
doc-router = { path = "../doc-router" }
doc-router-jev = { path = "../doc-router-jev" }
```

```rust
use doc_router::classify_with;
use doc_router_jev::{JevJudge, JevMode};

let judge = JevJudge::from_env()           // None when no key is set
    .expect("set TYPESAFE_API_KEY")
    .with_mode(JevMode::Gated)             // call Jev only when the evidence is ambiguous
    .with_strict(true);                    // a failed call is an error, not a fallback

let classification = classify_with(&pdf_bytes, &judge)?;
# Ok::<(), doc_router::Error>(())
```

From the benchmark harness, which registers it under two names, no code needed:

```sh
export TYPESAFE_API_KEY=...
cargo run -q -p doc-router-bench -- --judge jev --judge jev_gated
```

With no key set the harness says so and carries on with the baseline; it does not fail:

```console
skipped judge `jev`: no API key in the environment: set TYPESAFE_API_KEY (or JEV_API_KEY), or put it in a .env file at the workspace root
```

## Configuration

`JevJudge::from_env()` returns `None` unless a key is set. A blank or whitespace-only
value counts as unset.

| variable | purpose | default |
|----------|---------|---------|
| `TYPESAFE_API_KEY` | the API key. Checked first | — |
| `JEV_API_KEY` | the API key. Checked second | — |
| `TYPESAFE_BASE_URL` | API origin. `/v1/systemone` is appended and is **not** configurable | `https://api.typesafe.ai` |
| `TYPESAFE_MODEL` | model id | `jev-latest` |

Everything else is a builder method: `with_base_url`, `with_model`, `with_timeout`,
`with_mode`, `with_threshold`, `with_strict`, `with_failure_threshold`,
`with_cool_off_seconds`.

Defaults, all public constants: timeout `30.0` s, threshold `0.5`, failure threshold `3`,
cool-off `30.0` s.

## Modes

| mode | judge name | behaviour |
|------|------------|-----------|
| `JevMode::Always` | `jev` | every page goes to the model |
| `JevMode::Gated` | `jev_gated` | Jev runs only when `is_ambiguous(evidence)`, otherwise the structural answer stands |

`is_ambiguous` is public, and small on purpose. It fires when any page has
`has_encoding_issues`, or the inspector flagged *some but not all* pages, or a flagged
page carries no reason at all. It deliberately does **not** fire on `has_tables` or
`has_columns`: a table is a layout problem, not a "is this text real" problem, and paying
a model to look at every document with a table is how gating stops saving anything.

A page the gate skips gets the reason `jev_not_escalated`, so a verdict never claims Jev
looked at a page it did not.

## The wire protocol

`wire.rs` is deliberately free of HTTP and of `JevJudge`: it turns `&[PageEvidence]` into
request bodies and turns a response back into a probability for a named page, which is
what makes the chunker and the answer matcher testable without a socket.

One request carries a `state` (the pages) and a `questions` map with **one entry per
page**, keyed `page_<n>`. Answers come back under the same keys, which is why a whole
document can ride in one call:

```json
{
  "model": "jev-latest",
  "state": [
    {
      "page": 0,
      "text": "…",
      "text_chars": 1841,
      "text_truncated": false,
      "inspector_flagged": false,
      "inspector_reasons": [],
      "has_tables": false,
      "has_columns": false,
      "has_encoding_issues": false
    }
  ],
  "questions": {
    "page_0": { "type": "noul", "instructions": "…", "criteria": { "true": "…", "false": "…" } }
  }
}
```

A `noul` question answers with a single calibrated probability in 0..1 — the model's
belief that the statement is true — which is exactly the shape `PageVerdict::confidence`
wants, and exactly what the structural judge has none of.

Two decisions worth knowing:

* **The statement is phrased so that true means "needs OCR"**, matching the direction
  `PageVerdict::needs_ocr` reads. Flipping it here would mean flipping it back in the
  judge: twice as many places to get a `1.0 -` wrong.
* **Answers are never read positionally.** `answers` is a JSON object; nothing promises
  key order, and long documents are split across several requests, so "the third answer"
  is not a page number under any reading. `noul_for(page)` looks the key up. A key that is
  absent, answered with the wrong question type, or carrying a non-number is a
  `ProtocolError` — never a quietly-defaulted `false`. A judge that invents `false` for a
  page the vendor did not answer is a judge that silently routes scans to the local text
  extractor, which is the failure this crate exists to catch.

### Limits and chunking

| constant | value | what it bounds |
|----------|-------|----------------|
| `MAX_TEXT_CHARS` | `2000` | characters of one page's text sent. A clipped page sets `text_truncated`, so the model is never told a fragment is the whole page |
| `MAX_PAGES_PER_REQUEST` | `50` | pages per request |
| `MAX_REQUEST_BYTES` | `256 * 1024` | serialised request size |

`chunks()` is greedy and bounded on both axes. The size bound is measured by **actually
serialising** each page rather than estimating from the character cap, because
`MAX_TEXT_CHARS` is a character cap and a page of CJK or of mojibake costs several bytes
per character. One oversized page still gets its own request rather than an infinite
sequence of empty ones.

Chunking is invisible to the rest of the crate: page 73 is `page_73` whether it rides in
the first request or the fourth.

## Failure behaviour

**Strict** (`with_strict(true)`) turns any call failure into an `Error`. The benchmark
registry builds the judge strict on purpose: a silent fallback would report the
heuristic's score under Jev's name.

**Non-strict** (the default) falls back to the structural verdict and records *why* in the
reason, so a fallback is always visible in the output rather than being indistinguishable
from a real answer:

| reason | meaning |
|--------|---------|
| `jev_needs_ocr` | Jev answered, above the threshold |
| `jev_clear` | Jev answered, below the threshold |
| `jev_not_escalated` | gated mode, evidence unambiguous, Jev never called |
| `jev_fallback_timeout` | the call timed out |
| `jev_fallback_http` | transport or non-2xx status |
| `jev_fallback_protocol` | 2xx with a body this crate cannot read |
| `jev_fallback_breaker_open` | the circuit breaker was open; no call was made |

A **circuit breaker** opens after 3 consecutive failures and stays open for 30 seconds, so
a burst of 429s degrades to structural verdicts instead of hammering the API.
`calls()` returns a `CallRecord` per attempt for inspection.

## Tests

```sh
cargo test -p doc-router-jev
```

The suite runs against an `httpmock` server — the wire format, chunking, the answer
matcher, the fallback reasons and the breaker. No test calls the real API, and none needs
a key.
