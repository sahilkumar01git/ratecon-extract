# RateCon Extract

LLM-powered pipeline that turns unstructured freight rate confirmations into validated, schema-checked JSON, with confidence scoring designed around the real cost of getting it wrong.

**Results (live, 4 October 2026):** on a 16-document held-out set, 0 wrong fields out of 192, composite score 0.92, and no high-confidence record with a wrong critical field. See [`RESULTS.md`](RESULTS.md) for full numbers, commands and caveats.

## Problem

Rate confirmations aren't standardized — every carrier and shipper formats them differently, and they carry financially sensitive data. The failure modes aren't symmetric:

- A broker who catches a bad extraction loses ~30 seconds.
- A broker who doesn't can eat a **$500–$5,000+ carrier dispute** from a wrong auto-booked rate.

So the system is built around that asymmetry: hallucinated values must cost more than honest nulls, and anything uncertain must be routed to a human instead of auto-populated.

## Solution

An LLM does what it's good at — reading messy text — while deterministic Python does everything a few lines of code can do more reliably:

| Layer | Responsibility |
|-------|----------------|
| **LLM** (any OpenAI-compatible model via `OPENAI_MODEL`) | Field extraction only, under a strict JSON schema |
| **Strict schema rewrite** | Makes the Pydantic schema legal for OpenAI `strict: true` (full `required`, closed objects, rejected keywords stripped) |
| **Pydantic validation** | Types, ranges, enum coercion — malformed output degrades instead of crashing |
| **Business validation** | Missing-critical detection · date-ambiguity detection on raw source text · totals reconciliation (`line_haul + fuel + itemized accessorials ≈ total`) · equipment-synonym normalization |
| **Confidence checklist** | Documented weights over the validation signals → HIGH / MEDIUM / LOW |
| **Routing** | HIGH = auto-populate · MEDIUM = human review · LOW = reject, manual entry |

### Normal path

```
        Rate confirmation (raw text)
                    │
                    ▼
        ┌────────────────────────┐
        │   LLM structured       │  strict JSON schema;
        │   extraction           │  temperature 0
        └───────────┬────────────┘
                    ▼
        ┌────────────────────────┐
        │  Pydantic validation   │  types, enums, ranges
        └───────────┬────────────┘
                    ▼
     ┌──────────────┼──────────────────┐
     ▼              ▼                  ▼
 missing       date ambiguity     totals reconcile?
 critical      (raw "3/4/26")     line_haul + fuel
 fields                           + accessorials vs total
     └──────────────┼──────────────────┘
                    ▼
          Confidence checklist
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        HIGH      MEDIUM      LOW
          │         │         │
     auto-fill   review    reject
```

### Failure path

LLM and validation failures are handled gracefully — converted into explicit, low-confidence results with human-readable reasons (filesystem errors or genuine bugs can still surface as normal exceptions):

```
LLM call ──┬── timeout / connection / 429 / 5xx ──→ backoff retry (1s, 2s)
           ├── auth / bad request ────────────────→ fail fast (no pointless retries)
           ├── malformed JSON ────────────────────→ retry, then empty extraction
           └── valid JSON but bad values ─────────→ Pydantic coercion / OTHER fallback
                        │
                        ▼
                 business validation
                        │
        ┌───────────────┼────────────────┐
        ▼               ▼                ▼
  critical field   totals conflict   ambiguous date
  missing          flagged w/ $diff  flagged
        └───────────────┼────────────────┘
                        ▼
        LOW/MEDIUM confidence + human-readable
        validation errors[] explaining why
```

## Example

Input (`samples/LD64392.txt`):

```
Reference ID LD64392 Booked On 28-Jul-2026 Rate Con Date 28-Jul-2026 Pickup Date 30-Jul-2026
...
1 Pickup ... Illinois State Police, 100 W Randolph St, Chicago, IL 60601, USA ...
2 Drop ... New University ..., 50 W 4th Street, New York, NY 10012, USA ...
Rate Breakdown: Base Carrier Rate 50.00 USD Total 50.00 USD
```

Output:

```json
{
  "load_id": "LD64392",
  "origin": { "city": "Chicago", "state": "IL", "zip": "60601" },
  "destination": { "city": "New York", "state": "NY", "zip": "10012" },
  "pickup_date": "2026-07-30",
  "delivery_date": "2026-08-01",
  "equipment_type": "flatbed",
  "line_haul_rate": 50.0,
  "fuel_surcharge": null,
  "accessorials": null,
  "total_rate": 50.0,
  "weight_lbs": 182,
  "commodity": "Ceramics",
  "confidence": "high"
}
```

Money is carried as `Decimal` through the entire business-logic layer and converted to float only at the JSON serialization boundary. Extra charges (carrier charge, lumper, detention, ...) are **itemized in `accessorials[]`, never folded into `line_haul_rate`** — the financial model is:

```
line_haul_rate + fuel_surcharge + Σ accessorials ≈ total_rate
```

A con like `Base $500 + Carrier Charge $200 = Total $700` reconciles honestly, with the carrier charge visible as its own line item.

## Confidence scoring

A documented weighted checklist over the validation signals above (weights sum to 1.0):

| Signal | Effect |
|--------|--------|
| All critical fields present | +0.30 / missing any: −0.40 |
| Total present & reconciles | +0.25 / conflict: −0.30 |
| Both dates parsed, unambiguous | +0.15 / ambiguous: −0.20 |
| Equipment mapped to a known type | +0.10 |
| Optional fields present (0–4) | up to +0.10 |
| First-attempt success | +0.10 / each retry: −0.15 |

Buckets: **HIGH ≥ 0.75** → auto-populate · **MEDIUM ≥ 0.45** → review · **LOW** → reject.

These thresholds are expert judgment, not statistically calibrated against labeled outcomes — see [Limitations](#limitations).

## Evaluation

[`eval/evaluate.py`](eval/README.md) implements the partial-credit rubric and the **FABR proxy** — share of high-confidence extractions containing a wrong critical field — defined in [`docs/EVAL.md`](docs/EVAL.md). FABR, not accuracy/F1, is the metric tied to real dollars; the target on a proper holdout set is < 0.5%. The bundled golden set — **16 annotated records: 3 supplied samples + 13 crafted edge cases** — exercises the scoring machinery end-to-end but is crafted data, not a quality benchmark.

**Live results** (4 October 2026, `openai/gpt-oss-120b` on Groq; details in [`RESULTS.md`](RESULTS.md)):

| Metric | Held-out set (16 docs) | Golden set (16 docs) |
|---|---:|---:|
| Average composite score | 0.919 | 0.878 |
| FABR proxy | 0.0% (0 of 14) | 0.0% (0 of 12) |
| Fields wrong / missed | 0 / 0 of 192 | 1 / 1 of 192 |
| Confidence: high / medium / low | 14 / 1 / 1 | 12 / 1 / 3 |

The held-out documents were written after the last fixes and never used for tuning. Both sets are small and crafted, so these are demo results, not production accuracy.

Reliability behaviors are covered by offline tests with scripted mock LLMs (malformed JSON, invalid enums, missing fields, conflicting rates, timeouts, rate limits, auth failures): `pytest` makes **zero API calls**.

## Quick start

```bash
# Install (Python 3.11+)
pip install -e ".[dev]"

# Configure
cp .env.example .env
# edit .env → OPENAI_API_KEY=...

# For reproducible evaluation, PIN the exact model version (docs/EVAL.md §3)
# rather than relying on the moving alias:
OPENAI_MODEL=gpt-4o-mini-2024-07-18

# Extract one file (--verbose adds latency/attempts/validation diagnostics)
ratecon-extract extract samples/LD64392.txt --verbose
# or: python -m ratecon_extract.cli extract samples/LD64392.txt

# Batch a directory
ratecon-extract batch samples/

# Offline test suite (no API key needed, ~1s)
pytest

# Live smoke tests + evaluation (calls your configured provider)
OPENAI_API_KEY=sk-... pytest tests/test_extraction.py -v -s
python eval/evaluate.py
python eval/evaluate.py eval/heldout.jsonl   # held-out set
```

**Any OpenAI-compatible provider works** — set `OPENAI_BASE_URL` + `OPENAI_MODEL` in `.env` and the whole pipeline (CLI, tests, eval) runs against it. One tested example, Groq's free tier:

```ini
OPENAI_API_KEY=gsk_...
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=llama-3.3-70b-versatile
```

The extraction schema is sent fully inlined (no `$defs`/`$ref`) for portability, and if a provider rejects strict structured outputs the request automatically degrades to schema-guided JSON — Pydantic validation remains the safety net either way. Expect smaller models to score worse on the eval; that's what the harness is there to measure.

Logs go to stderr (model, attempts, latency, validation issues — never document contents); JSON goes to stdout.

## Repo structure

```
src/ratecon_extract/
    models.py               Schemas: Extraction (LLM-facing), Output (contract), ValidationResult
    validators.py           Deterministic rules: dates, totals, equipment, missing fields
    confidence.py           Weighted checklist → HIGH/MEDIUM/LOW
    extractor.py            OpenAI client, strict-schema rewrite, retries/backoff, pipeline
    cli.py                  Typer CLI: extract/batch, logging, latency
tests/
    test_validation.py      Offline unit tests for the validation layer
    test_confidence.py      Offline unit tests for scoring buckets/penalties
    test_pipeline.py        Scripted mock-LLM failure-mode tests (zero API calls)
    test_extraction.py      Live smoke tests (skipped without OPENAI_API_KEY)
eval/
    evaluate.py             Partial-credit scoring + FABR proxy vs golden annotations
    golden.jsonl            16 annotated records (3 samples + 13 crafted corpus cases)
    corpus/                 Crafted edge-case rate confirmations for the golden set
    heldout.jsonl           16 held-out annotated records, written after tuning
    heldout/                Held-out rate confirmations
docs/
    EVAL.md                 Eval-set design, FABR metric, drift detection, HITL UX
    CARRIER_MATCH_DESIGN.md System design proposal — not implemented
samples/                    Example rate confirmations
```

## Limitations

Stated plainly:

- **Plain-text in.** PDF/email/fax ingestion and OCR are upstream concerns, intentionally outside this exercise — the assignment's contract is raw text → validated JSON.
- **The confidence checklist is expert-weighted, not calibrated.** Thresholds encode judgment about cost asymmetry; calibrating them against labeled outcomes (the FABR eval in docs/EVAL.md) is future work.
- **The eval sets are small.** 16 golden plus 16 held-out documents, all crafted and cleaner than real freight paperwork. They show the pipeline works on varied cases; they are not production accuracy.
- **Date-ambiguity heuristic assumes US conventions.** `M/D/YY` slash formats trigger it; month-name and ISO dates are treated as unambiguous by design.
- **Single-provider coupling.** Model/timeouts are env-configurable, but prompts and the strict-schema path assume OpenAI-compatible structured outputs.
- **Totals reconciliation trusts itemization.** The check sums whatever the LLM captured (`line_haul + fuel + accessorials` vs printed total). If a source document itself omits a charge from its printed total, or the model misses an itemized line, reconciliation can flag a false conflict — flagged for review, not silently accepted.

## License

MIT — see [LICENSE](LICENSE).
