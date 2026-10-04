# Evaluation results

Run date: 2026-10-04. Every number below is copied from the output of
`eval/evaluate.py`; nothing is estimated.

The raw per-record output behind the tables is in `eval/last_run.json`
(original set, after the fixes) and `eval/heldout_run.json` (held-out set).

## What was measured

- **Dataset:** `eval/golden.jsonl`, 16 hand-crafted records (3 samples plus
  13 edge-case documents in `eval/corpus/`), 12 scored fields each, 192
  fields in total. This is a small demonstration set, **not a benchmark**.
- **Model:** `openai/gpt-oss-120b` on Groq (OpenAI-compatible endpoint),
  temperature 0, strict JSON schema.
- **Tests:** `pytest` reports 58 passed.

`llama-3.3-70b-versatile` was the intended model but Groq returned
`model_not_found` for it on this account, so `openai/gpt-oss-120b` was used.

## Commands

```bash
pip install -e ".[dev]"
pytest
```

`.env`:

```
OPENAI_API_KEY=<Groq key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_MODEL=openai/gpt-oss-120b
```

```bash
python eval/evaluate.py --json --delay 20
```

`--delay 20` paces the run for Groq's free tier (8,000 tokens per minute).
Without it most records fail with HTTP 429 and are scored as empty.

## Results

| Metric | Result |
| --- | --- |
| Average partial-credit (composite) score | **0.878** |
| FABR proxy (high-confidence record with a wrong critical field) | **0.0%** (0 of 12) |
| Fields correct | 155 of 192 (80.7%) |
| Fields correctly left null (gold is also null) | 35 of 192 (18.2%) |
| Fields wrong | 1 of 192 (0.5%) |
| Fields missed (null, gold has a value) | 1 of 192 (0.5%) |
| Accuracy on fields the model filled in | 155 of 156 (99.4%) |
| Confidence: high / medium / low | 12 / 1 / 3 |

The two remaining errors:

- `LD64408` `commodity`: extracted "Ceramics"; the gold annotation is null
  because the document lists conflicting commodities per stop.
- `COR008` `line_haul_rate`: returned null for a zero-rate load where the
  gold value is 0.0.

The composite score is below 1.0 mainly because a correctly-null field earns
0 points under the rubric in `docs/EVAL.md`, not because of wrong values.

## Before and after the fixes made on this date

| | Before | After |
| --- | --- | --- |
| Average composite score | 0.841 | 0.878 |
| FABR proxy | 16.7% (2 of 12) | 0.0% (0 of 12) |
| Wrong fields | 4 | 1 |

Two changes produced this:

1. **Scorer fix** (`eval/evaluate.py`): city/state comparison is now
   case-insensitive. "ONTARIO, CA" was being marked wrong against the gold
   "Ontario, CA" in `COR013`.
2. **Prompt rule** (`src/ratecon_extract/extractor.py`): for multi-stop
   loads, origin is the first pickup and destination is the final delivery
   stop. In `COR004` the model had returned the first of two drops.

## Held-out check (documents never used for tuning)

`eval/heldout.jsonl` and `eval/heldout/` were written after the fixes: 16
new documents (multi-drop, multi-pickup, forwarded email, missing fields,
month-name dates, itemized accessorials, unusual equipment, conflicting
totals, unstated values, M/D/YY dates, terse all-caps, prose). No pipeline
code or gold answer was changed after this run.

```bash
python eval/evaluate.py eval/heldout.jsonl --json --delay 20
```

| Metric | Held-out set | Original set |
| --- | --- | --- |
| Average composite score | **0.919** | 0.878 |
| FABR proxy | **0.0%** (0 of 14) | 0.0% (0 of 12) |
| Fields wrong | 0 of 192 | 1 of 192 |
| Fields missed | 0 of 192 | 1 of 192 |
| Fields correct | 167 of 192 | 155 of 192 |
| Fields correctly left null | 25 of 192 | 35 of 192 |
| Confidence: high / medium / low | 14 / 1 / 1 | 12 / 1 / 3 |

The conflicting-totals document was rated low confidence and the M/D/YY
document medium, as intended.

## Caveats

- Both fixes were found by looking at failures on the original 16-record
  set, so its "after" numbers are optimistic. The held-out set above is the
  fairer measure.
- The held-out documents were written by the same person who made the
  fixes, and they are cleaner than real freight paperwork. This shows the
  fixes generalise beyond the original 16; it is not production accuracy.
- 12 high-confidence records is far too few to demonstrate the <0.5% FABR
  target in `docs/EVAL.md`; 0 of 12 only shows no failure was observed.
- LLM output varies slightly between runs even at temperature 0. An
  intermediate run got one pickup date wrong on `LD64408` (a document with
  two different pickup dates) while a stricter date rule was in the prompt;
  that rule was removed.
