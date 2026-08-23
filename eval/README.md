# Eval Harness

Scores pipeline output against annotated **golden extractions** using the
partial-credit rubric from [../docs/EVAL.md](../docs/EVAL.md) §1 — per-field
weighted scoring with explicit penalties for hallucinated values, plus the
**FABR proxy** (high-confidence extractions that contain a wrong critical
field — the expensive mistake this whole system exists to prevent).

## Usage

```bash
cp .env.example .env          # evaluation calls your configured provider
python eval/evaluate.py                      # bundled golden set
python eval/evaluate.py my_golden.jsonl      # your annotations
python eval/evaluate.py --json               # machine-readable summary
```

## Golden file format

JSONL — one record per line. `source` resolves relative to the JSONL file;
`golden` uses the same field names as pipeline output (`pickup_date` /
`delivery_date` are the resolved ISO dates, not raw text):

```json
{"id": "COR007", "source": "corpus/COR007_carrier_charge.txt", "category": "carrier_charge", "golden": {...}}
```

## Coverage

The bundled set contains **16 annotated documents** (3 samples + 13 crafted
corpus cases) covering: normal confirmations · missing fields · multiple
pickups · multiple drops · all-in rates · fuel surcharge · itemized carrier
charges/accessorials · zero-dollar rates · ambiguous dates (`3/4/26`) ·
malformed values · unusual equipment · conflicting totals · noisy email
formatting.

## Scoring rubric

| Tier | Fields | Exact | Null | Wrong |
|------|--------|-------|------|-------|
| Critical (×2.0) | `load_id`, `origin`, `destination`, `pickup_date`, `total_rate` | +2.0 | 0 | −1.0 |
| Secondary (×1.0) | `delivery_date`, `equipment_type`, `accessorials` | +1.0 | 0 | −0.5 |
| Tertiary (×0.5) | `line_haul_rate`, `fuel_surcharge`, `weight_lbs`, `commodity` | +0.5 (within 1%) | 0 | −0.25 |

- Money compares to ~0.1% tolerance; weight to 1%; `accessorials` must match
  as an itemized set (description case-insensitive, amount to the cent).
- Nulls score 0 by design — including "correctly null" fields. This follows
  the documented rubric and applies uniformly, so relative comparisons are
  unaffected; it does slightly deflate absolute composites.
- **Composite** = Σ(field scores) / Σ(max possible), clamped to [0, 1].
- **FABR proxy** = share of `confidence=high` outputs with ≥1 wrong critical
  field. Target on a real holdout: **< 0.5%** (docs/EVAL.md §2).

## Honest status

The bundled corpus was **crafted to cover edge-case categories — it is not
real production freight data**, so its numbers demonstrate methodology, not
model quality claims. The eval-set design needed for meaningful numbers
(~275 human-annotated real rate cons; 200 dev / 75 holdout) is specified in
[../docs/EVAL.md](../docs/EVAL.md).
