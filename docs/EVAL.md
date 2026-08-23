# Evaluation & Reliability Plan

> **Status:** design doc. Section §1 (partial-credit scoring + FABR) has a
> runnable implementation in [../eval/](../eval/) — currently demonstrated on
> 3 annotated samples; the full eval set below is the plan, not shipped data.

## 1. Eval Set Construction

**Ground truth**: Human-annotated extractions from **real rate confirmations** (PDFs/emails), not synthetic data. Each example = (raw_text, golden_json) where golden_json matches our output schema exactly.

**Sources**:
- 200 historical rate cons from our top 20 shippers/carriers (covers format diversity)
- 50 "pathological" cases: handwritten notes, scanned faxes, email threads with quoted history, conflicting numbers
- 25 adversarial: missing critical fields, ambiguous dates (3/4/26), totals that don't reconcile

**Total**: ~275 examples. Split: 200 dev / 75 holdout test. Stratified by shipper/carrier to detect format-specific regressions.

**Partial correctness scoring** (per-field, not all-or-nothing):
| Field | Weight | Scoring |
|-------|--------|---------|
| `load_id`, `origin`, `destination`, `pickup_date`, `total_rate` | 2.0 | Exact match = 2, null = 0, wrong = -1 (penalize hallucination) |
| `delivery_date`, `equipment_type` | 1.0 | Exact = 1, null = 0, wrong = -0.5 |
| `line_haul_rate`, `fuel_surcharge`, `weight_lbs`, `commodity` | 0.5 | Within 1% = 0.5, null = 0, wrong = -0.25 |

**Composite score** = Σ(field_score) / Σ(max_possible). Normalized to [0, 1].

---

## 2. The Metric That Matters: **False Auto-Book Rate (FABR)**

**Definition**: % of loads where confidence=`high` (auto-populate) but extraction has ≥1 critical error.

**Why not accuracy/F1?**
- Broker cost asymmetry:
  - **False negative** (flag for review): ~30 sec human time → **\$0.50**
  - **False positive** (wrong rate auto-books): carrier dispute, reputational damage, potential double-pay → **\$500–$5,000+**
- **FABR directly measures the expensive error**. Target: **< 0.5%** on holdout.

**Operational proxy**: Track `confidence=high` rate + human override rate in review queue. If override rate spikes → FABR likely rising.

---

## 3. Drift / Regression Detection

| Signal | Detection | Alert Threshold |
|--------|-----------|-----------------|
| **Format drift** | Cluster embeddings of incoming raw texts; flag new clusters >5% volume | New cluster >10% of daily volume |
| **Confidence distribution shift** | Monitor % high/med/low daily (7-day rolling) | `high`% drops >15% WoW |
| **Field-level null rate** | Track null rate per critical field | Any critical field null rate >20% (baseline ~5%) |
| **Totals conflict rate** | % of extractions with `_totals_conflict=true` | >8% (baseline ~3%) |
| **Model provider change** | Pin model version (`gpt-4o-mini-2024-07-18`); re-run eval set on any version bump | Composite score drop >0.05 |
| **Shipper-specific accuracy** | Join extraction → load → shipper; track per-shipper FABR proxy | Any shipper override rate >2x global median |

**Implementation**: Lightweight SQL + cron job running nightly. Slack alert to #ml-ops. No complex monitoring stack — debuggable at 2am.

---

## 4. Human-in-the-Loop (HITL)

**Where**: **Only on `confidence=medium`**. `high` → auto-populate. `low` → reject + log (broker sees "could not extract, manual entry required").

**UI Moment** (embedded in TMS load create flow):


┌─────────────────────────────────────────────────────┐ │ ⚠️ Review suggested fields (medium confidence) │ ├─────────────────────────────────────────────────────┤ │ Load ID: RC-2024-08921 ✓ verified │ │ Origin: Chicago, IL 60601 ✓ verified │ │ Destination: Atlanta, GA 30301 [EDIT] ← zip? │ │ Pickup: 2024-03-15 ✓ verified │ │ Total Rate: $3,150.00 [EDIT] ← check │ │ Equipment: Van ✓ verified │ ├─────────────────────────────────────────────────────┤ │ [Accept All] [Edit & Accept] [Reject → Manual]│ └─────────────────────────────────────────────────────┘



**Key UX decisions**:
- **Pre-fill everything** — human only corrects, never types from scratch
- **Highlight uncertain fields** (yellow) with *why*: "zip missing from source", "totals don't reconcile"
- **One-click accept** if all looks right — 90% of medium cases
- **Rejection feeds eval set** — every "Reject → Manual" becomes a new golden example

**No separate review queue** — embedded in the broker's existing flow. They don't context-switch.
