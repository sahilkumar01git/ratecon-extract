# Carrier Match — Design Sketch

> **Status: system design proposal — NOT implemented.** This document is a
> companion architecture exercise covering data sources, LLM-vs-classical-ML
> tradeoffs, cold-start strategy, and cost/latency estimates. There is no
> carrier-ranking code in this repository; the working code is the rate
> confirmation extraction pipeline in `src/ratecon_extract/`.

## 1. Data Sources & Features

| Source | Raw Data | Derived Features |
|--------|----------|------------------|
| **TMS (our loads)** | Historical loads: lane, equipment, rate, weight, commodity, pickup/delivery dates, assigned carrier, outcome (covered/failed) | • Carrier lane affinity (last 90d loads on lane / total)<br>• Equipment match rate<br>• Rate acceptance distribution (p10/p50/p90)<br>• On-time pickup/delivery %<br>• Cancellation / fall-off rate |
| **DAT / Truckstop** | Posted trucks: location, equipment, preferred lanes, rate expectations | • Real-time capacity density (trucks/load ratio per 150mi radius)<br>• Lane competition index<br>• Carrier's posted rate vs. market |
| **Highway** | Carrier identity: MC/DOT, authority status, insurance, inspections, crashes, fraud flags, factoring company | • **Fraud risk score** (0–1): authority age, inspection density, crash rate, factoring co. reputation<br>• **Capacity reliability**: active trucks vs. authority size<br>• **Payment risk**: factoring fees, days-to-pay |
| **QuickBooks (AP/AR)** | Invoices, payments, disputes, factoring fees per carrier | • Avg days-to-pay<br>• Dispute rate %<br>• Factoring cost passthrough<br>• Net revenue per load (rate - factoring - fuel) |
| **External** | Weather, fuel index (DOE), produce seasons, holidays | • Lane risk multiplier (weather/disruption)<br>• Seasonal rate adjustment |

**Carrier feature vector** (per load posting): ~40 features. Mostly numerical, some categorical (equipment, region).

---

## 2. Where LLMs Add Value (and Where They Don't)

| Task | Tool | Why |
|------|------|-----|
| **Parse unstructured carrier packets** (insurance certs, authority letters, W-9s) | **LLM** (extraction) | High format variance, semi-structured, one-time ingestion |
| **Normalize lane names** ("LA Basin" → "Los Angeles, CA") | **LLM** (few-shot) | Fuzzy geography, context-dependent |
| **Generate carrier outreach scripts** ("Hey, you ran this lane last week...") | **LLM** (generation) | Personalization at scale, low risk if slightly off |
| **Rank carriers for a load** | **Gradient Boosted Trees (XGBoost/LightGBM)** | Tabular features, need calibrated probabilities, interpretable, fast, cheap inference |
| **Predict cover probability** | **GBT** | Same — structured prediction, not language |
| **Detect fraud patterns in Highway data** | **GBT + rules** | Highway already scores; we combine with our outcome data |
| **Explain ranking to broker** | **LLM** (post-hoc) | "Top 3 because: 1) ran this lane 12x last quarter, 2) low fraud score, 3) accepts ~\$2.85/mi" |

**Principle**: LLMs for **unstructured → structured** and **human-facing text**. Classical ML for **ranking, scoring, prediction** on tabular data. Heuristics for **hard constraints** (equipment match, authority active, insurance valid).

---

## 3. Cold-Start for New Brokerage

**No load history → no carrier outcome data**. Strategy:

1. **Industry priors** (pre-trained on anonymized freight-network data aggregated across brokerages):
   - Global lane-carrier affinity matrix (which carriers run which lanes)
   - Equipment-type conversion rates
   - Seasonal rate curves per lane

2. **Carrier self-reported preferences** (from DAT/Truckstop profiles + onboarding survey):
   - Preferred lanes, equipment, rate floors, max deadhead

3. **Highway signals as proxy**:
   - Active authority + clean inspections + reasonable factoring = "likely reliable"
   - Use Highway's own carrier scores as features

4. **Exploration bandit** (Thompson Sampling):
   - For first 50 loads: 70% exploit (priors), 30% explore (random eligible carrier)
   - Log outcome → retrain weekly
   - Converges to broker-specific model in ~200 loads

5. **Manual seed**: Broker imports last 6 months of loads from previous TMS (CSV upload) → instant history.

---

## 4. Latency, Cost, Caching

**SLA**: Ranking must return in **< 500ms** (broker hits "Post Load" → suggestions appear).

| Component | Latency | Cost | Strategy |
|-----------|---------|------|----------|
| Feature fetch (carrier × load) | 50–150ms | Low | **Pre-compute carrier features nightly**; load features computed at post time (trivial) |
| GBT inference | 5–10ms | Negligible | Model < 100 trees, depth ≤ 6 → ~50KB, runs in-process |
| Highway API (fraud check) | 100–300ms | \$0.02/call | **Cache 24h**; async pre-fetch for active carriers |
| DAT/Truckstop capacity | 200–500ms | Included | **Cache 15min**; stale capacity still directional |
| LLM outreach gen | 1–3s | \$0.001 | **Async, post-rank** — broker sees ranked list immediately, scripts populate in background |

**Architecture**:
Load Posted │ ▼ ┌─────────────────────────────────────┐ │ Feature Assembly (sync, <100ms) │ │ - Load features (computed) │ │ - Carrier features (cached Redis) │ │ - Highway fraud score (cached) │ │ - DAT capacity (cached) │ └─────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────┐ │ GBT Ranker (in-process, <10ms) │ │ → Top 20 carriers + scores │ └─────────────────────────────────────┘ │ ▼ ┌─────────────────────────────────────┐ │ Hard Filter (heuristics, <1ms) │ │ - Equipment match │ │ - Authority active │ │ - Insurance valid │ │ - Fraud score < threshold │ └─────────────────────────────────────┘ │ ▼ Top 5 → Broker UI (synchronous) │ ├─→ Async: LLM generates outreach scripts └─→ Async: Log features + ranking for retraining

**Cost estimate** (100 loads/day):
- GBT inference: ~\$0
- Highway API: 500 carriers × \$0.02 × 1/day (cached) = **\$10/day**
- LLM scripts: 100 × 5 carriers × \$0.001 = **\$0.50/day**
- Redis/cache: **~\$20/mo**

**Total: < \$350/mo** for 3,000 loads/mo. Scales linearly.

---

## 5. Honest Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Highway API down | Cache 24h + fallback to last known score; alert if stale >6h |
| New carrier not in Highway | Treat as "unknown risk" — don't penalize, don't boost |
| Broker ignores suggestions | Track suggestion acceptance rate; if <10%, surface "why" (explainability) |
| Rate prediction drift | Nightly retrain on last 90d outcomes; champion/challenger A/B |
| Carrier gaming (posting fake capacity) | DAT/Truckstop data is noisy — weight our own outcome data 3x higher |

---

## Diagram: Data Flow
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ │ TMS │ │ Highway │ │ DAT/ │ │ (Loads) │ │ (Identity) │ │ Truckstop │ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ │ │ │ ▼ ▼ ▼ ┌─────────────────────────────────────────────────────┐ │ NIGHTLY FEATURE PIPELINE │ │ - Carrier lane affinity │ │ - Equipment match rates │ │ - Rate acceptance curves │ │ - Fraud risk scores (24h cache) │ │ - Payment history (QuickBooks) │ └────────────────────────┬────────────────────────────┘ │ ▼ ┌─────────────────────┐ │ Redis: carrier_id │ │ → feature vector │ │ (TTL 24h) │ └──────────┬──────────┘ │ Load Posted │ Broker UI │ │ │ ▼ ▼ ▼ ┌──────────────────────────────┐ │ Feature Assembly (<100ms) │ └──────────────┬───────────────┘ │ ▼ ┌──────────────────────────────┐ │ GBT Ranker + Hard Filters │ └──────────────┬───────────────┘ │ ▼ Top 5 Carriers → UI │ ┌────────────┴────────────┐ ▼ ▼ Async LLM Log for Outreach Gen Retrain

