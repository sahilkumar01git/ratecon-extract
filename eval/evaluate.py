"""Evaluation harness: scores pipeline output against human-annotated
golden extractions, implementing the partial-credit rubric and the FABR
proxy defined in docs/EVAL.md.

Usage:
    python eval/evaluate.py [path/to/golden.jsonl] [--json]

This calls the OpenAI API once per record — set OPENAI_API_KEY first.
The bundled golden.jsonl (3 annotated samples) demonstrates the methodology;
it is NOT a benchmark. See docs/EVAL.md for the ~275-example eval-set design.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ratecon_extract.extractor import MODEL, extract_and_validate  # noqa: E402

# Per-field weights and scores exactly as specified in docs/EVAL.md §1:
# critical fields  → exact = +2.0, null = 0, wrong   = -1.0
# secondary fields → exact = +1.0, null = 0, wrong   = -0.5
# tertiary fields  → within-1% = +0.5, null = 0, wrong = -0.25
FIELD_WEIGHTS = {
    "load_id": 2.0,
    "origin": 2.0,
    "destination": 2.0,
    "pickup_date": 2.0,
    "total_rate": 2.0,
    "delivery_date": 1.0,
    "equipment_type": 1.0,
    "accessorials": 1.0,   # itemized charges feed the financial model
    "line_haul_rate": 0.5,
    "fuel_surcharge": 0.5,
    "weight_lbs": 0.5,
    "commodity": 0.5,
}
WRONG_PENALTY_FACTOR = {
    "critical": -0.5,   # × weight → -1.0
    "secondary": -0.5,  # × weight → -0.5
    "tertiary": -0.5,   # × weight → -0.25
}
NULL_SCORE = 0.0
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_WAIT_S = 30


def _tier(field: str) -> str:
    if FIELD_WEIGHTS[field] >= 2.0:
        return "critical"
    if FIELD_WEIGHTS[field] >= 1.0:
        return "secondary"
    return "tertiary"


def _norm_str(v) -> str | None:
    return v.strip() if isinstance(v, str) else v


def _loc_match(out, gold) -> bool:
    if not isinstance(out, dict) or not isinstance(gold, dict):
        return out == gold
    # Case-insensitive: sources often print locations in capitals
    # ("ONTARIO CA"), and that is the same city as the gold "Ontario".
    def _key(v):
        return v.strip().casefold() if isinstance(v, str) else v

    same = (
        _key(out.get("city")) == _key(gold.get("city"))
        and _key(out.get("state")) == _key(gold.get("state"))
    )
    if same and gold.get("zip"):
        same = out.get("zip") == gold["zip"]
    return same


def _num_match(out_v, gold_v, tol_pct) -> bool:
    try:
        o, g = float(out_v), float(gold_v)
    except (TypeError, ValueError):
        return False
    if g == 0:
        return o == 0
    return abs(o - g) <= abs(g) * tol_pct


def _acc_norm(v):
    """Normalize an accessorials list to sorted (description, amount) pairs.
    None/[] → None (no itemized charges)."""
    if not v:
        return None
    return sorted(
        (str(a.get("description", "")).strip().lower(), round(float(a["amount"]), 2))
        for a in v
    )


def field_score(name: str, out_v, gold_v) -> tuple[float, str]:
    """Returns (score, verdict) where verdict ∈ {exact, close, null, wrong}."""
    w = FIELD_WEIGHTS[name]
    if out_v is None or (name == "accessorials" and not out_v):
        return NULL_SCORE, "null"
    tier = _tier(name)

    if name in ("origin", "destination"):
        ok = _loc_match(out_v, gold_v)
    elif name == "accessorials":
        ok = _acc_norm(out_v) == _acc_norm(gold_v)
    elif name in ("total_rate", "line_haul_rate", "fuel_surcharge"):
        ok = _num_match(out_v, gold_v, 0.001)      # money: exact to ~0.1%
    elif name == "weight_lbs":
        ok = _num_match(out_v, gold_v, 0.01)       # tertiary: within 1%
    else:
        ok = _norm_str(str(out_v)) == _norm_str(str(gold_v))

    if ok:
        return w, "exact" if tier != "tertiary" else "close"
    return round(w * WRONG_PENALTY_FACTOR[tier], 2), "wrong"


def evaluate_record(rec: dict, out_dict: dict) -> dict:
    golden = rec["golden"]
    per_field, total, max_possible = {}, 0.0, 0.0
    critical_errors = []

    for name, gold_v in golden.items():
        max_possible += FIELD_WEIGHTS[name]
        score, verdict = field_score(name, out_dict.get(name), gold_v)
        per_field[name] = {"score": score, "verdict": verdict, "got": out_dict.get(name)}
        total += score
        if verdict == "wrong" and _tier(name) == "critical":
            critical_errors.append(name)

    composite = max(0.0, total / max_possible) if max_possible else 0.0
    return {
        "id": rec["id"],
        "composite": round(composite, 3),
        "per_field": per_field,
        "critical_errors": critical_errors,
        "confidence": out_dict.get("confidence"),
        "auto_book_error": out_dict.get("confidence") == "high" and bool(critical_errors),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score extraction against golden annotations.")
    ap.add_argument("golden", nargs="?", default=str(Path(__file__).parent / "golden.jsonl"))
    ap.add_argument("--json", action="store_true", help="Machine-readable summary.")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="Seconds to wait between records (for providers with low tokens-per-minute limits).")
    args = ap.parse_args()

    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set — evaluation calls the live API. "
              "Copy .env.example to .env and add your key.", file=sys.stderr)
        return 1

    records = [json.loads(line) for line in Path(args.golden).read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("//")]
    base = Path(args.golden).parent

    results = []
    for i, rec in enumerate(records):
        if i and args.delay:
            time.sleep(args.delay)
        text = (base / rec["source"]).resolve().read_text(encoding="utf-8")
        out, diag = extract_and_validate(text)
        # A rate-limited call yields an all-null record, which would be scored
        # as a (false) abstention — wait and retry instead of scoring it.
        for _ in range(RATE_LIMIT_RETRIES):
            if not (diag.api_error and "RateLimitError" in diag.api_error):
                break
            time.sleep(RATE_LIMIT_WAIT_S)
            out, diag = extract_and_validate(text)
        if diag.api_error:
            print(f"[{rec['id']}] API error: {diag.api_error}", file=sys.stderr)
        results.append(evaluate_record(rec, out.to_dict()))

    n = len(results)
    avg_composite = sum(r["composite"] for r in results) / n if n else 0.0
    high = [r for r in results if r["confidence"] == "high"]
    fabr = (sum(1 for r in high if r["auto_book_error"]) / len(high)) if high else None

    if args.json:
        print(json.dumps({"model": MODEL, "n": n, "avg_composite": round(avg_composite, 4),
                          "fabr_proxy": fabr, "results": results}, indent=2))
        return 0

    print(f"model={MODEL}  records={n}")
    print(f"{'record':<12} {'composite':>9} {'conf':>7}  critical errors / notes")
    for r in results:
        flags = []
        if r["critical_errors"]:
            flags.append("wrong: " + ",".join(r["critical_errors"]))
        if r["auto_book_error"]:
            flags.append("FABR HIT")
        print(f"{r['id']:<12} {r['composite']:>9.3f} {r['confidence']:>7}  {'; '.join(flags)}")

    def pct(x): return f"{x * 100:.1f}%"
    print(f"\navg composite score: {avg_composite:.3f}")
    print(f"FABR proxy (high-confidence with a wrong critical field): "
          + (pct(fabr) if fabr is not None else "n/a (nothing scored high)"))
    print("\nNote: 3 bundled samples demonstrate methodology only — see docs/EVAL.md "
          "for the real eval-set design and the <0.5% FABR target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
