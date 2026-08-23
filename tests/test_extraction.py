"""Live smoke tests against samples/ — the ONLY tests that call the OpenAI
API. Skipped automatically when OPENAI_API_KEY is not set, so `pytest` stays
green and free as an offline suite in CI or on a fresh clone.

To run for real:
    cp .env.example .env   # add your key
    pytest tests/test_extraction.py -v -s

Expected behavior on samples/:
- LD64392 → HIGH   (clean: all critical fields, totals reconcile)
- LD64407 → HIGH   (near-duplicate of LD64392, different weights/dates)
- LD64408 → MEDIUM (multiple pickups, unusable weights, carrier charge ≠ fuel)
"""
import os

import pytest

from conftest import SAMPLES_DIR
from ratecon_extract import ConfidenceLevel, run

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — live extraction tests skipped",
)

SAMPLES = {p.name: p.read_text(encoding="utf-8") for p in sorted(SAMPLES_DIR.glob("*.txt"))}


@pytest.mark.parametrize("name,text", SAMPLES.items())
def test_sample_extracts(name, text):
    out = run(text)
    print(f"\n{'=' * 60}")
    print(f"{name} → {out.confidence.value.upper()} | load_id={out.load_id} | total=${out.total_rate}")
    print(f"  origin: {out.origin.city if out.origin else None}, {out.origin.state if out.origin else None}")
    print(f"  dest:   {out.destination.city if out.destination else None}, {out.destination.state if out.destination else None}")
    print(f"  pickup: {out.pickup_date} | delivery: {out.delivery_date}")
    print(f"  equipment: {out.equipment_type} | weight: {out.weight_lbs} | commodity: {out.commodity}")
    print(f"  line_haul: {out.line_haul_rate} | fuel: {out.fuel_surcharge}")

    # The pipeline must never crash and must always return a schema-shaped result.
    assert out.confidence in (
        ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW,
    )
