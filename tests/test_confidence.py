"""Offline tests for the confidence-scoring checklist.

The scorer is pure logic — no API involved. These tests pin the documented
bucket boundaries and penalties, because the HIGH/MEDIUM/LOW split is the
product decision the whole pipeline exists to make.
Run: pytest tests/test_confidence.py -v
"""
from decimal import Decimal

from ratecon_extract.confidence import score_confidence
from ratecon_extract.models import ConfidenceLevel, Extraction
from ratecon_extract.validators import validate

CRITICAL = ["load_id", "origin", "destination", "pickup_date", "total_rate"]


def make_ext(**overrides) -> Extraction:
    """A clean, complete extraction — everything present and consistent."""
    fields = dict(
        load_id="RC-1",
        origin={"city": "Chicago", "state": "IL", "zip": "60601"},
        destination={"city": "Atlanta", "state": "GA", "zip": "30301"},
        pickup_date_raw="March 15, 2026",   # month name → unambiguous
        delivery_date_raw="2026-03-18",     # ISO → unambiguous
        equipment_type="van",
        line_haul_rate=Decimal("2800"),
        fuel_surcharge=Decimal("350"),
        total_rate=Decimal("3150"),
        weight_lbs=42000,
        commodity="Auto parts",
    )
    fields.update(overrides)
    return Extraction(**fields)


def score(ext, *, retries=0) -> ConfidenceLevel:
    return score_confidence(ext, validate(ext), retries=retries)


def test_clean_complete_extraction_scores_high():
    assert score(make_ext()) == ConfidenceLevel.HIGH


def test_totals_conflict_blocks_high():
    ext = make_ext(total_rate=Decimal("9999"))  # no longer reconciles
    out = score(ext)
    assert out in (ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)
    # Same extraction with a reconciling total returns to HIGH.
    assert score(make_ext(total_rate=Decimal("3150"))) == ConfidenceLevel.HIGH


def test_ambiguous_date_lands_in_review_bucket():
    # Ambiguity penalty (-0.20) on an otherwise clean extraction → MEDIUM.
    out = score(make_ext(pickup_date_raw="3/4/26"))
    assert out == ConfidenceLevel.MEDIUM


def test_two_retries_land_in_review_bucket():
    # Retry penalty (-0.30 replaces the +0.10 first-try bonus) → MEDIUM.
    out = score(make_ext(), retries=2)
    assert out == ConfidenceLevel.MEDIUM


def test_missing_one_critical_field_is_rejected_not_reviewed():
    # Heavy -0.40 penalty dominates even when everything else is clean.
    res = validate(make_ext())
    res.missing_critical_fields = ["total_rate"]
    out = score_confidence(make_ext(), res, retries=0)
    assert out == ConfidenceLevel.LOW


def test_total_failure_never_scores_medium_or_above():
    # What llm_extract returns when every attempt fails: empty extraction.
    empty = Extraction()
    res = validate(empty)
    out = score_confidence(empty, res, retries=3)
    assert out == ConfidenceLevel.LOW


def test_zero_total_rate_is_a_value_not_an_absence():
    # $0 total must earn the totals bonus (is-not-None check), not be treated
    # as a missing field by the scorer.
    ext = make_ext(
        line_haul_rate=Decimal("0"), fuel_surcharge=Decimal("0"), total_rate=Decimal("0"),
    )
    assert score(ext) == ConfidenceLevel.HIGH
