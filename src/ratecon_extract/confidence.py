"""Confidence scoring: a documented, weighted checklist — not an
LLM-reported confidence number.

Weights sum to 1.0; penalties subtract. Buckets are the product decision:
  HIGH   (>= 0.75) → auto-populate
  MEDIUM (>= 0.45) → human review
  LOW    (<  0.45) → reject, manual entry

The weights are expert judgment, NOT statistically calibrated against
labeled outcomes — see docs/EVAL.md for the plan (FABR metric) that would
calibrate them.
"""
from __future__ import annotations

from ratecon_extract.models import ConfidenceLevel, EquipmentType, Extraction, ValidationResult

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.45


def score_confidence(
    ext: Extraction, result: ValidationResult, *, retries: int
) -> ConfidenceLevel:
    score = 0.0

    if not result.missing_critical_fields:
        score += 0.30
    else:
        score -= 0.40  # missing critical = heavy penalty

    if result.totals_conflict:
        score -= 0.30
    elif ext.total_rate is not None:  # is-not-None, not truthiness: $0 is valid
        score += 0.25

    if result.pickup_parsed and result.delivery_parsed \
            and not result.ambiguous_pickup_date and not result.ambiguous_delivery_date:
        score += 0.15
    elif result.ambiguous_pickup_date or result.ambiguous_delivery_date:
        score -= 0.20

    # OTHER means the raw text named equipment we couldn't map; None means it
    # wasn't stated at all. Neither earns the bonus.
    if ext.equipment_type and ext.equipment_type != EquipmentType.OTHER:
        score += 0.10

    optional = sum(
        1 for v in (
            ext.line_haul_rate, ext.fuel_surcharge, ext.weight_lbs, ext.commodity,
        ) if v is not None
    )
    if optional >= 3:
        score += 0.10
    elif optional >= 1:
        score += 0.05

    if retries == 0:
        score += 0.10
    else:
        score -= 0.15 * min(retries, 2)

    score = max(0.0, min(1.0, score))
    if score >= HIGH_THRESHOLD:
        return ConfidenceLevel.HIGH
    if score >= MEDIUM_THRESHOLD:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
