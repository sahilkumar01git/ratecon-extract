"""Deterministic validation: date parsing/ambiguity, equipment
normalization, totals reconciliation, missing-critical detection.

Everything here runs in plain Python against the raw extraction — the LLM
does extraction, never judgment calls a few lines of code can do more
reliably. No network, no I/O, fully unit-testable.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from dateutil import parser as dateparser

from ratecon_extract.models import EquipmentType, Extraction, ValidationResult

CRITICAL_FIELDS = ("load_id", "origin", "destination", "pickup_date", "total_rate")

# A $1 absolute floor plus 1% relative slack absorbs rounding in the source
# document ("$3,149.99" vs "$3,150.00") without masking a real mismatch.
TOTALS_TOL_ABS = Decimal("1")
TOTALS_TOL_REL = Decimal("0.01")


def parse_date(v: str | date | None) -> tuple[Optional[date], bool]:
    """Returns (parsed_date, is_ambiguous).

    Ambiguity must be computed on the RAW text (e.g. "3/4/26") — once the
    date is resolved to ISO the formatting cue that made it ambiguous is gone.
    A slash date with a 2-digit year where day and month are both <= 12 AND
    different can be read two ways; "3/3/26" reads the same either way, so it
    is NOT ambiguous.
    """
    if not v:
        return None, False
    if isinstance(v, date):
        return v, False
    s = str(v).strip()
    try:
        return date.fromisoformat(s), False
    except ValueError:
        pass
    try:
        dt = dateparser.parse(s, dayfirst=False, yearfirst=False, fuzzy=True)
        parts = s.split("/")
        amb = (
            dt.day <= 12 and dt.month <= 12 and dt.day != dt.month
            and len(parts) == 3 and len(parts[-1]) == 2
        )
        return dt.date(), amb
    except Exception:
        return None, False


def normalize_equipment(v) -> Optional[EquipmentType]:
    """Map an unrecognized synonym to OTHER instead of letting enum
    validation kill the whole extraction over one field."""
    if v is None or isinstance(v, EquipmentType):
        return v
    s = str(v).strip().lower()
    try:
        return EquipmentType(s)
    except ValueError:
        pass
    if "reef" in s or "refrig" in s:
        return EquipmentType.REEFER
    if "flat" in s or "deck" in s:
        return EquipmentType.FLATBED
    if "van" in s:
        return EquipmentType.VAN
    return EquipmentType.OTHER


def check_totals(ext: Extraction) -> tuple[bool, Optional[float]]:
    """Financial model: line_haul + fuel + itemized accessorials ≈ total.

    Only compares when an actual line-item breakdown exists — an all-in rate
    with no components is not a conflict, it's just incomplete. Concretely:
    compare when line_haul AND total are present AND at least one component
    beyond line_haul was itemized (fuel, or any accessorial). Missing fuel
    with itemized accessorials is a 0 fuel line, not missing data.
    """
    lh, fuel, tot = ext.line_haul_rate, ext.fuel_surcharge, ext.total_rate
    accs = ext.accessorials or []
    if lh is None or tot is None or (fuel is None and not accs):
        return False, None
    expected = lh + (fuel or 0) + sum(a.amount for a in accs)
    diff = abs(expected - tot)
    tol = max(TOTALS_TOL_ABS, tot * TOTALS_TOL_REL)
    if diff > tol:
        return True, float(diff)
    return False, None


def validate(ext: Extraction) -> ValidationResult:
    """Compute every deterministic signal for one extraction."""
    res = ValidationResult()

    pu_dt, pu_amb = parse_date(ext.pickup_date_raw)
    de_dt, de_amb = parse_date(ext.delivery_date_raw)
    res.pickup_parsed = pu_dt is not None
    res.delivery_parsed = de_dt is not None
    res.ambiguous_pickup_date = pu_amb
    res.ambiguous_delivery_date = de_amb
    res.invalid_equipment = ext.equipment_type == EquipmentType.OTHER

    values = {
        "load_id": ext.load_id,
        "origin": ext.origin,
        "destination": ext.destination,
        "pickup_date": pu_dt,
        "total_rate": ext.total_rate,
    }
    res.missing_critical_fields = [k for k, v in values.items() if v is None]

    res.totals_conflict, res.totals_diff = check_totals(ext)

    if ext.load_id is None:
        res.errors.append("load_id missing from source")
    if ext.origin is None:
        res.errors.append("origin missing from source")
    if ext.destination is None:
        res.errors.append("destination missing from source")
    if pu_dt is None:
        res.errors.append("pickup date unparseable or missing")
    elif pu_amb:
        res.errors.append(f'pickup date "{ext.pickup_date_raw}" is ambiguous (M/D vs D/M)')
    if de_dt is None:
        res.errors.append("delivery date unparseable or missing")
    elif de_amb:
        res.errors.append(f'delivery date "{ext.delivery_date_raw}" is ambiguous (M/D vs D/M)')
    if ext.total_rate is None:
        res.errors.append("total_rate missing from source")
    if res.totals_conflict:
        res.errors.append(
            f"line_haul + fuel + accessorials differs from total by ${res.totals_diff:.2f}"
        )
    return res
