"""Offline tests for the deterministic validation layer.

Zero API calls — these pin exactly the logic that makes the pipeline
trustworthy: date parsing/ambiguity, totals reconciliation, equipment
normalization, ValidationResult signals, and the strict-schema contract
sent to OpenAI.
Run: pytest tests/test_validation.py -v
"""
from datetime import date as _date
from decimal import Decimal

import pytest

from ratecon_extract.extractor import SCHEMA
from ratecon_extract.models import EquipmentType, Extraction, Location, Output
from ratecon_extract.validators import check_totals, parse_date, validate


# ─── Date parsing & ambiguity ──────────────────────────────────────

def test_iso_date_not_ambiguous():
    assert parse_date("2026-07-30") == (_date(2026, 7, 30), False)


def test_two_digit_year_slash_dates_are_ambiguous():
    d, amb = parse_date("3/4/26")
    assert (d.month, d.day) == (3, 4)
    assert amb is True  # could equally be read April 3


def test_unambiguous_us_format():
    assert parse_date("07/30/2026") == (_date(2026, 7, 30), False)  # 4-digit year
    d, amb = parse_date("7/30/26")
    assert amb is False  # day=30 > 12 → swapped reading would be invalid


def test_day_equals_month_is_not_ambiguous():
    # 3/3 reads the same either way — flagging it would be a false penalty.
    assert parse_date("03/03/26") == (_date(2026, 3, 3), False)


def test_month_name_is_not_ambiguous():
    assert parse_date("30-Jul-2026") == (_date(2026, 7, 30), False)
    assert parse_date("March 15, 2024")[1] is False


def test_garbage_and_missing_dates_degrade_to_none():
    assert parse_date(None) == (None, False)
    assert parse_date("")[0] is None
    assert parse_date("not a date")[0] is None


# ─── Totals reconciliation ─────────────────────────────────────────

def test_all_in_rate_alone_is_not_a_conflict():
    e = Extraction(total_rate=Decimal("3200"))
    assert check_totals(e) == (False, None)


def test_real_mismatch_is_flagged_with_diff():
    e = Extraction(
        line_haul_rate=Decimal("1000"),
        fuel_surcharge=Decimal("100"),
        total_rate=Decimal("3200"),
    )
    conflict, diff = check_totals(e)
    assert conflict is True
    assert abs(diff - 2100.0) < 1e-9


def test_reconciling_breakdown_passes():
    e = Extraction(
        line_haul_rate=Decimal("2800"),
        fuel_surcharge=Decimal("350"),
        total_rate=Decimal("3150"),
    )
    assert check_totals(e) == (False, None)


def test_small_rounding_difference_within_tolerance():
    e = Extraction(
        line_haul_rate=Decimal("2800.00"),
        fuel_surcharge=Decimal("350.00"),  # $0.005 off → inside tolerance
        total_rate=Decimal("3150.005"),
    )
    assert check_totals(e)[0] is False


def test_zero_rates_are_still_comparable_values():
    # $0 components are legitimate values, not "missing".
    e = Extraction(line_haul_rate=Decimal("0"), fuel_surcharge=Decimal("0"),
                   total_rate=Decimal("0"))
    assert check_totals(e) == (False, None)


def test_itemized_accessorials_reconcile_with_total():
    # line_haul + fuel + accessorials = total — carrier charges are itemized,
    # never folded into line haul.
    e = Extraction(
        line_haul_rate=Decimal("1800"),
        fuel_surcharge=Decimal("220"),
        accessorials=[
            {"description": "Carrier Charge", "amount": "200"},
            {"description": "Lumper", "amount": "150"},
        ],
        total_rate=Decimal("2370"),
    )
    assert check_totals(e) == (False, None)


def test_accessorial_only_breakdown_is_compared():
    # No fuel line, but accessorials are itemized → missing fuel means $0 fuel.
    e = Extraction(
        line_haul_rate=Decimal("500"),
        accessorials=[{"description": "Carrier Charge", "amount": "200"}],
        total_rate=Decimal("700"),
    )
    assert check_totals(e) == (False, None)
    bad = Extraction(
        line_haul_rate=Decimal("500"),
        accessorials=[{"description": "Carrier Charge", "amount": "200"}],
        total_rate=Decimal("999"),
    )
    assert check_totals(bad)[0] is True


def test_all_in_rate_without_any_breakdown_still_not_a_conflict():
    e = Extraction(line_haul_rate=None, fuel_surcharge=None,
                   accessorials=[], total_rate=Decimal("2450"))
    assert check_totals(e) == (False, None)


# ─── Equipment normalization ────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("Dry Van", EquipmentType.VAN),
    ("53' dry van", EquipmentType.VAN),
    ("Reefer", EquipmentType.REEFER),
    ("refrigerated trailer", EquipmentType.REEFER),
    ("Flatbed", EquipmentType.FLATBED),
    ("step deck", EquipmentType.FLATBED),
    ("Sprinter box truck", EquipmentType.OTHER),
    ("hotshot", EquipmentType.OTHER),
])
def test_equipment_synonyms_coerce_without_failing(raw, expected):
    # An unrecognized synonym must degrade to OTHER — never kill the extraction.
    assert Extraction(equipment_type=raw).equipment_type == expected


def test_equipment_none_stays_none():
    assert Extraction(equipment_type=None).equipment_type is None


# ─── Location normalization ────────────────────────────────────────

def test_state_uppercased_and_zip_truncated_to_zip5():
    loc = Location(city="chicago", state="il", zip="60601-1234")
    assert loc.state == "IL"
    assert loc.zip == "60601"


# ─── ValidationResult signals ──────────────────────────────────────

def clean_ext(**over):
    fields = dict(
        load_id="RC-1",
        origin={"city": "Chicago", "state": "IL"},
        destination={"city": "Atlanta", "state": "GA"},
        pickup_date_raw="March 15, 2026",
        delivery_date_raw="2026-03-18",
        equipment_type="van",
        line_haul_rate=Decimal("2800"),
        fuel_surcharge=Decimal("350"),
        total_rate=Decimal("3150"),
        weight_lbs=42000,
        commodity="Auto parts",
    )
    fields.update(over)
    return Extraction(**fields)


def test_clean_extraction_has_no_problems():
    res = validate(clean_ext())
    assert res.missing_critical_fields == []
    assert not res.totals_conflict
    assert res.pickup_parsed and res.delivery_parsed
    assert not res.ambiguous_pickup_date and not res.ambiguous_delivery_date
    assert not res.invalid_equipment
    assert res.errors == []
    assert not res.has_critical_problems


def test_missing_fields_are_listed_individually():
    res = validate(Extraction())
    assert set(res.missing_critical_fields) == {
        "load_id", "origin", "destination", "pickup_date", "total_rate",
    }
    assert res.has_critical_problems
    assert len(res.errors) >= 5


def test_totals_conflict_recorded_in_result_and_errors():
    res = validate(clean_ext(total_rate=Decimal("9999")))
    assert res.totals_conflict
    assert any("differs from total" in e for e in res.errors)


def test_accessorial_mismatch_recorded_in_errors():
    res = validate(clean_ext(accessorials=[{"description": "Detention", "amount": "75"}]))
    # 2800 + 350 + 75 vs 3150 → $75 off
    assert res.totals_conflict
    assert any("differs from total" in e for e in res.errors)


def test_ambiguity_flags_surface_in_errors():
    res = validate(clean_ext(pickup_date_raw="3/4/26"))
    assert res.ambiguous_pickup_date
    assert any("ambiguous" in e for e in res.errors)


def test_other_equipment_marks_invalid_but_not_critical():
    res = validate(clean_ext(equipment_type="hyperloop"))
    assert res.invalid_equipment
    assert "equipment" not in " ".join(res.missing_critical_fields)


# ─── Strict-schema contract ────────────────────────────────────────
# The schema actually sent to the API must satisfy OpenAI strict mode:
# every object closes with additionalProperties:false + full required, and
# carries none of the keywords strict mode rejects (default/title/pattern/
# minimum/…). A violation here means a 400 on every single API call.

FORBIDDEN_KEYS = {
    "default", "title", "format", "pattern",
    "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "uniqueItems",
}


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_strict_schema_has_no_rejected_keywords():
    for node in _walk(SCHEMA):
        if isinstance(node, dict):
            assert not (set(node) & FORBIDDEN_KEYS), f"rejected keyword in: {node}"


def test_strict_schema_objects_are_closed_and_total():
    for node in _walk(SCHEMA):
        if isinstance(node, dict) and node.get("type") == "object" and "properties" in node:
            assert node["additionalProperties"] is False
            assert sorted(node["required"]) == sorted(node["properties"])


def test_strict_schema_is_ref_free_for_provider_portability():
    # $defs/$ref are unsupported on several OpenAI-compatible providers
    # (Groq, Gemini, Ollama) — the schema must be fully inlined.
    assert "$defs" not in SCHEMA
    for node in _walk(SCHEMA):
        if isinstance(node, dict):
            assert "$ref" not in node


# ─── Output serialization boundary ─────────────────────────────────

def test_output_serialization_converts_decimal_only_at_boundary():
    o = Output(confidence="low", total_rate=Decimal("0.00"), line_haul_rate=Decimal("1234.56"))
    d = o.to_dict()
    assert d["total_rate"] == 0.0          # $0 survives — never silently nulled
    assert d["line_haul_rate"] == 1234.56  # float only at serialization
    assert o.total_rate == Decimal("0.00")  # internal precision intact


def test_output_serializes_itemized_accessorials():
    o = Output(
        confidence="medium",
        accessorials=[{"description": "Carrier Charge", "amount": "200.00"}],
    )
    d = o.to_dict()
    assert d["accessorials"] == [{"description": "Carrier Charge", "amount": 200.0}]
    # Absent stays null, not []:
    assert Output(confidence="low").to_dict()["accessorials"] is None
