"""Pipeline tests with a scripted (mocked) LLM — zero API calls.

Proves the reliability contract: malformed model output must not crash the
pipeline and must not silently produce confident bad data. Each test scripts
exact client behaviors via conftest's FakeClient and asserts on what survives.

Run: pytest tests/test_pipeline.py -v
"""
import json
from decimal import Decimal

from ratecon_extract import ConfidenceLevel, extract_and_validate
from ratecon_extract.extractor import MAX_RETRIES

# Strict mode requires EVERY property in every response, so realistic fake
# payloads include all keys. This one is a fully valid extraction.
VALID = json.dumps({
    "load_id": "RC-100",
    "origin": {"city": "Chicago", "state": "IL", "zip": "60601"},
    "destination": {"city": "Atlanta", "state": "GA", "zip": "30301"},
    "pickup_date_raw": "03/15/2026",
    "delivery_date_raw": "March 18, 2026",
    "equipment_type": "Dry Van",
    "line_haul_rate": 2800,
    "fuel_surcharge": 350,
    "accessorials": [],
    "total_rate": 3150,
    "weight_lbs": 42000,
    "commodity": "Auto parts",
})

CONFLICTING = json.dumps({
    **json.loads(VALID),
    "load_id": "RC-BAD",
    "total_rate": 9999,
})


def test_valid_llm_json_produces_high_confidence_output(fake_llm):
    fake_llm(VALID)
    out, diag = extract_and_validate("any text — the mock ignores it")
    assert diag.api_error is None
    assert diag.attempts == 1
    assert out.load_id == "RC-100"
    assert out.total_rate == Decimal("3150")  # Decimal internally…
    assert out.to_dict()["total_rate"] == 3150.0  # …float only in JSON
    assert out.pickup_date == "2026-03-15"
    assert out.confidence == ConfidenceLevel.HIGH
    assert diag.validation.errors == []


def test_malformed_then_valid_json_recovers_on_retry(fake_llm):
    client = fake_llm("not JSON at all {{{", VALID)
    out, diag = extract_and_validate("text")
    assert diag.attempts == 2  # first attempt failed, second succeeded
    assert out.load_id == "RC-100"
    assert len(client.completions.calls) == 2


def test_always_malformed_degrades_to_empty_low_confidence(fake_llm):
    client = fake_llm("garbage", "still garbage", "nope")
    out, diag = extract_and_validate("text")
    assert out.load_id is None
    assert out.confidence == ConfidenceLevel.LOW
    assert diag.api_error is not None
    assert diag.attempts == MAX_RETRIES + 1
    assert len(client.completions.calls) == MAX_RETRIES + 1


def test_invalid_enum_value_coerces_to_other_not_crash(fake_llm):
    weird = json.dumps({**json.loads(VALID), "equipment_type": "hyperloop tanker"})
    fake_llm(weird)
    out, diag = extract_and_validate("text")
    assert str(out.equipment_type.value) == "other"
    # Degraded equipment keeps the load reviewable but extractable:
    assert out.load_id == "RC-100"


def test_missing_critical_fields_flagged_and_penalized(fake_llm):
    sparse = json.dumps({
        "load_id": None, "origin": None, "destination": None,
        "pickup_date_raw": None, "delivery_date_raw": None,
        "equipment_type": None, "line_haul_rate": None, "fuel_surcharge": None,
        "accessorials": None, "total_rate": None, "weight_lbs": None,
        "commodity": None,
    })
    fake_llm(sparse)
    out, diag = extract_and_validate("text")
    assert out.load_id is None
    assert set(diag.validation.missing_critical_fields) == {
        "load_id", "origin", "destination", "pickup_date", "total_rate",
    }
    assert out.confidence == ConfidenceLevel.LOW


def test_conflicting_rates_detected_never_auto_booked(fake_llm):
    fake_llm(CONFLICTING)
    out, diag = extract_and_validate("text")
    assert diag.validation.totals_conflict is True
    assert any("differs from total" in e for e in diag.validation.errors)
    assert out.confidence != ConfidenceLevel.HIGH


class FakeTimeoutError(Exception):
    """Name triggers the retryable classifier without needing real SDK types."""


class FakeAuthError(Exception):
    pass


class FakeRateLimitError(Exception):
    pass


class FakeSchemaBadRequest(Exception):
    """Name contains 'BadRequest' → non-retryable; message mimics a provider
    rejecting strict structured outputs (common on Groq/Ollama-tier APIs)."""


def test_provider_rejecting_strict_schema_degrades_and_succeeds(fake_llm):
    client = fake_llm(
        FakeSchemaBadRequest("Invalid schema for response_format: 'strict' is not supported"),
        VALID,
    )
    out, diag = extract_and_validate("text")
    assert out.load_id == "RC-100"
    assert len(client.completions.calls) == 2
    first = client.completions.calls[0]["response_format"]["json_schema"]["strict"]
    second = client.completions.calls[1]["response_format"]["json_schema"]["strict"]
    assert first is True and second is False  # fell back to non-strict
    # Pydantic still validated the result locally either way:
    assert out.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW)


def test_strict_env_false_skips_strict_entirely(monkeypatch, fake_llm):
    monkeypatch.setattr("ratecon_extract.extractor.STRICT_MODE", False)
    client = fake_llm(VALID)
    extract_and_validate("text")
    assert client.completions.calls[0]["response_format"]["json_schema"]["strict"] is False


def test_empty_env_values_fall_back_to_defaults(monkeypatch):
    # .env files routinely contain `KEY=` lines; those must behave like unset,
    # not crash float() parsing or produce an empty base_url.
    import importlib
    import ratecon_extract.extractor as ex

    monkeypatch.setenv("OPENAI_MODEL", "")
    monkeypatch.setenv("OPENAI_TIMEOUT_S", "")
    monkeypatch.setenv("OPENAI_STRICT", "")
    try:
        importlib.reload(ex)
        assert ex.MODEL == "gpt-4o-mini"
        assert ex.REQUEST_TIMEOUT_S == 30.0
        assert ex.STRICT_MODE is True
    finally:
        importlib.reload(ex)


def test_timeout_is_retried_then_succeeds(fake_llm):
    client = fake_llm(FakeTimeoutError("timed out"), VALID)
    out, diag = extract_and_validate("text")
    assert diag.attempts == 2
    assert out.load_id == "RC-100"
    assert len(client.completions.calls) == 2


def test_rate_limit_is_retried(fake_llm):
    client = fake_llm(FakeRateLimitError("429"), FakeRateLimitError("429"), VALID)
    out, diag = extract_and_validate("text")
    assert diag.attempts == 3
    assert out.load_id == "RC-100"


def test_auth_error_fails_fast_without_retry(fake_llm):
    client = fake_llm(FakeAuthError("bad api key"))
    out, diag = extract_and_validate("text")
    # Exactly ONE call — no point hammering an API that rejected our key.
    assert len(client.completions.calls) == 1
    assert diag.attempts == 1
    assert out.load_id is None
    assert out.confidence == ConfidenceLevel.LOW
    assert "FakeAuthError" in diag.api_error


def test_backoff_schedule_is_exponential(monkeypatch, fake_llm):
    sleeps = []
    monkeypatch.setattr("ratecon_extract.extractor.time.sleep", sleeps.append)
    fake_llm(FakeTimeoutError("t"), FakeTimeoutError("t"), FakeTimeoutError("t"))
    _, diag = extract_and_validate("text")
    assert sleeps == [1.0, 2.0]  # wait 1s → attempt → wait 2s → attempt
    assert diag.api_error is not None
