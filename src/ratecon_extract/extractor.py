"""LLM extraction via OpenAI structured outputs, wrapped so it can never
crash a batch run: strict schema → retry with exponential backoff on
retryable errors only → graceful fallback to an empty extraction.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from ratecon_extract.confidence import score_confidence
from ratecon_extract.models import Extraction, Output, ValidationResult
from ratecon_extract.validators import parse_date, validate

load_dotenv()

# Env-driven (docs/EVAL.md §3 recommends pinning the model version; this makes
# the pin a config decision instead of a code change).
def _env_str(name: str, default: str) -> str:
    """Env read that treats an empty value (e.g. `OPENAI_BASE_URL=` in .env)
    as unset — python-dotenv loads empty strings, which would otherwise
    crash numeric parsing or produce an invalid empty base_url."""
    v = os.getenv(name)
    return v if v else default


MODEL = _env_str("OPENAI_MODEL", "gpt-4o-mini")
REQUEST_TIMEOUT_S = float(_env_str("OPENAI_TIMEOUT_S", "30"))
MAX_RETRIES = 2  # our retries; the SDK's internal retry is disabled below
STRICT_MODE = _env_str("OPENAI_STRICT", "true").lower() != "false"

# ─── Strict-schema helper ───────────────────────────────────────────
# OpenAI's `strict: true` mode requires, recursively on every object:
# (1) every property listed in "required" (optionality is expressed via a
# `null` type, not omission), and (2) "additionalProperties": false. Pydantic
# emits neither by default. Strict mode also REJECTS keywords Pydantic emits
# freely — "default" on every optional field, "pattern" (Decimal renders as a
# regex-constrained string branch), "minimum"/"maximum" (Field(ge=...)) — each
# one is a guaranteed 400 on every call. They are stripped here: the LLM only
# needs the shape; Pydantic re-validates actual values locally anyway.

UNSUPPORTED_STRICT_KEYS = (
    "default", "title", "format", "pattern",
    "minLength", "maxLength",
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minItems", "maxItems", "uniqueItems",
    "minProperties", "maxProperties", "patternProperties",
)


def _inline_refs(schema: dict) -> dict:
    """Resolve $defs/$ref into plain nested schemas.

    OpenAI accepts refs, but many OpenAI-compatible providers (Groq, Gemini,
    Ollama, ...) don't — inlining makes the schema portable everywhere.
    """
    import copy

    defs = schema.pop("$defs", {})

    def visit(node):
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/$defs/"):
                target = defs.get(ref.rsplit("/", 1)[-1])
                if target is not None:
                    node.clear()
                    node.update(copy.deepcopy(target))
            for v in list(node.values()):
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return schema


def to_strict_schema(model: type[BaseModel]) -> dict:
    schema = _inline_refs(model.model_json_schema())

    def visit(node):
        if isinstance(node, dict):
            for k in UNSUPPORTED_STRICT_KEYS:
                node.pop(k, None)
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for v in node.values():
                visit(v)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(schema)
    return schema


SCHEMA = to_strict_schema(Extraction)


# ─── Client (lazy + explicit key validation) ────────────────────────

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    """Lazily build the client so importing this package never crashes when
    the key is absent — the failure surfaces at call time with instructions,
    and llm_extract() degrades gracefully.

    Any OpenAI-compatible endpoint works: point OPENAI_BASE_URL at it
    (Groq, Google Gemini, OpenRouter, Ollama, ...) and set OPENAI_MODEL.
    With OPENAI_BASE_URL unset, the official OpenAI API is used."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add "
                "your key (see README, Setup section)."
            )
        # max_retries=0: the SDK's internal retries are disabled so OUR policy
        # (backoff schedule + retryability classification) governs entirely.
        _client = OpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,  # None → official OpenAI
            timeout=REQUEST_TIMEOUT_S,
            max_retries=0,
        )
    return _client


def reset_client() -> None:
    """Test hook / reconfiguration hook."""
    global _client
    _client = None


# ─── Prompt ─────────────────────────────────────────────────────────

SYSTEM = """Extract rate confirmation data to JSON. Rules:
- All fields optional → null if missing. Do not guess.
- pickup_date_raw / delivery_date_raw: copy the date EXACTLY as it appears in
  the source text (e.g. "3/4/26", "07/30/2026", "March 15, 2024"). Do not
  resolve, reformat, or guess the year — the raw text is parsed downstream.
- Equipment: van | reefer | flatbed | other (normalize synonyms).
- Rates: numbers only. line_haul_rate = base line haul. fuel_surcharge = fuel only.
- accessorials: itemize EVERY other charge separately (detention, lumper,
  carrier charge, stop-off, layover...) with its exact description and amount.
  NEVER fold an extra charge into line_haul_rate or fuel_surcharge.
- If no breakdown exists and only an all-in total is stated, return just
  total_rate. line_haul + fuel + accessorials should ≈ total when all present.
- Weight in lbs. load_id = rate con / load / BOL / PRO / reference number.
"""

FEW_SHOTS = [
    ("""RATE CONFIRMATION #RC-2024-08921
CARRIER: ABC Trucking
ORIGIN: Chicago, IL 60601
DESTINATION: Atlanta, GA 30301
PICKUP: 03/15/2024
DELIVERY: 03/18/2024
EQUIPMENT: Dry Van
COMMODITY: Auto parts
WEIGHT: 42,000 lbs
LINE HAUL: $2,800.00
FUEL SURCHARGE: $350.00
TOTAL RATE: $3,150.00""",
     {"load_id": "RC-2024-08921", "origin": {"city": "Chicago", "state": "IL", "zip": "60601"},
      "destination": {"city": "Atlanta", "state": "GA", "zip": "30301"},
      "pickup_date_raw": "03/15/2024", "delivery_date_raw": "03/18/2024",
      "equipment_type": "van", "commodity": "Auto parts", "weight_lbs": 42000,
      "line_haul_rate": 2800, "fuel_surcharge": 350, "accessorials": [],
      "total_rate": 3150}),

    ("""Load ID: LD-7742
Pickup: Los Angeles, CA 90001 on 3/4/24
Delivery: Dallas, TX 75201 by 3/7/24
Reefer set at 38F
Rate: $3,200 all-in (includes fuel)
Weight: ~38k lbs
Commodity: Fresh produce""",
     {"load_id": "LD-7742", "origin": {"city": "Los Angeles", "state": "CA", "zip": "90001"},
      "destination": {"city": "Dallas", "state": "TX", "zip": "75201"},
      "pickup_date_raw": "3/4/24", "delivery_date_raw": "3/7/24",
      "equipment_type": "reefer", "commodity": "Fresh produce", "weight_lbs": 38000,
      "line_haul_rate": None, "fuel_surcharge": None, "accessorials": None,
      "total_rate": 3200}),

    ("""REF: XZ-88131 | Dry Van
Pickup: Fresno, CA 93701 on July 2, 2025
Drop: Phoenix, AZ 85043 by 7/4/25
LINEHAUL $1,850.00   FSC $210.00
LUMPER FEE: $125.00 (at receiver)
DETENTION (2 hrs): $150.00
TOTAL CHARGES: $2,335.00
Weight 38000 lbs, commodity: canned goods""",
     {"load_id": "XZ-88131", "origin": {"city": "Fresno", "state": "CA", "zip": "93701"},
      "destination": {"city": "Phoenix", "state": "AZ", "zip": "85043"},
      "pickup_date_raw": "July 2, 2025", "delivery_date_raw": "7/4/25",
      "equipment_type": "van", "commodity": "canned goods", "weight_lbs": 38000,
      "line_haul_rate": 1850, "fuel_surcharge": 210,
      "accessorials": [
          {"description": "LUMPER FEE", "amount": 125},
          {"description": "DETENTION (2 hrs)", "amount": 150}
      ],
      "total_rate": 2335}),
]


# ─── Retryability classification ────────────────────────────────────
# Retry only errors where retrying can help. Auth/schema problems fail fast.

def _is_retryable(exc: Exception) -> bool:
    import openai
    if isinstance(exc, (
        openai.AuthenticationError, openai.PermissionDeniedError,
        openai.BadRequestError, openai.NotFoundError,
    )):
        return False
    if isinstance(exc, (
        openai.APITimeoutError, openai.APIConnectionError, openai.RateLimitError,
    )):
        return True
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return True
    # Name-based fallback keeps the policy intact across SDK versions and lets
    # tests inject plain fakes instead of constructing real SDK exceptions.
    name = type(exc).__name__
    if any(t in name for t in ("Auth", "Permission", "BadRequest", "NotFound", "Invalid")):
        return False
    if any(t in name for t in ("Timeout", "RateLimit", "Connection", "ServerError")):
        return True
    return False  # unknown → fail fast rather than spin


def _backoff(attempt: int) -> float:
    """1s after the first failure, 2s after the second."""
    return float(2 ** attempt)


def _is_schema_rejection(exc: Exception) -> bool:
    """Provider rejected our response_format/schema (not a transient error)."""
    m = str(exc).lower()
    return any(t in m for t in ("response_format", "json_schema", "strict", "'schema'"))


# ─── Extraction ─────────────────────────────────────────────────────

@dataclass
class RunDiagnostics:
    attempts: int = 0
    api_error: Optional[str] = None
    extraction_ms: float = 0.0
    validation_ms: float = 0.0
    total_ms: float = 0.0
    validation: Optional[ValidationResult] = None


def llm_extract(
    text: str, max_retries: int = MAX_RETRIES, client: Optional[OpenAI] = None,
) -> tuple[Extraction, int, Optional[str]]:
    """Returns (extraction, attempts_used, error_if_failed). Never raises.

    Schedule: attempt → wait 1s → attempt → wait 2s → attempt → give up with
    an explicit empty extraction. Retryable errors (timeout, connection,
    429, 5xx) back off; auth/schema errors abort immediately.
    """
    user = "Examples:\n" + "\n\n".join(
        f"IN:\n{inp}\n\nOUT:\n{json.dumps(out)}" for inp, out in FEW_SHOTS
    ) + f"\n\n---\nIN:\n{text}\n\nOUT:"

    make_client = client or get_client
    last_err: Optional[Exception] = None
    strict = STRICT_MODE

    for attempt in range(max_retries + 1):
        try:
            resp = make_client().chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "ext", "schema": SCHEMA, "strict": strict},
                },
                temperature=0,
            )
            return Extraction.model_validate_json(resp.choices[0].message.content), attempt + 1, None
        except Exception as e:
            last_err = e
            # Some OpenAI-compatible providers reject strict schemas outright.
            # Degrade to non-strict (the schema still guides generation; local
            # Pydantic validation remains the safety net) and retry immediately.
            if strict and _is_schema_rejection(e):
                strict = False
                continue
            # Malformed/invalid model output is transient — the next attempt
            # may parse fine — so it retries with backoff like network errors.
            retryable = isinstance(e, ValidationError) or _is_retryable(e)
            if not retryable or attempt == max_retries:
                return Extraction(), attempt + 1, f"{type(e).__name__}: {e}"
            time.sleep(_backoff(attempt))

    return Extraction(), max_retries + 1, f"{type(last_err).__name__}: {last_err}"  # pragma: no cover


def extract_and_validate(text: str) -> tuple[Output, RunDiagnostics]:
    """Full pipeline over raw text. Never raises."""
    diag = RunDiagnostics()
    t0 = perf_counter()
    ext, attempts, err = llm_extract(text)
    diag.attempts = attempts
    diag.api_error = err
    diag.extraction_ms = (perf_counter() - t0) * 1000

    t1 = perf_counter()
    result = validate(ext)
    conf = score_confidence(ext, result, retries=max(attempts - 1, 0))
    diag.validation_ms = (perf_counter() - t1) * 1000
    diag.validation = result

    out = build_output(ext, conf)
    diag.total_ms = (perf_counter() - t0) * 1000
    return out, diag


def run(text: str) -> Output:
    """Public convenience wrapper — output only, no diagnostics."""
    return extract_and_validate(text)[0]


def build_output(ext: Extraction, confidence: ConfidenceLevel) -> Output:
    def fmt(raw: Optional[str]) -> Optional[str]:
        d, _ = parse_date(raw)
        return d.isoformat() if d else None

    return Output(
        load_id=ext.load_id,
        origin=ext.origin,
        destination=ext.destination,
        pickup_date=fmt(ext.pickup_date_raw),
        delivery_date=fmt(ext.delivery_date_raw),
        equipment_type=ext.equipment_type,
        line_haul_rate=ext.line_haul_rate,
        fuel_surcharge=ext.fuel_surcharge,
        accessorials=ext.accessorials,
        total_rate=ext.total_rate,
        weight_lbs=ext.weight_lbs,
        commodity=ext.commodity,
        confidence=confidence,
    )
