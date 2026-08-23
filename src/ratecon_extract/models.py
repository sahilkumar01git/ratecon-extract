"""Pydantic models: the LLM-facing schema (Extraction) and the public
output contract (Output), plus the shared ValidationResult that carries
every deterministic signal the rest of the pipeline consumes.

This module is deliberately logic-free apart from field-level coercion —
cross-field business rules live in validators.py, scoring in confidence.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class EquipmentType(str, Enum):
    VAN = "van"
    REEFER = "reefer"
    FLATBED = "flatbed"
    OTHER = "other"


class ConfidenceLevel(str, Enum):
    HIGH = "high"      # >= 0.75 → auto-populate
    MEDIUM = "medium"  # >= 0.45 → human review
    LOW = "low"        # below   → reject, manual entry


class Location(BaseModel):
    city: str
    state: str
    zip: Optional[str] = None

    @field_validator("state", mode="before")
    @classmethod
    def up(cls, v):
        return v.strip().upper() if isinstance(v, str) else v

    @field_validator("zip", mode="before")
    @classmethod
    def zip5(cls, v):
        return v.strip()[:5] if isinstance(v, str) else None


class Accessorial(BaseModel):
    """One itemized extra charge (detention, lumper, carrier charge, ...).

    Itemized separately instead of being folded into line_haul_rate: a
    carrier charge is not line haul, and silently relabeling it corrupts
    the exact number this pipeline exists to protect.
    """
    description: str
    amount: Decimal = Field(ge=0)


class Extraction(BaseModel):
    """Raw LLM output. Money stays Decimal until the serialization boundary."""

    load_id: Optional[str] = None
    origin: Optional[Location] = None
    destination: Optional[Location] = None
    pickup_date_raw: Optional[str] = None
    delivery_date_raw: Optional[str] = None
    equipment_type: Optional[EquipmentType] = None
    line_haul_rate: Optional[Decimal] = None
    fuel_surcharge: Optional[Decimal] = None
    accessorials: Optional[list[Accessorial]] = None
    total_rate: Optional[Decimal] = None
    weight_lbs: Optional[int] = None
    commodity: Optional[str] = None

    # Runs BEFORE Pydantic's enum validation so an unrecognized synonym
    # degrades to OTHER instead of failing the whole extraction.
    # Lazy import avoids a models ↔ validators import cycle.
    @field_validator("equipment_type", mode="before")
    @classmethod
    def coerce_equip(cls, v):
        from ratecon_extract.validators import normalize_equipment
        return normalize_equipment(v)


class Output(BaseModel):
    """Final public contract. Money fields stay Decimal internally; to_dict()
    is the only place they are converted (to float) for JSON serialization —
    financial math must never round-trip through binary floats."""

    load_id: Optional[str] = None
    origin: Optional[Location] = None
    destination: Optional[Location] = None
    pickup_date: Optional[str] = None
    delivery_date: Optional[str] = None
    equipment_type: Optional[EquipmentType] = None
    line_haul_rate: Optional[Decimal] = None
    fuel_surcharge: Optional[Decimal] = None
    accessorials: Optional[list[Accessorial]] = None
    total_rate: Optional[Decimal] = None
    weight_lbs: Optional[int] = None
    commodity: Optional[str] = None
    confidence: ConfidenceLevel

    def to_dict(self) -> dict:
        return {
            "load_id": self.load_id,
            "origin": self.origin.model_dump(mode="json") if self.origin else None,
            "destination": self.destination.model_dump(mode="json") if self.destination else None,
            "pickup_date": self.pickup_date,
            "delivery_date": self.delivery_date,
            "equipment_type": self.equipment_type.value if self.equipment_type else None,
            "line_haul_rate": float(self.line_haul_rate) if self.line_haul_rate is not None else None,
            "fuel_surcharge": float(self.fuel_surcharge) if self.fuel_surcharge is not None else None,
            "accessorials": (
                [{"description": a.description, "amount": float(a.amount)} for a in self.accessorials]
                if self.accessorials is not None else None
            ),
            "total_rate": float(self.total_rate) if self.total_rate is not None else None,
            "weight_lbs": self.weight_lbs,
            "commodity": self.commodity,
            "confidence": self.confidence.value,
        }


@dataclass
class ValidationResult:
    """Every deterministic signal computed by validators.validate().

    Confidence scoring consumes this instead of poking at private attributes,
    and each problem is also rendered as a human-readable `errors[]` entry so
    a reviewer can see *why* something scored the way it did.
    """

    missing_critical_fields: list[str] = field(default_factory=list)
    totals_conflict: bool = False
    totals_diff: Optional[float] = None
    pickup_parsed: bool = False
    delivery_parsed: bool = False
    ambiguous_pickup_date: bool = False
    ambiguous_delivery_date: bool = False
    invalid_equipment: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def has_critical_problems(self) -> bool:
        return bool(self.missing_critical_fields or self.errors)
