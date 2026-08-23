"""RateCon Extract — LLM-powered extraction pipeline for freight rate
confirmations, with deterministic validation and cost-aware confidence
scoring.

Public API:
    run(text) -> Output
    extract_and_validate(text) -> (Output, RunDiagnostics)
"""
from ratecon_extract.confidence import score_confidence
from ratecon_extract.extractor import extract_and_validate, llm_extract, run
from ratecon_extract.models import (
    ConfidenceLevel,
    EquipmentType,
    Extraction,
    Location,
    Output,
    ValidationResult,
)
from ratecon_extract.validators import parse_date, validate

__all__ = [
    "run",
    "extract_and_validate",
    "llm_extract",
    "validate",
    "parse_date",
    "score_confidence",
    "Extraction",
    "Output",
    "Location",
    "EquipmentType",
    "ConfidenceLevel",
    "ValidationResult",
]
