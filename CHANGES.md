# Changelog

## v0.3 — Architecture & reliability pass

**Itemized accessorials replace line-haul folding.** New `accessorials[]
[{description, amount}]` field on both the extraction schema and output
contract. The extraction prompt no longer folds carrier charges/lumpers/
detention into `line_haul_rate`; the financial model is now
`line_haul + fuel + Σ accessorials ≈ total`, and reconciliation runs whenever
line haul, total, and any breakdown item (fuel or an accessorial) are
present — so `Base $500 + Carrier Charge $200 = Total $700` reconciles with
the carrier charge visible as its own line (previously mis-modeled as fuel).

**Golden corpus expanded to 16 annotated records** (3 samples + 13 crafted
cases in `eval/corpus/`) covering: missing fields, multiple pickups/drops,
all-in rates, fuel surcharge, itemized carrier charges, zero-dollar rates,
ambiguous dates, malformed values, unusual equipment, conflicting totals,
and noisy email formatting. `accessorials` scored as a secondary field
(item-set match). Still crafted data — a methodology harness, not a quality
benchmark.

**Provider portability.** `OPENAI_BASE_URL` support for any OpenAI-compatible
endpoint; extraction schema sent fully inlined (`$defs`/`$ref` resolved) since
several providers don't support refs; automatic degradation from strict to
schema-guided JSON when a provider rejects strict mode (`OPENAI_STRICT`
override); `.env.example` slimmed to exactly what the app consumes. README
provider guidance reduced to one tested example instead of a table of models
that will age.

**Documentation honesty pass:** softened "never throws" to "LLM and
validation failures are handled gracefully" (filesystem/OS/bug failures can
still throw); model-version pinning (`gpt-4o-mini-2024-07-18`) documented for
reproducible evaluation per docs/EVAL.md §3.

**Packaged the pipeline (`src/ratecon_extract/`)** and split it by
responsibility: `models.py` (Extraction / Output / ValidationResult),
`validators.py` (dates, totals, equipment, missing fields), `confidence.py`
(scoring), `extractor.py` (OpenAI client, strict schema, retries, pipeline),
`cli.py`. Added a console script (`ratecon-extract`) and src-layout packaging.

**Replaced dynamic attribute hacks with `ValidationResult`.** The
`_totals_conflict` attribute attached via `object.__setattr__` is gone;
`validators.validate()` now returns a structured result (missing critical
fields, totals conflict + diff, date parsed/ambiguous flags, equipment flag,
human-readable `errors[]`) that confidence scoring consumes.

**Fixed bugs found by exercising the code:**
- Importing the package crashed when `OPENAI_API_KEY` was absent
  (module-level OpenAI client construction). The client is now lazy, with an
  explicit, actionable error at call time; `python-dotenv` is actually loaded.
- The strict-mode schema still violated OpenAI constraints after v0.2's fix:
  Pydantic emits `default`, `pattern` (Decimal string branch), and
  `minimum` (from `ge=0`) — all rejected keywords. `to_strict_schema()` now
  strips them recursively. Without this every API call 400s and the pipeline
  "gracefully" returns empty extractions forever.
- Falsy-zero bugs: $0 rates were treated as missing (`if x` → `is not None`)
  in output conversion, critical-field detection, optional-field counting,
  and scoring.
- `"3/3/26"` was flagged as an ambiguous date; identical day/month reads the
  same either way, so it no longer triggers the penalty.

**Reliability hardening:**
- Retries now use exponential backoff (1s, 2s) and a retryability classifier:
  timeout/connection/429/5xx retry; auth/bad-request fail fast; malformed
  model JSON retries then degrades. SDK-internal retries disabled so our
  policy governs.
- Request timeout (default 30s, env-configurable).
- Model is env-driven (`OPENAI_MODEL`), consistent with docs/EVAL.md §3's
  pin-the-version recommendation.

**Precision:** money stays `Decimal` through validation and scoring;
float conversion happens only in `Output.to_dict()` at the serialization
boundary.

**Observability:** structured logging to stderr (model, attempts, latency,
validation issues, API errors — never document contents); per-run latency
diagnostics via `extract_and_validate()`; CLI `--verbose`.

**Tests reorganized for zero-API determinism:** `tests/test_validation.py`
and `tests/test_confidence.py` are fully offline; new
`tests/test_pipeline.py` scripts mock LLM responses (malformed JSON, invalid
enums, missing fields, conflicting rates, timeouts, rate limits, auth
failures) to prove the never-crash contract; live smoke tests live in
`tests/test_extraction.py` and skip automatically without a key.

**Evaluation harness:** `eval/evaluate.py` + `eval/golden.jsonl` implement
the partial-credit rubric and FABR proxy from docs/EVAL.md on annotated
golden records (bundled: 3 samples — methodology demo, not a benchmark).

**Docs:** README rewritten around problem/solution/failure-path/limitations;
Carrier Match doc explicitly labeled as an unimplemented design proposal;
EVAL.md linked to its runnable subset.

## v0.2 — Correctness pass

A second pass focused on making the pipeline correct against real OpenAI
structured-output constraints and realistic rate-confirmation formats, not
just the happy path. Each item below was verified by exercising the actual
code, not just inspecting the diff.

**Schema now satisfies OpenAI's `strict: true` requirements.** Strict mode
requires every field to be listed in `required` (optionality is expressed via
a `null` type, not by omission) and every object to set `additionalProperties:
false`, recursively. `Extraction.model_json_schema()` did neither on its own —
added `to_strict_schema()` to rewrite the schema before it's sent to the API.

**Fixed a false-positive in totals reconciliation.** An all-in rate with no
line-item breakdown (a normal, common case) was being compared against a
missing line-haul/fuel value treated as `0`, so it was always flagged as a
conflict. Now only compares when both components are actually present.
Verified: `total_rate=3200` alone → no conflict; `total_rate=3200,
line_haul_rate=1000, fuel_surcharge=100` → still correctly flags the real
mismatch.

**Fixed equipment-type normalization so it actually runs.** It was a
`mode="after"` model validator, but Pydantic's own enum validation rejects an
unrecognized string during field validation, before an "after" validator ever
sees it — so an unrecognized synonym killed the whole extraction instead of
degrading to `OTHER`. Moved to a `mode="before"` field validator, which
intercepts the raw value first. Verified: `"Dry Van"` → `VAN`, `"Sprinter box
truck"` → `OTHER`, neither crashes.

**Date-ambiguity detection is now wired up and runs on raw text.** The
ambiguity heuristic needs the original, unformatted date string (e.g.
`"3/4/26"`) — once a date is resolved to ISO format, the formatting cue that
made it ambiguous is gone. The LLM now returns the raw date substring as
extracted; ambiguity is computed locally in Python. Verified: `"3/4/26"` →
ambiguous; `"March 15, 2026"` → not ambiguous.

**Removed a duplicated computation.** `score_confidence` now uses the
`missing_critical` list passed in from the caller instead of silently
recomputing the same thing internally.

**Minor:** simplified an `except` clause that listed `Exception` alongside
two of its own subclasses; removed unnecessary `\$` escapes in example text;
`test_extract.py` now resolves `samples/` relative to the test file instead
of the current working directory, so it collects correctly regardless of
where pytest is invoked from.
