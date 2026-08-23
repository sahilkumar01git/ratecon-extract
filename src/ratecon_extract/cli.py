"""Typer CLI: single-file and batch extraction.

Machine-readable JSON on stdout; structured logs (model, attempts, latency,
validation failures, API errors) on stderr via the logging module. Rate
confirmation bodies are never logged — only file names and pipeline signals.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import typer

from ratecon_extract.extractor import MODEL, extract_and_validate

logger = logging.getLogger("ratecon_extract")

app = typer.Typer(help="Extract validated, confidence-scored JSON from freight rate confirmations.")


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@app.command()
def extract(
    file: str = typer.Argument(..., help="Path to a plain-text rate confirmation."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="JSON + diagnostics after it."),
):
    """Extract one rate confirmation -> JSON on stdout."""
    _configure_logging(verbose)
    text = Path(file).read_text(encoding="utf-8", errors="replace")
    t0 = time.perf_counter()
    out, diag = extract_and_validate(text)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    print(json.dumps(out.to_dict(), indent=2))

    if diag.api_error:
        logger.error("api_error model=%s attempts=%d error=%s", MODEL, diag.attempts, diag.api_error)
    else:
        logger.info(
            "extracted file=%s model=%s attempts=%d confidence=%s total_ms=%.0f",
            Path(file).name, MODEL, diag.attempts, out.confidence.value,
            elapsed_ms,
        )
        if diag.validation is not None:
            for err in diag.validation.errors:
                logger.warning("validation_failure file=%s issue=%s", Path(file).name, err)

    if verbose:
        print(
            f"\n# diagnostics\n"
            f"# model: {MODEL}\n"
            f"# attempts: {diag.attempts}\n"
            f"# extraction latency: {diag.extraction_ms:.0f} ms\n"
            f"# validation latency: {diag.validation_ms:.1f} ms\n"
            f"# total latency: {elapsed_ms:.0f} ms",
            file=sys.stderr,
        )


@app.command()
def batch(
    dir: str = typer.Argument(..., help="Directory containing .txt rate confirmations."),
):
    """Process every .txt in a directory. One result line per file on stdout."""
    _configure_logging(False)
    files = sorted(Path(dir).glob("*.txt"))
    if not files:
        logger.warning("no .txt files found in %s", dir)
        raise typer.Exit(code=1)

    for f in files:
        out, diag = extract_and_validate(f.read_text(encoding="utf-8", errors="replace"))
        print(f"{f.name}: {out.confidence.value} | load_id={out.load_id} | total={out.total_rate}")
        if diag.api_error:
            logger.error("api_error file=%s attempts=%d error=%s", f.name, diag.attempts, diag.api_error)
        else:
            logger.info(
                "extracted file=%s attempts=%d validation_issues=%d",
                f.name, diag.attempts,
                len(diag.validation.errors) if diag.validation else 0,
            )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
