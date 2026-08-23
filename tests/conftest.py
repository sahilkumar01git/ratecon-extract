"""Shared pytest setup and fixtures.

- Puts src/ on sys.path so `ratecon_extract` imports without requiring
  `pip install -e .` first (pytest's pythonpath ini covers this too; this
  makes direct invocation from any cwd work as well).
- Provides a fake OpenAI client factory so pipeline tests can script exact
  LLM behaviors — malformed JSON, timeouts, rate limits — with zero API calls.
"""
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
for p in (str(SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

SAMPLES_DIR = REPO_ROOT / "samples"


# ─── Fake OpenAI plumbing ───────────────────────────────────────────

class FakeChoice:
    def __init__(self, content):
        self.message = SimpleNamespace(content=content)


class FakeResponse:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    """Plays back a scripted list of contents/exceptions, one per call."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError("fake client called more times than scripted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return FakeResponse(item)


class FakeClient:
    def __init__(self, *script):
        self.chat = SimpleNamespace(completions=FakeCompletions(script))

    @property
    def completions(self):
        return self.chat.completions


@pytest.fixture
def fake_llm(monkeypatch):
    """Returns a factory: fake_llm(*script) patches get_client to a FakeClient."""
    def _install(*script) -> FakeClient:
        client = FakeClient(*script)
        monkeypatch.setattr("ratecon_extract.extractor.get_client", lambda: client)
        return client
    return _install


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Retry backoff must not slow the test suite."""
    monkeypatch.setattr("ratecon_extract.extractor.time.sleep", lambda s: None)
