"""Reasoning off by default, and never at the cost of working at all.

LM Studio's counterpart of the ollama provider's `think:false`. Measured on
qwen3.8-27b with one trivial prompt: 100 completion tokens with reasoning on,
15 with it off. Extraction pays that difference on every chunk.
"""
import pytest

from kgb.clients.providers.lmstudio import LMStudioLanguageModel, _rejects_parameter

BASE = "http://localhost:1234/v1"


class FakeCreate:
    """Records calls; optionally rejects the first one like a picky server."""

    def __init__(self, reject=None):
        self.calls = []
        self.reject = reject

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.reject and len(self.calls) == 1:
            raise self.reject
        return "response"


def _model(monkeypatch, create, **kw):
    m = LMStudioLanguageModel(model_id="qwen", api_key="k", base_url=BASE, **kw)
    monkeypatch.setattr(m._client.chat.completions, "create", create)
    return m


class Rejected(Exception):
    status_code = 400

    def __str__(self):
        return "Unknown parameter: reasoning_effort"


def test_sent_flat_because_the_nested_form_is_ignored(monkeypatch):
    # langextract's _normalize_reasoning_params rewrites a flat reasoning_effort
    # into {"reasoning": {"effort": ..}}. Measured against LM Studio: nested is
    # ignored (57 reasoning tokens), flat works (0). Normalising would silently
    # undo the setting, so this path must bypass it.
    create = FakeCreate()
    m = _model(monkeypatch, create)
    m._create({"model": "qwen"}, m.reasoning_effort)
    assert create.calls[0]["reasoning_effort"] == "none"
    assert "reasoning" not in create.calls[0]


def test_default_is_off():
    assert LMStudioLanguageModel(model_id="q", api_key="k",
                                 base_url=BASE).reasoning_effort == "none"


def test_a_picky_server_costs_one_request_not_every_request(monkeypatch):
    create = FakeCreate(reject=Rejected())
    m = _model(monkeypatch, create)
    assert m._create({"model": "qwen"}, m.reasoning_effort) == "response"
    assert "reasoning_effort" not in create.calls[1]   # retried without it
    m._create({"model": "qwen"}, m.reasoning_effort)
    assert "reasoning_effort" not in create.calls[2]   # and stays off
    assert len(create.calls) == 3                      # no second failed attempt


def test_a_real_failure_is_not_swallowed(monkeypatch):
    class Overloaded(Exception):
        status_code = 500

    create = FakeCreate(reject=Overloaded())
    m = _model(monkeypatch, create)
    with pytest.raises(Overloaded):
        m._create({"model": "qwen"}, m.reasoning_effort)
    assert len(create.calls) == 1  # a 500 must not look like success after retry


def test_a_400_about_something_else_is_not_swallowed():
    class BadRequest(Exception):
        status_code = 400

        def __str__(self):
            return "context length exceeded"

    assert not _rejects_parameter(BadRequest())


def test_explicitly_disabled_sends_nothing(monkeypatch):
    create = FakeCreate()
    m = _model(monkeypatch, create, reasoning_effort=None)
    m._create({"model": "qwen"}, m.reasoning_effort)
    assert "reasoning_effort" not in create.calls[0]
