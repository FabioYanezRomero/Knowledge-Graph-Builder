"""LM Studio cannot grow its window, so it has to refuse what will not fit.

Ollama takes a per-request num_ctx, so the ollama provider sizes the window to
the prompt. LM Studio fixes the window when the model is loaded and truncates an
over-long prompt server-side, returning a well-formed partial answer that is
indistinguishable from a model that simply answered badly. The only defence left
is to read the window and refuse.
"""
import pytest
import requests

from kgb.clients import LLMClientError
from kgb.clients.providers.lmstudio import assert_prompt_fits, context_limit

BASE = "http://localhost:1234/v1"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def _serving(monkeypatch, entries):
    seen = {}

    def fake_get(url, **kwargs):
        seen["url"] = url
        return FakeResponse({"data": entries})

    monkeypatch.setattr(requests, "get", fake_get)
    return seen


def test_reads_the_loaded_window_not_the_maximum(monkeypatch):
    # A model can be loaded with a window far below what it supports — the
    # loaded one is what truncates, so it is the one that matters.
    seen = _serving(monkeypatch, [
        {"id": "qwen", "state": "loaded", "max_context_length": 131072,
         "loaded_context_length": 8192},
    ])
    assert context_limit(BASE, "qwen") == 8192
    # /v1/models does not expose context length; the native API does.
    assert seen["url"] == "http://localhost:1234/api/v0/models"


def test_falls_back_to_the_only_loaded_model(monkeypatch):
    # The shipped default model_id is the placeholder "local-model", and LM Studio
    # answers with whatever is in memory regardless of the id we send.
    _serving(monkeypatch, [
        {"id": "nomic-embed", "state": "not-loaded", "max_context_length": 2048},
        {"id": "qwen", "state": "loaded", "loaded_context_length": 32768},
    ])
    assert context_limit(BASE, "local-model") == 32768


def test_ambiguous_load_is_not_guessed(monkeypatch):
    # Two loaded models and an id matching neither: guessing which one answers
    # would produce a limit we cannot stand behind.
    _serving(monkeypatch, [
        {"id": "a", "state": "loaded", "loaded_context_length": 4096},
        {"id": "b", "state": "loaded", "loaded_context_length": 65536},
    ])
    assert context_limit(BASE, "local-model") is None


def test_unreachable_server_disables_the_guard(monkeypatch):
    # A guard that breaks extraction when it cannot introspect is worse than no
    # guard: older LM Studio builds have no /api/v0 at all.
    def boom(url, **kwargs):
        raise requests.RequestException("no route")

    monkeypatch.setattr(requests, "get", boom)
    assert context_limit(BASE, "qwen") is None


def test_fits_comfortably_is_silent():
    assert_prompt_fits(32768, "x" * 1000, "extraction")


def test_unknown_limit_is_silent():
    assert_prompt_fits(None, "x" * 10_000_000, "extraction")


def test_overflow_refuses_and_says_how_to_fix_it():
    with pytest.raises(LLMClientError) as e:
        assert_prompt_fits(8192, "x" * 200_000, "augmentation")
    msg = str(e.value)
    assert "truncate it silently" in msg
    assert "8,192" in msg                    # the window it would have hit
    assert "Raise the context length" in msg  # and what to do about it


def test_headroom_is_reserved_for_the_answer():
    # A prompt that technically fits but leaves no room to reply is still broken.
    just_under_the_window = "x" * int(8192 * 3.5 * 0.95)
    with pytest.raises(LLMClientError):
        assert_prompt_fits(8192, just_under_the_window, "extraction")
