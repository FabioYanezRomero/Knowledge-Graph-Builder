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
    # Measured on a JIT-loaded qwen3.8-27b: LM Studio gave it a 4,096 window on a
    # model that supports 262,144. Reading the maximum would wave through prompts
    # 64x too big for the window actually in use.
    seen = _serving(monkeypatch, [
        {"id": "qwen", "state": "loaded", "max_context_length": 262144,
         "loaded_context_length": 4096},
    ])
    assert context_limit(BASE, "qwen") == 4096
    # /v1/models does not expose context length; the native API does.
    assert seen["url"] == "http://localhost:1234/api/v0/models"


def test_a_model_not_yet_loaded_reports_no_window(monkeypatch):
    # Its max_context_length is a ceiling, not the window LM Studio will pick
    # when it loads on the first request. Better unguarded than confidently wrong:
    # the load happens on that request, and every later one is checked for real.
    _serving(monkeypatch, [
        {"id": "qwen", "state": "not-loaded", "max_context_length": 262144},
    ])
    assert context_limit(BASE, "qwen") is None


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


def test_extraction_path_is_actually_guarded(monkeypatch):
    # The guard is only worth anything if it sits where the fully-rendered chunk
    # prompt exists. Measured live: a 16,000-char chunk renders to ~5,510 tokens,
    # so max_char_buffer alone cannot tell you whether it fits -- the
    # instructions and few-shot examples ride along.
    from kgb.clients.providers.lmstudio import LMStudioLanguageModel

    model = LMStudioLanguageModel(
        model_id="qwen", api_key="k", base_url=BASE, context_limit=4096,
    )
    with pytest.raises(LLMClientError):
        model._process_single_prompt("x" * 60_000, {})


def test_augment_path_is_actually_guarded(monkeypatch):
    # Consolidation's prompt scales with the entity count, so it outgrows a fixed
    # window first -- and it never touches the extraction seam.
    from kgb.clients import ClientConfig, ClientFactory

    _serving(monkeypatch, [{"id": "qwen", "state": "loaded",
                            "loaded_context_length": 4096}])

    def no_inference(*a, **k):
        raise AssertionError("refusal must happen before anything is sent")

    monkeypatch.setattr(requests, "post", no_inference)

    class Item(__import__("pydantic").BaseModel):
        head: str

    client = ClientFactory.create(ClientConfig(
        client_type="lmstudio", model_id="qwen", base_url=BASE, show_progress=False))
    with pytest.raises(LLMClientError, match="truncate it silently"):
        client.augment(text="x" * 60_000, prompt_description="p", format_type=Item)


def test_configured_timeout_reaches_the_http_client():
    # langextract's OpenAI provider builds its client with no timeout, so ours
    # used to be stored and ignored -- every extraction ran on the SDK default of
    # 600s and a slow local model surfaced it as "Request timed out".
    from kgb.clients.providers.lmstudio import LMStudioLanguageModel

    model = LMStudioLanguageModel(model_id="qwen", api_key="k", base_url=BASE,
                                  timeout=900)
    assert model._client.timeout == 900
