"""LM Studio cannot size its window per request, so all we can do is report it.

Ollama takes a per-request num_ctx and the ollama provider grows it to fit the
prompt. LM Studio picks the window when the model is loaded, so there is nothing
to grow -- we read what it reports and say when a prompt is bigger.

We say it rather than refuse it. The first version raised, assuming LM Studio
enforces the reported window. Measured against LM Studio on the MLX runtime it
does not: an 8,340-token prompt against a reported 4,096 window came back
correct, including a fact planted in its last line. Refusing would have blocked
work that demonstrably succeeds.
"""
import requests

from kgb.clients.providers.lmstudio import context_limit, warn_if_prompt_exceeds_window

BASE = "http://localhost:1234/v1"


def _completion(content):
    """The shape LMStudioLanguageModel reads out of the OpenAI SDK response."""
    import types
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content=content))])


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


def test_fits_comfortably_is_silent(capsys):
    warn_if_prompt_exceeds_window(32768, "x" * 1000, "extraction")
    assert capsys.readouterr().out == ""


def test_unknown_limit_is_silent(capsys):
    warn_if_prompt_exceeds_window(None, "x" * 10_000_000, "extraction")
    assert capsys.readouterr().out == ""


def test_overflow_reports_without_blocking(capsys):
    # Warn, do not raise: an over-window prompt was measured to succeed, so
    # refusing would block real work on a number we cannot enforce.
    warn_if_prompt_exceeds_window(8192, "x" * 200_000, "augmentation")
    out = capsys.readouterr().out
    assert "8,192" in out                  # the window it went over
    assert "if results look thin" in out   # and what to do if it mattered


def test_headroom_counts_the_answer_too(capsys):
    # A prompt that fits with nothing left to reply into is worth mentioning.
    warn_if_prompt_exceeds_window(8192, "x" * int(8192 * 3.5 * 0.95), "extraction")
    assert "[LM Studio]" in capsys.readouterr().out


def test_extraction_path_reports_and_carries_on(monkeypatch, capsys):
    # The report is only worth anything where the fully-rendered chunk prompt
    # exists. Measured live: a 16,000-char chunk renders to ~5,510 tokens, so
    # max_char_buffer alone cannot tell you the size -- the instructions and
    # few-shot examples ride along.
    from kgb.clients.providers.lmstudio import LMStudioLanguageModel

    model = LMStudioLanguageModel(
        model_id="qwen", api_key="k", base_url=BASE, context_limit=4096)
    monkeypatch.setattr(model, "_create", lambda *a, **k: _completion("[]"))
    model._process_single_prompt("x" * 60_000, {})
    assert "[LM Studio] extraction prompt" in capsys.readouterr().out


def test_augment_path_reports_and_carries_on(monkeypatch, capsys):
    # Consolidation's prompt scales with the entity count, so it goes over first
    # -- and it never touches the extraction seam.
    from kgb.clients import ClientConfig, ClientFactory

    _serving(monkeypatch, [{"id": "qwen", "state": "loaded",
                            "loaded_context_length": 4096}])

    class Resp:
        status_code = 200

        def raise_for_status(self): pass

        def json(self): return {"choices": [{"message": {"content": "[]"}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())

    class Item(__import__("pydantic").BaseModel):
        head: str

    client = ClientFactory.create(ClientConfig(
        client_type="lmstudio", model_id="qwen", base_url=BASE, show_progress=False))
    assert client.augment(text="x" * 60_000, prompt_description="p",
                          format_type=Item) == []
    assert "[LM Studio] augmentation prompt" in capsys.readouterr().out


def test_configured_timeout_reaches_the_http_client():
    # langextract's OpenAI provider builds its client with no timeout, so ours
    # used to be stored and ignored -- every extraction ran on the SDK default of
    # 600s and a slow local model surfaced it as "Request timed out".
    from kgb.clients.providers.lmstudio import LMStudioLanguageModel

    model = LMStudioLanguageModel(model_id="qwen", api_key="k", base_url=BASE,
                                  timeout=900)
    assert model._client.timeout == 900
