"""langextract's fuzzy aligner must stay off, and our kwargs must be accepted.

The aligner is difflib.SequenceMatcher(autojunk=False); on chunks past ~16k chars
it degrades to hours of pure CPU with the LLM already idle — a hang that looks
exactly like a wedged local backend. Nothing is lost by disabling it:
extract_triples discards langextract's prompt-relative offsets and re-anchors
every span to the source document by exact find.

The flag is NOT a top-level lx.extract() parameter — it is read out of
resolver_params — so the fake below binds against the real signature. A test that
only asserted "we passed the flag" happily passed while every real extraction
died on `unexpected keyword argument`.
"""

import inspect

import pytest
from langextract import extraction as lx_extraction

from kgb.clients import ClientConfig, ClientFactory

# NOT lx.extract — the public alias is a (*args, **kwargs) passthrough that
# accepts anything and only fails deep inside, which is how a bad kwarg reached
# production in the first place.
_REAL_SIGNATURE = inspect.signature(lx_extraction.extract)


@pytest.fixture
def captured(monkeypatch):
    """Record each provider's lx.extract kwargs, rejecting ones it wouldn't take."""
    seen = {}

    class _Result:
        extractions = []

    def fake_extract(**kwargs):
        _REAL_SIGNATURE.bind_partial(**kwargs)  # TypeError on an unknown kwarg
        seen.update(kwargs)
        return _Result()

    for provider in ("ollama", "lmstudio", "gemini"):
        module = __import__(f"kgb.clients.providers.{provider}", fromlist=["lx"])
        monkeypatch.setattr(module.lx, "extract", fake_extract)
    return seen


@pytest.mark.parametrize("client_type", ["ollama", "lmstudio", "gemini"])
def test_fuzzy_alignment_off_by_default(client_type, captured):
    client = ClientFactory.create(ClientConfig(
        client_type=client_type, model_id="m", api_key="k",
        base_url="http://localhost:1", show_progress=False,
    ))
    client.extract(text="some text", prompt_description="d", examples=[])
    assert captured["resolver_params"]["enable_fuzzy_alignment"] is False


def test_alignment_flag_is_a_resolver_param_not_a_top_level_kwarg():
    # Guards the mistake this test file exists because of.
    assert "enable_fuzzy_alignment" not in _REAL_SIGNATURE.parameters
    assert "resolver_params" in _REAL_SIGNATURE.parameters
