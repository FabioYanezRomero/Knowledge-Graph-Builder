"""Every backend's augment() survives the same malformed output.

augment() bypasses langextract in all three providers, so each parses raw model
output on its own. Ollama had salvage; the other two raised on the first bad
character, so one stray comma cost the whole consolidation. The malformations
are not provider-specific, so the repair should not be either -- and the only
way to know they all really got it is to feed all three the same broken bodies.
"""
import json
import sys
import types

import pytest
import requests
from pydantic import BaseModel

from kgb.clients import ClientConfig, ClientFactory


class Item(BaseModel):
    head: str
    relation: str
    tail: str


GOOD = [{"head": "a", "relation": "r", "tail": "b"},
        {"head": "c", "relation": "r", "tail": "d"}]

BROKEN = [
    pytest.param('[{"head": "a", "relation": "r", "tail": "b"},'
                 ' {"head": "c", "relation": "r", "tail": "d"},]',
                 id="trailing comma"),
    pytest.param('[{"head": "a", "relation": "r", "tail": "b"},'
                 ' {"head": "c", "relation": "r", "tail": "d"}, {"head": "e", "rela',
                 id="cut off at max tokens"),
    # Fatal for gemini only: ollama and lmstudio already stripped fences, gemini
    # fed them straight to json.loads despite asking for response_mime_type JSON.
    pytest.param('```json\n[{"head": "a", "relation": "r", "tail": "b"},'
                 ' {"head": "c", "relation": "r", "tail": "d"}]\n```',
                 id="markdown fences"),
    pytest.param('[{"head": "a\x00", "relation": "r", "tail": "b"},'
                 ' {"head": "c", "relation": "r", "tail": "d"}]',
                 id="control character"),
]


def _ollama(monkeypatch, body):
    class Resp:
        status_code = 200

        def raise_for_status(self): pass

        def json(self): return {"response": body}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    return ClientFactory.create(ClientConfig(
        client_type="ollama", model_id="m", timeout=10, show_progress=False))


def _lmstudio(monkeypatch, body):
    class Resp:
        status_code = 200

        def raise_for_status(self): pass

        def json(self): return {"choices": [{"message": {"content": body}}]}

    monkeypatch.setattr(requests, "post", lambda *a, **k: Resp())
    # No LM Studio behind this, so the context guard must find no limit and stay
    # out of the way -- exactly what an older build without /api/v0 looks like.
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(
        requests.RequestException("no server")))
    return ClientFactory.create(ClientConfig(
        client_type="lmstudio", model_id="m", timeout=10, show_progress=False))


def _gemini(monkeypatch, body):
    fake = types.ModuleType("google.generativeai")
    fake.configure = lambda **kw: None
    fake.GenerativeModel = lambda model_id: types.SimpleNamespace(
        generate_content=lambda *a, **k: types.SimpleNamespace(text=body))
    monkeypatch.setitem(sys.modules, "google.generativeai", fake)
    return ClientFactory.create(ClientConfig(
        client_type="gemini", model_id="m", api_key="k", show_progress=False))


@pytest.mark.parametrize("build", [_ollama, _lmstudio, _gemini],
                         ids=["ollama", "lmstudio", "gemini"])
@pytest.mark.parametrize("body", BROKEN)
def test_broken_augment_response_is_salvaged(monkeypatch, build, body):
    client = build(monkeypatch, body)
    out = client.augment(text="t", prompt_description="p", format_type=Item)
    # The control-character case rewrites the byte it strips; compare the shape
    # and the intact second object, which every case must recover in full.
    assert len(out) == 2
    assert {k: out[1][k] for k in ("head", "relation", "tail")} == GOOD[1]


@pytest.mark.parametrize("build", [_ollama, _lmstudio, _gemini],
                         ids=["ollama", "lmstudio", "gemini"])
def test_healthy_response_untouched(monkeypatch, build):
    client = build(monkeypatch, json.dumps(GOOD))
    out = client.augment(text="t", prompt_description="p", format_type=Item)
    assert [{k: o[k] for k in ("head", "relation", "tail")} for o in out] == GOOD
