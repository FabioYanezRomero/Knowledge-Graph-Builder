"""num_ctx is sized to the prompt actually being sent, not to a config guess.

Ollama truncates an over-long prompt with no error and no flag, so an undersized
window looks exactly like a model that answered badly. This is what silently
truncated consolidation once a stronger model found 3x the entities on the same
document under the same config.
"""
from kgb.clients.providers.ollama import _MAX_NUM_CTX, _fit_num_ctx
from kgb.clients.providers.ollama import _ThinkInjectingRequests
from tests.test_ollama_reset import FakeRequests, _no_cooldown


def _payload(prompt_chars, num_ctx=None):
    p = {"model": "m", "prompt": "x" * prompt_chars}
    if num_ctx:
        p["options"] = {"num_ctx": num_ctx}
    return p


def test_comfortable_prompt_left_alone():
    p = _payload(1_000, num_ctx=8192)
    _fit_num_ctx(p)
    assert p["options"]["num_ctx"] == 8192


def test_never_shrinks_a_generous_window():
    # A caller who deliberately asked for a big window keeps it.
    p = _payload(1_000, num_ctx=65536)
    _fit_num_ctx(p)
    assert p["options"]["num_ctx"] == 65536


def test_grows_to_fit_and_leaves_room_to_answer():
    # The real case: 363 entities -> ~112k chars of consolidation prompt, sent
    # under a num_ctx of 32768 that was ample when the graph had 126 entities.
    p = _payload(112_000, num_ctx=32768)
    _fit_num_ctx(p)
    fitted = p["options"]["num_ctx"]
    assert fitted > 32768
    assert fitted >= 112_000 / 3.5 / 0.85   # prompt fits AND ~15% is left over


def test_sets_a_window_when_none_was_configured():
    # No num_ctx means Ollama's default (commonly 4096) truncates in silence.
    p = _payload(112_000)
    _fit_num_ctx(p)
    assert p["options"]["num_ctx"] >= 112_000 / 3.5


def test_past_the_cap_it_says_so(capsys):
    p = _payload(_MAX_NUM_CTX * 5, num_ctx=8192)
    _fit_num_ctx(p)
    assert p["options"]["num_ctx"] == _MAX_NUM_CTX
    assert "truncate it silently" in capsys.readouterr().out


def test_applied_on_the_wire(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests()
    shim = _ThinkInjectingRequests(real, think=False, options={"num_ctx": 8192})
    shim.post("http://x/api/generate", json={"model": "m", "prompt": "x" * 200_000})
    assert real.generates()[0]["options"]["num_ctx"] > 8192
