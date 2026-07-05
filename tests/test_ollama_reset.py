"""Wedge recovery for local Ollama backends: retry-with-reset + proactive reset."""

import kgb.clients.providers.ollama as ollama_mod
from kgb.clients.providers.ollama import _ThinkInjectingRequests, OllamaClient


class FakeExc(Exception):
    pass


class _Exceptions:
    RequestException = FakeExc
    ReadTimeout = FakeExc


class FakeResp:
    def __init__(self, status_code=200):
        self.status_code = status_code


class FakeRequests:
    """Records every POST; fails the first ``fail_times`` real generate calls,
    then succeeds. Unload calls (keep_alive:0) always ack 200."""

    def __init__(self, fail_times=0, wedge_times=0):
        self.exceptions = _Exceptions()
        self.calls = []          # list of json payloads
        self._fail = fail_times
        self._wedge = wedge_times

    def post(self, url, *args, **kwargs):
        payload = kwargs.get("json", {}) or {}
        self.calls.append(payload)
        if payload.get("keep_alive") == 0:
            return FakeResp(200)  # unload ack
        if self._fail > 0:
            self._fail -= 1
            raise FakeExc("read timeout")
        if self._wedge > 0:
            self._wedge -= 1
            return FakeResp(503)  # server wedge
        return FakeResp(200)

    def unloads(self):
        return [c for c in self.calls if c.get("keep_alive") == 0]

    def generates(self):
        return [c for c in self.calls if c.get("keep_alive") != 0]


def _no_cooldown(monkeypatch):
    monkeypatch.setattr(ollama_mod, "_RESET_COOLDOWN", 0)


def test_retry_with_reset_on_timeout(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests(fail_times=1)
    shim = _ThinkInjectingRequests(real, think=False, options=None)
    resp = shim.post("http://x/api/generate", json={"model": "m", "prompt": "p"})
    assert resp.status_code == 200
    assert len(real.unloads()) == 1          # wedge cleared once
    assert len(real.generates()) == 2        # original + one retry


def test_retry_with_reset_on_5xx(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests(wedge_times=1)
    shim = _ThinkInjectingRequests(real, think=None, options=None)
    resp = shim.post("http://x/api/generate", json={"model": "m", "prompt": "p"})
    assert resp.status_code == 200
    assert len(real.unloads()) == 1


def test_gives_up_after_max_retries(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests(fail_times=99)
    shim = _ThinkInjectingRequests(real, think=None, options=None)
    try:
        shim.post("http://x/api/generate", json={"model": "m", "prompt": "p"})
        assert False, "should have raised after exhausting retries"
    except FakeExc:
        pass
    assert len(real.generates()) == ollama_mod._MAX_RESET_RETRIES + 1


def test_think_and_options_still_injected(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests()
    shim = _ThinkInjectingRequests(real, think=False, options={"num_ctx": 4096})
    shim.post("http://x/api/generate", json={"model": "m", "prompt": "p"})
    sent = real.generates()[0]
    assert sent["think"] is False
    assert sent["options"]["num_ctx"] == 4096


def test_proactive_reset_every_n(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests()
    monkeypatch.setattr(ollama_mod, "requests", real)
    c = OllamaClient(model_id="m", base_url="http://x", reset_every=2)
    for _ in range(4):
        c._maybe_reset()
    assert len(real.unloads()) == 2          # reset after call 2 and call 4


def test_no_reset_when_disabled(monkeypatch):
    _no_cooldown(monkeypatch)
    real = FakeRequests()
    monkeypatch.setattr(ollama_mod, "requests", real)
    c = OllamaClient(model_id="m", base_url="http://x", reset_every=None)
    for _ in range(5):
        c._maybe_reset()
    assert real.unloads() == []
