"""Per-step client overrides in YAML pipeline configs."""

import json

import pytest

from kgb.pipeline import config as pipeline_config


class DummyClient:
    def __init__(self, cfg):
        self.cfg = cfg


@pytest.fixture
def created(monkeypatch):
    """Stub ClientFactory.create and record every ClientConfig it receives."""
    configs = []

    def fake_create(cfg):
        configs.append(cfg)
        return DummyClient(cfg)

    monkeypatch.setattr(pipeline_config.ClientFactory, "create", staticmethod(fake_create))
    return configs


def _base_config(tmp_path, steps):
    data = tmp_path / "in.jsonl"
    data.write_text(json.dumps({"id": "r1", "text": "some text"}) + "\n")
    return {
        "client": {"type": "ollama", "model": "gemma3:1b", "temperature": 0.0},
        "domain": "default",
        "input": {"file": str(data)},
        "steps": steps,
    }


def test_step_client_override(tmp_path, created):
    raw = _base_config(tmp_path, [
        "extract",
        {"augment": {"client": {"model": "gemma3:27b", "temperature": 0.5}}},
    ])
    runner, _ = pipeline_config.build_pipeline_from_config(raw)

    assert len(created) == 2  # default + augment override
    assert created[0].model_id == "gemma3:1b"
    assert created[1].model_id == "gemma3:27b"
    assert created[1].temperature == 0.5
    extract_step, augment_step = runner.steps[0], runner.steps[1]
    assert extract_step.client.cfg is created[0]
    assert augment_step.client.cfg is created[1]
    assert augment_step.temperature == 0.5


def test_identical_override_shares_default_client(tmp_path, created):
    raw = _base_config(tmp_path, [
        {"extract": {"client": {"model": "gemma3:1b"}}},
        "augment",
    ])
    runner, _ = pipeline_config.build_pipeline_from_config(raw)
    assert len(created) == 1  # override equals default -> cached
    assert runner.steps[0].client is runner.steps[1].client


def test_provider_switch_drops_provider_fields(tmp_path, created):
    raw = _base_config(tmp_path, ["extract"])
    raw["client"]["base_url"] = "http://localhost:11434"
    raw["steps"] = [{"extract": {"client": {"type": "gemini", "model": "gemini-2.0-flash"}}}]
    pipeline_config.build_pipeline_from_config(raw)

    override_cfg = created[-1]
    assert override_cfg.client_type == "gemini"
    assert override_cfg.model_id == "gemini-2.0-flash"
    # ollama's base_url must not leak into the gemini client
    assert not getattr(override_cfg, "base_url", None)
