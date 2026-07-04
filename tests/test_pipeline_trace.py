"""Per-stage trace: snapshot triples after each transforming stage."""

import json

import pytest

from kgb.pipeline import config as pipeline_config
from kgb.pipeline.steps.export import ExportJSONStep


@pytest.fixture
def stub_client(monkeypatch):
    monkeypatch.setattr(
        pipeline_config.ClientFactory, "create", staticmethod(lambda cfg: object())
    )


def _config(tmp_path, trace, steps):
    data = tmp_path / "in.jsonl"
    data.write_text(json.dumps({"id": "r1", "text": "t"}) + "\n")
    return {
        "client": {"type": "ollama", "model": "m"},
        "domain": "default",
        "input": {"file": str(data)},
        "output_dir": str(tmp_path / "out"),
        "trace": trace,
        "steps": steps,
    }


def test_trace_inserts_labeled_snapshots(tmp_path, stub_client):
    raw = _config(tmp_path, True, ["extract", "consolidate", "augment", "convert"])
    runner, _ = pipeline_config.build_pipeline_from_config(raw)

    # A snapshot ExportJSONStep is inserted after each transforming stage.
    snapshot_dirs = [
        s.output_dir.name for s in runner.steps
        if isinstance(s, ExportJSONStep) and s.output_dir.parent.name == "trace"
    ]
    assert snapshot_dirs == ["01_extract", "02_consolidate", "03_augment"]
    # convert (non-transforming) gets no snapshot
    assert not any("convert" in d for d in snapshot_dirs)


def test_no_trace_no_snapshots(tmp_path, stub_client):
    raw = _config(tmp_path, False, ["extract", "augment"])
    runner, _ = pipeline_config.build_pipeline_from_config(raw)
    assert not any(
        isinstance(s, ExportJSONStep) and s.output_dir.parent.name == "trace"
        for s in runner.steps
    )
