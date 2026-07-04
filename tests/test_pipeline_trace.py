"""Per-stage trace: checkpoint triples + metadata after each transforming stage."""

import json

import pytest

from kgb.domains import Triple
from kgb.pipeline import config as pipeline_config
from kgb.pipeline.context import PipelineContext
from kgb.pipeline.steps.checkpoint import CheckpointStep


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


def test_trace_inserts_labeled_checkpoints(tmp_path, stub_client):
    raw = _config(tmp_path, True, ["extract", "consolidate", "augment", "convert"])
    runner, _ = pipeline_config.build_pipeline_from_config(raw)

    checkpoint_dirs = [
        s.output_dir.name for s in runner.steps
        if isinstance(s, CheckpointStep) and s.output_dir.parent.name == "trace"
    ]
    assert checkpoint_dirs == ["01_extract", "02_consolidate", "03_augment"]  # convert excluded


def test_no_trace_no_checkpoints(tmp_path, stub_client):
    raw = _config(tmp_path, False, ["extract", "augment"])
    runner, _ = pipeline_config.build_pipeline_from_config(raw)
    assert not any(
        isinstance(s, CheckpointStep) and s.output_dir.parent.name == "trace"
        for s in runner.steps
    )


def test_checkpoint_writes_triples_and_metadata(tmp_path):
    ctx = PipelineContext(record_id="r1", text="t")
    ctx.triples = [Triple(head="a", relation="r", tail="b")]
    ctx.metadata["consolidation_entity_resolution"] = {
        "vetoed_mappings": [("IL-10", "IL-6")],
        "acronym_merges": 1,
    }
    CheckpointStep(output_dir=tmp_path / "01_consolidate").process(ctx)

    triples = json.loads((tmp_path / "01_consolidate" / "r1.json").read_text())
    meta = json.loads((tmp_path / "01_consolidate" / "r1.meta.json").read_text())
    assert triples[0]["head"] == "a"
    assert meta["consolidation_entity_resolution"]["acronym_merges"] == 1
