"""Consolidate vs augment taxonomy: strategy kinds, domain folders, pipeline step."""

import json

import pytest

from kgb.builder.strategies import list_strategies, strategy_kind
from kgb.domains import get_domain
from kgb.pipeline import config as pipeline_config
from kgb.pipeline.steps.augmentation import AugmentationStep, ConsolidationStep


def test_strategy_kinds():
    assert strategy_kind("connectivity") == "augment"
    assert strategy_kind("entity_resolution") == "consolidate"
    assert "connectivity" in list_strategies(kind="augment")
    assert "entity_resolution" in list_strategies(kind="consolidate")
    assert set(list_strategies()) >= {"connectivity", "entity_resolution"}


def test_domain_loads_strategy_from_consolidation_folder():
    legal = get_domain("legal")
    assert "entity_resolution" in legal.list_augmentation_strategies()
    component = legal.get_augmentation("entity_resolution")
    assert component.prompt  # loaded from consolidation/entity_resolution/


def test_domain_backcompat_augmentation_folder(tmp_path):
    # external domains that keep strategies under augmentation/ still work
    domain_dir = tmp_path / "old_style"
    (domain_dir / "extraction").mkdir(parents=True)
    (domain_dir / "extraction" / "prompt_open.md").write_text("extract")
    strategy = domain_dir / "augmentation" / "entity_resolution"
    strategy.mkdir(parents=True)
    (strategy / "prompt.md").write_text("resolve entities")
    d = get_domain(str(domain_dir))
    assert d.get_augmentation("entity_resolution").prompt == "resolve entities"


def test_consolidate_pipeline_step(tmp_path, monkeypatch):
    class DummyClient:
        def __init__(self, cfg):
            self.cfg = cfg

    monkeypatch.setattr(
        pipeline_config.ClientFactory, "create", staticmethod(lambda cfg: DummyClient(cfg))
    )
    data = tmp_path / "in.jsonl"
    data.write_text(json.dumps({"id": "r1", "text": "t"}) + "\n")
    raw = {
        "client": {"type": "ollama", "model": "gemma3:1b"},
        "domain": "default",
        "input": {"file": str(data)},
        "steps": ["extract", "consolidate", "augment"],
    }
    runner, _ = pipeline_config.build_pipeline_from_config(raw)
    consolidate = runner.steps[1]
    assert isinstance(consolidate, ConsolidationStep)
    assert consolidate.strategy == "entity_resolution"
    assert isinstance(runner.steps[2], AugmentationStep)


def test_kind_mismatch_warns(capsys):
    AugmentationStep(client=object(), domain=get_domain("default"), strategy="entity_resolution")
    assert "consolidate" in capsys.readouterr().err


def test_missing_strategy_skips_without_erroring(tmp_path, capsys):
    # domain with only extraction, no augmentation/consolidation strategies
    domain_dir = tmp_path / "bare"
    (domain_dir / "extraction").mkdir(parents=True)
    (domain_dir / "extraction" / "prompt_open.md").write_text("extract")
    domain = get_domain(str(domain_dir))

    from kgb.domains import Triple
    from kgb.pipeline.context import PipelineContext

    step = ConsolidationStep(client=object(), domain=domain, strategy="entity_resolution")
    ctx = PipelineContext(record_id="r1", text="t")
    ctx.triples = [Triple(head="a", relation="r", tail="b")]
    out = step.process(ctx)

    assert out.errors == []                      # record NOT failed
    assert len(out.triples) == 1                 # triples preserved for downstream steps
    assert "consolidation_skipped" in out.metadata
    assert "no 'entity_resolution' strategy" in capsys.readouterr().err
