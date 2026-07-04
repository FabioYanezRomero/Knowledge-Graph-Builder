"""Checkpoint step: snapshot the full record state at a point in the pipeline.

Unlike export-json (which writes only triples for downstream conversion), a
checkpoint also writes the accumulated per-stage metadata — the decisions each
stage made (e.g. entity_resolution's acronym_merges, vetoed_mappings,
rejected_mappings). That is what makes per-stage failures diagnosable: you can
compare triples across checkpoints AND read why they changed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..context import PipelineContext
from ..step import register_step


@register_step("checkpoint")
class CheckpointStep:
    """Write a record's triples and metadata to a labeled checkpoint directory."""

    def __init__(self, output_dir: Path | str):
        self.output_dir = Path(output_dir)

    def process(self, context: PipelineContext, **kwargs: Any) -> PipelineContext:
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)

            triples_path = self.output_dir / f"{context.record_id}.json"
            with open(triples_path, "w", encoding="utf-8") as f:
                json.dump([t.model_dump() for t in context.triples], f, ensure_ascii=False, indent=2)

            meta_path = self.output_dir / f"{context.record_id}.meta.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(context.metadata, f, ensure_ascii=False, indent=2, default=str)

            context.artifacts[f"checkpoint::{self.output_dir.name}"] = str(triples_path)
        except Exception as e:
            context.errors.append(f"Checkpoint failed: {str(e)}")

        return context


__all__ = ["CheckpointStep"]
