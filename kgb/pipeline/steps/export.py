"""JSON export pipeline step."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..context import PipelineContext
from ..step import register_step


@register_step("export-json")
class ExportJSONStep:
    """Pipeline step for saving extracted graph triples formatted to JSON."""
    
    def __init__(self, output_dir: Path | str):
        """Initialize the export step.
        
        Args:
            output_dir: Directory to save the output JSON elements.
        """
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir

    def process(self, context: PipelineContext, **kwargs: Any) -> PipelineContext:
        """Export the triples mapped into JSON.
        
        Args:
            context: The pipeline context containing the triples payload.
            **kwargs: Unused.
            
        Returns:
            PipelineContext updated with artifact paths.
        """
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{context.record_id}.json"
            
            records = [t.model_dump() for t in context.triples]
            records += [{"head": e, "relation": "", "tail": ""} for e in context.entities]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
                
            # Log output artifacts mapping
            context.artifacts["export_json_path"] = str(output_path)
            
        except Exception as e:
            context.errors.append(f"JSON Export failed: {str(e)}")
            
        return context

__all__ = ["ExportJSONStep"]
