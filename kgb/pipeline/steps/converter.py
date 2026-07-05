"""GraphML conversion pipeline step."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ...io.writers.graphml import json_to_graphml
from ..context import PipelineContext
from ..step import register_step


@register_step("convert")
class ConverterStep:
    """Pipeline step for converting graph triples into GraphML format."""
    
    def __init__(self, output_dir: Path | str):
        """Initialize the converter step.
        
        Args:
            output_dir: Directory where the generated GraphML file should be saved.
        """
        self.output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir

    def process(self, context: PipelineContext, **kwargs: Any) -> PipelineContext:
        """Convert accumulated logic triples into GraphML format and save.
        
        Args:
            context: The pipeline context with current triples.
            **kwargs: Unused.
            
        Returns:
            PipelineContext with graphml extraction artifact metadata attached.
        """
        if not context.triples and not context.entities:
            context.metadata["convert_skipped"] = "True (no triples or entities to convert)"
            return context

        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = self.output_dir / f"{context.record_id}.graphml"
            
            # Using the json_to_graphml method which accepts lists of Triples directly
            json_to_graphml(triples=context.triples, output_path=output_path, entities=context.entities)
            
            context.artifacts["graphml_path"] = str(output_path)
            
        except Exception as e:
            context.errors.append(f"GraphML Conversion failed: {str(e)}")
            
        return context

__all__ = ["ConverterStep"]
