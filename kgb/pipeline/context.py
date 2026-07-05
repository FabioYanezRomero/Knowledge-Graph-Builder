"""Core data structures for the knowledge graph pipeline."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from ..domains.models import Triple


class PipelineContext(BaseModel):
    """Encapsulates the state of a document through varying pipeline steps.
    
    This object is passed from step to step in the pipeline. Each step reads
    from and potentially modifies this context.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    record_id: str = Field(description="Unique identifier for the input record/document.")
    text: str = Field(description="The source text of the document.")
    
    triples: list[Triple] = Field(
        default_factory=list,
        description="The extracted/augmented knowledge graph triples for this document."
    )

    entities: list[str] = Field(
        default_factory=list,
        description="Grounded standalone entities with no relation stated in the text; "
                    "carried as isolated graph nodes (extraction is grounding-only, so it "
                    "never fabricates a relation just to attach them)."
    )
    
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="General purpose dictionary for storing arbitrary intermediate step metadata."
    )
    
    errors: list[str] = Field(
        default_factory=list,
        description="List of error messages encountered during pipeline execution for this document."
    )
    
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Dictionary mapping artifact names (e.g., 'graphml_path') to artifact data or file paths."
    )
