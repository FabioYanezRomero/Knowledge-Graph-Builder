"""Extraction pipeline step."""

from __future__ import annotations

import json
from typing import Any
from pathlib import Path

from ...builder import extract_triples
from ...clients import BaseLLMClient
from ...domains import KnowledgeDomain

from ..context import PipelineContext
from ..step import register_step


@register_step("extract")
class ExtractionStep:
    """Pipeline step for initial knowledge graph extraction."""
    
    def __init__(
        self, 
        client: BaseLLMClient, 
        domain: KnowledgeDomain, 
        temperature: float = 0.0,
        prompt_override: str | None = None
    ):
        """Initialize the extraction step.
        
        Args:
            client: Instantiated LLM client to use.
            domain: Knowledge domain defining the context and extraction prompt.
            temperature: Sampling temperature for inference.
            prompt_override: Optional override for the prompt text.
        """
        self.client = client
        self.domain = domain
        self.temperature = temperature
        self.prompt_override = prompt_override

    def process(self, context: PipelineContext, **kwargs: Any) -> PipelineContext:
        """Execute text extraction and update context triples.
        
        Args:
            context: The pipeline context containing input text and record id.
            **kwargs: Unused for extraction.
            
        Returns:
            PipelineContext updated with extracted triples.
        """
        try:
            triples, entities = extract_triples(
                client=self.client,
                domain=self.domain,
                text=context.text,
                record_id=context.record_id,
                temperature=self.temperature,
                prompt_override=self.prompt_override
            )
            context.triples.extend(triples)
            context.entities.extend(entities)
        except Exception as e:
            context.errors.append(f"Extraction failed: {str(e)}")
            
        return context

__all__ = ["ExtractionStep"]
