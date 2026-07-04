"""Graph-operation pipeline steps: augment (adds knowledge) and
consolidate (merges/cleans without adding any)."""

from __future__ import annotations

import sys
from typing import Any

from ...builder import augment_triples
from ...builder.augmentation import strategy_kind
from ...clients import BaseLLMClient
from ...domains import KnowledgeDomain

from ..context import PipelineContext
from ..step import register_step


@register_step("augment")
class AugmentationStep:
    """Pipeline step for augmentation strategies (add new triples)."""

    EXPECTED_KIND = "augment"
    METADATA_PREFIX = "augmentation_"

    def __init__(
        self, 
        client: BaseLLMClient, 
        domain: KnowledgeDomain, 
        strategy: str = "connectivity",
        max_disconnected: int = 3,
        max_iterations: int = 2,
        temperature: float = 0.0,
        augmentation_prompt_override: str | None = None
    ):
        """Initialize the augmentation step.
        
        Args:
            client: Instantiated LLM client to use.
            domain: Knowledge domain defining the augmentation prompt context.
            strategy: The augmentation strategy to employ (default: connectivity).
            max_disconnected: Constraint parameter for connectivity tracking.
            max_iterations: Max retry attempts parameter.
            temperature: LLM temperature setting.
            augmentation_prompt_override: Optional custom prompt text.
        """
        self.client = client
        self.domain = domain
        self.strategy = strategy
        self.max_disconnected = max_disconnected
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.augmentation_prompt_override = augmentation_prompt_override

        actual_kind = strategy_kind(strategy)
        if actual_kind != self.EXPECTED_KIND:
            print(
                f"Warning: strategy '{strategy}' is a '{actual_kind}' operation; "
                f"prefer the '{'consolidate' if actual_kind == 'consolidate' else 'augment'}' step for it.",
                file=sys.stderr,
            )

    def process(self, context: PipelineContext, **kwargs: Any) -> PipelineContext:
        """Execute augmentation to connect graph components via inference.
        
        Args:
            context: The pipeline context with existing triples to refine.
            **kwargs: Unused.
            
        Returns:
            PipelineContext with refined graph triples.
        """
        if not context.triples:
            context.metadata["augmentation_skipped"] = "True (no initial triples found)"
            return context

        try:
            triples, metadata = augment_triples(
                client=self.client,
                domain=self.domain,
                text=context.text,
                record_id=context.record_id,
                initial_triples=context.triples,
                temperature=self.temperature,
                max_disconnected=self.max_disconnected,
                max_iterations=self.max_iterations,
                augmentation_strategy=self.strategy,
                augmentation_prompt_override=self.augmentation_prompt_override
            )
            # Override former triples state with augmented ones
            context.triples = triples
            
            # Store metadata
            context.metadata[self.METADATA_PREFIX + self.strategy] = metadata

        except Exception as e:
            context.errors.append(f"Strategy '{self.strategy}' failed: {str(e)}")

        return context


@register_step("consolidate")
class ConsolidationStep(AugmentationStep):
    """Pipeline step for consolidation strategies (merge/clean existing
    triples without adding knowledge) — e.g. entity_resolution.

    Same machinery as AugmentationStep; the split is the taxonomy: run
    consolidate after extract and/or after augment in the YAML steps list.
    """

    EXPECTED_KIND = "consolidate"
    METADATA_PREFIX = "consolidation_"

    def __init__(
        self,
        client: BaseLLMClient,
        domain: KnowledgeDomain,
        strategy: str = "entity_resolution",
        **kwargs: Any,
    ):
        super().__init__(client, domain, strategy=strategy, **kwargs)


__all__ = ["AugmentationStep", "ConsolidationStep"]
