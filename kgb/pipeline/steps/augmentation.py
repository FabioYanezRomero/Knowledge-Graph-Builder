"""Augmentation pipeline step: add knowledge not explicit in the text."""

from __future__ import annotations

from ...clients import BaseLLMClient
from ...domains import KnowledgeDomain

from ..step import register_step
from .strategy import StrategyStep


@register_step("augment")
class AugmentationStep(StrategyStep):
    """Pipeline step for augmentation strategies (add new triples), e.g.
    connectivity bridging of disconnected components."""

    EXPECTED_KIND = "augment"
    METADATA_PREFIX = "augmentation_"
    DEFAULT_STRATEGY = "connectivity"

    def __init__(
        self,
        client: BaseLLMClient,
        domain: KnowledgeDomain,
        strategy: str | None = None,
        max_disconnected: int = 3,
        max_iterations: int = 2,
        temperature: float = 0.0,
        augmentation_prompt_override: str | None = None,
    ):
        """Initialize the augmentation step.

        Args:
            client: Instantiated LLM client to use.
            domain: Knowledge domain defining the augmentation prompt context.
            strategy: Augmentation strategy to employ (default: connectivity).
            max_disconnected: Target max components for connectivity.
            max_iterations: Max refinement iterations for connectivity.
            temperature: LLM temperature setting.
            augmentation_prompt_override: Optional custom prompt text.
        """
        super().__init__(
            client,
            domain,
            strategy=strategy,
            temperature=temperature,
            augmentation_prompt_override=augmentation_prompt_override,
            max_disconnected=max_disconnected,
            max_iterations=max_iterations,
        )


__all__ = ["AugmentationStep"]
