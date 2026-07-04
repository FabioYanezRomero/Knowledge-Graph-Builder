"""Shared base for pipeline steps that run a graph strategy.

AugmentationStep and ConsolidationStep are siblings, not one a subclass of the
other: consolidation is NOT a kind of augmentation. They share only the neutral
mechanics of running a registered strategy over a context's triples, captured
here and parameterized by the subclass's kind, metadata prefix, and default
strategy.
"""

from __future__ import annotations

import sys
from typing import Any

from ...builder import augment_triples
from ...builder.strategies import strategy_kind, has_builtin_prompt
from ...clients import BaseLLMClient
from ...domains import KnowledgeDomain

from ..context import PipelineContext


class StrategyStep:
    """Runs a registered strategy over a context's triples.

    Subclasses set EXPECTED_KIND, METADATA_PREFIX and DEFAULT_STRATEGY, and may
    contribute strategy-specific parameters (e.g. max_disconnected) via
    **strategy_params, which are forwarded to the strategy.
    """

    EXPECTED_KIND: str = ""
    METADATA_PREFIX: str = ""
    DEFAULT_STRATEGY: str = ""

    def __init__(
        self,
        client: BaseLLMClient,
        domain: KnowledgeDomain,
        strategy: str | None = None,
        temperature: float = 0.0,
        augmentation_prompt_override: str | None = None,
        **strategy_params: Any,
    ):
        self.client = client
        self.domain = domain
        self.strategy = strategy or self.DEFAULT_STRATEGY
        self.temperature = temperature
        self.augmentation_prompt_override = augmentation_prompt_override
        self.strategy_params = strategy_params

        actual_kind = strategy_kind(self.strategy)
        if actual_kind != self.EXPECTED_KIND:
            print(
                f"Warning: strategy '{self.strategy}' is a '{actual_kind}' operation; "
                f"prefer the '{actual_kind}' step for it.",
                file=sys.stderr,
            )

    def process(self, context: PipelineContext, **kwargs: Any) -> PipelineContext:
        """Run the strategy over the context's triples.

        Skips (without erroring the record) when there are no triples yet or the
        domain doesn't define the strategy, so an optional refinement step never
        discards upstream work.
        """
        if not context.triples:
            context.metadata[self.METADATA_PREFIX + "skipped"] = "True (no initial triples found)"
            return context

        if (
            self.strategy not in self.domain.list_augmentation_strategies()
            and not has_builtin_prompt(self.strategy)
        ):
            print(
                f"Warning: domain has no '{self.strategy}' strategy; skipping "
                f"{self.__class__.__name__} for record {context.record_id}.",
                file=sys.stderr,
            )
            context.metadata[self.METADATA_PREFIX + "skipped"] = f"True (domain lacks '{self.strategy}')"
            return context

        try:
            triples, metadata = augment_triples(
                client=self.client,
                domain=self.domain,
                text=context.text,
                record_id=context.record_id,
                initial_triples=context.triples,
                temperature=self.temperature,
                augmentation_strategy=self.strategy,
                augmentation_prompt_override=self.augmentation_prompt_override,
                **self.strategy_params,
            )
            context.triples = triples
            context.metadata[self.METADATA_PREFIX + self.strategy] = metadata
        except Exception as e:
            context.errors.append(f"Strategy '{self.strategy}' failed: {str(e)}")

        return context


__all__ = ["StrategyStep"]
