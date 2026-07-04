"""Shared registry for graph-operation strategies.

A strategy is a named function that transforms a triple set. Each declares a
kind:
- "augment"     — adds knowledge not explicit in the text (new triples)
- "consolidate" — merges/cleans existing knowledge without adding any

The strategies themselves live in augmentation.py (augment kind) and
consolidation.py (consolidate kind). This module holds only the kind-neutral
registry they share, so neither depends on the other.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol

from ..clients import BaseLLMClient
from ..domains import KnowledgeDomain, Triple


class GraphStrategy(Protocol):
    """Protocol for graph-operation strategies (augment or consolidate).

    Strategy-specific parameters (e.g., max_iterations, max_disconnected) are
    passed via **kwargs and handled by each strategy individually.
    """
    def __call__(
        self,
        client: BaseLLMClient,
        domain: KnowledgeDomain,
        text: str,
        triples: list[Triple],
        **kwargs: Any,
    ) -> tuple[list[Triple], dict[str, Any]]:
        ...


STRATEGIES: dict[str, GraphStrategy] = {}

# Taxonomy of graph operations:
#   "augment"     — adds knowledge that was not explicit (new triples,
#                   typically marked as contextual inference)
#   "consolidate" — merges/cleans existing knowledge without adding any
#                   (entity resolution, deduplication, canonicalization)
STRATEGY_KINDS: dict[str, str] = {}


def register_strategy(name: str, kind: str = "augment") -> Callable[[GraphStrategy], GraphStrategy]:
    """Decorator for registering graph-operation strategies.

    Args:
        name: Strategy name (also the domain resource folder name).
        kind: "augment" (adds new triples) or "consolidate" (merges/cleans
            existing ones without adding knowledge).

    Usage:
        @register_strategy("my_strategy")
        def my_strategy(client, domain, text, triples, **kwargs):
            ...
            return refined_triples, metadata
    """
    if kind not in ("augment", "consolidate"):
        raise ValueError(f"Unknown strategy kind '{kind}'. Expected 'augment' or 'consolidate'.")

    def decorator(fn: GraphStrategy) -> GraphStrategy:
        STRATEGIES[name] = fn
        STRATEGY_KINDS[name] = kind
        return fn
    return decorator


def list_strategies(kind: str | None = None) -> list[str]:
    """List registered strategies, optionally filtered by kind."""
    if kind is None:
        return list(STRATEGIES.keys())
    return [n for n in STRATEGIES if STRATEGY_KINDS.get(n) == kind]


def strategy_kind(name: str) -> str:
    """Return the kind of a registered strategy (defaults to 'augment')."""
    return STRATEGY_KINDS.get(name, "augment")


__all__ = [
    "GraphStrategy",
    "STRATEGIES",
    "STRATEGY_KINDS",
    "register_strategy",
    "list_strategies",
    "strategy_kind",
]
