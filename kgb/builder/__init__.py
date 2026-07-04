"""Knowledge Graph Builder module.

This module provides the core engines for constructing a knowledge graph:
- Extraction: Converting raw text into initial triples.
- Augmentation: Adding knowledge not explicit in the text (connectivity).
- Consolidation: Merging/cleaning existing knowledge (entity resolution).

Extensibility:
- Use `@register_strategy(name, kind)` to add a strategy (kind: augment/consolidate).
- Use `list_strategies()` to discover available strategies.
"""

from .extraction import extract_triples
from .strategies import (
    GraphStrategy,
    STRATEGIES,
    register_strategy,
    list_strategies,
    strategy_kind,
)
from .augmentation import augment_triples, connectivity_strategy
# Import consolidation to register its strategies (entity_resolution).
from . import consolidation
from .consolidation import enforce_closed_set

__all__ = [
    "extract_triples",
    "augment_triples",
    "connectivity_strategy",
    "enforce_closed_set",
    "GraphStrategy",
    "register_strategy",
    "list_strategies",
    "strategy_kind",
    "STRATEGIES",
]
