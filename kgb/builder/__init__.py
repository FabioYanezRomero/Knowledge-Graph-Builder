"""Knowledge Graph Builder module.

Three operation types, each with a home in this package:
- Extraction   text -> triples. The SOURCE of triples (extraction.py, via
               langextract). Not a strategy; a future ExtractStrategy protocol +
               registry lands here when a second extractor (e.g. multi-model
               consensus) exists.
- Augmentation triples -> triples that ADD knowledge not explicit in the text
               (augmentation.py: connectivity).
- Consolidation triples -> triples that MERGE/CLEAN without adding knowledge
               (consolidation/: entity_resolution + Schwartz-Hearst + guard).

Shared core:
- strategies.py  GraphStrategy protocol + registry (augment/consolidate) and
                 the strategy orchestrator (augment_triples).
- validation.py  schema constraints, validation, prompt rendering.

Folder-structure plan: an operation becomes a subpackage (like consolidation/)
when it grows past ~2 files or gains a second implementation; until then it stays
a single .py (extraction.py, augmentation.py). Migrate area by area, driven by
real growth, not up front.

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
