"""Consolidation: merge/clean existing knowledge without adding any.

Consolidation operates on an already-extracted graph and never introduces
entities or relations absent from it (the closed-set invariant in guard.py).

Modules:
- guard.py            closed-set invariant (enforce_closed_set)
- schwartz_hearst.py  deterministic acronym/expansion extraction from text
- entity_resolution.py the entity_resolution strategy (SH pre-pass + LLM)
"""

from .guard import enforce_closed_set
from .schwartz_hearst import extract_abbreviation_pairs
from .entity_resolution import entity_resolution_strategy, acronym_mapping

__all__ = [
    "enforce_closed_set",
    "extract_abbreviation_pairs",
    "entity_resolution_strategy",
    "acronym_mapping",
]
