"""Consolidation: merge/clean existing knowledge without adding any.

Consolidation operates on an already-extracted graph and never introduces
entities or relations absent from it (the closed-set invariant in guard.py).

Modules:
- guard.py            closed-set invariant (enforce_closed_set)
- schwartz_hearst.py  deterministic acronym/expansion extraction from text
- sieves.py           deterministic sieve bag (multi-pass, precision-ordered)
- veto.py             discriminative-signature veto
- entity_resolution.py the strategy (sieve bag -> LLM -> veto -> guard)
"""

from .guard import enforce_closed_set
from .schwartz_hearst import extract_abbreviation_pairs
from .sieves import (
    exact_match_sieve,
    acronym_sieve,
    acronym_mapping,
    DEFAULT_SIEVES,
    run_sieves,
    resolve_chains,
)
from .veto import discriminative_signature, merge_allowed, apply_discriminative_veto
from .entity_resolution import entity_resolution_strategy

__all__ = [
    "enforce_closed_set",
    "extract_abbreviation_pairs",
    "exact_match_sieve",
    "acronym_sieve",
    "acronym_mapping",
    "DEFAULT_SIEVES",
    "run_sieves",
    "resolve_chains",
    "discriminative_signature",
    "merge_allowed",
    "apply_discriminative_veto",
    "entity_resolution_strategy",
]
