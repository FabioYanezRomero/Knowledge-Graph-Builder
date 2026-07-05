"""Consolidation: merge/clean an existing graph without adding knowledge.

Consolidation operates on an already-extracted graph and never introduces
entities or relations absent from it (the closed-set invariant in
``layers/guard.py``).

Structure mirrors the architecture — strategies are pipelines, layers are the
reusable steps they compose:

- ``entity_resolution.py`` / ``relation_resolution.py`` — the STRATEGIES the
  pipeline invokes (a ``consolidate`` step dispatches to one of these).
- ``layers/`` — the reusable building blocks each strategy composes, ordered
  ``deterministic-merge → fuzzy-propose → LLM-decide → deterministic-filter``.

Composition (what each strategy actually uses):
- entity_resolution   = sieves + fuzzy + LLM + veto + guard   (all five)
- relation_resolution = fuzzy + LLM + guard                   (no sieves/veto —
  relation labels have no source spans and the fuzzy veto already declines the
  pairs veto would catch)
"""

from .layers.guard import enforce_closed_set
from .layers.schwartz_hearst import extract_abbreviation_pairs
from .layers.sieves import (
    exact_match_sieve,
    acronym_sieve,
    acronym_mapping,
    DEFAULT_SIEVES,
    run_sieves,
    resolve_chains,
)
from .layers.veto import discriminative_signature, merge_allowed, apply_discriminative_veto
from .layers.fuzzy import fuzzy_candidates
from .entity_resolution import entity_resolution_strategy
from .relation_resolution import relation_resolution_strategy

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
    "fuzzy_candidates",
    "entity_resolution_strategy",
    "relation_resolution_strategy",
]
