"""Reusable consolidation layers — the building blocks that strategies compose.

A consolidation strategy (entity_resolution, relation_resolution) is a *pipeline*
over these layers. Each layer plays one role in the ``deterministic → fuzzy → LLM
→ filter`` flow; the LLM step is not here (it is ``client.augment`` in the
strategy). Order by role:

MERGE (deterministic, high precision, runs first):
- sieves.py           multi-pass exact/acronym sieve bag (closed-set)
- schwartz_hearst.py  acronym/expansion extraction from text (used by the sieve bag)

PROPOSE (fuzzy blocking, high recall — never merges, only surfaces candidates):
- fuzzy.py            difflib look-alike pairs for the LLM to judge

FILTER (deterministic, runs last on the LLM's output):
- veto.py             discriminative-signature veto (numeric/staging siblings)
- guard.py            closed-set invariant (no invented entity/relation survives)

Strategies pick the subset they need: entity_resolution uses all five;
relation_resolution uses fuzzy + guard only (relation labels have no source spans
for sieves, and the fuzzy veto already declines the pairs veto would catch).
"""

from .fuzzy import fuzzy_candidates
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

__all__ = [
    "fuzzy_candidates",
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
]
