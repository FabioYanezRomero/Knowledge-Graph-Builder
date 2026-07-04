"""Deterministic sieve bag for entity resolution.

A cascade of high-precision deterministic passes applied BEFORE the LLM
(multi-pass sieve architecture; Raghunathan et al. 2010, Lee et al. 2013).
Each sieve maps variant -> canonical over EXISTING entities only (closed-set).
The bag composes sieves in precision-descending order (earlier wins on conflict)
and `resolve_chains` computes the transitive closure, so the result plugs
directly into the semantic LLM layer: the LLM then works on the reduced,
already-consolidated entity set, and its merges compose with the sieve merges.

Sieve signature: (text: str, entities: list[str]) -> dict[str, str]
Only surface-layer, general, no-training, no-dependency passes belong here.
Fuzzy similarity / stemming (lower precision) stay for the LLM; blocking/LSH is
a scaling concern, not a precision one.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Callable

from .schwartz_hearst import extract_abbreviation_pairs

Sieve = Callable[[str, list[str]], dict[str, str]]


def _normalize(name: str) -> str:
    """Unicode(NFKC) + casefold + drop punctuation + collapse whitespace."""
    n = unicodedata.normalize("NFKC", name).casefold()
    n = re.sub(r"[^\w\s]", " ", n)
    return re.sub(r"\s+", " ", n).strip()


def _pick_canonical(members: list[str]) -> str:
    """Deterministic canonical for a group: most cased, then longest, then max lexicographic."""
    return max(members, key=lambda m: (sum(c.isupper() for c in m), len(m), m))


def exact_match_sieve(text: str, entities: list[str]) -> dict[str, str]:
    """Merge entities identical after Unicode/case/whitespace/punctuation normalization.

    Highest-precision pass: "Bladder"/"bladder", "T-cell"/"T cell" collapse.
    """
    groups: dict[str, list[str]] = {}
    for e in entities:
        groups.setdefault(_normalize(e), []).append(e)
    mapping: dict[str, str] = {}
    for members in groups.values():
        if len(members) < 2:
            continue
        canonical = _pick_canonical(members)
        for m in members:
            if m != canonical:
                mapping[m] = canonical
    return mapping


def acronym_mapping(text: str, entities: list[str]) -> dict[str, str]:
    """Schwartz-Hearst acronym → expansion merges, closed to existing entities.

    Keeps only pairs where BOTH forms match an existing entity; the expansion is
    canonical. Prefers an exact-case match over a case-insensitive one, so it
    binds to the right entity when both "PSA" and "psa" exist (and the exact-match
    sieve then chains them). Never introduces an entity absent from the graph.
    """
    entity_set = set(entities)
    lower = {e.lower(): e for e in entities}

    def find(name: str) -> str | None:
        return name if name in entity_set else lower.get(name.lower())

    mapping: dict[str, str] = {}
    for short, long in extract_abbreviation_pairs(text):
        short_e = find(short)
        long_e = find(long)
        if short_e and long_e and short_e != long_e:
            mapping[short_e] = long_e
    return mapping


# alias so this reads as a sieve in the bag
acronym_sieve: Sieve = acronym_mapping

# Precision-descending order: exact match (highest) then acronym expansion.
DEFAULT_SIEVES: list[Sieve] = [exact_match_sieve, acronym_sieve]


def run_sieves(
    text: str,
    entities: list[str],
    sieves: list[Sieve] = DEFAULT_SIEVES,
) -> dict[str, str]:
    """Apply sieves in precision order; earlier (higher-precision) wins per variant."""
    mapping: dict[str, str] = {}
    for sieve in sieves:
        for variant, canonical in sieve(text, entities).items():
            mapping.setdefault(variant, canonical)
    return mapping


def resolve_chains(mapping: dict[str, str]) -> dict[str, str]:
    """Transitive closure: A->B, B->C  ==>  A->C, B->C.

    Lets sieve merges and LLM merges compose. Cycles are broken by returning the
    first repeated node (sieves don't produce cycles; guards downstream catch the
    rest).
    """
    def terminal(x: str) -> str:
        seen = {x}
        while x in mapping:
            nxt = mapping[x]
            if nxt in seen:
                break
            seen.add(nxt)
            x = nxt
        return x

    return {v: terminal(v) for v in mapping if terminal(v) != v}


__all__ = [
    "Sieve",
    "exact_match_sieve",
    "acronym_sieve",
    "acronym_mapping",
    "DEFAULT_SIEVES",
    "run_sieves",
    "resolve_chains",
]
