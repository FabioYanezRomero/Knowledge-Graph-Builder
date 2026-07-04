"""Fuzzy candidate generation for the LLM (blocking, not matching).

Surfaces pairs of EXISTING entities that look similar, as CANDIDATES for the
semantic LLM stage to judge — they are never merged directly. This is the safe
way to use fuzzy matching (the same difflib langextract uses for grounding): a
false fuzzy match just becomes a question the LLM answers, and the veto + guard
still filter the final merges. Because it only generates candidates, a permissive
threshold (high recall) is safe — precision is the LLM's job.

Fills the gap between the exact_match sieve (identical after normalization) and
the semantic layer: morphological / near-surface variants like "prostatic
adenocarcinoma" vs "prostate adenocarcinoma".
"""

from __future__ import annotations

import difflib

from .veto import merge_allowed


def _ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def fuzzy_candidates(
    entities: list[str],
    threshold: float = 0.82,
    veto_filter: bool = True,
) -> list[tuple[str, str, float]]:
    """Pairs of entities with difflib similarity >= threshold, as LLM candidates.

    Args:
        entities: existing entity strings (already sieve-reduced upstream).
        threshold: minimum similarity; permissive is fine — the LLM decides.
        veto_filter: drop pairs the discriminative veto would block (numeric /
            staging siblings), so the LLM isn't asked about obvious non-merges.

    Returns:
        (a, b, score) triples, most similar first. Candidates only — never merged.

    Note: O(n^2) over the (small, per-document) reduced entity set. For
    cross-document consolidation at scale, replace with blocking / a C++-backed
    similarity lib (RapidFuzz) before this becomes the bottleneck.
    """
    ents = sorted(set(entities))
    cands: list[tuple[str, str, float]] = []
    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            a, b = ents[i], ents[j]
            if veto_filter and not merge_allowed(a, b):
                continue
            r = _ratio(a, b)
            if r >= threshold:
                cands.append((a, b, round(r, 3)))
    cands.sort(key=lambda c: c[2], reverse=True)
    return cands


__all__ = ["fuzzy_candidates"]
