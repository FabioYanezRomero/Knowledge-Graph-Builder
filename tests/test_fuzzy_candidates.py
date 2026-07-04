"""Fuzzy candidate generation: blocking for the LLM, never merges directly."""

from kgb.builder.consolidation import fuzzy_candidates


def _pairs(cands):
    return {frozenset((a, b)) for a, b, _ in cands}


def test_surfaces_morphological_variant():
    # exact_match can't catch this (prostatic != prostate) — fuzzy should propose it.
    cands = fuzzy_candidates(["prostatic adenocarcinoma", "prostate adenocarcinoma", "bladder"])
    assert frozenset(("prostatic adenocarcinoma", "prostate adenocarcinoma")) in _pairs(cands)


def test_veto_filters_numeric_siblings():
    # "Gleason 3+4" vs "Gleason 4+3" are similar but must NOT be proposed —
    # the veto would block the merge anyway. Low threshold isolates the veto
    # from the similarity cutoff (their difflib ratio is ~0.82).
    cands = fuzzy_candidates(["Gleason 3+4", "Gleason 4+3"], threshold=0.7)
    assert cands == []


def test_veto_filter_can_be_disabled():
    cands = fuzzy_candidates(["Gleason 3+4", "Gleason 4+3"], threshold=0.7, veto_filter=False)
    assert frozenset(("Gleason 3+4", "Gleason 4+3")) in _pairs(cands)


def test_threshold_respected():
    # unrelated entities should not be proposed
    cands = fuzzy_candidates(["kidney", "bladder tumor recurrence"], threshold=0.82)
    assert cands == []


def test_candidates_never_merge_only_propose():
    # fuzzy returns pairs, not a mapping — it never decides a canonical
    cands = fuzzy_candidates(["prostatic adenocarcinoma", "prostate adenocarcinoma"])
    assert all(isinstance(c, tuple) and len(c) == 3 for c in cands)
