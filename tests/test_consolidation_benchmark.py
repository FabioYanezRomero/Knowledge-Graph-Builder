"""Regression guards on the consolidation benchmark's deterministic components."""

from kgb.eval.consolidation_benchmark import run, generate_hard_negatives


def test_veto_never_blocks_a_true_merge():
    # The critical property: FN=0. A false veto would silently drop good merges.
    veto = run()["veto"]
    assert veto["fn"] == 0, veto["false_vetoes"]


def test_veto_blocks_all_numeric_and_staging_negatives():
    veto = run()["veto"]
    # Every uncovered negative (FP) must be a known-gap category, never numeric/staging.
    gap_categories = {"laterality", "polarity"}
    for p in veto["misses"]:
        assert p.category in gap_categories, f"veto missed a numeric/staging negative: {p}"


def test_generated_hard_negatives_are_blocked():
    from kgb.builder.consolidation import merge_allowed
    negs = generate_hard_negatives()
    assert negs, "expected some generated hard negatives"
    for p in negs:
        assert not merge_allowed(p.a, p.b), f"veto let a generated negative through: {p}"


def test_schwartz_hearst_full_recall_on_definitions():
    sh = run()["sh"]
    assert sh["found"] == sh["total"], sh["missed"]


def test_sieve_bag_no_regression_precision_preserved():
    # The full bag must never merge something it shouldn't (precision == 1.0).
    full = run()["sieve_full_bag"]
    assert full["precision"] == 1.0, full["false_merges"]
    assert full["fp"] == 0, full["false_merges"]


def test_sieve_bag_improves_recall_over_sh_only():
    r = run()
    # Adding exact_match strictly increases recall on surface positives...
    assert r["sieve_full_bag"]["recall"] > r["sieve_sh_only"]["recall"]
    # ...without losing any (full bag catches every surface positive).
    assert r["sieve_full_bag"]["recall"] == 1.0
