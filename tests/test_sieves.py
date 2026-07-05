"""Deterministic sieve bag: exact-match, composition, transitive closure."""

from kgb.builder.consolidation import (
    exact_match_sieve,
    run_sieves,
    resolve_chains,
    acronym_mapping,
)
from kgb.builder.consolidation.layers.sieves import _normalize


def test_exact_match_normalizes_case_and_punctuation():
    m = exact_match_sieve("", ["Bladder", "bladder", "T-cell", "T cell"])
    # both groups collapse to a single canonical each
    assert m["bladder"] == "Bladder" or m["Bladder"] == "bladder"
    assert _normalize("T-cell") == _normalize("T cell")
    tcell_canon = {v: c for v, c in m.items() if _normalize(v) == "t cell"}
    assert len(set(tcell_canon.values())) <= 1


def test_run_sieves_precision_order_earlier_wins():
    # if two sieves disagree on a variant, the earlier (higher-precision) one wins
    def sieve_a(text, entities):
        return {"x": "A"}
    def sieve_b(text, entities):
        return {"x": "B"}
    assert run_sieves("", ["x", "A", "B"], sieves=[sieve_a, sieve_b])["x"] == "A"


def test_resolve_chains_transitive_closure():
    # sieve: A->B ; LLM: B->C  =>  A->C, B->C  (the layer-compatibility property)
    assert resolve_chains({"A": "B", "B": "C"}) == {"A": "C", "B": "C"}


def test_resolve_chains_handles_cycle():
    out = resolve_chains({"A": "B", "B": "A"})
    # no crash, no infinite loop; both remain mapped to the other
    assert set(out.keys()) == {"A", "B"}


def test_acronym_sieve_closed_set():
    text = "Prostate-specific antigen (PSA) rose."
    assert acronym_mapping(text, ["PSA", "Prostate-specific antigen"]) == {
        "PSA": "Prostate-specific antigen"
    }
    # expansion not an entity -> no merge
    assert acronym_mapping(text, ["PSA"]) == {}


def test_bag_composes_exact_and_acronym():
    # "psa" (lowercase) exact-matches "PSA"; "PSA" acronym-expands.
    # Composition + closure should route "psa" all the way to the expansion.
    text = "Prostate-specific antigen (PSA) rose."
    entities = ["psa", "PSA", "Prostate-specific antigen"]
    mapping = resolve_chains(run_sieves(text, entities))
    assert mapping.get("psa") == "Prostate-specific antigen"
    assert mapping.get("PSA") == "Prostate-specific antigen"


def test_strategy_composes_sieves_with_llm_layer():
    # End-to-end layer compatibility: exact-match + acronym sieves + an LLM
    # merge all collapse to one canonical via transitive closure.
    from kgb.domains import get_domain, Triple
    from kgb.builder.consolidation import entity_resolution_strategy

    class StubClient:
        def augment(self, text, prompt_description, format_type, temperature, max_tokens):
            return [{"canonical": "Prostate-specific antigen", "variants": ["PSA antigen"]}]

    triples = [
        Triple(head="psa", relation="rose_in", tail="patient"),
        Triple(head="PSA", relation="measured_in", tail="serum"),
        Triple(head="Prostate-specific antigen", relation="is_a", tail="biomarker"),
        Triple(head="PSA antigen", relation="synonym_of", tail="marker"),
    ]
    out, meta = entity_resolution_strategy(
        StubClient(), get_domain("pathology"), "Prostate-specific antigen (PSA) rose.", triples
    )
    heads = {t.head for t in out}
    assert not ({"psa", "PSA", "PSA antigen"} & heads)
    assert "Prostate-specific antigen" in heads
    assert meta["sieve_merges"] == 2
