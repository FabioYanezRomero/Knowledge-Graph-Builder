"""Relation resolution: canonicalize predicates, closed-set guard, built-in prompt."""

from kgb.builder.consolidation import relation_resolution_strategy
from kgb.builder.strategies import strategy_kind, has_builtin_prompt
from kgb.domains import Triple, get_domain


class CapturingClient:
    """Stub LLM: records the prompt it saw, returns a canned merge group."""

    def __init__(self, groups):
        self.groups = groups
        self.prompt = None

    def augment(self, text, **kwargs):
        self.prompt = text
        return self.groups


def _triples():
    # Two predicates linking the SAME (head, tail) pair -> synonymy evidence.
    return [
        Triple(head="prostate ca", relation="graded_as", tail="gleason 7"),
        Triple(head="prostate ca", relation="has_grade", tail="gleason 7"),
        Triple(head="tumor", relation="located_in", tail="bladder"),
    ]


def test_synonym_merge_collapses_and_dedupes():
    client = CapturingClient([{"canonical": "graded_as", "variants": ["graded_as", "has_grade"]}])
    out, meta = relation_resolution_strategy(client, get_domain("default"), "", _triples())
    rels = {t.relation for t in out}
    assert "has_grade" not in rels and "graded_as" in rels
    # the two grading triples now collide on (head, relation, tail) -> deduped to one
    assert meta["duplicates_removed"] == 1
    assert len(out) == 2


def test_closed_set_guard_rejects_invented_canonical():
    # LLM tries to canonicalize onto a label not present in the graph.
    client = CapturingClient([{"canonical": "assigned_grade", "variants": ["graded_as", "has_grade"]}])
    out, meta = relation_resolution_strategy(client, get_domain("default"), "", _triples())
    assert meta["status"] == "no_merges"
    assert meta["rejected_mappings"]  # guard fired
    assert len(out) == 3  # unchanged


def test_builtin_prompt_carries_context_and_candidates():
    # add a surface-similar pair so fuzzy blocking has something to propose;
    # graded_as/has_grade are semantic (not surface) synonyms, left to the LLM.
    triples = _triples() + [Triple(head="tumor", relation="is_located_in", tail="wall")]
    client = CapturingClient([])
    relation_resolution_strategy(client, get_domain("default"), "", triples)
    p = client.prompt
    assert "Relations to Resolve" in p                 # built-in default prompt used
    assert "(prostate ca) -> (gleason 7)" in p         # co-occurrence context present
    assert "Candidate pairs to evaluate" in p          # fuzzy proposed located_in vs is_located_in
    assert "located_in" in p and "is_located_in" in p


def test_runs_on_bare_domain_without_folder():
    # builtin_prompt=True means the pipeline guard won't skip it, and the
    # strategy falls back to its default prompt (no relation_resolution folder).
    assert has_builtin_prompt("relation_resolution")
    assert strategy_kind("relation_resolution") == "consolidate"
    client = CapturingClient([])
    out, meta = relation_resolution_strategy(client, get_domain("default"), "", _triples())
    assert meta["status"] in ("no_merges", "success")


def test_single_relation_skips():
    client = CapturingClient([])
    out, meta = relation_resolution_strategy(
        client, get_domain("default"), "",
        [Triple(head="a", relation="r", tail="b")],
    )
    assert meta["status"] == "skipped"
    assert client.prompt is None  # never called the LLM
