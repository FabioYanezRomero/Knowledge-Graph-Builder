"""Grounding-only extraction: entities with no stated relation become isolated nodes."""

from kgb.builder.extraction import extract_triples
from kgb.domains import get_domain
from kgb.io.writers.graphml import json_to_graphml


class StubClient:
    """Returns a fixed mix of grounded triples and standalone-entity dicts."""

    def __init__(self, rows):
        self.rows = rows

    def extract(self, text, prompt_description, examples=None, format_type=None,
                temperature=0.0, max_tokens=None, **kwargs):
        return self.rows


def test_extract_splits_triples_and_standalone_entities():
    client = StubClient([
        {"head": "carcinoma", "relation": "located_in", "tail": "bladder"},
        {"head": "TURBT", "relation": "", "tail": ""},          # standalone: no relation
        {"head": "Foley catheter", "relation": "", "tail": ""},  # standalone
        {"head": "x", "relation": "rel"},                         # missing tail -> standalone
    ])
    triples, entities = extract_triples(client, get_domain("default"), "some text")
    assert [(t.head, t.relation, t.tail) for t in triples] == [("carcinoma", "located_in", "bladder")]
    assert set(entities) == {"TURBT", "Foley catheter", "x"}


def test_no_fabricated_relation_never_becomes_a_triple():
    client = StubClient([{"head": "TURBT", "relation": "", "tail": ""}])
    triples, entities = extract_triples(client, get_domain("default"), "t")
    assert triples == []
    assert entities == ["TURBT"]


def test_graphml_adds_standalone_entities_as_isolated_nodes():
    from kgb.domains import Triple
    G = json_to_graphml(
        triples=[Triple(head="carcinoma", relation="located_in", tail="bladder")],
        entities=["TURBT", "lamina propria"],
    )
    assert G.degree("TURBT") == 0 and G.degree("lamina propria") == 0
    assert G.has_edge("carcinoma", "bladder")


def test_graphml_routes_empty_relation_dicts_to_isolated_nodes():
    G = json_to_graphml(triples=[
        {"head": "carcinoma", "relation": "located_in", "tail": "bladder"},
        {"head": "TURBT", "relation": "", "tail": ""},
    ])
    assert "TURBT" in G.nodes() and G.degree("TURBT") == 0


def test_triple_model_accepts_the_standalone_shape():
    # The model must admit what the pipeline actually produces, or `kgb domain
    # lint` rejects a domain whose examples are correct (pathology's did).
    from kgb.domains import Triple
    t = Triple(head="TURBT", relation="", tail="")
    assert (t.head, t.relation, t.tail) == ("TURBT", "", "")


def test_triple_model_rejects_half_a_relation():
    import pytest
    from kgb.domains import Triple
    for bad in ({"head": "a", "relation": "does", "tail": ""},
                {"head": "a", "relation": "", "tail": "b"},
                {"head": "", "relation": "", "tail": ""}):
        with pytest.raises(Exception):
            Triple(**bad)


def test_provenance_offsets_remapped_to_document():
    # langextract returns prompt-relative offsets; extract_triples must re-anchor
    # to the source document via the exact span text.
    doc = "The 75-year-old man had prostate adenocarcinoma."
    client = StubClient([
        {"head": "man", "relation": "has_age", "tail": "75",
         "extraction_text": "75-year-old man", "char_start": 999, "char_end": 1014},  # bogus prompt offsets
        {"head": "x", "relation": "rel", "tail": "y", "extraction_text": "not in the doc"},
    ])
    triples, _ = extract_triples(client, get_domain("default"), doc)
    grounded = {t.head: t for t in triples}
    m = grounded["man"]
    assert doc[m.char_start:m.char_end] == "75-year-old man"   # re-anchored & exact
    assert grounded["x"].char_start is None                     # span absent -> no offset
