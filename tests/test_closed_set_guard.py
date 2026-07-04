"""Closed-set guard: consolidation may only relabel to existing entities."""

from kgb.builder.consolidation import enforce_closed_set


def test_keeps_mappings_to_existing_entities():
    entities = {"Michael J. Salvino", "Salvino", "DXC", "DXC Technology Company"}
    mapping = {"Salvino": "Michael J. Salvino", "DXC": "DXC Technology Company"}
    kept, rejected = enforce_closed_set(mapping, entities)
    assert kept == mapping
    assert rejected == []


def test_rejects_invented_canonical():
    # LLM invents a canonical ("John Smith Jr.") absent from the graph.
    entities = {"John Smith", "J. Smith"}
    mapping = {"J. Smith": "John Smith Jr."}
    kept, rejected = enforce_closed_set(mapping, entities)
    assert kept == {}
    assert rejected == [("J. Smith", "John Smith Jr.")]


def test_partial_rejection():
    entities = {"a", "b", "c"}
    mapping = {"a": "b", "c": "z"}  # z is not an entity
    kept, rejected = enforce_closed_set(mapping, entities)
    assert kept == {"a": "b"}
    assert rejected == [("c", "z")]
