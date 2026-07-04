"""Merging per-record JSON triples into one cross-document GraphML."""

import json

from kgb.io.writers import merge_json_directories


def test_merge_shares_entities_across_documents(tmp_path):
    doc1 = [{"head": "Aspirin", "relation": "treats", "tail": "Headache"}]
    doc2 = [{"head": "aspirin ", "relation": "reduces", "tail": "Fever"}]  # case/space variant
    (tmp_path / "doc1.json").write_text(json.dumps(doc1))
    (tmp_path / "doc2.json").write_text(json.dumps(doc2))
    (tmp_path / "bad.json").write_text("{not json")

    out = tmp_path / "graphml" / "merged.graphml"
    G = merge_json_directories(tmp_path, out)

    assert out.exists()
    # "Aspirin" and "aspirin " collapse into one node -> 3 nodes, 2 edges
    assert G.number_of_nodes() == 3
    assert G.number_of_edges() == 2
