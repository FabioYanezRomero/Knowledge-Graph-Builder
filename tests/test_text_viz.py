"""Regression test: text visualization must accept Triple objects.

Previously text_viz.py used getattr(t, 'head', t.get(...)) whose default was
evaluated eagerly, calling .get() on Triple objects and crashing save_html
with "'Triple' object has no attribute 'get'" (the last pipeline step).
"""

from kgb.domains import Triple
from kgb.visualization.text_viz import TextVisualizer


def test_save_html_accepts_triple_objects(tmp_path):
    triples = [
        Triple(head="prostate biopsy", relation="shows", tail="adenocarcinoma"),
        Triple(head="adenocarcinoma", relation="has_grade", tail="Gleason 3+4"),
    ]
    out = tmp_path / "viz.html"
    TextVisualizer().save_html(
        text="The prostate biopsy shows adenocarcinoma, Gleason 3+4.",
        triples=triples,
        output_path=out,
        document_id="t1",
    )
    html = out.read_text(encoding="utf-8")
    assert "<b>3</b>" in html  # 3 distinct entities, no crash


def test_save_html_still_accepts_dicts(tmp_path):
    out = tmp_path / "viz.html"
    TextVisualizer().save_html(
        text="a r b",
        triples=[{"head": "a", "relation": "r", "tail": "b"}],
        output_path=out,
    )
    assert "<b>2</b>" in out.read_text(encoding="utf-8")
