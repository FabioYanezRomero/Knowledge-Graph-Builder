"""Local models mis-case field keys ("Tail" vs "tail"); those values must not
be silently dropped. Regression from real gemma4:12b-mlx output on report 11."""

from kgb.builder.extraction import extract_triples
from kgb.builder.validation import canonicalize_triple_keys
from kgb.domains import get_domain


class StubClient:
    def __init__(self, rows):
        self.rows = rows

    def extract(self, text, prompt_description, examples=None, format_type=None,
                temperature=0.0, max_tokens=None, **kwargs):
        return self.rows


def test_canonicalize_lowercases_known_keys():
    d = canonicalize_triple_keys({"head": "a", "relation": "r", "Tail": "b",
                                  "Inference": "explicit", "Weird": 1})
    assert d["tail"] == "b"
    assert d["inference"] == "explicit"
    assert d["Weird"] == 1  # unknown keys untouched


def test_real_lowercase_wins_over_miscased_duplicate():
    # If both "tail" and "Tail" arrive, the real lowercase value survives.
    assert canonicalize_triple_keys({"tail": "real", "Tail": "junk"})["tail"] == "real"
    assert canonicalize_triple_keys({"Tail": "junk", "tail": "real"})["tail"] == "real"


def test_miscased_tail_becomes_grounded_triple_not_isolated_node():
    # Before the fix: "Tail" -> t.get("tail") == "" -> degraded to standalone.
    client = StubClient([
        {"head": "patient", "relation": "started_medication", "Tail": "Cialis 10 mg"},
    ])
    triples, entities = extract_triples(client, get_domain("default"), "some text")
    assert [(t.head, t.relation, t.tail) for t in triples] == [
        ("patient", "started_medication", "Cialis 10 mg")]
    assert entities == []  # NOT dropped to an isolated node


def test_report11_full_response_recovers_all_triples():
    # The exact key-casing mix captured from gemma4:12b-mlx on report 11:
    # 1 object used "tail", 20 used "Tail"; 3 are standalone (empty rel/tail).
    rows = [
        {"head": "patient", "relation": "has_history_of", "tail": "erectile dysfunction"},
    ] + [
        {"head": h, "relation": r, "Tail": t}
        for h, r, t in [
            ("patient", "started_medication", "Cialis 10 mg"),
            ("erectile dysfunction", "has_status", "mild improvement"),
            ("Cialis 10 mg", "has_side_effect", "none"),
            ("patient", "denies", "nitroglycerin usage"),
            ("patient", "denies", "cardiac issues"),
            ("patient", "has_history_of", "elevated PSA"),
            ("biopsy", "performed_date", "June of this year"),
            ("biopsy", "shows", "high grade PIN"),
            ("PIN", "has_grade", "high grade"),
            ("biopsy", "located_in", "mid left"),
            ("specimen", "count", "2"),
            ("specimen", "has_status", "too small to evaluate"),
            ("PSA", "has_value", "11.6"),
            ("patient", "considering", "prostate ultrasound"),
            ("patient", "considering", "biopsy"),
            ("patient", "prescribed", "Cialis 20 mg"),
            ("patient", "provided_samples", "Levitra 10 mg"),
        ]
    ] + [
        {"head": "penile prosthesis", "relation": "", "Tail": ""},
        {"head": "Caverject injection", "relation": "", "Tail": ""},
        {"head": "penile pump", "relation": "", "Tail": ""},
    ]
    triples, entities = extract_triples(StubClient(rows), get_domain("default"), "x")
    assert len(triples) == 18       # was 1 before the fix
    assert set(entities) == {"penile prosthesis", "Caverject injection", "penile pump"}
