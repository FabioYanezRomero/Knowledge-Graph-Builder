"""Hardened JSON repair for local-model extraction output."""
import json
from kgb.clients.providers.ollama import _repair_json_text


def _parse(s):
    return json.loads(_repair_json_text(s))


def test_clean_json_untouched():
    s = '[{"head": "a", "relation": "r", "tail": "b"}]'
    assert json.loads(_repair_json_text(s)) == [{"head": "a", "relation": "r", "tail": "b"}]


def test_trailing_comma_fixed():
    assert _parse('[{"head": "a", "relation": "r", "tail": "b",}]') == [
        {"head": "a", "relation": "r", "tail": "b"}
    ]


def test_truncated_response_keeps_complete_objects():
    # model hit max_tokens mid-object -> the incomplete last object is dropped
    s = '[{"head": "a", "relation": "r", "tail": "b"}, {"head": "c", "relat'
    out = _parse(s)
    assert out == [{"head": "a", "relation": "r", "tail": "b"}]


def test_malformed_object_skipped_others_kept():
    # middle object has an unquoted key; salvage keeps the valid neighbours
    s = '[{"head":"a","relation":"r","tail":"b"}, {head:"x"}, {"head":"c","relation":"r","tail":"d"}]'
    out = _parse(s)
    assert {"head": "a", "relation": "r", "tail": "b"} in out
    assert {"head": "c", "relation": "r", "tail": "d"} in out
    assert len(out) == 2


def test_fenced_json_parses():
    s = '```json\n[{"head": "a", "relation": "r", "tail": "b"}]\n```'
    # fenced healthy body is recognized as parseable (returned as-is, fences kept)
    assert '"head": "a"' in _repair_json_text(s)


def test_control_chars_stripped():
    s = '[{"head": "a\x00b", "relation": "r", "tail": "c"}]'
    out = _parse(s)
    assert out[0]["head"] == "a b"


def test_strings_with_braces_not_confused():
    # a brace inside a string value must not break object boundary detection
    s = '[{"head": "a {b} c", "relation": "r", "tail": "d",}]'
    assert _parse(s) == [{"head": "a {b} c", "relation": "r", "tail": "d"}]
