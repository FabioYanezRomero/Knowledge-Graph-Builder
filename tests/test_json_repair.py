"""Hardened JSON repair for local-model extraction output."""
import json
import re
from kgb.clients.providers.ollama import _repair_json_text, _coerce_scalar_values


def _parse(s):
    return json.loads(_repair_json_text(s))


def _fenced_body(s):
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    return json.loads(m.group(1) if m else s)


def test_nonscalar_value_coerced_to_string():
    # langextract aborts the whole doc if any value is a dict/list; coerce them.
    raw = '```json\n[{"head": "prostate", "relation": "has_grade", "tail": {"gleason": 7}}]\n```'
    out = _coerce_scalar_values(raw)
    obj = _fenced_body(out)[0]
    assert isinstance(obj["tail"], str)
    assert json.loads(obj["tail"]) == {"gleason": 7}  # info preserved, not dropped


def test_scalar_values_untouched():
    raw = '[{"head": "a", "relation": "r", "tail": "b"}]'
    assert _coerce_scalar_values(raw) == raw  # no change -> identical text


def test_list_value_coerced():
    raw = '[{"head": "specimen", "relation": "sites", "tail": ["left", "right"]}]'
    obj = json.loads(_coerce_scalar_values(raw))[0]
    assert obj["tail"] == '["left", "right"]'


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
