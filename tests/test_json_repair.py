"""Hardened JSON repair for local-model extraction output."""
import json
import re
from kgb.clients.providers.ollama import _repair_json_text, _coerce_scalar_values


def _parse(s):
    return json.loads(_repair_json_text(s))


def _fenced(s):
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", s)
    return json.loads(m.group(1) if m else s)


# langextract renders each extraction as {"Triple": <text>, "Triple_attributes": {..}}.
# The resolver requires the *_attributes value to stay a dict and every other value
# to be scalar; coercion must respect BOTH or it aborts the whole document.

def test_attributes_dict_never_stringified():
    # The regression that broke all 15 reports: don't touch *_attributes dicts.
    raw = ('```json\n[{"Triple": "cystoscopy", "Triple_attributes": '
           '{"head": "patient", "relation": "underwent", "tail": "cystoscopy"}}]\n```')
    out = _coerce_scalar_values(raw)
    obj = _fenced(out)[0]
    assert isinstance(obj["Triple_attributes"], dict)   # still a dict, untouched
    assert obj["Triple_attributes"]["relation"] == "underwent"


def test_nonscalar_extraction_text_coerced():
    # The report-07/10 failure: a nested value in the scalar-required field.
    raw = '[{"Triple": {"nested": 1}, "Triple_attributes": {"head": "a"}}]'
    obj = json.loads(_coerce_scalar_values(raw))[0]
    assert isinstance(obj["Triple"], str)               # nested text -> string
    assert isinstance(obj["Triple_attributes"], dict)   # attrs still a dict
    assert json.loads(obj["Triple"]) == {"nested": 1}   # info preserved


def test_all_scalar_unchanged():
    raw = '[{"Triple": "x", "Triple_attributes": {"head": "a"}}]'
    assert _coerce_scalar_values(raw) == raw            # no change -> identical text


def test_list_extraction_text_coerced():
    raw = '[{"Triple": ["a", "b"], "Triple_attributes": {"head": "a"}}]'
    obj = json.loads(_coerce_scalar_values(raw))[0]
    assert obj["Triple"] == '["a", "b"]'


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
