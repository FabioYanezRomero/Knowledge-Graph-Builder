"""Smoke tests for kgb domain lint."""

import json

from kgb.domains.lint import lint_domain


def test_packaged_domains_are_clean():
    for name in ("default", "legal"):
        errors, _ = lint_domain(name)
        assert errors == [], f"{name}: {errors}"


def test_unknown_domain():
    errors, _ = lint_domain("does-not-exist")
    assert errors and "not found" in errors[0]


def test_broken_domain(tmp_path):
    domain = tmp_path / "broken"
    (domain / "extraction").mkdir(parents=True)
    (domain / "extraction" / "prompt_open.md").write_text("extract stuff")
    # invalid: extractions items missing required 'attributes'
    (domain / "extraction" / "examples.json").write_text(json.dumps(
        [{"text": "abc", "extractions": [{"extraction_text": "abc"}]}]
    ))
    (domain / "schema.json").write_text("{not json")
    errors, warnings = lint_domain(str(domain))
    assert any("examples.json[0]" in e for e in errors)
    assert any("schema.json" in e for e in errors)
    assert any("prompt_constrained" in w for w in warnings)


def test_char_span_warning(tmp_path):
    domain = tmp_path / "spans"
    (domain / "extraction").mkdir(parents=True)
    (domain / "extraction" / "prompt_open.md").write_text("extract stuff")
    (domain / "extraction" / "examples.json").write_text(json.dumps([{
        "text": "short",
        "extractions": [{
            "extraction_text": "short",
            "char_start": 0,
            "char_end": 999,
            "attributes": {"head": "a", "relation": "b", "tail": "c"},
        }],
    }]))
    errors, warnings = lint_domain(str(domain))
    assert errors == []
    assert any("char span" in w for w in warnings)
