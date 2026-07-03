"""Smoke test for filesystem-based domain discovery."""

import os
from pathlib import Path

from kgb.domains import get_domain, list_available_domains
from kgb.domains.base import DomainResourceError


def test_packaged_domains_discovered():
    domains = list_available_domains()
    assert "legal" in domains
    assert "default" in domains


def test_get_domain_by_name():
    legal = get_domain("legal", extraction_mode="open")
    assert legal.extraction.prompt  # loads from kgb/domains/legal/extraction/
    assert "connectivity" in legal.list_augmentation_strategies()


def test_get_domain_by_path():
    path = Path(__file__).parent.parent / "kgb" / "domains" / "legal"
    legal = get_domain(str(path))
    assert legal.extraction.prompt


def test_external_domains_via_env(tmp_path, monkeypatch):
    domain_dir = tmp_path / "finance"
    (domain_dir / "extraction").mkdir(parents=True)
    (domain_dir / "extraction" / "prompt_open.md").write_text("extract finance triples")
    monkeypatch.setenv("KGB_DOMAINS_PATH", str(tmp_path))
    assert "finance" in list_available_domains()
    finance = get_domain("finance")
    assert finance.extraction.prompt == "extract finance triples"


def test_unknown_domain_raises():
    try:
        get_domain("does-not-exist")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "does-not-exist" in str(e)


if __name__ == "__main__":
    test_packaged_domains_discovered()
    test_get_domain_by_name()
    test_get_domain_by_path()
    test_unknown_domain_raises()
    print("ok")
