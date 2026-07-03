"""Smoke tests for the plain-text reader (single file and directory)."""

from pathlib import Path

import pytest

from kgb.io.readers import DataLoadError, detect_format, load_records


def test_single_txt_file(tmp_path):
    f = tmp_path / "report_001.txt"
    f.write_text("The patient was prescribed aspirin.", encoding="utf-8")
    records = load_records(f)
    assert len(records) == 1
    assert records[0]["id"] == "report_001"
    assert records[0]["text"] == "The patient was prescribed aspirin."


def test_directory_of_txt_files(tmp_path):
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("x,y", encoding="utf-8")
    records = load_records(tmp_path)
    assert [r["id"] for r in records] == ["a", "b"]  # sorted, empty skipped


def test_directory_respects_limit(tmp_path):
    for i in range(5):
        (tmp_path / f"doc{i}.txt").write_text(f"text {i}", encoding="utf-8")
    assert len(load_records(tmp_path, limit=2)) == 2


def test_empty_dir_and_empty_file_raise(tmp_path):
    with pytest.raises(DataLoadError):
        load_records(tmp_path)
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    with pytest.raises(DataLoadError):
        load_records(f)


def test_detect_format():
    assert detect_format(Path("x.txt")) == "txt"
    assert detect_format(Path("x.jsonl")) == "jsonl"
