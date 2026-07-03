"""Data loaders for the extraction pipeline.

Supports multiple input formats:
- JSONL (recommended): One JSON object per line
- JSON: Array of objects
- CSV: Comma-separated values
- TXT: A plain-text file (one record), or a directory of .txt files
  (one record per file, id = filename stem)

Structured formats should have at least 'id' and 'text' fields.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Iterator


class DataLoadError(Exception):
    """Raised when data loading fails."""
    def __init__(self, message: str, path: Path, line_number: int | None = None):
        super().__init__(message)
        self.path = path
        self.line_number = line_number


def detect_format(path: Path) -> str:
    """Detect file format from extension. A directory is treated as a
    collection of plain-text files.

    Returns:
        One of: 'jsonl', 'json', 'csv', 'txt'

    Raises:
        DataLoadError: If format cannot be determined
    """
    if path.is_dir():
        return 'txt'

    suffix = path.suffix.lower()
    if suffix == '.jsonl':
        return 'jsonl'
    elif suffix == '.json':
        return 'json'
    elif suffix == '.csv':
        return 'csv'
    elif suffix == '.txt':
        return 'txt'
    else:
        raise DataLoadError(
            f"Unknown file format: {suffix}. Supported: .jsonl, .json, .csv, .txt (or a directory of .txt files)",
            path
        )


def load_records(
    path: Path,
    text_field: str = "text",
    id_field: str = "id",
    record_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load records from a file.

    Supports JSONL, JSON, and CSV formats (auto-detected by extension).

    Args:
        path: Path to input file
        text_field: Name of the field containing text (default: "text")
        id_field: Name of the field containing record IDs (default: "id")
        record_id: Optional list of record IDs to load
        limit: Optional limit on number of records

    Returns:
        List of records, each with at least 'id' and 'text' keys (normalized)

    Raises:
        DataLoadError: If file cannot be loaded or parsed
    """
    if not path.exists():
        raise DataLoadError(f"File not found: {path}", path)

    format_type = detect_format(path)

    if format_type == 'jsonl':
        records = _load_jsonl(path)
    elif format_type == 'json':
        records = _load_json(path)
    elif format_type == 'txt':
        records = _load_txt(path)
    else:  # csv
        records = _load_csv(path)

    # Normalize field names and validate
    normalized = []
    for i, record in enumerate(records):
        if record_ids and str(record[id_field]) not in record_ids:
            continue
        if limit and len(normalized) >= limit:
            break

        if text_field not in record:
            raise DataLoadError(
                f"Missing text field '{text_field}' in record {i}",
                path,
                line_number=i + 1
            )
        if id_field not in record:
            raise DataLoadError(
                f"Missing id field '{id_field}' in record {i}",
                path,
                line_number=i + 1
            )

        # Normalize to standard field names
        normalized_record = dict(record)
        if text_field != "text":
            normalized_record["text"] = record[text_field]
        if id_field != "id":
            normalized_record["id"] = record[id_field]

        normalized.append(normalized_record)

    return normalized


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from JSONL file (one JSON object per line)."""
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DataLoadError(
                    f"Invalid JSON on line {line_num}: {e}",
                    path,
                    line_number=line_num
                ) from e
    return records


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load records from JSON file (array of objects)."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise DataLoadError(f"Invalid JSON: {e}", path) from e

    if not isinstance(data, list):
        raise DataLoadError("JSON file must contain an array of objects", path)

    return data


def _load_txt(path: Path) -> list[dict[str, Any]]:
    """Load a plain-text file as one record, or a directory of .txt files
    as one record per file (id = filename stem, sorted by name)."""

    def read_one(file_path: Path) -> dict[str, Any]:
        try:
            text = file_path.read_text(encoding='utf-8-sig').strip()
        except UnicodeDecodeError:
            text = file_path.read_text(encoding='latin-1').strip()
        return {
            "id": file_path.stem,
            "text": text,
            "source_file": str(file_path),
        }

    if path.is_dir():
        txt_files = sorted(path.glob("*.txt"))
        if not txt_files:
            raise DataLoadError("No .txt files found in directory", path)
        records = [read_one(f) for f in txt_files]
        empty = [r["source_file"] for r in records if not r["text"]]
        if empty:
            print(f"Warning: skipping {len(empty)} empty .txt file(s): {', '.join(empty[:5])}", file=sys.stderr)
        return [r for r in records if r["text"]]

    record = read_one(path)
    if not record["text"]:
        raise DataLoadError("TXT file is empty", path)
    return [record]


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load records from CSV file."""
    records = []
    with open(path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(dict(row))
    return records


__all__ = ["load_records", "detect_format", "DataLoadError"]
