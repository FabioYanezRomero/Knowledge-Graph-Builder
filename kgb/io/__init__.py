"""Unified I/O module: readers (input loading) and writers (output conversion)."""

from .readers import load_records, detect_format, DataLoadError
from .writers import json_to_graphml, convert_json_directory, merge_json_directories

__all__ = [
    "load_records",
    "detect_format",
    "DataLoadError",
    "json_to_graphml",
    "convert_json_directory",
    "merge_json_directories",
]
