#!/usr/bin/env python3
"""Prepare sample pathology reports for the pathology_ollama pipeline.

Downloads the public mtsamples clinical-transcription dataset, filters to
oncology/pathology-flavored reports, and writes them as .txt files into
data/pathology/reports/ (the input directory referenced by
kgb/pipeline/configs/pathology_ollama.yaml).

Public stand-in for private clinical reports — no PHI. Usage:
    python scripts/prepare_pathology_data.py [--n 15]
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://huggingface.co/datasets/tchebonenko/MedicalTranscriptions/resolve/main/mtsamples.csv.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "pathology" / "reports"


def _rows() -> list[dict]:
    with urllib.request.urlopen(URL, timeout=120) as r:  # noqa: S310 (trusted host)
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    with zf.open("mtsamples.csv") as f:
        return list(csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")))


def _is_pathology(row: dict) -> bool:
    spec = row["medical_specialty"].strip()
    kw = (row["keywords"] or "").lower()
    txt = (row["transcription"] or "").strip()
    flavored = spec == "Hematology - Oncology" or any(
        k in kw for k in ("biopsy", "pathology", "carcinoma")
    )
    return flavored and 600 <= len(txt) <= 3500


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=15, help="number of reports to write")
    n = ap.parse_args().n

    OUT.mkdir(parents=True, exist_ok=True)
    picked, seen = [], set()
    for row in _rows():
        if not _is_pathology(row):
            continue
        txt = row["transcription"].strip()
        if txt[:80] in seen:
            continue
        seen.add(txt[:80])
        picked.append(row)
        if len(picked) >= n:
            break

    for i, row in enumerate(picked, 1):
        name = re.sub(r"[^a-z0-9]+", "_", (row["sample_name"] or f"report_{i}").strip().lower()).strip("_")
        (OUT / f"{i:02d}_{name}.txt").write_text(row["transcription"].strip() + "\n", encoding="utf-8")

    print(f"Wrote {len(picked)} pathology reports to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
