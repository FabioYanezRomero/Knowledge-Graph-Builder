"""Validate a domain directory's resources before spending tokens on them.

Checks prompts exist and are non-empty, examples validate against the
Pydantic models, char spans fit the example text, and schema.json parses.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from .models import AugmentationExample, DomainSchema, ExtractionExample
from .registry import _DOMAIN_REGISTRY, _find_domain_dir


def _resolve_root(name_or_path: str) -> Path | None:
    if name_or_path in _DOMAIN_REGISTRY:
        return _DOMAIN_REGISTRY[name_or_path]()._root_dir
    return _find_domain_dir(name_or_path)


def _load_json_items(path: Path, errors: list[str]) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"{path}: invalid JSON — {e}")
        return []
    if not isinstance(data, list):
        errors.append(f"{path}: expected a JSON array of examples")
        return []
    return data


def lint_domain(name_or_path: str) -> tuple[list[str], list[str]]:
    """Lint a domain by name or directory path.

    Returns:
        (errors, warnings) — errors make the domain unusable or unreliable;
        warnings are optional resources that are missing or suspicious.
    """
    errors: list[str] = []
    warnings: list[str] = []

    root = _resolve_root(name_or_path)
    if root is None:
        return [f"Domain '{name_or_path}' not found (no directory with an extraction/ subfolder)"], []

    # Extraction prompts
    open_prompt = root / "extraction" / "prompt_open.md"
    constrained_prompt = root / "extraction" / "prompt_constrained.md"
    if not open_prompt.is_file() or not open_prompt.read_text(encoding="utf-8").strip():
        errors.append(f"{open_prompt}: missing or empty (required)")
    if not constrained_prompt.is_file():
        warnings.append(f"{constrained_prompt}: missing — 'constrained' mode will fail for this domain")

    # Extraction examples
    ext_examples = root / "extraction" / "examples.json"
    if not ext_examples.is_file():
        warnings.append(f"{ext_examples}: missing — extraction will run zero-shot")
    else:
        for i, item in enumerate(_load_json_items(ext_examples, errors)):
            try:
                example = ExtractionExample(**item)
            except ValidationError as e:
                errors.append(f"{ext_examples}[{i}]: {e.error_count()} validation error(s) — {e.errors()[0]['msg']} at {'.'.join(str(p) for p in e.errors()[0]['loc'])}")
                continue
            for j, extraction in enumerate(example.extractions):
                if extraction.char_start is not None and extraction.char_end is not None:
                    if not (0 <= extraction.char_start <= extraction.char_end <= len(example.text)):
                        warnings.append(f"{ext_examples}[{i}].extractions[{j}]: char span ({extraction.char_start}, {extraction.char_end}) outside text bounds (len={len(example.text)})")
                if extraction.extraction_text and extraction.extraction_text not in example.text:
                    warnings.append(f"{ext_examples}[{i}].extractions[{j}]: extraction_text not found verbatim in text")

    # Augmentation strategies
    aug_dir = root / "augmentation"
    strategies = [d for d in aug_dir.iterdir() if d.is_dir()] if aug_dir.is_dir() else []
    if not strategies:
        warnings.append(f"{aug_dir}: no augmentation strategies — 'kgb augment' will fail for this domain")
    for strategy in strategies:
        prompt = strategy / "prompt.md"
        if not prompt.is_file() or not prompt.read_text(encoding="utf-8").strip():
            errors.append(f"{prompt}: missing or empty (required per strategy)")
        aug_examples = strategy / "examples.json"
        if not aug_examples.is_file():
            warnings.append(f"{aug_examples}: missing — augmentation will run zero-shot")
        else:
            items = _load_json_items(aug_examples, errors)
            # Only connectivity has a fixed model; other strategies (e.g.
            # entity_resolution) define their own example shape, so we only
            # check they are valid JSON arrays.
            if strategy.name == "connectivity":
                for i, item in enumerate(items):
                    try:
                        AugmentationExample(**item)
                    except ValidationError as e:
                        errors.append(f"{aug_examples}[{i}]: {e.error_count()} validation error(s) — {e.errors()[0]['msg']} at {'.'.join(str(p) for p in e.errors()[0]['loc'])}")

    # Schema
    schema_path = root / "schema.json"
    if not schema_path.is_file():
        warnings.append(f"{schema_path}: missing — no type constraints for 'constrained' mode")
    else:
        try:
            DomainSchema(**json.loads(schema_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            errors.append(f"{schema_path}: invalid JSON — {e}")
        except ValidationError as e:
            errors.append(f"{schema_path}: {e.errors()[0]['msg']} at {'.'.join(str(p) for p in e.errors()[0]['loc'])}")

    return errors, warnings


__all__ = ["lint_domain"]
