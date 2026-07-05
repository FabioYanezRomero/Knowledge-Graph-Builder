"""Validation and normalization utilities for triples.

Shared by both the extraction and augmentation builders, this module
provides schema-constraint collection, triple normalization, and
post-generation validation.
"""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from typing import Any

from ..domains import DomainSchema, ExtractionMode, KnowledgeDomain, Triple


@dataclass(frozen=True)
class SchemaConstraints:
    """Normalized schema constraints used by the builder orchestrators."""

    enforce: bool
    entity_types: tuple[str, ...]
    relation_types: tuple[str, ...]
    normalized_entity_types: frozenset[str]
    normalized_relation_types: frozenset[str]
    normalized_type_relations: frozenset[str]


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def normalize_constraint_label(value: str) -> str:
    """Normalize labels so schema validation tolerates case and punctuation variance."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return normalized.strip("_")


# ---------------------------------------------------------------------------
# Example helpers
# ---------------------------------------------------------------------------

def iter_example_triples(raw_example: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract triple payloads from domain example formats."""
    triples: list[dict[str, Any]] = []

    for extraction in raw_example.get("extractions", []):
        attributes = extraction.get("attributes")
        if isinstance(attributes, dict):
            triples.append(attributes)

    for output_triple in raw_example.get("output", []):
        if isinstance(output_triple, dict):
            triples.append(output_triple)

    return triples


# ---------------------------------------------------------------------------
# Schema constraint collection
# ---------------------------------------------------------------------------

def collect_schema_constraints(
    domain: KnowledgeDomain,
    raw_examples: list[dict[str, Any]] | None = None,
) -> SchemaConstraints:
    """Build a normalized constraint set from the schema and domain examples."""
    if domain.extraction_mode != ExtractionMode.CONSTRAINED:
        return SchemaConstraints(False, tuple(), tuple(), frozenset(), frozenset(), frozenset())

    schema: DomainSchema = domain.schema

    entity_types: list[str] = []
    normalized_entity_types: set[str] = set()
    for entity_type in schema.entity_types:
        normalized = normalize_constraint_label(entity_type)
        if normalized and normalized not in normalized_entity_types:
            entity_types.append(entity_type.strip())
            normalized_entity_types.add(normalized)

    relation_types: list[str] = []
    normalized_relation_types: set[str] = set()

    def _register_relation(label: str) -> None:
        normalized = normalize_constraint_label(label)
        if normalized and normalized not in normalized_relation_types:
            relation_types.append(label.strip())
            normalized_relation_types.add(normalized)

    for relation_type in schema.relation_types:
        _register_relation(relation_type)

    for raw_example in raw_examples or []:
        for triple in iter_example_triples(raw_example):
            relation = triple.get("relation")
            if isinstance(relation, str):
                _register_relation(relation)

    normalized_type_relations = frozenset(
        normalize_constraint_label(label) for label in schema.type_relations
    ) - {""}

    return SchemaConstraints(
        enforce=bool(entity_types or relation_types),
        entity_types=tuple(entity_types),
        relation_types=tuple(relation_types),
        normalized_entity_types=frozenset(normalized_entity_types),
        normalized_relation_types=frozenset(normalized_relation_types),
        normalized_type_relations=normalized_type_relations,
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------

def build_schema_guidance(constraints: SchemaConstraints) -> str:
    """Build prompt guidance from normalized constraints."""
    if not constraints.enforce:
        return ""

    lines = ["Schema constraints:"]

    if constraints.entity_types:
        lines.append("Allowed entity types:")
        lines.extend(f"- {entity_type}" for entity_type in constraints.entity_types)
        lines.append(
            "If entity types are implicit in the text, keep extracted entities semantically within these categories."
        )

    if constraints.relation_types:
        lines.append("Allowed relation labels:")
        lines.extend(f"- {relation_type}" for relation_type in constraints.relation_types)
        lines.append("Use only relation labels from this list. Non-matching labels will be discarded.")

    return "\n".join(lines)


def render_prompt_template(
    prompt_template: str,
    record: dict[str, Any],
    schema_guidance: str = "",
) -> str:
    """Render a prompt template with the current record payload."""
    prompt = prompt_template.replace(
        "{{record_json}}",
        json.dumps(record, ensure_ascii=False, indent=2)
    )
    prompt = prompt.replace("{{schema_constraints}}", schema_guidance)
    if schema_guidance and "{{schema_constraints}}" not in prompt_template:
        prompt = f"{prompt.rstrip()}\n\n{schema_guidance}"
    return prompt


# ---------------------------------------------------------------------------
# Triple normalization
# ---------------------------------------------------------------------------

def normalize_triple(raw_triple: dict[str, Any]) -> Triple | None:
    """Normalize a raw extraction to standard Triple format."""
    try:
        return Triple(
            head=raw_triple.get("head", ""),
            relation=raw_triple.get("relation", ""),
            tail=raw_triple.get("tail", ""),
            inference=raw_triple.get("inference", "explicit"),
            justification=raw_triple.get("justification"),
            extraction_text=raw_triple.get("extraction_text"),
            char_start=raw_triple.get("char_start"),
            char_end=raw_triple.get("char_end"),
        )
    except Exception as e:
        print(f"Warning: Skipping invalid triple: {e}")
        return None


# ---------------------------------------------------------------------------
# Entity-type helpers
# ---------------------------------------------------------------------------

def extract_explicit_entity_type_labels(raw_triple: dict[str, Any]) -> list[str]:
    """Collect explicit entity type annotations if a client/provider emitted them."""
    labels: list[str] = []
    candidate_keys = ("head_type", "tail_type", "entity_type", "source_type", "target_type")

    for key in candidate_keys:
        value = raw_triple.get(key)
        if isinstance(value, str) and value.strip():
            labels.append(value.strip())
        elif isinstance(value, list):
            labels.extend(
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            )

    return labels


def validate_entity_types(
    triple: Triple,
    raw_triple: dict[str, Any],
    constraints: SchemaConstraints,
) -> tuple[bool | None, list[str]]:
    """Validate entity type information when the builder can do so defensibly."""
    if not constraints.normalized_entity_types:
        return True, []

    explicit_labels = extract_explicit_entity_type_labels(raw_triple)
    if explicit_labels:
        invalid_labels = [
            label for label in explicit_labels
            if normalize_constraint_label(label) not in constraints.normalized_entity_types
        ]
        return not invalid_labels, invalid_labels

    if normalize_constraint_label(triple.relation) in constraints.normalized_type_relations:
        candidate_labels = [triple.head, triple.tail]
        is_valid = any(
            normalize_constraint_label(label) in constraints.normalized_entity_types
            for label in candidate_labels
        )
        return is_valid, candidate_labels if not is_valid else []

    return None, []


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def validate_triples_against_schema(
    triples: list[Triple],
    constraints: SchemaConstraints,
    *,
    raw_triples: list[dict[str, Any]] | None = None,
) -> tuple[list[Triple], dict[str, Any]]:
    """Filter triples against the normalized schema constraints."""
    summary = {
        "applied": constraints.enforce,
        "allowed_entity_types": list(constraints.entity_types),
        "allowed_relation_types": list(constraints.relation_types),
        "accepted_triples": 0,
        "rejected_triples": 0,
        "rejected_due_to_relation": 0,
        "rejected_due_to_entity_type": 0,
        "entity_type_validation_skipped": 0,
        "rejected_samples": [],
    }

    if not constraints.enforce:
        summary["accepted_triples"] = len(triples)
        return triples, summary

    accepted: list[Triple] = []
    raw_triples = raw_triples or []

    for index, triple in enumerate(triples):
        raw_triple = raw_triples[index] if index < len(raw_triples) and isinstance(raw_triples[index], dict) else {}
        rejected_for: list[str] = []

        if constraints.normalized_relation_types:
            normalized_relation = normalize_constraint_label(triple.relation)
            if normalized_relation not in constraints.normalized_relation_types:
                rejected_for.append("relation")
                summary["rejected_due_to_relation"] += 1

        entity_validation_result, invalid_entity_labels = validate_entity_types(triple, raw_triple, constraints)
        if entity_validation_result is None:
            summary["entity_type_validation_skipped"] += 1
        elif not entity_validation_result:
            rejected_for.append("entity_type")
            summary["rejected_due_to_entity_type"] += 1

        if rejected_for:
            summary["rejected_triples"] += 1
            if len(summary["rejected_samples"]) < 5:
                summary["rejected_samples"].append({
                    "triple": triple.model_dump(),
                    "reasons": rejected_for,
                    "invalid_entity_labels": invalid_entity_labels,
                })
            continue

        accepted.append(triple)

    summary["accepted_triples"] = len(accepted)
    return accepted, summary


def warn_on_schema_validation(stage: str, summary: dict[str, Any]) -> None:
    """Emit a compact warning when schema post-validation discards output."""
    if not summary.get("applied"):
        return

    rejected = summary.get("rejected_triples", 0)
    if rejected == 0:
        return

    warnings.warn(
        f"Schema validation during {stage} discarded {rejected} triple(s).",
        stacklevel=2,
    )
