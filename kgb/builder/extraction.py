"""Core extraction logic for converting text to triples."""

from __future__ import annotations

from typing import Any

import langextract as lx
from ..clients import BaseLLMClient
from ..domains import KnowledgeDomain, Triple
from .validation import (
    SchemaConstraints,
    build_schema_guidance,
    collect_schema_constraints,
    normalize_triple,
    render_prompt_template,
    validate_triples_against_schema,
    warn_on_schema_validation,
)


def _build_examples(domain: KnowledgeDomain) -> list[lx.data.ExampleData]:
    """Build extraction examples from the domain configuration.

    Converts raw example dicts to proper langextract ExampleData objects
    with Extraction objects (not plain dicts).
    """
    raw_examples = domain.extraction.examples
    examples = []

    # Valid fields for lx.data.Extraction
    valid_extraction_fields = {
        "extraction_text", "extraction_class", "attributes",
        "char_interval", "description", "extraction_index",
        "group_index", "alignment_status"
    }

    for ex_data in raw_examples:
        # Make a copy to avoid mutating original
        ex_data = dict(ex_data)

        # Convert dict extractions to Extraction objects
        if "extractions" in ex_data:
            extractions = []
            for ext in ex_data["extractions"]:
                if isinstance(ext, dict):
                    ext = dict(ext)  # Copy to avoid mutation

                    # Convert char_start/char_end to char_interval
                    if "char_start" in ext and "char_end" in ext:
                        char_start = ext.pop("char_start")
                        char_end = ext.pop("char_end")
                        if char_start is not None and char_end is not None:
                            ext["char_interval"] = lx.data.CharInterval(
                                start_pos=char_start,
                                end_pos=char_end
                            )

                    # Filter keys to valid fields
                    filtered_ext = {k: v for k, v in ext.items() if k in valid_extraction_fields}
                    extractions.append(lx.data.Extraction(**filtered_ext))
                else:
                    extractions.append(ext)
            ex_data["extractions"] = extractions
        examples.append(lx.data.ExampleData(**ex_data))
    return examples


def extract_triples(
    client: BaseLLMClient,
    domain: KnowledgeDomain,
    text: str,
    record_id: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    prompt_override: str | None = None
) -> tuple[list[Triple], list[str]]:
    """Extract triples from a single text.

    Args:
        client: LLM client to use
        domain: Knowledge domain providing prompts and examples
        text: Input text
        record_id: Optional record ID for logging
        temperature: Sampling temperature
        max_tokens: Max tokens to generate
        prompt_override: Optional prompt template override

    Returns:
        (triples, standalone_entities) — triples are grounded relations; standalone
        entities are grounded nodes with no relation stated in the text.
    """
    record = {"text": text}
    if record_id:
        record["id"] = record_id

    prompt_template = prompt_override or domain.extraction.prompt
    raw_examples = domain.extraction.examples
    constraints = collect_schema_constraints(domain, raw_examples)
    final_prompt = render_prompt_template(
        prompt_template,
        record,
        schema_guidance=build_schema_guidance(constraints),
    )
    examples = _build_examples(domain)

    # Use langextract for extraction (required for visualizations)
    raw_triples = client.extract(
        text=final_prompt,
        prompt_description="Extract meaningful knowledge graph triples from the text, focusing on explicit relationships between entities.",
        examples=examples,
        format_type=Triple,
        temperature=temperature,
        max_tokens=max_tokens
    )

    triples = []
    normalized_raw_triples: list[dict[str, Any]] = []
    entities: list[str] = []
    seen_entities: set[str] = set()
    for t in raw_triples:
        if not isinstance(t, dict):
            continue
        head = str(t.get("head", "")).strip()
        relation = str(t.get("relation", "")).strip()
        tail = str(t.get("tail", "")).strip()
        # Grounding-only: a standalone entity (relation/tail absent because the
        # text states no relation for it) becomes an isolated node, never a
        # fabricated triple.
        if head and (not relation or not tail):
            if head not in seen_entities:
                seen_entities.add(head)
                entities.append(head)
            continue
        normalized = normalize_triple(t)
        if normalized:
            triples.append(normalized)
            normalized_raw_triples.append(t)

    validated_triples, validation_summary = validate_triples_against_schema(
        triples,
        constraints,
        raw_triples=normalized_raw_triples,
    )
    warn_on_schema_validation("extraction", validation_summary)
    return validated_triples, entities
