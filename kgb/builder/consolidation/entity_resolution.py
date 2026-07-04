"""Entity resolution: canonicalize entity names across triples.

Two layers, deterministic-first:
1. A Schwartz-Hearst pre-pass reads the source text for acronym/expansion pairs
   and merges them when BOTH forms already exist as entities (closed-set, no
   LLM, no cost).
2. The LLM clusters the remaining variants (synonyms, morphological variants)
   using edge context, and its output is filtered through the closed-set guard.
"""

from __future__ import annotations

from typing import Any

from ...clients import BaseLLMClient
from ...domains import KnowledgeDomain, Triple
from ..strategies import register_strategy
from ..validation import (
    build_schema_guidance,
    collect_schema_constraints,
    render_prompt_template,
)
from .guard import enforce_closed_set
from .schwartz_hearst import extract_abbreviation_pairs


def _collect_unique_entities(triples: list[Triple]) -> list[str]:
    """Collect all unique entity strings from triple heads and tails."""
    entities: set[str] = set()
    for t in triples:
        if t.head:
            entities.add(t.head.strip())
        if t.tail:
            entities.add(t.tail.strip())
    return sorted(entities)


def _build_entity_context(entities: list[str], triples: list[Triple]) -> dict[str, list[str]]:
    """Build a context map: entity → list of triples it appears in.

    This gives the LLM evidence about how each entity is used, so it can
    make informed decisions about whether two entity names are the same
    real-world thing (e.g., "Salvino" appearing as head of "served_as → CEO"
    confirms it's the same as "Michael J. Salvino").
    """
    context: dict[str, list[str]] = {e: [] for e in entities}
    for t in triples:
        triple_str = f"({t.head}) --[{t.relation}]--> ({t.tail})"
        h = t.head.strip() if t.head else ""
        tl = t.tail.strip() if t.tail else ""
        if h in context:
            context[h].append(triple_str)
        if tl in context and tl != h:
            context[tl].append(triple_str)
    # Cap per entity to keep prompt manageable
    for e in context:
        context[e] = context[e][:8]
    return context


def acronym_mapping(text: str, entities: list[str]) -> dict[str, str]:
    """Deterministic acronym → expansion merges via Schwartz-Hearst.

    Extracts (short, long) pairs from the text, then keeps only those where
    BOTH forms match an existing entity (case-insensitive). Returns a
    short_entity -> long_entity mapping (the expansion is canonical). Never
    introduces an entity absent from the graph.
    """
    by_norm = {e.lower(): e for e in entities}
    mapping: dict[str, str] = {}
    for short, long in extract_abbreviation_pairs(text):
        short_e = by_norm.get(short.lower())
        long_e = by_norm.get(long.lower())
        if short_e and long_e and short_e != long_e:
            mapping[short_e] = long_e
    return mapping


def _apply_entity_mapping(
    triples: list[Triple], mapping: dict[str, str]
) -> list[Triple]:
    """Rewrite triple heads/tails using the canonical mapping and deduplicate.

    Matching is case-insensitive because LLMs often return lowercased variants
    even when the original entities had proper casing.
    """
    ci_mapping: dict[str, str] = {}
    for variant, canonical in mapping.items():
        ci_mapping[variant.lower().strip()] = canonical

    seen: set[tuple[str, str, str]] = set()
    resolved: list[Triple] = []
    for t in triples:
        head = t.head.strip() if t.head else t.head
        tail = t.tail.strip() if t.tail else t.tail
        head = ci_mapping.get(head.lower(), head) if head else head
        tail = ci_mapping.get(tail.lower(), tail) if tail else tail
        key = (head.lower(), t.relation.lower().strip(), tail.lower())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(t.model_copy(update={"head": head, "tail": tail}))
    return resolved


@register_strategy("entity_resolution", kind="consolidate")
def entity_resolution_strategy(
    client: BaseLLMClient,
    domain: KnowledgeDomain,
    text: str,
    triples: list[Triple],
    *,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    augmentation_prompt_override: str | None = None,
    **kwargs: Any,
) -> tuple[list[Triple], dict[str, Any]]:
    """Entity resolution: merge entity name variants into canonical names.

    Does NOT generate new triples — it only merges existing entities and removes
    resulting duplicates, and never introduces an entity absent from the graph.

    Returns:
        Tuple of (resolved_triples, metadata)
    """
    entities = _collect_unique_entities(triples)
    if len(entities) <= 1:
        return triples, {"strategy": "entity_resolution", "status": "skipped", "reason": "<=1 entity"}

    entity_set = set(entities)

    # 1. Deterministic pre-pass: Schwartz-Hearst acronym/expansion merges,
    #    already closed to existing entities.
    mapping = acronym_mapping(text, entities)
    acronym_merges = len(mapping)

    # 2. LLM pass: cluster the remaining variants using edge context.
    entity_context = _build_entity_context(entities, triples)
    er_component = domain.get_augmentation("entity_resolution")
    prompt_template = augmentation_prompt_override or er_component.prompt
    constraints = collect_schema_constraints(domain, er_component.examples)

    entity_entries = [{"name": e, "edges": entity_context.get(e, [])} for e in entities]
    record: dict[str, Any] = {"entities": entity_entries}
    if text:
        record["source_text_excerpt"] = text[:4000] + ("..." if len(text) > 4000 else "")

    final_prompt = render_prompt_template(
        prompt_template,
        record,
        schema_guidance=build_schema_guidance(constraints),
    )

    print(f"  Entity resolution: {len(entities)} entities ({acronym_merges} acronym merges), asking LLM to cluster...", flush=True)

    raw_results = client.augment(
        text=final_prompt,
        prompt_description="Identify entity name variants and map them to canonical names",
        format_type=Triple,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    for group in raw_results or []:
        if not isinstance(group, dict) or "canonical" not in group:
            continue
        canonical = str(group.get("canonical", "")).strip()
        variants = group.get("variants", [])
        if canonical and isinstance(variants, list):
            for v in variants:
                v_str = str(v).strip()
                if v_str and v_str != canonical:
                    mapping[v_str] = canonical

    # 3. Closed-set guard over the combined mapping.
    mapping, rejected_mappings = enforce_closed_set(mapping, entity_set)
    if rejected_mappings:
        print(
            f"  Entity resolution: rejected {len(rejected_mappings)} merge(s) with "
            f"non-existent canonical (closed-set guard)",
            flush=True,
        )

    if not mapping:
        print("  Entity resolution: no merges", flush=True)
        return triples, {
            "strategy": "entity_resolution",
            "status": "no_merges",
            "entities_analyzed": len(entities),
            "acronym_merges": acronym_merges,
            "rejected_mappings": rejected_mappings,
        }

    resolved_triples = _apply_entity_mapping(triples, mapping)

    merged_entities = len(mapping)
    canonical_targets = len(set(mapping.values()))
    triples_before = len(triples)
    triples_after = len(resolved_triples)
    deduped = triples_before - triples_after

    print(f"  Entity resolution: {merged_entities} variants -> {canonical_targets} canonical names", flush=True)
    print(f"  Triples: {triples_before} -> {triples_after} ({deduped} duplicates removed)", flush=True)

    metadata = {
        "strategy": "entity_resolution",
        "status": "success",
        "entities_analyzed": len(entities),
        "acronym_merges": acronym_merges,
        "merge_groups": canonical_targets,
        "variants_mapped": merged_entities,
        "triples_before": triples_before,
        "triples_after": triples_after,
        "duplicates_removed": deduped,
        "mapping": mapping,
        "rejected_mappings": rejected_mappings,
    }

    return resolved_triples, metadata


__all__ = ["entity_resolution_strategy", "acronym_mapping"]
