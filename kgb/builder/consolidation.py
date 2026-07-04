"""Consolidation strategies: merge/clean existing knowledge without adding any.

Consolidation operates on an already-extracted graph and never introduces
entities or relations absent from it (the closed-set invariant enforced by
enforce_closed_set). Currently provides entity_resolution; future stages
(acronym expansion, discriminative veto) will land here too.
"""

from __future__ import annotations

from typing import Any

from ..clients import BaseLLMClient
from ..domains import KnowledgeDomain, Triple
from .strategies import register_strategy
from .validation import (
    build_schema_guidance,
    collect_schema_constraints,
    render_prompt_template,
)


def enforce_closed_set(
    mapping: dict[str, str],
    entity_set: set[str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Closed-set guard for consolidation: normalization may relabel entities
    only to names that already exist in the graph.

    A merge whose canonical target is not an existing entity would introduce a
    node with no extraction grounding, so it is rejected. This is the invariant
    that keeps text-reading and LLM stages from smuggling ungrounded entities
    into the graph.

    Args:
        mapping: Proposed variant -> canonical rewrites.
        entity_set: The entities that actually exist in the graph.

    Returns:
        (kept, rejected) — kept mappings, and the (variant, canonical) pairs
        dropped because their canonical is not an existing entity.
    """
    kept: dict[str, str] = {}
    rejected: list[tuple[str, str]] = []
    for variant, canonical in mapping.items():
        if canonical in entity_set:
            kept[variant] = canonical
        else:
            rejected.append((variant, canonical))
    return kept, rejected


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


def _apply_entity_mapping(
    triples: list[Triple], mapping: dict[str, str]
) -> list[Triple]:
    """Rewrite triple heads/tails using the canonical mapping and deduplicate.

    Matching is case-insensitive because LLMs often return lowercased variants
    even when the original entities had proper casing.
    """
    # Build a case-insensitive lookup
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
    """Entity resolution: canonicalize entity names across triples.

    Collects all unique entity strings, asks the LLM to cluster variants
    and pick canonical names, then rewrites every triple and deduplicates.

    This strategy does NOT generate new triples — it only merges existing
    entities and removes resulting duplicates.

    Args:
        client: LLM client
        domain: Knowledge domain (must have entity_resolution strategy folder)
        text: Source text (context for ambiguous cases)
        triples: Existing triples to resolve
        temperature: Sampling temperature
        max_tokens: Max tokens for LLM
        augmentation_prompt_override: Override the default prompt

    Returns:
        Tuple of (resolved_triples, metadata)
    """
    # 1. Collect unique entities and their context (triples they appear in)
    entities = _collect_unique_entities(triples)
    if len(entities) <= 1:
        return triples, {"strategy": "entity_resolution", "status": "skipped", "reason": "<=1 entity"}

    entity_context = _build_entity_context(entities, triples)

    # 2. Load prompt from domain
    er_component = domain.get_augmentation("entity_resolution")
    prompt_template = augmentation_prompt_override or er_component.prompt
    constraints = collect_schema_constraints(domain, er_component.examples)

    # 3. Build prompt with entity list, their graph context, and source text
    #    The LLM needs to see HOW each entity is used in order to decide
    #    whether "Salvino" and "Michael J. Salvino" are truly the same.
    entity_entries = []
    for e in entities:
        edges = entity_context.get(e, [])
        entry = {"name": e, "edges": edges}
        entity_entries.append(entry)

    record: dict[str, Any] = {"entities": entity_entries}
    # Include a text excerpt so the LLM has document context for ambiguous cases
    if text:
        # Truncate to ~4000 chars to keep prompt size reasonable
        record["source_text_excerpt"] = text[:4000] + ("..." if len(text) > 4000 else "")

    final_prompt = render_prompt_template(
        prompt_template,
        record,
        schema_guidance=build_schema_guidance(constraints),
    )

    # 4. Call LLM
    print(f"  Entity resolution: {len(entities)} unique entities, asking LLM to cluster...", flush=True)

    # We need a raw text response (not structured extraction), so use augment()
    # with Triple as a dummy format_type. We'll parse the JSON ourselves.
    raw_results = client.augment(
        text=final_prompt,
        prompt_description="Identify entity name variants and map them to canonical names",
        format_type=Triple,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 5. Parse mapping from response into variant -> canonical. augment()
    #    returns list[dict]; process every {canonical, variants} group.
    mapping: dict[str, str] = {}
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

    # 5b. Closed-set guard: reject merges whose canonical isn't an existing
    #     entity, so the LLM can only collapse real nodes, never invent one.
    mapping, rejected_mappings = enforce_closed_set(mapping, set(entities))
    if rejected_mappings:
        print(
            f"  Entity resolution: rejected {len(rejected_mappings)} merge(s) with "
            f"non-existent canonical (closed-set guard)",
            flush=True,
        )

    if not mapping:
        print("  Entity resolution: LLM returned no merge groups", flush=True)
        return triples, {
            "strategy": "entity_resolution",
            "status": "no_merges",
            "entities_analyzed": len(entities),
            "rejected_mappings": rejected_mappings,
        }

    # 6. Apply mapping
    resolved_triples = _apply_entity_mapping(triples, mapping)

    # Count stats
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
        "merge_groups": canonical_targets,
        "variants_mapped": merged_entities,
        "triples_before": triples_before,
        "triples_after": triples_after,
        "duplicates_removed": deduped,
        "mapping": mapping,
        "rejected_mappings": rejected_mappings,
    }

    return resolved_triples, metadata


__all__ = ["enforce_closed_set", "entity_resolution_strategy"]
