"""Entity resolution: canonicalize entity names across triples.

Layered, deterministic-first, and composable with the semantic layer:
1. A deterministic SIEVE BAG (sieves.py) runs high-precision passes over the
   source text and entity strings, merging only existing entities (closed-set).
2. The entity set is REDUCED to the sieve canonicals, and the LLM clusters the
   remaining variants (synonyms, morphological) on that smaller set.
3. Sieve and LLM merges are composed via transitive closure, then filtered by
   the discriminative veto and the closed-set guard.
"""

from __future__ import annotations

from typing import Any, Callable

from ...clients import BaseLLMClient
from ...domains import KnowledgeDomain, Triple
from ..strategies import register_strategy
from ..validation import (
    build_schema_guidance,
    collect_schema_constraints,
    render_prompt_template,
)
from .fuzzy import fuzzy_candidates
from .guard import enforce_closed_set
from .sieves import run_sieves, resolve_chains
from .veto import apply_discriminative_veto


def _collect_unique_entities(triples: list[Triple]) -> list[str]:
    """Collect all unique entity strings from triple heads and tails."""
    entities: set[str] = set()
    for t in triples:
        if t.head:
            entities.add(t.head.strip())
        if t.tail:
            entities.add(t.tail.strip())
    return sorted(entities)


def _build_entity_context(
    entities: list[str],
    triples: list[Triple],
    resolve: Callable[[str], str] = lambda e: e,
) -> dict[str, list[str]]:
    """Build a context map: entity → triples it appears in, endpoints resolved
    through `resolve` so edges reference the (post-sieve) canonical names the LLM
    is being asked about."""
    context: dict[str, list[str]] = {e: [] for e in entities}
    for t in triples:
        h = resolve(t.head.strip()) if t.head else ""
        tl = resolve(t.tail.strip()) if t.tail else ""
        triple_str = f"({h}) --[{t.relation}]--> ({tl})"
        if h in context:
            context[h].append(triple_str)
        if tl in context and tl != h:
            context[tl].append(triple_str)
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
    """Merge entity name variants into canonical names (deterministic sieves +
    LLM), never introducing an entity absent from the graph.

    Returns:
        Tuple of (resolved_triples, metadata)
    """
    entities = _collect_unique_entities(triples)
    if len(entities) <= 1:
        return triples, {"strategy": "entity_resolution", "status": "skipped", "reason": "<=1 entity"}

    entity_set = set(entities)

    # 1. Deterministic sieve bag (closed-set), then resolve chains to get the
    #    canonical each entity currently maps to.
    sieve_mapping = run_sieves(text, entities)
    sieve_resolved = resolve_chains(sieve_mapping)
    resolve = lambda e: sieve_resolved.get(e, e)
    sieve_merges = len(sieve_resolved)

    # 2. Reduce the entity set to the sieve canonicals — the LLM works on the
    #    already-consolidated set, and its merges compose with the sieves'.
    reduced_entities = sorted({resolve(e) for e in entities})
    entity_context = _build_entity_context(reduced_entities, triples, resolve)

    er_component = domain.get_augmentation("entity_resolution")
    prompt_template = augmentation_prompt_override or er_component.prompt
    constraints = collect_schema_constraints(domain, er_component.examples)

    entity_entries = [{"name": e, "edges": entity_context.get(e, [])} for e in reduced_entities]
    record: dict[str, Any] = {"entities": entity_entries}
    if text:
        record["source_text_excerpt"] = text[:4000] + ("..." if len(text) > 4000 else "")

    final_prompt = render_prompt_template(
        prompt_template,
        record,
        schema_guidance=build_schema_guidance(constraints),
    )

    # Fuzzy blocking: surface look-alike pairs as CANDIDATES for the LLM to
    # judge (never merged here). Appended as an explicit instruction so the LLM
    # actually attends to them — directs it to morphological variants the
    # exact-match sieve can't catch. Veto-filtered, so it isn't asked about
    # numeric/staging siblings. Works for any domain without editing its prompt.
    candidates = fuzzy_candidates(reduced_entities)
    if candidates:
        lines = "\n".join(f"- {a}  vs  {b}" for a, b, _ in candidates)
        final_prompt += (
            "\n\n## Candidate pairs to evaluate\n"
            "These entity names are surface-similar. For each pair, decide whether "
            "they refer to the SAME real-world entity, and if so include them in the "
            "same merge group. If they are genuinely different, do not merge them.\n"
            f"{lines}"
        )

    print(
        f"  Entity resolution: {len(entities)} entities -> {len(reduced_entities)} after "
        f"{sieve_merges} sieve merge(s); asking LLM to cluster the rest...",
        flush=True,
    )

    raw_results = client.augment(
        text=final_prompt,
        prompt_description="Identify entity name variants and map them to canonical names",
        format_type=Triple,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # 3. Combine: sieves first (higher precision wins), then LLM additions.
    mapping = dict(sieve_mapping)
    for group in raw_results or []:
        if not isinstance(group, dict) or "canonical" not in group:
            continue
        canonical = str(group.get("canonical", "")).strip()
        variants = group.get("variants", [])
        if canonical and isinstance(variants, list):
            for v in variants:
                v_str = str(v).strip()
                if v_str and v_str != canonical:
                    mapping.setdefault(v_str, canonical)

    # 4. Transitive closure so sieve and LLM merges chain (A->B->C => A->C).
    mapping = resolve_chains(mapping)

    # 5. Discriminative veto, then closed-set guard.
    mapping, vetoed_mappings = apply_discriminative_veto(mapping)
    if vetoed_mappings:
        print(
            f"  Entity resolution: vetoed {len(vetoed_mappings)} merge(s) differing "
            f"in a discriminative token (numeric/staging)",
            flush=True,
        )
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
            "sieve_merges": sieve_merges,
            "vetoed_mappings": vetoed_mappings,
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
        "sieve_merges": sieve_merges,
        "fuzzy_candidates": len(candidates),
        "merge_groups": canonical_targets,
        "variants_mapped": merged_entities,
        "triples_before": triples_before,
        "triples_after": triples_after,
        "duplicates_removed": deduped,
        "mapping": mapping,
        "vetoed_mappings": vetoed_mappings,
        "rejected_mappings": rejected_mappings,
    }

    return resolved_triples, metadata


__all__ = ["entity_resolution_strategy"]
