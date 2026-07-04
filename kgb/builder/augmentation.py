"""Augmentation strategies for iterative graph refinement.

This module provides:
- Strategy Protocol for extensibility
- Registry pattern for strategy dispatch
- Built-in connectivity strategy
"""

from __future__ import annotations

from typing import Any, Protocol, Callable

import networkx as nx

from ..clients import BaseLLMClient
from ..domains import KnowledgeDomain, Triple, InferenceType
from .extraction import extract_triples
from .validation import (
    build_schema_guidance,
    collect_schema_constraints,
    render_prompt_template,
    validate_triples_against_schema,
    warn_on_schema_validation,
)

# =============================================================================
# Strategy Protocol & Registry
# =============================================================================

class AugmentationStrategy(Protocol):
    """Protocol for augmentation strategies.
    
    Strategy-specific parameters (e.g., max_iterations, max_disconnected)
    are passed via **kwargs and handled by each strategy individually.
    """
    def __call__(
        self,
        client: BaseLLMClient,
        domain: KnowledgeDomain,
        text: str,
        triples: list[Triple],
        **kwargs: Any
    ) -> tuple[list[Triple], dict[str, Any]]:
        """Execute the strategy.
        
        Args:
            client: LLM client for generation
            domain: Knowledge domain with prompts/examples
            text: Original source text
            triples: Current list of triples
            **kwargs: Strategy-specific parameters
            
        Returns:
            Tuple of (refined_triples, metadata)
        """
        ...


STRATEGIES: dict[str, AugmentationStrategy] = {}

# Taxonomy of graph operations:
#   "augment"     — adds knowledge that was not explicit (new triples,
#                   typically marked as contextual inference)
#   "consolidate" — merges/cleans existing knowledge without adding any
#                   (entity resolution, deduplication, canonicalization)
STRATEGY_KINDS: dict[str, str] = {}


def register_strategy(name: str, kind: str = "augment") -> Callable[[AugmentationStrategy], AugmentationStrategy]:
    """Decorator for registering graph-operation strategies.

    Args:
        name: Strategy name (also the domain resource folder name).
        kind: "augment" (adds new triples) or "consolidate" (merges/cleans
            existing ones without adding knowledge).

    Usage:
        @register_strategy("my_strategy")
        def my_strategy(client, domain, text, triples, **kwargs):
            ...
            return refined_triples, metadata
    """
    if kind not in ("augment", "consolidate"):
        raise ValueError(f"Unknown strategy kind '{kind}'. Expected 'augment' or 'consolidate'.")

    def decorator(fn: AugmentationStrategy) -> AugmentationStrategy:
        STRATEGIES[name] = fn
        STRATEGY_KINDS[name] = kind
        return fn
    return decorator


def list_strategies(kind: str | None = None) -> list[str]:
    """List registered strategies, optionally filtered by kind."""
    if kind is None:
        return list(STRATEGIES.keys())
    return [n for n in STRATEGIES if STRATEGY_KINDS.get(n) == kind]


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


def strategy_kind(name: str) -> str:
    """Return the kind of a registered strategy (defaults to 'augment')."""
    return STRATEGY_KINDS.get(name, "augment")


# =============================================================================
# Shared Utilities
# =============================================================================

def _build_graph_from_triples(triples: list[Triple]) -> nx.DiGraph:
    """Build NetworkX graph from Triple objects for analysis."""
    G = nx.DiGraph()
    for t in triples:
        if t.head and t.tail:
            G.add_edge(t.head, t.tail, relation=t.relation)
    return G


# =============================================================================
# Built-in Connectivity Strategy
# =============================================================================

def _format_components(components: list[set], G: nx.DiGraph, triples: list) -> str:
    """Format disconnected components with their triples for the augmentation prompt.
    
    Args:
        components: List of sets of node names (from nx.weakly_connected_components)
        G: The NetworkX graph
        triples: List of Triple objects
        
    Returns:
        Formatted string showing each component with its entities and triples
    """
    formatted = []
    
    for i, nodes in enumerate(components, 1):
        # Get entities in this component
        node_list = list(nodes)[:15]  # Limit to 15 entities for readability
        truncated = len(nodes) > 15
        
        # Find triples that belong to this component (both head and tail in this component)
        component_triples = []
        for t in triples:
            head = getattr(t, 'head', t.get('head', '')) if isinstance(t, dict) else t.head
            tail = getattr(t, 'tail', t.get('tail', '')) if isinstance(t, dict) else t.tail
            if head in nodes or tail in nodes:
                relation = getattr(t, 'relation', t.get('relation', '')) if isinstance(t, dict) else t.relation
                component_triples.append(f"  ({head}) --[{relation}]--> ({tail})")
        
        # Format component output
        entity_str = ", ".join(node_list)
        if truncated:
            entity_str += f"... (+{len(nodes) - 15} more)"
        
        comp_output = f"Component {i}:\n"
        comp_output += f"  Entities: [{entity_str}]\n"
        comp_output += f"  Triples ({len(component_triples)}):\n"
        
        # Limit triples shown per component
        for triple_str in component_triples[:10]:
            comp_output += f"    {triple_str}\n"
        if len(component_triples) > 10:
            comp_output += f"    ... (+{len(component_triples) - 10} more triples)\n"
        
        formatted.append(comp_output)
    
    return "\n".join(formatted)


@register_strategy("connectivity")
def connectivity_strategy(
    client: BaseLLMClient,
    domain: KnowledgeDomain,
    text: str,
    triples: list[Triple],
    *,
    max_disconnected: int = 3,
    max_iterations: int = 2,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    augmentation_prompt_override: str | None = None,
    **kwargs: Any
) -> tuple[list[Triple], dict[str, Any]]:
    """Connectivity augmentation: Reduce disconnected graph components.
    
    Iteratively prompts the LLM to find bridging relationships between
    disconnected components until the target is reached or max iterations.
    
    Args:
        client: LLM client
        domain: Knowledge domain
        text: Source text
        triples: Initial triples
        max_disconnected: Target max number of components (default: 3)
        max_iterations: Max refinement iterations (default: 2)
        temperature: Sampling temperature
        max_tokens: Max tokens for LLM
        augmentation_prompt_override: Override the default prompt
        
    Returns:
        Tuple of (all_triples, metadata)
    """
    augmentation_component = domain.get_augmentation("connectivity")
    aug_prompt_template = augmentation_prompt_override or augmentation_component.prompt
    constraints = collect_schema_constraints(domain, augmentation_component.examples)
    
    all_triples = list(triples)  # Copy to avoid mutating input
    iterations_data = []
    error_occurred = False

    for i in range(max_iterations):
        try:
            G = _build_graph_from_triples(all_triples)
            components = list(nx.weakly_connected_components(G))
            
            if len(components) <= max_disconnected:
                break

            # Prepare augmentation prompt
            comp_text = _format_components(components, G, all_triples)
            current_triples_dicts = [t.model_dump() for t in all_triples]
            
            record = {
                "text": text,
                "current_triples": current_triples_dicts,
                "disconnected_components": comp_text
            }
            
            final_prompt = render_prompt_template(
                aug_prompt_template,
                record,
                schema_guidance=build_schema_guidance(constraints),
            )
            
            # Call LLM for bridge triples using augment (NOT extract)
            # Augmentation generates NEW bridging triples that don't need source grounding
            # The extract() method uses langextract's grounding which fails for augmentation
            new_triples_raw = client.augment(
                text=final_prompt,
                prompt_description=f"Generate bridging triples to connect disconnected components",
                format_type=Triple,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            new_triples = []
            normalized_new_triples_raw: list[dict[str, Any]] = []
            for t_raw in new_triples_raw:
                try:
                    t_dict = t_raw if isinstance(t_raw, dict) else t_raw.model_dump()
                    t_dict["inference"] = InferenceType.CONTEXTUAL
                    new_triples.append(Triple(**t_dict))
                    normalized_new_triples_raw.append(t_dict)
                except Exception as e:
                    print(f"Warning: Skipping invalid augmented triple: {e}")

            validated_new_triples, validation_summary = validate_triples_against_schema(
                new_triples,
                constraints,
                raw_triples=normalized_new_triples_raw,
            )
            warn_on_schema_validation("augmentation", validation_summary)

            all_triples.extend(validated_new_triples)
            iterations_data.append({
                "iteration": i + 1,
                "components_before": len(components),
                "new_triples_count": len(validated_new_triples),
                "status": "success",
                "schema_validation": validation_summary,
            })
            
        except Exception as e:
            print(f"Error during augmentation iteration {i+1}: {e}")
            iterations_data.append({
                "iteration": i + 1,
                "status": "failed",
                "error": str(e)
            })
            error_occurred = True
            break

    final_G = _build_graph_from_triples(all_triples)
    final_components = list(nx.weakly_connected_components(final_G))

    metadata = {
        "strategy": "connectivity",
        "iterations": iterations_data, 
        "final_components": len(final_components),
        "partial_result": error_occurred,
        "schema_constraints_applied": constraints.enforce,
        "allowed_entity_types": list(constraints.entity_types),
        "allowed_relation_types": list(constraints.relation_types),
    }

    return all_triples, metadata


# =============================================================================
# Built-in Entity Resolution Strategy
# =============================================================================

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


def _parse_entity_mapping(response_text: str) -> dict[str, str]:
    """Parse LLM response into a variant→canonical mapping.

    The LLM returns a JSON array of {canonical, variants} groups.
    We invert that into a flat lookup: variant_string → canonical_string.
    """
    import json as _json
    import re as _re

    text = response_text.strip()

    # Strip markdown fences
    if text.startswith("```"):
        match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if match:
            text = match.group(1).strip()

    # Find JSON array
    json_match = _re.search(r'\[[\s\S]*\]', text)
    if not json_match:
        return {}

    try:
        groups = _json.loads(json_match.group(0))
    except _json.JSONDecodeError:
        return {}

    mapping: dict[str, str] = {}
    for group in groups:
        if not isinstance(group, dict):
            continue
        canonical = group.get("canonical", "").strip()
        variants = group.get("variants", [])
        if not canonical or not isinstance(variants, list):
            continue
        for v in variants:
            v_str = str(v).strip()
            if v_str and v_str != canonical:
                mapping[v_str] = canonical
    return mapping


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
    """Entity resolution augmentation: Canonicalize entity names across triples.

    Collects all unique entity strings, asks the LLM to cluster variants
    and pick canonical names, then rewrites every triple and deduplicates.

    This strategy does NOT generate new triples — it only merges existing
    entities and removes resulting duplicates.

    Args:
        client: LLM client
        domain: Knowledge domain (must have entity_resolution augmentation folder)
        text: Source text (unused but kept for protocol compatibility)
        triples: Existing triples to resolve
        temperature: Sampling temperature
        max_tokens: Max tokens for LLM
        augmentation_prompt_override: Override the default prompt

    Returns:
        Tuple of (resolved_triples, metadata)
    """
    import json as _json

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

    # 5. Parse mapping from response
    # augment() returns list[dict] — we need to reconstruct the JSON text
    # The raw response should already be parsed by the client, but the
    # schema won't match Triple. Fall back to raw response parsing.
    mapping: dict[str, str] = {}

    # If augment returned dicts with canonical/variants keys, use them directly
    if raw_results and isinstance(raw_results[0], dict) and "canonical" in raw_results[0]:
        for group in raw_results:
            canonical = group.get("canonical", "").strip()
            variants = group.get("variants", [])
            if canonical and isinstance(variants, list):
                for v in variants:
                    v_str = str(v).strip()
                    if v_str and v_str != canonical:
                        mapping[v_str] = canonical
    else:
        # Fallback: try to interpret raw_results as the mapping
        # This handles the case where augment() couldn't parse into Triple schema
        # and returned raw dicts
        for item in raw_results:
            if isinstance(item, dict) and "canonical" in item:
                canonical = item["canonical"].strip()
                for v in item.get("variants", []):
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


# =============================================================================
# Main Orchestrator
# =============================================================================

def augment_triples(
    client: BaseLLMClient,
    domain: KnowledgeDomain,
    text: str,
    record_id: str | None = None,
    initial_triples: list[Triple] | list[dict[str, Any]] | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    max_disconnected: int = 3,
    max_iterations: int = 2,
    augmentation_strategy: str = "connectivity",
    prompt_override: str | None = None,
    augmentation_prompt_override: str | None = None
) -> tuple[list[Triple], dict[str, Any]]:
    """Extract triples with iterative improvement via a registered strategy.

    This function orchestrates:
    1. Initial extraction (if no triples provided)
    2. Triple validation
    3. Strategy dispatch

    Args:
        client: LLM client
        domain: Knowledge domain
        text: Input text
        record_id: Record ID
        initial_triples: Optional existing triples to start from
        temperature: Sampling temperature
        max_tokens: Max tokens
        max_disconnected: Target max components (passed to strategy)
        max_iterations: Max iterations (passed to strategy)
        augmentation_strategy: Strategy name (default: "connectivity")
        prompt_override: Extraction prompt override
        augmentation_prompt_override: Augmentation prompt override

    Returns:
        Tuple of (all_triples, metadata)
        
    Raises:
        ValueError: If the strategy is not registered
    """
    initial_schema_validation: dict[str, Any] | None = None

    # 1. Initial Extraction (if needed)
    if initial_triples:
        validated_triples = []
        raw_validated_triples: list[dict[str, Any]] = []
        for t in initial_triples:
            if isinstance(t, Triple):
                validated_triples.append(t)
                raw_validated_triples.append(t.model_dump())
            else:
                try:
                    triple = Triple(**t)
                    validated_triples.append(triple)
                    raw_validated_triples.append(t)
                except Exception as e:
                    print(f"Warning: Skipping invalid initial triple: {e}")
        extraction_constraints = collect_schema_constraints(domain, domain.extraction.examples)
        triples, validation_summary = validate_triples_against_schema(
            validated_triples,
            extraction_constraints,
            raw_triples=raw_validated_triples,
        )
        initial_schema_validation = validation_summary
        warn_on_schema_validation("augmentation bootstrap", validation_summary)
    else:
        triples = extract_triples(
            client, domain, text, record_id, temperature, max_tokens, prompt_override
        )

    # 2. Strategy Dispatch
    strategy_fn = STRATEGIES.get(augmentation_strategy)
    if not strategy_fn:
        available = ", ".join(list_strategies())
        raise ValueError(
            f"Unknown augmentation strategy: '{augmentation_strategy}'. "
            f"Available: [{available}]"
        )

    triples, metadata = strategy_fn(
        client=client,
        domain=domain,
        text=text,
        triples=triples,
        max_disconnected=max_disconnected,
        max_iterations=max_iterations,
        temperature=temperature,
        max_tokens=max_tokens,
        augmentation_prompt_override=augmentation_prompt_override
    )
    if initial_schema_validation is not None:
        metadata["initial_schema_validation"] = initial_schema_validation
    return triples, metadata



__all__ = [
    "AugmentationStrategy",
    "register_strategy",
    "list_strategies",
    "strategy_kind",
    "enforce_closed_set",
    "STRATEGIES",
    "STRATEGY_KINDS",
    "connectivity_strategy",
    "entity_resolution_strategy",
    "augment_triples",
]
