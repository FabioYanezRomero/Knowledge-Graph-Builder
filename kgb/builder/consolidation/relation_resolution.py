"""Relation resolution: canonicalize relation labels across triples.

Sibling of entity_resolution, on the relation axis. Runs AFTER entity
resolution so endpoints are already canonical — then two predicates that
connect the SAME (head, tail) pairs are strong, precise evidence of synonymy
("prostate ca --graded_as--> gleason 7" and "prostate ca --has_grade-->
gleason 7" clearly share a meaning). That co-occurrence signal is masked when
endpoints are still messy, which is why order matters.

Simpler than entity_resolution: relation labels don't live in the source text
as spans, so there are no text sieves (exact-match / acronym / Schwartz-Hearst)
and no discriminative veto (relations lack the opaque numeric/staging tokens
that trip the veto — grade_1 vs grade_2 are just different relations, and the
fuzzy veto already declines to propose them). Flow: collect relations -> fuzzy
candidates + co-occurrence context -> LLM cluster -> transitive closure ->
closed-set guard -> apply.

The prompt is built in and domain-agnostic (relation clustering doesn't depend
on a domain's naming conventions the way entity naming does), so no domain
needs a relation_resolution folder. A domain MAY still override it by adding
one. When a domain declares relation_types in its schema, that closed
vocabulary is surfaced to the LLM as canonical guidance.
"""

from __future__ import annotations

from typing import Any

from ...clients import BaseLLMClient
from ...domains import DomainResourceError, KnowledgeDomain, Triple
from ..strategies import register_strategy
from ..validation import (
    build_schema_guidance,
    collect_schema_constraints,
    render_prompt_template,
)
from .layers.fuzzy import fuzzy_candidates
from .layers.guard import enforce_closed_set
from .layers.sieves import resolve_chains


_DEFAULT_RELATION_PROMPT = """You are an expert in relation normalization for knowledge graphs.

## Task
Given a list of relation labels (predicates) extracted from a document, identify groups of labels that express the SAME relationship. For each group, choose the best canonical label.

## Rules

### 1. Synonymous predicates (MERGE)
Merge labels that mean the same thing, especially when they connect the same pairs of entities:
- "graded_as" / "has_grade" -> pick one canonical form
- "located_in" / "is_located_in" / "situated_in" -> one form
- "treats" / "used_to_treat" -> one form

### 2. Distinct predicates (DO NOT MERGE)
Keep genuinely different relationships apart, even when surface-similar:
- Opposite meaning: "increases" vs "decreases", "positive_for" vs "negative_for", "causes" vs "prevents"
- Different direction or role: "part_of" vs "has_part"
- Different granularity that changes meaning: "diagnosed_with" vs "at_risk_of"

### 3. Canonical label selection
For each group, prefer the clearest, most standard predicate form. If a list of allowed relation labels is given below, choose the canonical from that list.

## Evidence Format
Each relation below is shown with `connects` — example (head) -> (tail) pairs it links. Two relations that link the SAME pairs are strong evidence they mean the same thing.

## Output Format
Return a JSON array of merge groups. Each group is an object with:
- `canonical`: The chosen canonical label
- `variants`: Array of ALL labels in the group (including the canonical itself)

Only include groups with 2+ members. Return ONLY the JSON array. No explanation, no markdown fences.

{{schema_constraints}}

## Relations to Resolve (with connection context)
{{record_json}}"""


def _collect_unique_relations(triples: list[Triple]) -> list[str]:
    """Collect all unique relation labels from the triples."""
    relations: set[str] = set()
    for t in triples:
        if t.relation:
            relations.add(t.relation.strip())
    return sorted(relations)


def _build_relation_context(
    relations: list[str], triples: list[Triple]
) -> dict[str, list[str]]:
    """Map each relation to a few "(head) -> (tail)" pairs it connects — the
    co-occurrence evidence the LLM uses to judge synonymy."""
    context: dict[str, list[str]] = {r: [] for r in relations}
    for t in triples:
        rel = t.relation.strip() if t.relation else ""
        if rel in context:
            context[rel].append(f"({t.head}) -> ({t.tail})")
    for r in context:
        context[r] = context[r][:8]
    return context


def _apply_relation_mapping(
    triples: list[Triple], mapping: dict[str, str]
) -> list[Triple]:
    """Rewrite triple relations using the canonical mapping and deduplicate.

    Case-insensitive, mirroring entity resolution (LLMs return lowercased
    variants even when originals had proper casing).
    """
    ci_mapping = {v.lower().strip(): c for v, c in mapping.items()}

    seen: set[tuple[str, str, str]] = set()
    resolved: list[Triple] = []
    for t in triples:
        relation = t.relation.strip() if t.relation else t.relation
        relation = ci_mapping.get(relation.lower(), relation) if relation else relation
        key = (t.head.lower(), relation.lower().strip(), t.tail.lower())
        if key in seen:
            continue
        seen.add(key)
        resolved.append(t.model_copy(update={"relation": relation}))
    return resolved


@register_strategy("relation_resolution", kind="consolidate", builtin_prompt=True)
def relation_resolution_strategy(
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
    """Merge relation-label variants into canonical predicates (fuzzy candidates
    + LLM), never introducing a label absent from the graph (closed-set guard).

    Returns:
        Tuple of (resolved_triples, metadata)
    """
    relations = _collect_unique_relations(triples)
    if len(relations) <= 1:
        return triples, {"strategy": "relation_resolution", "status": "skipped", "reason": "<=1 relation"}

    relation_set = set(relations)

    # Prompt: a domain may override with its own relation_resolution folder;
    # otherwise the built-in, domain-agnostic default is used.
    examples: list[dict[str, Any]] | None = None
    try:
        component = domain.get_augmentation("relation_resolution")
        prompt_template = augmentation_prompt_override or component.prompt
        examples = component.examples
    except DomainResourceError:
        prompt_template = augmentation_prompt_override or _DEFAULT_RELATION_PROMPT

    # Surface the domain's declared relation vocabulary (if any) as canonical
    # guidance — the "closed list" the user provides via schema.relation_types.
    constraints = collect_schema_constraints(domain, examples)

    context = _build_relation_context(relations, triples)
    entries = [{"name": r, "connects": context.get(r, [])} for r in relations]
    record: dict[str, Any] = {"relations": entries}

    final_prompt = render_prompt_template(
        prompt_template,
        record,
        schema_guidance=build_schema_guidance(constraints),
    )

    # Fuzzy blocking: surface look-alike labels as CANDIDATES for the LLM to
    # judge (never merged here). Veto-filtered so numeric siblings aren't asked.
    candidates = fuzzy_candidates(relations)
    if candidates:
        lines = "\n".join(f"- {a}  vs  {b}" for a, b, _ in candidates)
        final_prompt += (
            "\n\n## Candidate pairs to evaluate\n"
            "These relation labels are surface-similar. For each pair, decide whether "
            "they express the SAME relationship (merge) or an opposite/different one "
            "(keep separate).\n"
            f"{lines}"
        )

    print(
        f"  Relation resolution: {len(relations)} relations; asking LLM to cluster synonyms...",
        flush=True,
    )

    raw_results = client.augment(
        text=final_prompt,
        prompt_description="Identify relation-label variants and map them to canonical labels",
        format_type=Triple,
        temperature=temperature,
        max_tokens=max_tokens,
    )

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
                    mapping.setdefault(v_str, canonical)

    # Transitive closure, then closed-set guard (canonical must already exist).
    mapping = resolve_chains(mapping)
    mapping, rejected_mappings = enforce_closed_set(mapping, relation_set)
    if rejected_mappings:
        print(
            f"  Relation resolution: rejected {len(rejected_mappings)} merge(s) with "
            f"non-existent canonical (closed-set guard)",
            flush=True,
        )

    if not mapping:
        print("  Relation resolution: no merges", flush=True)
        return triples, {
            "strategy": "relation_resolution",
            "status": "no_merges",
            "relations_analyzed": len(relations),
            "fuzzy_candidates": len(candidates),
            "rejected_mappings": rejected_mappings,
        }

    resolved_triples = _apply_relation_mapping(triples, mapping)

    merged = len(mapping)
    canonical_targets = len(set(mapping.values()))
    triples_before = len(triples)
    triples_after = len(resolved_triples)
    deduped = triples_before - triples_after

    print(f"  Relation resolution: {merged} variants -> {canonical_targets} canonical label(s)", flush=True)
    print(f"  Triples: {triples_before} -> {triples_after} ({deduped} duplicates removed)", flush=True)

    metadata = {
        "strategy": "relation_resolution",
        "status": "success",
        "relations_analyzed": len(relations),
        "fuzzy_candidates": len(candidates),
        "merge_groups": canonical_targets,
        "variants_mapped": merged,
        "triples_before": triples_before,
        "triples_after": triples_after,
        "duplicates_removed": deduped,
        "mapping": mapping,
        "rejected_mappings": rejected_mappings,
    }

    return resolved_triples, metadata


__all__ = ["relation_resolution_strategy"]
