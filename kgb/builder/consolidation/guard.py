"""Closed-set guard: the invariant that consolidation may only relabel entities
to names that already exist in the graph, never introduce a new one."""

from __future__ import annotations


def enforce_closed_set(
    mapping: dict[str, str],
    entity_set: set[str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Filter a variant -> canonical mapping to the closed-set invariant.

    A merge whose canonical target is not an existing entity would introduce a
    node with no extraction grounding, so it is rejected. This keeps text-reading
    (Schwartz-Hearst) and LLM stages from smuggling ungrounded entities into the
    graph.

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


__all__ = ["enforce_closed_set"]
