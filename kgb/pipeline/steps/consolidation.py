"""Consolidation pipeline step: merge/clean existing triples, add no knowledge.

A sibling of AugmentationStep (both are StrategyStep), not a subclass of it —
consolidation is a distinct kind of operation. Run it after extract and/or
after augment in the pipeline's steps list.
"""

from __future__ import annotations

from ..step import register_step
from .strategy import StrategyStep


@register_step("consolidate")
class ConsolidationStep(StrategyStep):
    """Pipeline step for consolidation strategies (merge/clean existing
    triples without adding knowledge), e.g. entity_resolution."""

    EXPECTED_KIND = "consolidate"
    METADATA_PREFIX = "consolidation_"
    DEFAULT_STRATEGY = "entity_resolution"


__all__ = ["ConsolidationStep"]
