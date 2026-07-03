"""Unified domain module for prompts and examples.

Domains are directories of resources (prompts, examples, schema) discovered
at runtime — no Python class needed. See registry.py for the resolution order
(registered classes, packaged directories, KGB_DOMAINS_PATH, direct paths).

Usage:
    from kgb.domains import get_domain

    legal = get_domain("legal", extraction_mode="open")
    custom = get_domain("/path/to/my_usecase")
"""

from .base import KnowledgeDomain, DomainComponent, DomainLike, DomainResourceError
from .models import DomainExamples, ExtractionMode, Triple, Extraction, ExtractionExample, AugmentationExample, DomainSchema, InferenceType
from .registry import domain, get_domain, register_domain, list_available_domains

__all__ = [
    # Base classes and protocols
    "KnowledgeDomain",
    "DomainComponent",
    "DomainLike",
    "DomainResourceError",
    # Models
    "DomainExamples",
    "DomainSchema",
    "ExtractionMode",
    "Triple",
    "Extraction",
    "ExtractionExample",
    "AugmentationExample",
    "InferenceType",
    # Registry
    "domain",
    "get_domain",
    "register_domain",
    "list_available_domains",
]
