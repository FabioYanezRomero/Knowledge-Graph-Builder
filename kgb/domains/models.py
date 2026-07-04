"""Pydantic models for domain data validation."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


class ExtractionMode(str, Enum):
    """Modes for graph extraction."""
    OPEN = "open"
    CONSTRAINED = "constrained"


class InferenceType(str, Enum):
    """Types of inference for triples."""
    EXPLICIT = "explicit"
    CONTEXTUAL = "contextual"


class Triple(BaseModel):
    """A single knowledge graph triple (head, relation, tail).
    
    This is the core data structure for representing relationships in the knowledge graph.
    All triples must have non-empty head, relation, and tail values.
    """
    model_config = ConfigDict(frozen=True)
    
    head: str = Field(description="The source entity in the relationship (person, organization, concept, etc.)")
    relation: str = Field(description="The relationship type connecting head to tail (e.g., works_at, filed_against, is_type, represents, etc.)")
    tail: str = Field(description="The target entity in the relationship")
    inference: InferenceType = Field(
        default=InferenceType.EXPLICIT,
        description="MUST be 'explicit' if directly stated, or 'contextual' if inferred for connectivity."
    )
    justification: Optional[str] = Field(
        default=None,
        description="Brief explanation for 'contextual' triples (optional for explicit)."
    )
    
    @field_validator('head', 'relation', 'tail')
    @classmethod
    def must_not_be_empty(cls, v: str, info) -> str:
        if not v or not v.strip():
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()


class Extraction(BaseModel):
    """A grounded extraction from text with character position information.
    
    This model represents a Triple that has been extracted from a specific
    location in the source text, enabling source verification.
    """
    model_config = ConfigDict(frozen=True)
    
    extraction_class: str = "Triple"
    extraction_text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    attributes: Triple


class ExtractionExample(BaseModel):
    """Few-shot example for extraction.
    
    Used to demonstrate the expected extraction format to the LLM.
    Each example contains source text and the expected extractions.
    """
    model_config = ConfigDict(frozen=True)
    
    text: str
    extractions: list[Extraction]


class Component(BaseModel):
    """A disconnected component in a graph, represented as a list of entity names."""
    model_config = ConfigDict(frozen=True)
    
    entities: list[str]


class AugmentationInput(BaseModel):
    """Input for augmentation/bridging step.
    
    Contains the original text and the list of disconnected components
    that need to be connected.
    """
    model_config = ConfigDict(frozen=True)
    
    text: str
    components: list[Component]


class AugmentationExample(BaseModel):
    """Few-shot example for augmentation.
    
    Demonstrates how to generate bridging triples that connect
    disconnected graph components.
    """
    model_config = ConfigDict(frozen=True)
    
    input: AugmentationInput
    output: list[Triple]


# Relations that express "X is-a-type-of Y". Used to validate entity types on
# typing triples. Declared here (domain layer) rather than hardcoded in the
# builder; a domain can override it in schema.json for its own conventions or
# language (e.g. "es_tipo_de").
DEFAULT_TYPE_RELATIONS = [
    "is_type",
    "is_type_of",
    "type_of",
    "instance_of",
    "entity_type",
    "category_of",
]


class DomainSchema(BaseModel):
    """Schema defining allowed entity and relation types for a domain.

    Used in 'constrained' extraction mode to limit the types of
    entities and relations the LLM can extract.
    """
    model_config = ConfigDict(frozen=True)

    entity_types: list[str] = Field(default_factory=list)
    relation_types: list[str] = Field(default_factory=list)
    type_relations: list[str] = Field(default_factory=lambda: list(DEFAULT_TYPE_RELATIONS))


class DomainExamples(BaseModel):
    """Collection of all examples for a domain.
    
    Groups extraction and augmentation examples together for
    convenient loading and validation.
    """
    model_config = ConfigDict(frozen=True)
    
    extraction: list[ExtractionExample] = Field(default_factory=list)
    augmentation: list[AugmentationExample] = Field(default_factory=list)


__all__ = [
    "ExtractionMode",
    "InferenceType",
    "Triple",
    "Extraction",
    "ExtractionExample",
    "AugmentationExample",
    "DomainSchema",
    "DomainExamples",
]
