"""Base classes for knowledge domains.

This module provides the foundation for the Unified Domain Pattern:
- KnowledgeDomain: Abstract base class for domain implementations
- DomainComponent: Groups prompt and examples for a single activity
- AugmentationStrategy: Named container for strategy-specific resources
- DomainLike: Protocol for consumers to depend on (instead of concrete class)
- DomainResourceError: Custom exception for resource loading failures
"""

from __future__ import annotations

import json
import inspect
from abc import ABC
from pathlib import Path
from typing import Any, Optional, Union, Protocol, runtime_checkable

from .models import ExtractionMode, DomainSchema


class DomainResourceError(Exception):
    """Raised when a domain resource cannot be loaded.
    
    Attributes:
        resource_path: The path to the resource that failed to load.
        domain_name: The name of the domain (if known).
    """
    def __init__(self, message: str, resource_path: Optional[Path] = None, domain_name: Optional[str] = None):
        super().__init__(message)
        self.resource_path = resource_path
        self.domain_name = domain_name


class DomainComponent:
    """Groups prompt and examples for a specific domain activity."""

    def __init__(
        self,
        prompt_path: Path,
        examples_path: Path,
        loader: "KnowledgeDomain",
    ) -> None:
        self._prompt_path = prompt_path
        self._examples_path = examples_path
        self._loader = loader
        self._prompt: Optional[str] = None
        self._examples: Optional[list[dict[str, Any]]] = None

    @property
    def prompt(self) -> str:
        """The prompt text for this component."""
        if self._prompt is None:
            self._prompt = self._loader._load_text(self._prompt_path)
        return self._prompt

    @property
    def examples(self) -> list[dict[str, Any]]:
        """The examples list for this component."""
        if self._examples is None:
            examples_data = self._loader._load_json(self._examples_path) if self._examples_path.exists() else []
            self._examples = examples_data
        return self._examples


@runtime_checkable
class DomainLike(Protocol):
    """Protocol defining the interface for domain consumers.
    
    This allows consumers (like KnowledgeGraphExtractor) to depend on
    an abstract interface rather than the concrete KnowledgeDomain class,
    improving testability and decoupling.
    """
    extraction: DomainComponent
    
    def get_augmentation(self, strategy: str) -> DomainComponent: ...
    def list_augmentation_strategies(self) -> list[str]: ...


class KnowledgeDomain(ABC):
    """Abstract base class for a knowledge domain.
    
    A domain manages resource-based prompts, validated examples, and schemas.
    
    Folder structure:
        domain_name/
            extraction/
                prompt_open.md
                prompt_constrained.md
                examples.json
            consolidation/
                entity_resolution/   # merges/cleans, adds no knowledge
                    prompt.md
                    examples.json
            augmentation/
                connectivity/        # adds new (contextual) triples
                    prompt.md
                    examples.json
            schema.json

    Strategy folders are looked up under both augmentation/ and
    consolidation/ — the split is taxonomy for humans; loading is uniform.
    """

    _STRATEGY_DIRS = ("augmentation", "consolidation")

    def __init__(
        self,
        extraction_mode: Union[ExtractionMode, str] = ExtractionMode.OPEN,
        root_dir: Optional[Union[str, Path]] = None,
        extraction_prompt_path: Optional[Union[str, Path]] = None,
        extraction_examples_path: Optional[Union[str, Path]] = None,
        schema_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.extraction_mode = ExtractionMode(extraction_mode)
        
        # 1. Automatic Root Resolution
        if root_dir:
            self._root_dir = Path(root_dir)
        else:
            # Fallback to the directory where the concrete subclass is defined
            self._root_dir = Path(inspect.getfile(self.__class__)).parent

        # 2. Extraction Resource Paths (with overrides)
        ext_mode_file = "prompt_open.md" if self.extraction_mode == ExtractionMode.OPEN else "prompt_constrained.md"
        
        self._ext_prompt_path = Path(extraction_prompt_path) if extraction_prompt_path else self._root_dir / "extraction" / ext_mode_file
        self._ext_examples_path = Path(extraction_examples_path) if extraction_examples_path else self._root_dir / "extraction" / "examples.json"
        
        self._schema_path = Path(schema_path) if schema_path else self._root_dir / "schema.json"

        # 3. Extraction Component
        self.extraction = DomainComponent(self._ext_prompt_path, self._ext_examples_path, self)

        # 4. Augmentation Strategy Cache
        self._augmentation_cache: dict[str, DomainComponent] = {}
        
        # Lazy loaded data
        self._schema: Optional[DomainSchema] = None

    def get_augmentation(self, strategy: str = "connectivity") -> DomainComponent:
        """Get augmentation resources for a specific strategy.
        
        Args:
            strategy: The augmentation strategy name (e.g., "connectivity")
            
        Returns:
            DomainComponent with prompt and examples for the strategy
            
        Raises:
            DomainResourceError: If the strategy folder doesn't exist
        """
        if strategy not in self._augmentation_cache:
            for parent in self._STRATEGY_DIRS:
                strategy_dir = self._root_dir / parent / strategy
                if strategy_dir.exists():
                    break
            else:
                available = self.list_augmentation_strategies()
                raise DomainResourceError(
                    f"Unknown strategy '{strategy}'. Available: {', '.join(available) or 'none'}",
                    resource_path=self._root_dir / "augmentation" / strategy
                )

            self._augmentation_cache[strategy] = DomainComponent(
                prompt_path=strategy_dir / "prompt.md",
                examples_path=strategy_dir / "examples.json",
                loader=self,
            )
        return self._augmentation_cache[strategy]

    def list_augmentation_strategies(self) -> list[str]:
        """List all available strategies for this domain.

        Returns:
            Sorted strategy names (folder names under augmentation/ and consolidation/)
        """
        names = set()
        for parent in self._STRATEGY_DIRS:
            parent_dir = self._root_dir / parent
            if parent_dir.exists():
                names.update(d.name for d in parent_dir.iterdir() if d.is_dir())
        return sorted(names)

    @property
    def schema(self) -> DomainSchema:
        """Get the validated schema for this domain."""
        if self._schema is None:
            self._schema = self._load_schema()
        return self._schema

    def _load_schema(self) -> DomainSchema:
        if not self._schema_path.exists():
            return DomainSchema()  # Empty schema if not provided
        data = self._load_json(self._schema_path)
        return DomainSchema(**data)

    @staticmethod
    def _load_text(path: Path) -> str:
        """Load text content from a file.
        
        Raises:
            DomainResourceError: If the file does not exist.
        """
        if not path.exists():
            raise DomainResourceError(
                f"Resource not found: {path}",
                resource_path=path
            )
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _load_json(path: Path) -> Any:
        """Load and parse JSON content from a file.
        
        Raises:
            DomainResourceError: If the file does not exist or contains invalid JSON.
        """
        if not path.exists():
            raise DomainResourceError(
                f"Resource not found: {path}",
                resource_path=path
            )
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise DomainResourceError(
                f"Invalid JSON in {path}: {e}",
                resource_path=path
            ) from e


__all__ = ["KnowledgeDomain", "DomainComponent", "DomainLike", "DomainResourceError"]
