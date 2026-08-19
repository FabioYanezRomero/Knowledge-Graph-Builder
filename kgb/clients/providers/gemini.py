"""Gemini client for knowledge graph extraction using langextract.

This module provides a client for Google Gemini models that properly leverages
the langextract library for:
- Source grounding (character-level positions for each extraction)
- Few-shot examples (schema enforcement via examples)
- Long document optimization (chunking, parallel processing)
- Controlled generation (native JSON schema constraints)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import langextract as lx

from ..base import BaseLLMClient, LLMClientError
from ..defaults import load_provider_defaults
from ..factory import client

if TYPE_CHECKING:
    from ..config import ClientConfig


@client("gemini")
class GeminiClient(BaseLLMClient):
    """Client for Google Gemini models using langextract.

    This client properly integrates with langextract to provide:
    - Source grounding: Each extraction includes character positions
    - Few-shot learning: Examples guide extraction quality
    - Long document handling: Automatic chunking and parallel processing
    - Controlled generation: Native JSON schema constraints
    """

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        max_workers: int | None = None,
        max_char_buffer: int = 8000,
        extraction_passes: int = 1,
        show_progress: bool = True,
        temperature: float = 0.0,
    ) -> None:
        """Initialize Gemini client with langextract.

        Args:
            model_id: Gemini model identifier (see configs/gemini.json for default)
            api_key: Google API key (or use LANGEXTRACT_API_KEY/GOOGLE_API_KEY env var)
            max_workers: Maximum parallel workers for long documents (see configs/gemini.json)
            max_char_buffer: Maximum characters per chunk for long documents
            extraction_passes: Number of extraction passes (higher = better recall)
            show_progress: Whether to show progress bar during extraction
            temperature: Default sampling temperature
        """
        _defaults = load_provider_defaults("gemini")
        self.model_id = model_id or _defaults["model_id"]
        self.api_key = api_key or os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.max_workers = max_workers if max_workers is not None else _defaults["max_workers"]
        self.max_char_buffer = max_char_buffer
        self.extraction_passes = extraction_passes
        self.show_progress = show_progress
        self.default_temperature = temperature

        if not self.api_key:
            raise LLMClientError(
                "No API key provided. Set api_key parameter or LANGEXTRACT_API_KEY/GOOGLE_API_KEY env var"
            )

        # Set API key in environment for langextract
        os.environ["LANGEXTRACT_API_KEY"] = self.api_key

    def extract(
        self,
        text: str,
        prompt_description: str,
        examples: list[Any] | None = None,
        format_type: type | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Extract knowledge graph triples using langextract.

        Uses langextract's full pipeline for extraction with source grounding,
        few-shot examples, and long document optimization.

        Args:
            text: Input text to analyze
            prompt_description: Extraction instructions
            examples: Few-shot examples (list of lx.data.ExampleData)
            format_type: Pydantic model (used to extract field names for schema)
            temperature: Sampling temperature (uses default if not specified)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional langextract parameters:
                - max_workers: Override parallel workers
                - max_char_buffer: Override chunk size
                - extraction_passes: Override number of passes

        Returns:
            List of extracted triples as dictionaries, each with:
                - head, relation, tail, inference, justification (from schema)
                - char_start, char_end (source grounding positions)
                - extraction_text (the text span that was extracted)

        Raises:
            LLMClientError: If extraction fails
        """
        try:
            # Build langextract parameters
            lx_kwargs = {
                "text_or_documents": text,
                "prompt_description": prompt_description,
                "examples": examples or [],
                "model_id": self.model_id,
                "format_type": lx.data.FormatType.JSON,
                "temperature": temperature if temperature is not None else self.default_temperature,
                "max_workers": kwargs.get("max_workers", self.max_workers),
                "max_char_buffer": kwargs.get("max_char_buffer", self.max_char_buffer),
                "extraction_passes": kwargs.get("extraction_passes", self.extraction_passes),
                "show_progress": kwargs.get("show_progress", self.show_progress),
                "use_schema_constraints": True,
                # See the ollama provider: langextract's fuzzy aligner is
                # quadratic on large chunks and its offsets are discarded by
                # extract_triples anyway.
                "resolver_params": {"enable_fuzzy_alignment": False},
            }

            # Add max_tokens if specified
            if max_tokens:
                lx_kwargs["language_model_params"] = {"max_output_tokens": max_tokens}

            # Perform extraction using langextract
            result = lx.extract(**lx_kwargs)

            # Convert langextract extractions to triple dictionaries
            triples = []
            if hasattr(result, "extractions"):
                for extraction in result.extractions:
                    triple = self._extraction_to_dict(extraction)
                    triples.append(triple)

            return triples

        # Return the entire traceback
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise LLMClientError(f"Langextract extraction failed: {e}") from e

    def augment(
        self,
        text: str,
        prompt_description: str,
        format_type: type,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Generate augmentation triples using Gemini's native JSON mode.

        This bypasses langextract to allow for ungrounded inference without
        the overhead or restrictions of character-level source grounding.

        Args:
            text: Input text/prompt
            prompt_description: Instructions for generation
            format_type: Pydantic model for schema definition
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            List of dictionaries matching the requested schema
        """
        import google.generativeai as genai
        import json

        try:
            # Configure genai with the API key
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_id)

            # Build the prompt
            schema_json = json.dumps(format_type.model_json_schema(), indent=2)
            full_prompt = f"""
{prompt_description}

Return the results as a JSON list of objects matching this JSON schema:
{schema_json}

Input Text:
{text}
"""

            # Generation configuration
            generation_config = {
                "temperature": temperature if temperature is not None else self.default_temperature,
                "response_mime_type": "application/json",
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            # Call the model
            response = model.generate_content(
                full_prompt,
                generation_config=generation_config
            )

            # Parse the JSON response
            if not response.text:
                return []

            try:
                data = json.loads(response.text)
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    # Some models might return {"items": [...]} or similar
                    for val in data.values():
                        if isinstance(val, list):
                            return val
                    return [data]
                return []
            except json.JSONDecodeError as e:
                raise LLMClientError(f"Failed to parse JSON response: {e}\nResponse text: {response.text}")

        except Exception as e:
            raise LLMClientError(f"Gemini JSON generation failed: {e}") from e

    def _extraction_to_dict(self, extraction: Any) -> dict[str, Any]:
        """Convert a langextract Extraction to a triple dictionary.

        Args:
            extraction: langextract Extraction object or dict

        Returns:
            Dictionary with triple fields + source grounding
        """
        # Handle cases where extraction is a dict (some langextract versions/configs)
        if isinstance(extraction, dict):
            # Access attributes from dict
            attrs = extraction.get("attributes", {})
            triple = dict(attrs) if attrs else {}
            
            # Add source grounding information
            char_interval = extraction.get("char_interval")
            if char_interval:
                # char_interval might also be a dict or object
                if isinstance(char_interval, dict):
                    triple["char_start"] = char_interval.get("start_pos")
                    triple["char_end"] = char_interval.get("end_pos")
                else:
                    triple["char_start"] = char_interval.start_pos
                    triple["char_end"] = char_interval.end_pos
            else:
                triple["char_start"] = None
                triple["char_end"] = None
            
            triple["extraction_text"] = extraction.get("extraction_text")
            triple["extraction_class"] = extraction.get("extraction_class")
            
            return triple

        # Standard object access
        # Start with attributes (contains head, relation, tail, etc.)
        triple = dict(extraction.attributes) if extraction.attributes else {}

        # Add source grounding information
        if extraction.char_interval:
            triple["char_start"] = extraction.char_interval.start_pos
            triple["char_end"] = extraction.char_interval.end_pos
        else:
            triple["char_start"] = None
            triple["char_end"] = None

        # Add extraction metadata
        triple["extraction_text"] = extraction.extraction_text
        triple["extraction_class"] = extraction.extraction_class

        return triple

    def extract_raw(
        self,
        text: str,
        prompt_description: str,
        examples: list[Any] | None = None,
        **kwargs: Any
    ) -> Any:
        """Extract and return the raw langextract AnnotatedDocument.

        This is useful when you need access to the full langextract result,
        including all metadata and the ability to generate visualizations.

        Args:
            text: Input text to analyze
            prompt_description: Extraction instructions
            examples: Few-shot examples
            **kwargs: Additional langextract parameters

        Returns:
            langextract AnnotatedDocument with full extraction results
        """
        try:
            lx_kwargs = {
                "text_or_documents": text,
                "prompt_description": prompt_description,
                "examples": examples or [],
                "model_id": self.model_id,
                "format_type": lx.data.FormatType.JSON,
                "temperature": kwargs.get("temperature", self.default_temperature),
                "max_workers": kwargs.get("max_workers", self.max_workers),
                "max_char_buffer": kwargs.get("max_char_buffer", self.max_char_buffer),
                "extraction_passes": kwargs.get("extraction_passes", self.extraction_passes),
                "show_progress": kwargs.get("show_progress", self.show_progress),
                "use_schema_constraints": True,
                # See the ollama provider: langextract's fuzzy aligner is
                # quadratic on large chunks and its offsets are discarded by
                # extract_triples anyway.
                "resolver_params": {"enable_fuzzy_alignment": False},
            }

            return lx.extract(**lx_kwargs)

        except Exception as e:
            raise LLMClientError(f"Langextract extraction failed: {e}") from e

    @classmethod
    def from_config(cls, config: "ClientConfig") -> "GeminiClient":
        """Create a GeminiClient from a ClientConfig."""
        return cls(
            model_id=config.model_id,
            api_key=config.api_key,
            max_workers=config.max_workers,
            max_char_buffer=config.max_char_buffer,
            extraction_passes=config.extraction_passes,
            show_progress=config.show_progress,
            temperature=config.temperature,
        )




__all__ = ["GeminiClient"]
