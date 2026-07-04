import dataclasses
from typing import TYPE_CHECKING, Any, Iterator, Sequence, Mapping
import langextract as lx
from langextract.providers.openai import OpenAILanguageModel
from langextract.core import types as core_types
from langextract.core import exceptions as lx_exceptions

from ..base import BaseLLMClient, LLMClientError
from ..defaults import load_provider_defaults
from ..factory import client

if TYPE_CHECKING:
    from ..config import ClientConfig


@dataclasses.dataclass(init=False)
class OllamaOpenAILanguageModel(OpenAILanguageModel):
    """Custom OpenAI model for Ollama that removes unsupported response_format."""

    def __init__(self, **kwargs):
        # Extract timeout before super().__init__ discards it via **kwargs
        self._request_timeout = kwargs.pop("timeout", 600)
        # Ollama generation controls, injected into the request via extra_body
        self._think = kwargs.pop("think", None)
        self._model_options = kwargs.pop("model_options", None)
        super().__init__(**kwargs)
        # Recreate OpenAI client with proper timeout for slow local models
        import openai
        self._client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            organization=self.organization,
            timeout=self._request_timeout,
        )
        
    @property
    def requires_fence_output(self) -> bool:
        """Ollama/LM Studio output needs fences when not using structured mode."""
        return True

    @staticmethod
    def _sanitize_control_chars(text: str) -> str:
        """Remove invalid JSON control characters from LLM output.

        Some local models emit raw control characters (tabs, newlines, etc.)
        inside JSON string values, which causes json.loads() to fail with
        'Invalid control character'.  We replace them with spaces.
        """
        import re
        # Replace control chars (U+0000–U+001F) except \n \r \t which are
        # commonly present in fenced output and handled by langextract.
        # Inside JSON *strings* these are illegal, but we can't easily
        # distinguish string-interior vs structural whitespace here, so we
        # only strip the truly unusual ones (NUL, BEL, BS, VT, FF, etc.).
        return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', text)

    def _process_single_prompt(self, prompt: str, config: dict[str, Any]) -> core_types.ScoredOutput:
        """Override to remove response_format and add logging."""
        try:
            # Get model configuration
            model_config = self.merge_kwargs(config)

            # Explicitly remove response_format as it can cause issues in Ollama/LM Studio
            model_config.pop('response_format', None)

            # Standard system message for JSON
            system_message = (
                "You are a helpful assistant that extracts information into structured JSON. "
                "Follow the provided format Exactly, matching the field names and structure of the examples. "
                "You may use ```json code fences. Do not include any preamble or extra explanations."
            )

            messages = [{'role': 'user', 'content': prompt}]
            messages.insert(0, {'role': 'system', 'content': system_message})

            api_params = {
                'model': self.model_id,
                'messages': messages,
                'n': 1,
            }

            temp = model_config.get('temperature', self.temperature)
            if temp is not None:
                api_params['temperature'] = temp

            if (v := model_config.get('max_output_tokens')) is not None:
                api_params['max_tokens'] = v

            # Ollama honors these via the OpenAI-compatible endpoint's extra body:
            # `think` toggles the reasoning phase, `options` sets num_ctx etc.
            extra_body: dict[str, Any] = {}
            if self._think is not None:
                extra_body['think'] = self._think
            if self._model_options:
                extra_body['options'] = self._model_options
            if extra_body:
                api_params['extra_body'] = extra_body

            response = self._client.chat.completions.create(**api_params)
            output_text = response.choices[0].message.content

            # Sanitize control characters that break JSON parsing
            if output_text:
                output_text = self._sanitize_control_chars(output_text)

            return core_types.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise lx_exceptions.InferenceRuntimeError(
                f'Ollama OpenAI API error: {str(e)}', original=e
            ) from e


@client("ollama")
class OllamaClient(BaseLLMClient):
    """Client for Ollama local models via langextract.

    This client uses langextract's Ollama provider to interact with
    locally hosted models for knowledge graph extraction.
    """

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        max_workers: int | None = None,
        batch_length: int | None = None,
        max_char_buffer: int = 8000,
        show_progress: bool = True,
        timeout: int = 120,
        think: bool | None = None,
        options: dict | None = None,
    ) -> None:
        """Initialize Ollama client.

        Args:
            model_id: Ollama model name (see configs/ollama.json for default)
            base_url: Ollama server URL (see configs/ollama.json for default)
            max_workers: Maximum parallel workers (see configs/ollama.json)
            batch_length: Number of chunks per batch (see configs/ollama.json)
            max_char_buffer: Maximum characters for inference
            show_progress: Whether to show progress bar
            timeout: Request timeout in seconds
            think: False disables the model's "thinking" phase (faster, more stable)
            options: Extra Ollama generation options (num_ctx, top_p, num_predict, ...)
        """
        _defaults = load_provider_defaults("ollama")
        self.model_id = model_id or _defaults["model_id"]
        self.base_url = base_url or _defaults["base_url"]
        self.max_workers = max_workers if max_workers is not None else _defaults["max_workers"]
        self.batch_length = batch_length if batch_length is not None else _defaults["batch_length"]
        self.max_char_buffer = max_char_buffer
        self.show_progress = show_progress
        self.timeout = timeout
        self.think = think
        self.options = options

    def extract(
        self,
        text: str,
        prompt_description: str,
        examples: list[Any] | None = None,
        format_type: type | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Extract knowledge graph triples using Ollama via langextract.

        Args:
            text: Input text to analyze
            prompt_description: Extraction instructions
            examples: Few-shot examples (list of lx.ExampleData)
            format_type: Pydantic model for structured output
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional langextract parameters

        Returns:
            List of extracted triples

        Raises:
            LLMClientError: If extraction fails
        """
        try:
            # Ensure base_url ends with /v1 for the OpenAI provider
            base_url = self.base_url
            if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
                base_url = base_url.rstrip('/') + '/v1'

            # Create Ollama OpenAI language model
            ollama_model = OllamaOpenAILanguageModel(
                model_id=self.model_id,
                api_key="ollama", # Placeholder for OpenAI provider
                base_url=base_url,
                timeout=self.timeout,
                think=self.think,
                model_options=self.options,
            )

            # Prepare langextract kwargs
            langextract_kwargs = {
                "model": ollama_model,
                "temperature": temperature,
                "max_workers": self.max_workers,
                "batch_length": self.batch_length,
                "max_char_buffer": self.max_char_buffer,
                "show_progress": self.show_progress,
                "use_schema_constraints": False, 
                "fence_output": True,  # Expect JSON in code fences
                "fetch_urls": False,
                "resolver_params": {
                    "require_extractions_key": False,
                }
            }

            if max_tokens:
                langextract_kwargs["language_model_params"] = {"max_tokens": max_tokens}

            langextract_kwargs.update(kwargs)

            # Perform extraction
            result = lx.extract(
                text_or_documents=text,
                prompt_description=prompt_description,
                examples=examples or [],
                **langextract_kwargs
            )

            # Extract triples from result
            triples = []
            if hasattr(result, 'extractions') and result.extractions:
                for extraction in result.extractions:
                    # Robust attribute extraction (handles both wrapped and flat formats)
                    attrs = extraction.attributes
                    
                    # If attributes is None, it might be a flat dict in extraction_text or data
                    if attrs is None:
                        # Some versions of langextract might put the dict in extraction_text if it's flat
                        if isinstance(extraction.extraction_text, str):
                            try:
                                import json
                                text_trimmed = extraction.extraction_text.strip()
                                if text_trimmed.startswith('{') and text_trimmed.endswith('}'):
                                    attrs = json.loads(text_trimmed)
                            except:
                                pass
                    
                    if attrs:
                        # Ensure it's a dict
                        triple = dict(attrs)
                        
                        # Add source grounding information from langextract
                        if extraction.char_interval:
                            triple["char_start"] = extraction.char_interval.start_pos
                            triple["char_end"] = extraction.char_interval.end_pos
                        else:
                            triple["char_start"] = None
                            triple["char_end"] = None
                        
                        # Add extraction metadata
                        triple["extraction_text"] = str(extraction.extraction_text)
                        triple["extraction_class"] = str(extraction.extraction_class)
                        
                        # Basic validation: must have head, relation, tail
                        if all(k in triple for k in ('head', 'relation', 'tail')):
                            triples.append(triple)

            return triples

        except Exception as e:
            raise LLMClientError(f"Ollama extraction failed: {e}") from e

    def augment(
        self,
        text: str,
        prompt_description: str,
        format_type: type,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Generate augmentation triples directly using Ollama.

        This bypasses langextract to allow for ungrounded inference without
        the overhead of character-level source grounding. Used for bridging step.

        Args:
            text: Input text/prompt
            prompt_description: Instructions for generation
            format_type: Pydantic model for schema definition
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            List of dictionaries matching the requested schema
        """
        import json
        import requests

        try:
            # Build the prompt with schema
            schema_json = json.dumps(format_type.model_json_schema(), indent=2)
            full_prompt = f"""{prompt_description}

Return the results as a JSON array of objects matching this schema:
{schema_json}

Input Text:
{text}

IMPORTANT: Respond with ONLY a valid JSON array. No markdown code blocks, no explanation, just the JSON array starting with [ and ending with ].
Each object MUST have at minimum: "head", "relation", "tail" fields."""

            # Debug: Show prompt length
            print(f"  [DEBUG Ollama] Prompt length: {len(full_prompt)} chars")

            # Call Ollama API directly
            # NOTE: Do NOT use format:"json" - it forces single object responses
            # We want arrays like LMStudio, so rely on prompt instructions instead
            payload: dict[str, Any] = {
                "model": self.model_id,
                "prompt": full_prompt,
                "stream": False,
                # "format": "json",  # DISABLED - forces single object, not array
                "options": {
                    "temperature": temperature if temperature is not None else 0.0,
                    **({"num_predict": max_tokens} if max_tokens else {}),
                    **(self.options or {}),  # num_ctx, top_p, ... from config
                },
            }
            if self.think is not None:
                payload["think"] = self.think  # top-level for /api/generate
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("response", "")

            # Debug: Show raw response
            print(f"  [DEBUG Ollama] Response length: {len(response_text)} chars")
            print(f"  [DEBUG Ollama] Response preview: {response_text[:500]}...")

            if not response_text:
                print("  [DEBUG Ollama] Empty response!")
                return []

            # Parse the JSON response with robust extraction
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            import re
            if response_text.startswith("```"):
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
                if match:
                    response_text = match.group(1).strip()
            
            # Find JSON array or object 
            json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', response_text)
            if json_match:
                response_text = json_match.group(1)
            
            try:
                data = json.loads(response_text)
                if isinstance(data, list):
                    items = data
                elif isinstance(data, dict):
                    # Some models might return {"items": [...]} or {"triples": [...]}
                    items = []
                    for key in ["items", "triples", "data", "results", "extractions"]:
                        if key in data and isinstance(data[key], list):
                            items = data[key]
                            break
                    if not items:
                        items = [data]
                else:
                    return []

                # Debug: Show parsed items
                print(f"  [DEBUG Ollama] Parsed {len(items)} items")
                for i, item in enumerate(items[:3]):  # Show first 3
                    print(f"    Item {i}: {item}")

                # Force inference to contextual for bridging (consistency across providers)
                for item in items:
                    if isinstance(item, dict):
                        item['inference'] = 'contextual'
                
                return items
            except json.JSONDecodeError as e:
                print(f"  [DEBUG Ollama] JSON parse error: {e}")
                raise LLMClientError(f"Failed to parse JSON response: {e}\nResponse text: {response_text[:500]}")

        except requests.RequestException as e:
            raise LLMClientError(f"Ollama request failed: {e}") from e
        except Exception as e:
            raise LLMClientError(f"Ollama JSON generation failed: {e}") from e

    @classmethod
    def from_config(cls, config: "ClientConfig") -> "OllamaClient":
        """Create an OllamaClient from a ClientConfig."""
        return cls(
            model_id=config.model_id,
            base_url=config.base_url,
            max_workers=config.max_workers,
            batch_length=config.batch_length,
            max_char_buffer=config.max_char_buffer,
            show_progress=config.show_progress,
            timeout=config.timeout,
            think=config.think,
            options=config.options,
        )




__all__ = ["OllamaClient"]
