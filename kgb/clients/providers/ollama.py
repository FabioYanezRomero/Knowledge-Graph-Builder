from typing import TYPE_CHECKING, Any
import requests
import langextract as lx
from langextract.providers.ollama import OllamaLanguageModel

from ..base import BaseLLMClient, LLMClientError
from ..defaults import load_provider_defaults
from ..factory import client

if TYPE_CHECKING:
    from ..config import ClientConfig


class _ThinkInjectingRequests:
    """Wraps the ``requests`` module so Ollama ``/api/generate`` calls carry a
    top-level ``think`` flag (and extra options).

    This is the ONLY reliable way to disable gemma-style reasoning on Ollama:
    the OpenAI-compatible endpoint ignores ``think`` entirely, and the native
    endpoint ignores it inside ``options`` — it must sit at the request-body top
    level, which langextract's native provider doesn't expose. The provider
    reads its HTTP client from ``self._requests``, so we swap in this shim.
    """

    def __init__(self, real, think: bool | None, options: dict | None):
        self._real = real
        self._think = think
        self._options = options

    def post(self, url, *args, **kwargs):
        payload = kwargs.get("json")
        if isinstance(payload, dict):
            if self._think is not None:
                payload["think"] = self._think
            if self._options:
                payload.setdefault("options", {}).update(self._options)
        return self._real.post(url, *args, **kwargs)

    def __getattr__(self, name):  # forward .exceptions and anything else
        return getattr(self._real, name)


def _json_loads_ok(text: str) -> bool:
    import json
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _salvage_json_objects(text: str) -> str | None:
    """Rebuild a clean JSON array from the top-level ``{...}`` objects in ``text``,
    keeping only the ones that parse. Handles the malformations a strict parser
    chokes on but that our lighter fixes miss: doubled commas, unquoted keys in a
    stray object, and — importantly — a response truncated mid-object because the
    model hit max_tokens (the incomplete trailing object is simply dropped)."""
    import json
    import re

    objs: list = []
    depth = 0
    start: int | None = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    frag = text[start : i + 1]
                    try:
                        objs.append(json.loads(frag))
                    except Exception:
                        try:  # one more try after dropping a trailing comma
                            objs.append(json.loads(re.sub(r",(\s*})", r"\1", frag)))
                        except Exception:
                            pass
                    start = None
    return json.dumps(objs) if objs else None


def _repair_json_text(text: str) -> str:
    """Make local-model JSON parseable by the native provider's strict parser.

    Applies cheap fixes first (strip control chars, drop trailing commas). If the
    result still doesn't parse, salvages the individual valid objects so a single
    malformed or truncated object doesn't lose the whole extraction. Guarantees
    the returned text is valid JSON whenever any object could be recovered.
    """
    import re

    fixed = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)  # control chars
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)                # trailing commas

    # Fast path: if it (or its fenced body) already parses, keep it as-is so the
    # common healthy case is untouched.
    body = fixed
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", fixed)
    if m:
        body = m.group(1)
    if _json_loads_ok(body):
        return fixed

    salvaged = _salvage_json_objects(fixed)
    return salvaged if salvaged is not None else fixed


class OllamaExtractionModel(OllamaLanguageModel):
    """Native Ollama provider (/api/generate) for extraction.

    Chosen over the OpenAI-compatible endpoint because only the native endpoint
    honors a top-level ``think`` flag (disabling gemma-style reasoning, which
    otherwise makes extraction ~4x slower and time out). Injects ``think`` +
    options via the ``_requests`` shim and repairs common JSON defects before
    langextract's strict parser sees them.
    """

    def __init__(self, *args, think: bool | None = None, model_options: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if think is not None or model_options:
            self._requests = _ThinkInjectingRequests(requests, think, dict(model_options or {}))

    def _ollama_query(self, *args, **kwargs):
        resp = super()._ollama_query(*args, **kwargs)
        if isinstance(resp, dict) and isinstance(resp.get("response"), str):
            resp = dict(resp)
            resp["response"] = _repair_json_text(resp["response"])
        return resp


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
            # Use langextract's NATIVE Ollama provider (/api/generate), not the
            # OpenAI-compatible endpoint: only the native endpoint honors a
            # top-level `think` flag to disable reasoning (the OpenAI-compat one
            # ignores it, so extraction would always run slow "thinking" mode).
            ollama_model = OllamaExtractionModel(
                model_id=self.model_id,
                model_url=self.base_url,
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
