"""LM Studio client using langextract for knowledge graph extraction."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

import langextract as lx
from langextract.providers.openai import OpenAILanguageModel
from langextract.core import types as core_types
from langextract.core import exceptions

from ..base import BaseLLMClient, LLMClientError
from ..json_repair import _repair_json_text
from ..defaults import load_provider_defaults
from ..factory import client

if TYPE_CHECKING:
    from ..config import ClientConfig


_CTX_HEADROOM = 0.85  # leave ~15% of the window for the answer


def _rejects_parameter(exc: Exception) -> bool:
    """Did the server refuse the request because of an unknown parameter?

    Narrow on purpose: a 400 about an unsupported field is worth retrying without
    it, but a 400 about anything else — or a timeout, or a connection error — is
    a real failure and must not be swallowed by a retry that looks like success.
    """
    if getattr(exc, "status_code", None) not in (400, 422):
        return False
    msg = str(exc).lower()
    return any(s in msg for s in
               ("reasoning_effort", "unknown parameter", "unrecognized",
                "unsupported", "extra fields", "not permitted"))


def context_limit(base_url: str, model_id: str) -> int | None:
    """Tokens LM Studio will actually accept, or None if we cannot find out.

    Unlike Ollama, LM Studio takes no per-request context size: the window is
    fixed when the model is loaded, and an over-long prompt is truncated by the
    server according to its overflow policy — quietly, and the reply still looks
    well-formed. So we cannot grow the window the way the ollama provider does;
    the only defence is to read the window and refuse to send something that
    won't fit.

    Read via LM Studio's native REST API (``/api/v0/models``), which reports
    ``loaded_context_length`` for the model currently in memory; the
    OpenAI-compatible ``/v1/models`` does not expose it.
    """
    import requests

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        entries = requests.get(f"{root}/api/v0/models", timeout=10).json()["data"]
    except Exception:
        return None  # a guard that breaks extraction is worse than no guard

    match = next((e for e in entries if e.get("id") == model_id), None)
    if match is None:
        # The default model_id is the placeholder "local-model", and LM Studio
        # serves whatever is loaded regardless of the id sent. Ask the model in
        # memory, which is the one that will answer.
        loaded = [e for e in entries if e.get("state") == "loaded"]
        match = loaded[0] if len(loaded) == 1 else None
    if match is None:
        return None
    # ONLY the loaded window, never max_context_length. Measured on a JIT-loaded
    # qwen3.8-27b: loaded_context_length 4,096 against max_context_length
    # 262,144 -- a 64x gap. Falling back to the maximum would wave through every
    # prompt the server then truncates, which is the exact failure this guards.
    # A model that is not loaded yet has no window to report; returning None
    # leaves the first request unguarded and the rest checked, because LM Studio
    # loads on that first request and reports the real window from then on.
    return match.get("loaded_context_length")


def assert_prompt_fits(limit: int | None, prompt: str, what: str) -> None:
    """Fail loudly rather than let LM Studio silently drop half the prompt.

    This is a configuration problem, not a data problem: if the window is too
    small it is too small for every chunk, so there is nothing to salvage by
    continuing. Truncated-but-plausible output is the failure mode we most want
    to avoid — it is indistinguishable from a model that just answered badly.
    """
    if not limit:
        return
    approx_tokens = len(prompt) / 3.5
    if approx_tokens > limit * _CTX_HEADROOM:
        raise LLMClientError(
            f"LM Studio {what} prompt is ~{int(approx_tokens):,} tokens but the loaded "
            f"model's context is {limit:,}. LM Studio would truncate it silently and "
            f"return a plausible-looking partial answer. Raise the context length when "
            f"loading the model, or send less at once (smaller --max-char-buffer for "
            f"extraction, fewer entities per consolidation)."
        )


@dataclasses.dataclass(init=False)
class LMStudioLanguageModel(OpenAILanguageModel):
    """Custom OpenAI-compatible model for LM Studio that doesn't use response_format.
    
    LM Studio rejects the response_format: json_object parameter that langextract's
    OpenAI provider sends. This subclass overrides the prompt processing to remove
    that parameter while keeping all other langextract functionality.
    """
    
    # Additional field for LM Studio
    timeout: int = 120

    def __init__(
        self,
        model_id: str = 'local-model',
        api_key: str | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        temperature: float | None = None,
        max_workers: int = 5,
        timeout: int = 120,
        context_limit: int | None = None,
        reasoning_effort: str | None = "none",
        **kwargs,
    ) -> None:
        """Initialize LM Studio language model with JSON format type."""
        from langextract.core.data import FormatType

        # Initialize parent with JSON format type
        super().__init__(
            model_id=model_id,
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            format_type=FormatType.JSON,
            temperature=temperature,
            max_workers=max_workers,
            **kwargs,
        )
        self.timeout = timeout
        self.context_limit = context_limit
        # langextract's OpenAI provider builds its client with no timeout, so
        # ours was stored and silently ignored: every extraction ran on the SDK
        # default of 600s regardless of what the config asked for. A big local
        # model on one chunk can exceed that, and it surfaces as "Request timed
        # out" rather than as the setting not being honoured.
        self._client = self._client.with_options(timeout=timeout)
        self.reasoning_effort = reasoning_effort

    @property
    def requires_fence_output(self) -> bool:
        """LM Studio doesn't use structured output, so we expect fenced JSON."""
        return True

    def _create(self, api_params: dict, reasoning_effort: str | None):
        """Send the request, dropping ``reasoning_effort`` if the server rejects it.

        This is the LM Studio counterpart of the ollama provider's ``think:false``:
        a reasoning model spends most of its output budget on a trace nobody reads,
        and extraction pays that on every chunk. But not every runtime accepts the
        parameter, and a setting that makes extraction impossible on some models is
        worse than a slow one. So try it, and fall back once — permanently for this
        model, so the cost is one failed request, not one per chunk.
        """
        if not reasoning_effort:
            return self._client.chat.completions.create(**api_params)
        try:
            return self._client.chat.completions.create(
                **api_params, reasoning_effort=reasoning_effort
            )
        except Exception as e:
            if not _rejects_parameter(e):
                raise
            print(f"  [LM Studio] server rejected reasoning_effort={reasoning_effort!r}; "
                  f"continuing without it (expect slower, longer generations)")
            self.reasoning_effort = None
            return self._client.chat.completions.create(**api_params)

    def _process_single_prompt(
        self, prompt: str, config: dict
    ) -> core_types.ScoredOutput:
        """Process a single prompt without sending response_format parameter."""
        # The one seam where the real, fully-rendered chunk prompt exists: the
        # chunk size is only part of it (instructions and few-shot examples ride
        # along), so max_char_buffer alone cannot tell us whether this fits.
        assert_prompt_fits(self.context_limit, prompt, "extraction")
        try:
            normalized_config = self._normalize_reasoning_params(config)

            # System message for JSON output (without using response_format API param)
            system_message = (
                "You are a helpful assistant that extracts information into structured JSON. "
                "Follow the provided format Exactly, matching the field names and structure of the examples. "
                "Do not include any preamble or extra explanations."
            )

            messages = [{'role': 'user', 'content': prompt}]
            messages.insert(0, {'role': 'system', 'content': system_message})

            api_params = {
                'model': self.model_id,
                'messages': messages,
                'n': 1,
            }

            temp = normalized_config.get('temperature', self.temperature)
            if temp is not None:
                api_params['temperature'] = temp

            # DO NOT add response_format - LM Studio doesn't support it properly
            # The system message and fence_output will handle JSON parsing

            if (v := normalized_config.get('max_output_tokens')) is not None:
                api_params['max_tokens'] = v
            if (v := normalized_config.get('top_p')) is not None:
                api_params['top_p'] = v
            for key in [
                'frequency_penalty',
                'presence_penalty',
                'seed',
                'stop',
                'logprobs',
                'top_logprobs',
                'reasoning',
                # Explicitly exclude 'response_format' from being passed
            ]:
                if (v := normalized_config.get(key)) is not None:
                    api_params[key] = v

            # Deliberately NOT via _normalize_reasoning_params, which rewrites a
            # flat reasoning_effort into the nested {"reasoning": {"effort": ..}}
            # of the Responses API. Measured against LM Studio on qwen3.8-27b:
            # the nested form is ignored (57 reasoning tokens), the flat one
            # works (0). Normalising here would silently undo the setting.
            response = self._create(api_params, self.reasoning_effort)
            output_text = response.choices[0].message.content

            # Sanitize control characters that break JSON parsing
            if output_text:
                import re
                output_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', output_text)

            return core_types.ScoredOutput(score=1.0, output=output_text)

        except Exception as e:
            raise exceptions.InferenceRuntimeError(
                f'LM Studio API error: {str(e)}', original=e
            ) from e


@client("lmstudio")
class LMStudioClient(BaseLLMClient):
    """Client for LM Studio local models via langextract.

    LM Studio provides an OpenAI-compatible API, so we use langextract's
    OpenAI provider with a custom base URL.
    """

    def __init__(
        self,
        model_id: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        max_workers: int | None = None,
        batch_length: int | None = None,
        max_char_buffer: int = 8000,
        show_progress: bool = True,
        timeout: int = 120,
        reasoning_effort: str | None = "none",
    ) -> None:
        """Initialize LM Studio client.

        Args:
            model_id: Model identifier (see configs/lmstudio.json for default)
            base_url: LM Studio server URL (see configs/lmstudio.json for default)
            api_key: API key (see configs/lmstudio.json for default)
            max_workers: Maximum parallel workers (see configs/lmstudio.json)
            batch_length: Number of chunks per batch (see configs/lmstudio.json)
            max_char_buffer: Maximum characters for inference
            show_progress: Whether to show progress bar
            timeout: Request timeout in seconds
        """
        _defaults = load_provider_defaults("lmstudio")
        self.model_id = model_id or _defaults["model_id"]
        self.base_url = base_url or _defaults["base_url"]
        self.api_key = api_key or _defaults["api_key"]
        self.max_workers = max_workers if max_workers is not None else _defaults["max_workers"]
        self.batch_length = batch_length if batch_length is not None else _defaults["batch_length"]
        self.max_char_buffer = max_char_buffer
        self.show_progress = show_progress
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort

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
        """Extract knowledge graph triples using LM Studio via langextract.

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
            # Use our custom LMStudioLanguageModel that removes response_format parameter
            lmstudio_model = LMStudioLanguageModel(
                model_id=self.model_id,
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                context_limit=context_limit(self.base_url, self.model_id),
                reasoning_effort=self.reasoning_effort,
            )

            # Prepare langextract kwargs
            # LM Studio has limited OpenAI API compatibility
            langextract_kwargs = {
                "model": lmstudio_model,
                "temperature": temperature,
                "max_workers": self.max_workers,
                "batch_length": self.batch_length,
                "max_char_buffer": self.max_char_buffer,
                "show_progress": self.show_progress,
                "use_schema_constraints": False,  # LM Studio doesn't support JSON schema
                "fence_output": True,  # Expect JSON in code fences
                "fetch_urls": False,
                # See the ollama provider: langextract's fuzzy aligner is
                # quadratic on large chunks and its offsets are discarded by
                # extract_triples anyway.
                "resolver_params": {"enable_fuzzy_alignment": False},
            }

            if max_tokens:
                langextract_kwargs["language_model_params"] = {"max_tokens": max_tokens}

            langextract_kwargs.update(kwargs)

            # Perform extraction
            # Note: Don't pass format_type here - the `format_type` parameter we receive is a Pydantic model
            # for structured output, not the FormatType enum that langextract expects.
            # Our LMStudioLanguageModel already has format_type=FormatType.JSON configured.
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
            # No traceback dump: the common failure here is now a context-window
            # refusal, which is an actionable config message, and burying it under
            # a langextract stack trace helps nobody. `raise ... from e` keeps the
            # chain for anyone who wants it.
            raise LLMClientError(f"LM Studio extraction failed: {e}") from e

    def augment(
        self,
        text: str,
        prompt_description: str,
        format_type: type,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Generate augmentation triples directly using LM Studio.

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
        import re
        import requests

        try:
            # Build the prompt with schema
            schema_json = json.dumps(format_type.model_json_schema(), indent=2)
            full_prompt = f"""
{prompt_description}

Return the results as a JSON array of objects matching this schema:
{schema_json}

Input Text:
{text}

IMPORTANT: Respond with ONLY a valid JSON array. No markdown code blocks, no explanation, just the JSON array starting with [ and ending with ].
"""

            # Consolidation's prompt scales with the ENTITY COUNT, not with the
            # document, so it is the one that outgrows a fixed window first.
            assert_prompt_fits(
                context_limit(self.base_url, self.model_id), full_prompt, "augmentation"
            )

            # Build request payload - don't use response_format as it's not universally supported
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "user", "content": full_prompt}
                ],
                "temperature": temperature if temperature is not None else 0.0,
            }
            if max_tokens:
                payload["max_tokens"] = max_tokens
            if self.reasoning_effort:
                # Same reason as extraction, and it matters more here: a
                # consolidation prompt is one big call, so a reasoning trace eats
                # the answer's room in the window rather than just time.
                payload["reasoning_effort"] = self.reasoning_effort

            # Call LM Studio's OpenAI-compatible API
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            result = response.json()
            response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not response_text:
                return []

            # Try to extract JSON from the response (handle markdown code blocks, etc.)
            response_text = response_text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                # Extract content between code blocks
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response_text)
                if match:
                    response_text = match.group(1).strip()
            
            # Find JSON array or object in the response
            json_match = re.search(r'(\[[\s\S]*\]|\{[\s\S]*\})', response_text)
            if json_match:
                response_text = json_match.group(1)

            # Parse the JSON response. Repair first: local models emit trailing
            # commas, control characters and responses cut off mid-object, and
            # raising on the first bad character costs the whole consolidation.
            try:
                data = json.loads(_repair_json_text(response_text))
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

                # Force inference to contextual for bridging (consistency across providers)
                for item in items:
                    if isinstance(item, dict):
                        item['inference'] = 'contextual'
                
                return items
            except json.JSONDecodeError as e:
                raise LLMClientError(f"Failed to parse JSON response: {e}\nResponse text: {response_text[:500]}")

        except requests.RequestException as e:
            raise LLMClientError(f"LM Studio request failed: {e}") from e
        except Exception as e:
            raise LLMClientError(f"LM Studio JSON generation failed: {e}") from e

    @classmethod
    def from_config(cls, config: "ClientConfig") -> "LMStudioClient":
        """Create an LMStudioClient from a ClientConfig."""
        return cls(
            model_id=config.model_id,
            base_url=config.base_url,
            api_key=config.api_key,
            max_workers=config.max_workers,
            batch_length=config.batch_length,
            max_char_buffer=config.max_char_buffer,
            show_progress=config.show_progress,
            timeout=config.timeout,
            reasoning_effort=config.reasoning_effort,
        )




__all__ = ["LMStudioClient"]
