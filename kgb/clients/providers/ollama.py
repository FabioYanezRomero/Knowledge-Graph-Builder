import time
from typing import TYPE_CHECKING, Any
import requests
import langextract as lx
from langextract.providers.ollama import OllamaLanguageModel

from ..base import BaseLLMClient, LLMClientError
from ..json_repair import _json_loads_ok, _repair_json_text, _salvage_json_objects
from ..defaults import load_provider_defaults
from ..factory import client

if TYPE_CHECKING:
    from ..config import ClientConfig


# Local backends (Ollama/MLX) wedge after sustained generation — the server stops
# responding (read timeout) or returns 5xx. Unloading the model (keep_alive:0)
# evicts it so the next request reloads a clean copy, which clears the wedge.
_MAX_RESET_RETRIES = 2      # wedge-recovery attempts before giving up
_RESET_COOLDOWN = 1.0       # seconds to let the server settle after an unload


def _unload_ollama(real, url: str, model: str | None, timeout: int | None) -> None:
    """Free a wedged model by unloading it (keep_alive:0). Best-effort."""
    if not model:
        return
    try:
        real.post(url, json={"model": model, "keep_alive": 0}, timeout=timeout)
    except Exception:
        pass
    time.sleep(_RESET_COOLDOWN)


def _is_wedged(resp) -> bool:
    """A 5xx from Ollama means the server (not the request) failed — retryable."""
    return getattr(resp, "status_code", 200) >= 500


_MAX_NUM_CTX = 131072


def _fit_num_ctx(payload: dict) -> None:
    """Grow ``num_ctx`` so the prompt fits, instead of letting Ollama truncate it.

    Ollama truncates an over-long prompt SILENTLY — no error, no flag, no field in
    the response. It used to show up as an obvious 1-char answer; now that we
    salvage partial output it shows up as plausible-looking partial results, which
    is worse. Nothing in the response says the model only saw half its input.

    A configured num_ctx is a guess about prompt size, and the guess breaks the
    moment either input grows. Consolidation is the sharp case: its prompt scales
    with the ENTITY COUNT (~310 chars each on legal prose), not with the document,
    so a stronger model that finds 3x the entities on the SAME document silently
    outgrows a window that was ample for a weaker one — the config is identical
    and the results quietly get worse.

    So don't guess: size the window to the prompt actually being sent, here, at
    the one seam every native call passes through. Only ever grows (a caller who
    asked for more keeps it), and leaves ~15% headroom for the answer. Past the
    cap we can only warn — but then the failure is at least loud.
    """
    prompt = payload.get("prompt") or ""
    if not prompt:
        return
    needed = int(len(prompt) / 3.5 / 0.85)  # ~3.5 chars/token, 15% left to answer
    options = payload.setdefault("options", {})
    current = options.get("num_ctx") or 0
    if needed <= current:
        return

    fitted = max(current, 8192)
    while fitted < needed and fitted < _MAX_NUM_CTX:
        fitted *= 2
    options["num_ctx"] = fitted
    if fitted < needed:
        print(
            f"  [Ollama] prompt is ~{int(len(prompt) / 3.5):,} tokens and does not fit "
            f"even at num_ctx {fitted:,} — Ollama will truncate it silently. "
            f"Send less at once (fewer entities per consolidation, smaller chunks)."
        )
    elif current:
        print(f"  [Ollama] raised num_ctx {current:,} -> {fitted:,} to fit a "
              f"~{int(len(prompt) / 3.5):,}-token prompt")


class _ThinkInjectingRequests:
    """Wraps the ``requests`` module so Ollama ``/api/generate`` calls carry a
    top-level ``think`` flag (and extra options), and recover from wedges.

    Injecting ``think`` at the request-body top level is the ONLY reliable way to
    disable gemma-style reasoning on Ollama: the OpenAI-compatible endpoint
    ignores ``think`` entirely, and the native endpoint ignores it inside
    ``options``. The provider reads its HTTP client from ``self._requests``, so we
    swap in this shim — which also makes it the single seam through which every
    native extraction call flows, so wedge recovery lives here too.
    """

    def __init__(self, real, think: bool | None, options: dict | None):
        self._real = real
        self._think = think
        self._options = options

    def post(self, url, *args, **kwargs):
        payload = kwargs.get("json")
        model = None
        if isinstance(payload, dict):
            if self._think is not None:
                payload["think"] = self._think
            if self._options:
                payload.setdefault("options", {}).update(self._options)
            model = payload.get("model")
            _fit_num_ctx(payload)
        timeout = kwargs.get("timeout")

        last_exc: Exception | None = None
        for attempt in range(_MAX_RESET_RETRIES + 1):
            try:
                resp = self._real.post(url, *args, **kwargs)
            except self._real.exceptions.RequestException as e:
                last_exc = e
                if attempt < _MAX_RESET_RETRIES:
                    _unload_ollama(self._real, url, model, timeout)
                    continue
                raise
            if _is_wedged(resp) and attempt < _MAX_RESET_RETRIES:
                _unload_ollama(self._real, url, model, timeout)
                continue
            return resp
        raise last_exc  # unreachable: loop either returns or raises

    def __getattr__(self, name):  # forward .exceptions and anything else
        return getattr(self._real, name)


def _pairs_hook(pairs: list[tuple[str, object]]):
    """json.loads hook that preserves duplicate keys instead of last-wins.

    A model that emits ``{"Triple": .., "Triple_attributes": {..}, "Triple": ..}``
    is writing several extractions into one flat object. Plain ``json.loads``
    silently keeps only the last; something downstream then collapses the repeats
    into a list value and langextract drops the whole document. Keeping the pairs
    lets us split them back into one object per extraction.
    """
    keys = [k for k, _ in pairs]
    return pairs if len(keys) != len(set(keys)) else dict(pairs)


def _pairs_to_items(pairs: list[tuple[str, object]]) -> list[dict]:
    """Split flat ``Class``/``Class_attributes`` pairs into one object each."""
    items: list[dict] = []
    current: dict = {}
    for k, v in pairs:
        if k in current:  # a repeated key starts the next extraction
            items.append(current)
            current = {}
        current[k] = v
    if current:
        items.append(current)
    return items


def _drop_unusable(items: list) -> tuple[list[dict], int]:
    """Drop the individual extractions langextract's resolver refuses.

    Enumerating malformed *containers* is whack-a-mole — every model breaks
    differently (gemma4:12b emits non-mapping items, 26b emits empty extraction
    text, e4b duplicated keys). So mirror the resolver's per-item contract
    instead: an item it would raise on is dropped, and the document survives with
    its other extractions rather than being lost whole.

    The contract: each item is a mapping, and its non-attributes value (the
    extraction text) is a non-empty scalar.
    """
    kept: list[dict] = []
    dropped = 0
    for obj in items:
        if not isinstance(obj, dict):
            dropped += 1  # "Each item in the sequence must be a mapping"
            continue
        texts = [v for k, v in obj.items() if not k.endswith("_attributes")]
        if not texts or not any(str(v).strip() for v in texts if v is not None):
            dropped += 1  # "Source tokens and extraction tokens cannot be empty"
            continue
        kept.append(obj)
    return kept, dropped


def _recover_escaped_payload(parsed: dict) -> list | None:
    """Recover extractions the model escaped into a JSON string.

    gemma4:26b double-encodes a whole response: several perfectly good
    extractions end up inside ONE string used as an object key, with no value.
    langextract then sees an extraction whose text is empty and raises, losing
    the document — while the payload sitting inside that string was fine and
    just needed unwrapping.
    """
    import json

    candidates = list(parsed.keys()) + [v for v in parsed.values() if isinstance(v, str)]
    for candidate in candidates:
        if not isinstance(candidate, str) or "_attributes" not in candidate:
            continue
        # The fragment usually starts *inside* an object (its opening brace was
        # the wrapper's), so try it both ways and salvage whatever parses.
        for wrapped in ("{" + candidate, candidate):
            salvaged = _salvage_json_objects(wrapped)
            if not salvaged:
                continue
            try:
                items = json.loads(salvaged)
            except Exception:
                continue
            if isinstance(items, list) and items:
                return items
    return None


def _extraction_items(parsed) -> tuple[list | None, bool]:
    """Normalize whatever the model emitted to a list of extraction objects.

    Returns (items, rewrapped) — ``rewrapped`` is True when the shape itself had
    to be repaired, so the caller knows to re-emit rather than pass the original
    text through. Returns (None, False) for shapes we don't recognise, which are
    left untouched for langextract to reject as before.

    Shapes observed from local models on a single document:
      1. ``[{...}, {...}]``                      — bare list (already fine)
      2. ``{"extractions": [...]}``              — the canonical wrapper
      3. ``{"triples": [...]}``                  — the model's own wrapper name
      4. ``{"Triple": .., "Triple_attributes": {..}}``          — one flat object
      5. the same with the pair repeated N times — N extractions in one object
    """
    # 5: duplicate keys survived as pairs (see _pairs_hook)
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], tuple):
        return _pairs_to_items(parsed), True
    # 1
    if isinstance(parsed, list):
        return parsed, False
    if not isinstance(parsed, dict):
        return None, False
    # 2
    if isinstance(parsed.get("extractions"), list):
        return parsed["extractions"], False
    # 4: a lone extraction, recognisable by its own attributes sibling
    if any(f"{k}_attributes" in parsed for k in parsed):
        return [parsed], True
    # 3: some other wrapper name around the list
    lists = [v for v in parsed.values() if isinstance(v, list)]
    if len(lists) == 1:
        return lists[0], True
    # 6: the whole answer escaped into a string
    recovered = _recover_escaped_payload(parsed)
    if recovered:
        return recovered, True
    return None, False


def _coerce_scalar_values(text: str) -> str:
    """Mirror langextract's resolver contract so one bad response can't abort a
    whole document. langextract renders each extraction as
    ``{"<Class>": <extraction_text>, "<Class>_attributes": {..triple..}}``, expects
    a LIST of those under an ``extractions`` key, and its resolver REQUIRES the
    ``*_attributes`` value to be a dict and every other value to be a scalar
    (str/int/float) — raising, and dropping the entire document, on a violation.

    Local models violate this two ways: they wrap the list under their own key (or
    omit the list entirely), and they nest the extraction_text value. We repair the
    shape and stringify ONLY the scalar-required fields, never touching the
    attributes dicts (blindly stringifying those is what broke all 15 reports once).
    """
    import json
    import re

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    body = m.group(1) if m else text
    try:
        parsed = json.loads(body, object_pairs_hook=_pairs_hook)
    except Exception:
        return text

    items, rewrapped = _extraction_items(parsed)
    if items is None:
        # A shape we can neither read nor repair. Passing it through means
        # langextract raises and the WHOLE document is lost; yielding nothing
        # loses only this chunk. Loud, because it is still lost data.
        print(f"  [Ollama] unrecognized response shape, dropping this chunk: {body[:120]!r}")
        empty = '{"extractions": []}'
        return f"```json\n{empty}\n```" if m else empty

    changed = rewrapped
    for obj in items:
        if not isinstance(obj, dict):
            continue
        for k, v in list(obj.items()):
            if k.endswith("_attributes"):
                continue  # resolver requires a dict here — never touch
            if not isinstance(v, (str, int, float, type(None))):
                obj[k] = json.dumps(v, ensure_ascii=False)
                changed = True
    items, dropped = _drop_unusable(items)
    if dropped:
        # Never silent: a dropped extraction is real data loss, just a much
        # smaller loss than the whole document.
        print(f"  [Ollama] dropped {dropped} unusable extraction(s) from this response")
        changed = True
        rewrapped = True  # the container must be rebuilt around the survivors

    if not changed:
        return text

    # Only re-shape what was actually broken: a payload that merely needed a
    # value stringified keeps its original container.
    dumped = json.dumps({"extractions": items} if rewrapped else parsed, ensure_ascii=False)
    return f"```json\n{dumped}\n```" if m else dumped


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
        # Always install the shim: besides injecting think/options, it carries the
        # wedge-recovery retry that keeps sustained extraction from hanging.
        self._requests = _ThinkInjectingRequests(requests, think, dict(model_options or {}))

    def _ollama_query(self, *args, **kwargs):
        resp = super()._ollama_query(*args, **kwargs)
        if isinstance(resp, dict) and isinstance(resp.get("response"), str):
            resp = dict(resp)
            resp["response"] = _coerce_scalar_values(_repair_json_text(resp["response"]))
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
        reset_every: int | None = None,
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
            reset_every: Proactively unload the model every N extractions. For
                low-parallelism local backends that wedge under sustained load;
                None disables it (retry-with-reset still recovers on failure).
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
        self.reset_every = reset_every
        self._extract_calls = 0  # ponytail: racy under workers>1, but this is a best-effort heuristic reset, not correctness

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
                    # langextract's fuzzy aligner is difflib.SequenceMatcher with
                    # autojunk=False, which degrades to hours on chunks past ~16k
                    # chars. We pay nothing to lose it: extract_triples discards
                    # langextract's (prompt-relative) offsets and re-anchors each
                    # span to the document by exact find, leaving None when it
                    # doesn't match — an honest gap beats a fuzzy guess.
                    "enable_fuzzy_alignment": False,
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
                        # Ensure it's a dict. Canonicalize key casing first: local
                        # models emit e.g. "Tail" instead of "tail" inconsistently,
                        # which would fail the case-sensitive guard below and drop
                        # the triple. Deferred import avoids a clients<->builder cycle.
                        from ...builder.validation import canonicalize_triple_keys
                        triple = canonicalize_triple_keys(dict(attrs))

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

            self._maybe_reset()
            return triples

        except Exception as e:
            raise LLMClientError(f"Ollama extraction failed: {e}") from e

    def _maybe_reset(self) -> None:
        """Unload the model every ``reset_every`` extractions so a low-parallelism
        local backend doesn't wedge under sustained load."""
        self._extract_calls += 1
        if self.reset_every and self._extract_calls % self.reset_every == 0:
            _unload_ollama(requests, f"{self.base_url}/api/generate", self.model_id, self.timeout)

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
            # augment() builds its own payload, so it misses the extraction shim.
            # This is the path consolidation runs on — the one that was silently
            # truncated — so it needs the fit at least as much.
            _fit_num_ctx(payload)
            url = f"{self.base_url}/api/generate"
            # Same wedge-recovery as extraction: unload + retry on timeout/5xx.
            last_exc: Exception | None = None
            response = None
            for attempt in range(_MAX_RESET_RETRIES + 1):
                try:
                    response = requests.post(url, json=payload, timeout=self.timeout)
                except requests.exceptions.RequestException as e:
                    last_exc = e
                    if attempt < _MAX_RESET_RETRIES:
                        _unload_ollama(requests, url, self.model_id, self.timeout)
                        continue
                    raise
                if _is_wedged(response) and attempt < _MAX_RESET_RETRIES:
                    _unload_ollama(requests, url, self.model_id, self.timeout)
                    continue
                break
            if response is None:  # unreachable: loop returns or raises
                raise last_exc
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
                # Same hardening the extraction path gets: a response truncated
                # mid-array (the usual symptom of a prompt that nearly fills
                # num_ctx) salvages to its complete objects instead of raising
                # and killing the run.
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
            reset_every=config.reset_every,
        )




__all__ = ["OllamaClient"]
