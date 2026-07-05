---
name: local-model-extraction
description: Running kgb extraction reliably on local LLM backends (Ollama/LMStudio/MLX) — the config that prevents wedging and the defensive JSON-parsing philosophy that keeps one malformed field from aborting a whole document. Use when running the pipeline on a local model or debugging extraction failures/hangs.
---

# Local-Model Extraction

Local backends (Ollama, LMStudio, MLX) are low-parallelism and loosely
schema-compliant. Two classes of problem and the practices that solve them,
learned on gemma4:12b-mlx over 15 pathology reports (0/15 → 15/15).

## Config that prevents wedging

The "MLX wedges after N extractions" symptom was a **misdiagnosis** — the real
cause was concurrency. Fixes, in order of impact:

- **Serialize the batch: `workers: 1`.** The pipeline's `execute_batch` uses a
  thread pool; with the CLI `--workers` unset it defaults to many, flooding the
  local server with concurrent generations until it hangs. `client.workers: 1` in
  the YAML now also caps batch concurrency. This single change eliminated all
  wedges.
- **`think: false`** (gemma-style models): disables the reasoning phase, ~4×
  faster. Only the **native** Ollama endpoint honors a top-level `think` flag; the
  OpenAI-compatible endpoint ignores it. kgb routes extraction through the native
  provider for this reason.
- **`num_ctx: 8192`** (explicit): bounds KV memory, fits typical 4.5k–11k char
  prompts.
- **`timeout: 300`**: fail fast instead of burning 15 min on a hung call.

`reset_every` (proactive model unload) exists as an opt-in knob but was proven
**unnecessary** once serialized — same result, extra cold-starts. Leave it off
(`None`). `retry-with-reset` (unload + retry on timeout/5xx) stays on as a free
safety net.

Minimal known-good YAML client block:
```yaml
client:
  type: ollama
  model: <model>
  workers: 1
  think: false
  timeout: 300
  options: { num_ctx: 8192 }
```

## Defensive parsing: mirror the parser's contract, never abort the document

Local models emit *almost*-valid JSON. The rule: **repair/coerce one bad field so
langextract's strict resolver doesn't drop the entire document** — but only in
ways that mirror what the downstream parser actually requires. All of this lives
in `kgb/clients/providers/ollama.py`.

Four real failure modes and their fixes:

1. **Malformed/truncated JSON** → salvage the individually-valid `{...}` objects
   (string-aware brace scan), dropping only the broken one. Never let a trailing
   comma or a mid-object `max_tokens` cutoff lose everything.
2. **Inconsistent key casing** (`"Tail"` vs `"tail"` — gemma did this on 20/21
   objects) → canonicalize recognized field keys to lowercase before any
   case-sensitive read. A real lowercase key wins over a mis-cased duplicate.
   Without this, 20/21 triples were silently dropped.
3. **Non-scalar value where a scalar is required** (`"Triple": {…}`) → stringify
   it. langextract renders extractions as
   `{"<Class>": <text>, "<Class>_attributes": {…triple…}}` and its resolver
   requires `*_attributes` to be a **dict** and every other value to be a
   **scalar**. **Coerce only the scalar-required fields; NEVER stringify a
   `*_attributes` dict** — a blanket coercion did exactly that and broke all 15
   reports. Mirror the contract, don't guess.
4. **Provider hang** → `retry-with-reset`: on timeout/connection error/5xx, unload
   the model (`keep_alive: 0`) so the next request reloads clean, then retry.

## Debugging method that worked

- **Capture the RAW model response** (tee `_repair_json_text`'s input to a file)
  rather than reasoning about it abstractly. A hand-rolled probe that skips the
  rendered few-shot examples is NOT representative — it produces a different
  (flatter) structure than the real langextract call.
- **Run serially and watch for concurrency in the log** (count simultaneous
  "Processing" bars); many bars = the wedge cause.
- **Isolate a hypothesis with a control run** (e.g. `reset_every` on vs off) before
  committing to a fix — but make sure the control isn't contaminated by an
  unrelated change first.
```
