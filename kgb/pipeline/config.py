"""YAML-based pipeline configuration loader.

Reads a declarative YAML file and assembles the same PipelineRunner + contexts
that the CLI ``run-pipeline`` command builds imperatively from boolean flags.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ..clients import ClientConfig, ClientFactory
from ..domains import get_domain, ExtractionMode
from ..io.readers import load_records
from .context import PipelineContext
from .runner import PipelineRunner
from .step import get_step


# Directory containing built-in example pipeline configs shipped with the package.
_CONFIGS_DIR = Path(__file__).parent / "configs"

# Maps step names to the output subdirectory they use (relative to output_dir).
_STEP_OUTPUT_SUBDIRS: dict[str, str] = {
    "export-json": "extracted_json",
    "convert": "graphml",
    "visualize-network": "visualizations",
    "visualize-extraction": "visualizations_extraction",
}

# Steps that require the shared LLM client and domain objects.
_STEPS_NEEDING_CLIENT = {"extract", "augment", "consolidate"}

# Steps that transform the triple set — snapshotted after each when trace is on.
_TRIPLE_TRANSFORMING_STEPS = {"extract", "augment", "consolidate"}


def load_pipeline_config(path: str | Path) -> dict[str, Any]:
    """Read and return the raw YAML pipeline configuration.

    Args:
        path: Path to a ``.yaml`` pipeline config file.

    Returns:
        Parsed YAML as a plain dict.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If required top-level keys are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Pipeline config must be a YAML mapping, got {type(raw).__name__}")

    for key in ("steps",):
        if key not in raw:
            raise ValueError(f"Pipeline config is missing required key: '{key}'")

    return raw


def build_pipeline_from_config(
    raw_config: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
) -> tuple[PipelineRunner, list[PipelineContext]]:
    """Assemble a PipelineRunner and input contexts from a parsed YAML config.

    CLI overrides (if provided) take precedence over values in the YAML file.
    Supported override keys mirror the YAML top-level keys:
    ``input_file``, ``output_dir``, ``domain``, ``mode``, ``client``, ``model``,
    ``api_key``, ``base_url``, ``temperature``, ``timeout``, ``workers``,
    ``text_field``, ``id_field``, ``limit``, ``no_progress``.

    Args:
        raw_config: The dict returned by :func:`load_pipeline_config`.
        cli_overrides: Optional dict of CLI-provided values that override the
            YAML configuration.

    Returns:
        A ``(runner, contexts)`` tuple ready for ``runner.execute_batch(contexts)``.
    """
    overrides = cli_overrides or {}

    def _resolve(yaml_section: dict | None, yaml_key: str, override_key: str, default: Any = None) -> Any:
        """Pick override > yaml > default."""
        if override_key in overrides and overrides[override_key] is not None:
            return overrides[override_key]
        if yaml_section and yaml_key in yaml_section:
            return yaml_section[yaml_key]
        return default

    # -- Client configuration --------------------------------------------------
    client_section = raw_config.get("client", {}) or {}

    client_type = _resolve(client_section, "type", "client", "gemini")
    model_id = _resolve(client_section, "model", "model")
    api_key = _resolve(client_section, "api_key", "api_key")
    base_url = _resolve(client_section, "base_url", "base_url")
    temperature = float(_resolve(client_section, "temperature", "temperature", 0.0))
    timeout = int(_resolve(client_section, "timeout", "timeout", 120))
    max_workers = _resolve(client_section, "workers", "workers")
    if max_workers is not None:
        max_workers = int(max_workers)
    no_progress = _resolve(None, "", "no_progress", False)

    default_client_vals: dict[str, Any] = {
        "type": client_type,
        "model": model_id,
        "api_key": api_key,
        "base_url": base_url,
        "temperature": temperature,
        "timeout": timeout,
        "workers": max_workers,
    }

    # Identical client configs (default or per-step) share one instance.
    client_cache: dict[tuple, Any] = {}

    def _client_for(vals: dict[str, Any]) -> Any:
        key = tuple(sorted((k, str(v)) for k, v in vals.items()))
        if key not in client_cache:
            ck: dict[str, Any] = {
                "client_type": vals["type"],
                "temperature": float(vals["temperature"]),
                "show_progress": not no_progress,
                "timeout": int(vals["timeout"]),
            }
            if vals.get("model"):
                ck["model_id"] = vals["model"]
            if vals.get("api_key"):
                ck["api_key"] = vals["api_key"]
            if vals.get("base_url"):
                ck["base_url"] = vals["base_url"]
            if vals.get("workers"):
                ck["max_workers"] = int(vals["workers"])
            client_cache[key] = ClientFactory.create(ClientConfig(**ck))
        return client_cache[key]

    def _merged_client_vals(override: dict[str, Any]) -> dict[str, Any]:
        merged = dict(default_client_vals)
        if override.get("type") and override["type"] != merged["type"]:
            # Switching provider: don't drag provider-specific fields along.
            for k in ("model", "api_key", "base_url"):
                merged[k] = None
        merged.update(override)
        return merged

    llm_client = _client_for(default_client_vals)

    # -- Domain ----------------------------------------------------------------
    domain_name = _resolve(raw_config, "domain", "domain", "default")
    mode_str = _resolve(raw_config, "mode", "mode", "open")
    mode = ExtractionMode(mode_str) if isinstance(mode_str, str) else mode_str
    domain_obj = get_domain(domain_name, extraction_mode=mode)

    # -- Input records ---------------------------------------------------------
    input_section = raw_config.get("input", {}) or {}

    input_file = _resolve(input_section, "file", "input_file")
    if input_file is None:
        raise ValueError("No input file specified (set input.file in YAML or pass --input)")
    input_file = Path(input_file)

    text_field = _resolve(input_section, "text_field", "text_field", "text")
    id_field = _resolve(input_section, "id_field", "id_field", "id")
    limit = _resolve(input_section, "limit", "limit")
    if limit is not None:
        limit = int(limit)
    record_ids = _resolve(input_section, "record_ids", "record_ids")

    records = load_records(input_file, text_field, id_field, record_ids, limit)
    contexts = [PipelineContext(record_id=str(r["id"]), text=str(r["text"])) for r in records]

    # -- Output directory ------------------------------------------------------
    output_dir = Path(_resolve(raw_config, "output_dir", "output_dir", "outputs/pipeline_run"))

    # -- Assemble steps --------------------------------------------------------
    steps_list: list = raw_config["steps"]
    steps_sequence = []

    # When trace is on, snapshot the triples after each transforming stage into
    # output_dir/trace/<NN>_<stage>/ so we can see where entities are lost or
    # wrongly merged, instead of only inspecting the final output.
    trace = bool(_resolve(raw_config, "trace", "trace", False))
    trace_counter = 0

    for entry in steps_list:
        if isinstance(entry, str):
            step_name = entry
            step_params: dict[str, Any] = {}
        elif isinstance(entry, dict):
            if len(entry) != 1:
                raise ValueError(f"Step mapping must have exactly one key, got: {list(entry.keys())}")
            step_name = next(iter(entry))
            step_params = entry[step_name] or {}
        else:
            raise ValueError(f"Invalid step entry (expected str or mapping): {entry!r}")

        step_cls = get_step(step_name)
        kwargs: dict[str, Any] = dict(step_params)

        # Inject shared objects based on step type. A step may carry its own
        # `client:` mapping that overrides fields of the default client
        # (e.g. a cheap model for extract, a strong one for augment).
        if step_name in _STEPS_NEEDING_CLIENT:
            step_client_section = kwargs.pop("client", None)
            step_temperature = temperature
            if isinstance(step_client_section, dict):
                step_vals = _merged_client_vals(step_client_section)
                step_temperature = float(step_vals["temperature"])
                kwargs["client"] = _client_for(step_vals)
            else:
                kwargs["client"] = llm_client
            kwargs.setdefault("domain", domain_obj)
            kwargs.setdefault("temperature", step_temperature)

        if step_name in _STEP_OUTPUT_SUBDIRS:
            subdir = _STEP_OUTPUT_SUBDIRS[step_name]
            kwargs.setdefault("output_dir", output_dir / subdir)

        steps_sequence.append(step_cls(**kwargs))

        if trace and step_name in _TRIPLE_TRANSFORMING_STEPS:
            trace_counter += 1
            snapshot_dir = output_dir / "trace" / f"{trace_counter:02d}_{step_name}"
            steps_sequence.append(get_step("checkpoint")(output_dir=snapshot_dir))

    if not steps_sequence:
        raise ValueError("Pipeline config defines no steps")

    runner = PipelineRunner(steps=steps_sequence)
    return runner, contexts


def list_pipeline_configs() -> list[tuple[str, str]]:
    """Scan the built-in configs directory and return ``(filename, description)`` pairs.

    The description is derived from the ``name`` and ``description`` keys in the
    YAML file.  Falls back to just the filename if the file cannot be parsed.

    Returns:
        Sorted list of ``(filename, description)`` tuples.
    """
    results: list[tuple[str, str]] = []
    if not _CONFIGS_DIR.is_dir():
        return results

    for yaml_file in sorted(_CONFIGS_DIR.glob("*.yaml")):
        try:
            with open(yaml_file, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f)
            name = raw.get("name", yaml_file.stem)
            desc = raw.get("description", "")
            label = f"{name}: {desc}" if desc else name
        except Exception:
            label = yaml_file.stem
        results.append((yaml_file.name, label))

    return results


__all__ = ["load_pipeline_config", "build_pipeline_from_config", "list_pipeline_configs"]
