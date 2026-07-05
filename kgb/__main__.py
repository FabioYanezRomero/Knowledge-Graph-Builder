"""Unified command-line interface for knowledge graph generation using Typer.

This CLI supports multiple LLM backends: Gemini API, Ollama, and LM Studio.
Supports JSONL (recommended), JSON, and CSV input formats.

Commands:
    extract              - Step 1: Extract triples from text
    augment connectivity - Step 2: Reduce disconnected graph components
    convert              - Convert JSON triples to GraphML
    visualize network    - Show interactive network graph (Plotly)
    visualize extraction - Show entity highlights in source text (langextract)
    list domains         - List available knowledge domains
    list clients         - List available LLM client types

Interactive mode:
    Run `kgb` with no arguments to enter the interactive shell.
"""

from __future__ import annotations

# Suppress absl warnings (e.g., langextract prompt alignment warnings)
import os
os.environ["ABSL_LOGGING_LEVEL"] = "ERROR"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import atexit
import json
try:
    import readline
except ImportError:
    readline = None  # readline unavailable on Windows
import shlex
import sys
from pathlib import Path
from typing import Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import typer
from rich.console import Console
from rich.table import Table

from .clients import ClientConfig, ClientFactory
from .io.readers import load_records
from .io.writers import convert_json_directory
from .visualization import batch_render_graphs, TextVisualizer
from .domains import list_available_domains, ExtractionMode
from .pipeline import (
    PipelineRunner, PipelineContext, get_step,
    load_pipeline_config, build_pipeline_from_config, list_pipeline_configs,
)

__version__ = "0.1.0"

# Initialize Typer apps
app = typer.Typer(
    help="Knowledge graph generation framework.",
)

augment_app = typer.Typer(
    help="Step 2: Augment the knowledge graph.",
    no_args_is_help=True
)
app.add_typer(augment_app, name="augment")

visualize_app = typer.Typer(
    help="Create interactive HTML visualizations.",
    no_args_is_help=True
)
app.add_typer(visualize_app, name="visualize")

list_app = typer.Typer(
    help="List available resources.",
    no_args_is_help=True
)
app.add_typer(list_app, name="list")

domain_app = typer.Typer(
    help="Domain utilities.",
    no_args_is_help=True
)
app.add_typer(domain_app, name="domain")

consolidate_app = typer.Typer(
    help="Step 2b: Consolidate the knowledge graph (merge/clean existing triples, adds no knowledge).",
    no_args_is_help=True
)
app.add_typer(consolidate_app, name="consolidate")

console = Console()

def _available_client_types() -> list[str]:
    """Return the registered client types."""
    return sorted(ClientFactory.get_available_clients())


def _validate_client_type(value: str) -> str:
    """Validate CLI client selection against the factory registry."""
    if ClientFactory.is_registered(value):
        return value
    available = ", ".join(_available_client_types()) or "none"
    raise typer.BadParameter(
        f"Unsupported client type '{value}'. Available types: {available}"
    )


def _build_client_config(
    client: str,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    temperature: float,
    no_progress: bool,
    max_workers: Optional[int],
    timeout: int
) -> ClientConfig:
    """Build ClientConfig from CLI options."""
    config_kwargs = {
        "client_type": client,
        "temperature": temperature,
        "show_progress": not no_progress,
        "timeout": timeout,
    }
    if model:
        config_kwargs["model_id"] = model
    if api_key:
        config_kwargs["api_key"] = api_key
    if base_url:
        config_kwargs["base_url"] = base_url
    if max_workers:
        config_kwargs["max_workers"] = max_workers
    return ClientConfig(**config_kwargs)


# =============================================================================
# LIST Commands
# =============================================================================

@list_app.command("domains")
def list_domains():
    """List available knowledge domains."""
    domains = list_available_domains()
    table = Table(title="Available Knowledge Domains")
    table.add_column("Domain Name", style="cyan")
    for d in domains:
        table.add_row(d)
    console.print(table)


@list_app.command("clients")
def list_clients():
    """List available LLM client types."""
    clients = ClientFactory.get_available_clients()
    table = Table(title="Available LLM Clients")
    table.add_column("Client Type", style="green")
    for c in clients:
        table.add_row(c)
    console.print(table)


@list_app.command("pipelines")
def list_pipelines():
    """List built-in YAML pipeline configurations."""
    configs = list_pipeline_configs()
    if not configs:
        console.print("[yellow]No built-in pipeline configs found.[/yellow]")
        return
    table = Table(title="Built-in Pipeline Configs")
    table.add_column("File", style="cyan")
    table.add_column("Description", style="white")
    for filename, description in configs:
        table.add_row(filename, description)
    console.print(table)
    console.print("\n[dim]Usage: kgb run-pipeline --config kgb/pipeline/configs/<file>[/dim]")


# =============================================================================
# DOMAIN Commands
# =============================================================================

@domain_app.command("lint")
def domain_lint(
    name: str = typer.Argument(..., help="Domain name (see 'kgb list domains') or path to a domain directory"),
):
    """Validate a domain's prompts, examples, and schema before using it."""
    from .domains.lint import lint_domain

    errors, warnings = lint_domain(name)
    for w in warnings:
        console.print(f"[yellow]warning[/yellow] {w}")
    for e in errors:
        console.print(f"[red]error[/red] {e}")
    if errors:
        console.print(f"\n[red]✗ {len(errors)} error(s), {len(warnings)} warning(s)[/red]")
        raise typer.Exit(1)
    console.print(f"\n[green]✓ Domain '{name}' is valid[/green] ({len(warnings)} warning(s))")


# =============================================================================
# PIPELINE Command
# =============================================================================

@app.command("run-pipeline")
def run_pipeline(
    config_file: Optional[Path] = typer.Option(None, "--config", "-f", help="Path to a YAML pipeline config file", exists=True),
    input_file: Optional[Path] = typer.Option(None, "--input", "-i", "--input-file", help="Path to input file (.jsonl, .json, .csv, .txt) or a directory of .txt files", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory to save pipeline artifacts"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Knowledge domain"),
    mode: Optional[ExtractionMode] = typer.Option(None, "--mode", "-m", help="Extraction mode"),
    client: Optional[str] = typer.Option(None, "--client", "-c", help="LLM client type"),
    model: Optional[str] = typer.Option(None, "--model", help="Model ID"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL"),
    text_field: Optional[str] = typer.Option(None, "--text-field", help="Field name containing text"),
    id_field: Optional[str] = typer.Option(None, "--id-field", help="Field name containing record IDs"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of records"),
    temperature: Optional[float] = typer.Option(None, "--temp", help="Sampling temperature"),
    # Logical Pipeline Steps Toggles (flag-based mode only)
    do_extract: bool = typer.Option(False, "--extract", help="Run extraction step"),
    do_augment: bool = typer.Option(False, "--augment", help="Run connectivity augmentation step"),
    do_convert: bool = typer.Option(False, "--convert", help="Convert resulting triples to GraphML"),
    do_visualize: bool = typer.Option(False, "--visualize", help="Create visual HTMLs from triples"),
    # General arguments
    no_progress: bool = typer.Option(False, "--no-progress", help="Hide progress bar"),
    max_workers: Optional[int] = typer.Option(None, "--workers", help="Max parallel workers"),
    timeout: Optional[int] = typer.Option(None, "--timeout", help="Request timeout (seconds)"),
):
    """Run a dynamically composed pipeline from a YAML config or logical flags.

    \b
    Examples:
        kgb run-pipeline --config pipeline.yaml
        kgb run-pipeline --config pipeline.yaml --input other.jsonl
        kgb run-pipeline --input data.jsonl --domain legal --extract --client ollama
        kgb run-pipeline --input data.jsonl --domain legal --extract --augment --convert --visualize --client ollama
    """
    # ----- YAML config-driven mode -------------------------------------------
    if config_file is not None:
        console.print(f"[bold blue]Pipeline Orchestrator Launching (config: {config_file.name})[/bold blue]")
        try:
            raw_config = load_pipeline_config(config_file)

            # Build overrides dict from any CLI flags that were explicitly set.
            cli_overrides: dict[str, Any] = {}
            if input_file is not None:
                cli_overrides["input_file"] = input_file
            if output_dir is not None:
                cli_overrides["output_dir"] = output_dir
            if domain is not None:
                cli_overrides["domain"] = domain
            if mode is not None:
                cli_overrides["mode"] = mode
            if client is not None:
                cli_overrides["client"] = client
            if model is not None:
                cli_overrides["model"] = model
            if api_key is not None:
                cli_overrides["api_key"] = api_key
            if base_url is not None:
                cli_overrides["base_url"] = base_url
            if temperature is not None:
                cli_overrides["temperature"] = temperature
            if timeout is not None:
                cli_overrides["timeout"] = timeout
            if max_workers is not None:
                cli_overrides["workers"] = max_workers
            if text_field is not None:
                cli_overrides["text_field"] = text_field
            if id_field is not None:
                cli_overrides["id_field"] = id_field
            if limit is not None:
                cli_overrides["limit"] = limit
            if no_progress:
                cli_overrides["no_progress"] = True

            runner, contexts = build_pipeline_from_config(raw_config, cli_overrides)

            config_name = raw_config.get("name", config_file.stem)
            resolved_output = cli_overrides.get("output_dir", raw_config.get("output_dir", "outputs/pipeline_run"))
            console.print(f"Pipeline [green]{config_name}[/green] | {len(contexts)} records | {len(runner.steps)} steps")

            # The YAML's client.workers governs the LLM client; without an explicit
            # --workers it should also cap pipeline-level (cross-document) concurrency,
            # else local low-parallelism backends (MLX) get flooded and wedge.
            batch_workers = max_workers
            if batch_workers is None:
                batch_workers = (raw_config.get("client", {}) or {}).get("workers")
            results = runner.execute_batch(contexts, max_workers=batch_workers, show_progress=not no_progress)

            successes = sum(1 for c in results if not c.errors)
            errors = sum(1 for c in results if c.errors)

            console.print(f"\n[bold green]Pipeline execution complete.[/bold green]")
            console.print(f"Success: {successes} | Errors: {errors}")
            console.print(f"Artifacts located at: {resolved_output}")

            if errors > 0:
                console.print("\n[bold red]Errors detail:[/bold red]")
                for c in results:
                    if c.errors:
                        console.print(f"  [{c.record_id}]: {c.errors[0]}")

        except Exception as e:
            console.print(f"[bold red]Pipeline Error:[/bold red] {e}")
            raise typer.Exit(code=1)
        return

    # ----- Flag-based mode (original behaviour) ------------------------------
    if not any([do_extract, do_augment, do_convert, do_visualize]):
        console.print("[yellow]Warning: No pipeline steps selected. Use --config <file> or --extract, --augment, --convert, --visualize.[/yellow]")
        return

    # Require mandatory flags in flag-based mode.
    if input_file is None:
        console.print("[red]Error: --input is required in flag-based mode.[/red]")
        raise typer.Exit(code=1)
    if domain is None:
        console.print("[red]Error: --domain is required in flag-based mode.[/red]")
        raise typer.Exit(code=1)

    # Apply defaults for optional flags.
    _output_dir = Path(output_dir) if output_dir is not None else Path("outputs/pipeline_run")
    _client = client or "gemini"
    _validate_client_type(_client)
    _temperature = temperature if temperature is not None else 0.0
    _timeout = timeout if timeout is not None else 120
    _text_field = text_field or "text"
    _id_field = id_field or "id"

    console.print(f"[bold blue]Pipeline Orchestrator Launching[/bold blue]")

    try:
        # Load records and initialize contexts
        records = load_records(input_file, _text_field, _id_field, None, limit)
        contexts = [PipelineContext(record_id=str(r["id"]), text=str(r["text"])) for r in records]
        console.print(f"Loaded {len(contexts)} contexts from {input_file.name}")

        # Setup cross-cutting utilities
        from .domains import get_domain as _get_domain
        _mode = mode if mode is not None else ExtractionMode.OPEN
        domain_obj = _get_domain(domain, extraction_mode=_mode)
        config = _build_client_config(_client, model, api_key, base_url, _temperature, no_progress, max_workers, _timeout)
        llm_client = ClientFactory.create(config)

        # Assemble Pipeline Steps Sequence
        steps_sequence = []

        if do_extract:
            steps_sequence.append(get_step("extract")(client=llm_client, domain=domain_obj, temperature=_temperature))

        if do_augment:
            steps_sequence.append(get_step("augment")(client=llm_client, domain=domain_obj, temperature=_temperature))

        if do_extract or do_augment:
            json_dir = _output_dir / "extracted_json"
            steps_sequence.append(get_step("export-json")(output_dir=json_dir))

        if do_convert:
            graphml_dir = _output_dir / "graphml"
            steps_sequence.append(get_step("convert")(output_dir=graphml_dir))

        if do_visualize:
            network_dir = _output_dir / "visualizations"
            extraction_viz_dir = _output_dir / "visualizations_extraction"
            steps_sequence.append(get_step("visualize-network")(output_dir=network_dir))
            steps_sequence.append(get_step("visualize-extraction")(output_dir=extraction_viz_dir))

        # Execute Pipeline Sequence
        console.print(f"[bold green]Assembled {len(steps_sequence)} pipeline steps.[/bold green]")
        runner = PipelineRunner(steps=steps_sequence)

        results = runner.execute_batch(contexts, max_workers=max_workers, show_progress=not no_progress)

        # Determine Success Criteria
        successes = sum(1 for c in results if not c.errors)
        errors = sum(1 for c in results if c.errors)

        console.print(f"\n[bold green]Pipeline execution complete.[/bold green]")
        console.print(f"Success: {successes} | Errors: {errors}")
        console.print(f"Artifacts located at: {_output_dir}")

        if errors > 0:
            console.print("\n[bold red]Errors detail:[/bold red]")
            for c in results:
                if c.errors:
                    console.print(f"  [{c.record_id}]: {c.errors[0]}")

    except Exception as e:
        console.print(f"[bold red]Pipeline Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# =============================================================================
# EXTRACT Command (Step 1)
# =============================================================================


@app.command()
def extract(
    input_file: Path = typer.Option(..., "--input", "-i", "--input-file", help="Path to input file (.jsonl, .json, .csv, .txt) or a directory of .txt files", exists=True),
    output_dir: Path = typer.Option("outputs/kg_extraction", "--output-dir", "-o", help="Directory to save outputs"),
    domain: str = typer.Option(..., "--domain", "-d", help="Knowledge domain [required] (use 'list domains' to see all)"),
    mode: ExtractionMode = typer.Option(ExtractionMode.OPEN, "--mode", "-m", help="Extraction mode"),
    client: str = typer.Option(
        "gemini",
        "--client",
        "-c",
        help="LLM client type",
        callback=_validate_client_type,
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model ID"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for Gemini"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL for Ollama/LM Studio"),
    text_field: str = typer.Option("text", "--text-field", help="Field name containing text"),
    id_field: str = typer.Option("id", "--id-field", help="Field name containing record IDs"),
    record_ids: Optional[list[str]] = typer.Option(None, "--record-ids", help="List of record IDs to process"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of records"),
    temperature: float = typer.Option(0.0, "--temp", help="Sampling temperature"),
    prompt_override: Optional[Path] = typer.Option(None, "--prompt", help="Override extraction prompt", exists=True),
    no_progress: bool = typer.Option(False, "--no-progress", help="Hide progress bar"),
    max_workers: Optional[int] = typer.Option(None, "--workers", help="Max parallel workers"),
    timeout: int = typer.Option(120, "--timeout", help="Request timeout (seconds)"),
):
    """Step 1: Extract knowledge graph triples from text.
    
    \b
    Examples:
        kgb extract --input data.jsonl --domain legal
    """
    console.print(f"[bold blue]Step 1: Extraction[/bold blue]")
    console.print(f"Input: [dim]{input_file}[/dim] | Domain: [green]{domain}[/green]")
    
    try:
        # Load records
        records = load_records(input_file, text_field, id_field, record_ids, limit)
        console.print(f"Loaded {len(records)} records")
        
        # Setup domain and client
        from .domains import get_domain
        domain_obj = get_domain(domain, extraction_mode=mode)
        config = _build_client_config(client, model, api_key, base_url, temperature, no_progress, max_workers, timeout)
        llm_client = ClientFactory.create(config)
        
        # Process
        json_dir = output_dir / "extracted_json"
        json_dir.mkdir(parents=True, exist_ok=True)
        
        from .builder import extract_triples
        
        output_files = {}
        for record in records:
            record_id = str(record["id"])
            text = str(record["text"])
            output_path = json_dir / f"{record_id}.json"
            
            console.print(f"Processing {record_id} (extract only)...")
            triples, entities = extract_triples(
                client=llm_client,
                domain=domain_obj,
                text=text,
                record_id=record_id,
                temperature=temperature,
                prompt_override=prompt_override.read_text() if prompt_override else None
            )
            # Standalone entities are preserved as empty-relation records so the
            # downstream converter can add them as isolated nodes.
            records = [t.model_dump() for t in triples]
            records += [{"head": e, "relation": "", "tail": ""} for e in entities]
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            output_files[record_id] = output_path
            console.print(f"  → {len(triples)} triples, {len(entities)} standalone entities saved")
        
        console.print(f"\n[bold green]✓ Extraction complete.[/bold green]")
        console.print(f"Output: {json_dir} ({len(output_files)} files)")
        console.print(f"\n[dim]Next: kgb augment connectivity --input {input_file} --domain {domain}[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# =============================================================================
# AUGMENT / CONSOLIDATE Subcommands (Step 2)
# =============================================================================

def _run_graph_strategy(
    strategy: str,
    input_file: Path,
    output_dir: Path,
    domain: str,
    mode: ExtractionMode,
    client: str,
    model: Optional[str],
    api_key: Optional[str],
    base_url: Optional[str],
    text_field: str,
    id_field: str,
    record_ids: Optional[list[str]],
    limit: Optional[int],
    temperature: float,
    no_progress: bool,
    max_workers: Optional[int],
    timeout: int,
    **strategy_kwargs,
) -> Path:
    """Shared runner for augment/consolidate strategies over extracted JSON."""
    records = load_records(input_file, text_field, id_field, record_ids, limit)

    from .domains import get_domain
    domain_obj = get_domain(domain, extraction_mode=mode)
    config = _build_client_config(client, model, api_key, base_url, temperature, no_progress, max_workers, timeout)
    llm_client = ClientFactory.create(config)

    json_dir = output_dir / "extracted_json"
    json_dir.mkdir(parents=True, exist_ok=True)

    from .builder import augment_triples

    count = 0
    for record in records:
        record_id = str(record["id"])
        text = str(record["text"])
        output_path = json_dir / f"{record_id}.json"

        existing_triples = None
        if output_path.exists():
            console.print(f"[dim]Loading existing triples for {record_id}[/dim]")
            with open(output_path, "r", encoding="utf-8") as f:
                existing_triples = json.load(f)

        console.print(f"Processing {record_id} ({strategy})...")
        triples, metadata = augment_triples(
            client=llm_client,
            domain=domain_obj,
            text=text,
            record_id=record_id,
            initial_triples=existing_triples,
            temperature=temperature,
            augmentation_strategy=strategy,
            **strategy_kwargs,
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([t.model_dump() for t in triples], f, ensure_ascii=False, indent=2)

        count += 1
        if metadata.get("partial_result"):
            console.print(f"  [yellow]⚠ Partial result saved due to iteration failure.[/yellow]")
        summary = f"  → {len(triples)} triples saved"
        if "final_components" in metadata:
            summary += f" (Final components: {metadata['final_components']})"
        console.print(summary)

    console.print(f"\n[bold green]✓ Strategy '{strategy}' complete.[/bold green]")
    console.print(f"Output: {json_dir} ({count} files)")
    console.print(f"\n[dim]Next: kgb convert --input {json_dir}[/dim]")
    return json_dir


@augment_app.command("connectivity")
def augment_connectivity(
    input_file: Path = typer.Option(..., "--input", "-i", "--input-file", help="Path to input file", exists=True),
    output_dir: Path = typer.Option("outputs/kg_extraction", "--output-dir", "-o", help="Directory with extracted JSON"),
    domain: str = typer.Option(..., "--domain", "-d", help="Knowledge domain (use 'list domains' to see all)"),
    mode: ExtractionMode = typer.Option(ExtractionMode.OPEN, "--mode", "-m", help="Extraction mode"),
    client: str = typer.Option(
        "gemini",
        "--client",
        "-c",
        help="LLM client type",
        callback=_validate_client_type,
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model ID"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL"),
    text_field: str = typer.Option("text", "--text-field", help="Field name containing text"),
    id_field: str = typer.Option("id", "--id-field", help="Field name containing IDs"),
    record_ids: Optional[list[str]] = typer.Option(None, "--record-ids", help="List of record IDs to process"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit records"),
    temperature: float = typer.Option(0.0, "--temp", help="Sampling temperature"),
    max_disconnected: int = typer.Option(3, "--max-disconnected", help="Target max disconnected components"),
    max_iterations: int = typer.Option(2, "--max-iterations", help="Max refinement iterations"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Hide progress bar"),
    max_workers: Optional[int] = typer.Option(None, "--workers", help="Max parallel workers"),
    timeout: int = typer.Option(120, "--timeout", help="Request timeout"),
):
    """Connectivity augmentation: Reduce disconnected graph components.
    
    \b
    Examples:
        kgb augment connectivity --input data.jsonl --domain legal
    """
    console.print(f"[bold blue]Step 2: Augmentation (Connectivity)[/bold blue]")
    console.print(f"Target: ≤ {max_disconnected} components | Max iterations: {max_iterations}")

    try:
        _run_graph_strategy(
            "connectivity",
            input_file, output_dir, domain, mode, client, model, api_key, base_url,
            text_field, id_field, record_ids, limit, temperature, no_progress, max_workers, timeout,
            max_disconnected=max_disconnected,
            max_iterations=max_iterations,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@consolidate_app.command("entity_resolution")
def consolidate_entity_resolution(
    input_file: Path = typer.Option(..., "--input", "-i", "--input-file", help="Path to input file", exists=True),
    output_dir: Path = typer.Option("outputs/kg_extraction", "--output-dir", "-o", help="Directory with extracted JSON"),
    domain: str = typer.Option(..., "--domain", "-d", help="Knowledge domain (use 'list domains' to see all)"),
    mode: ExtractionMode = typer.Option(ExtractionMode.OPEN, "--mode", "-m", help="Extraction mode"),
    client: str = typer.Option(
        "gemini",
        "--client",
        "-c",
        help="LLM client type",
        callback=_validate_client_type,
    ),
    model: Optional[str] = typer.Option(None, "--model", help="Model ID"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="Base URL"),
    text_field: str = typer.Option("text", "--text-field", help="Field name containing text"),
    id_field: str = typer.Option("id", "--id-field", help="Field name containing IDs"),
    record_ids: Optional[list[str]] = typer.Option(None, "--record-ids", help="List of record IDs to process"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit records"),
    temperature: float = typer.Option(0.0, "--temp", help="Sampling temperature"),
    no_progress: bool = typer.Option(False, "--no-progress", help="Hide progress bar"),
    max_workers: Optional[int] = typer.Option(None, "--workers", help="Max parallel workers"),
    timeout: int = typer.Option(120, "--timeout", help="Request timeout"),
):
    """Entity resolution: merge entity name variants into canonical names.

    Consolidation only — no new triples are generated.

    \b
    Examples:
        kgb consolidate entity_resolution --input data.jsonl --domain legal
    """
    console.print(f"[bold blue]Step 2b: Consolidation (Entity Resolution)[/bold blue]")

    try:
        _run_graph_strategy(
            "entity_resolution",
            input_file, output_dir, domain, mode, client, model, api_key, base_url,
            text_field, id_field, record_ids, limit, temperature, no_progress, max_workers, timeout,
        )
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@augment_app.callback(invoke_without_command=True)
def augment_default(ctx: typer.Context):
    """Show available augmentation strategies."""
    if ctx.invoked_subcommand is None:
        from .builder.strategies import list_strategies
        console.print("[yellow]Augmentation strategies (add new triples):[/yellow]")
        for s in list_strategies(kind="augment"):
            console.print(f"  • {s}")


@consolidate_app.callback(invoke_without_command=True)
def consolidate_default(ctx: typer.Context):
    """Show available consolidation strategies."""
    if ctx.invoked_subcommand is None:
        from .builder.strategies import list_strategies
        console.print("[yellow]Consolidation strategies (merge/clean, add no triples):[/yellow]")
        for s in list_strategies(kind="consolidate"):
            console.print(f"  • {s}")


# =============================================================================
# CONVERT Command
# =============================================================================

@app.command()
def convert(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Directory with JSON triples", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for GraphML"),
    merge: bool = typer.Option(False, "--merge", help="Merge all JSON files into a single cross-document GraphML"),
):
    """Convert JSON triples to GraphML format.

    \b
    Examples:
        kgb convert --input outputs/extracted_json
        kgb convert --input outputs/extracted_json --merge
    """
    console.print(f"[bold blue]Converting JSON to GraphML[/bold blue]")

    graphml_dir = output_dir or input_dir.parent / "graphml"

    try:
        if merge:
            from .io.writers import merge_json_directories
            output_path = graphml_dir / "merged_graph.graphml"
            merge_json_directories(input_dir, output_path)
            console.print(f"\n[bold green]✓ Merged graph: {output_path}[/bold green]")
        else:
            graphml_files = convert_json_directory(input_dir, graphml_dir)
            console.print(f"\n[bold green]✓ Converted {len(graphml_files)} files[/bold green]")
            console.print(f"Output: {graphml_dir}")
        console.print(f"\n[dim]Next: kgb visualize network --input {graphml_dir}[/dim]")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


# =============================================================================
# VISUALIZE Commands
# =============================================================================

@visualize_app.command("network")
def visualize_network(
    input_dir: Path = typer.Option(..., "--input", "-i", help="Directory with GraphML files", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for HTML"),
    dark_mode: bool = typer.Option(False, "--dark-mode", help="Enable premium dark mode theme"),
    layout: str = typer.Option("spring", "--layout", help="Graph layout (spring, circular, kamada_kawai, shell)"),
):
    """Create interactive network visualizations from GraphML.
    
    \b
    Examples:
        kgb visualize network --input outputs/graphml --dark-mode
    """
    console.print(f"[bold blue]Creating Network Visualizations[/bold blue]")
    
    viz_dir = output_dir or input_dir.parent / "visualizations"
    
    try:
        html_files = batch_render_graphs(input_dir, viz_dir, dark_mode=dark_mode, layout=layout)
        console.print(f"\n[bold green]✓ Created {len(html_files)} network visualizations[/bold green]")
        console.print(f"Output: {viz_dir}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@visualize_app.command("extraction")
def visualize_extraction(
    input_file: Path = typer.Option(..., "--input", "-i", "--input-file", help="Path to original text data (.jsonl, .json, or .csv)", exists=True),
    triples_dir: Path = typer.Option(..., "--triples", "-t", help="Directory with extracted JSON triples", exists=True),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for HTML"),
    text_field: str = typer.Option("text", "--text-field", help="Field name containing text"),
    id_field: str = typer.Option("id", "--id-field", help="Field name containing record IDs"),
    limit: Optional[int] = typer.Option(None, "--limit", "-l", help="Limit number of records"),
    animation_speed: float = typer.Option(1.0, "--speed", help="Animation speed for highlights"),
    group_by: str = typer.Option("entity_type", "--group-by", help="How to group highlights (entity_type, relation)"),
):
    """Create interactive text visualizations with entity highlights.
    
    \b
    Examples:
        kgb visualize extraction --input data.jsonl --triples outputs/extracted_json
    """
    console.print(f"[bold blue]Creating Extraction Visualizations[/bold blue]")
    
    viz_dir = output_dir or triples_dir.parent / "visualizations_extraction"
    
    try:
        records = load_records(input_file, text_field, id_field, limit=limit)
        visualizer = TextVisualizer(animation_speed=animation_speed)
        
        # Prepare records for batch visualizer
        record_map = {}
        for r in records:
            rid = str(r["id"])
            text = str(r["text"])
            triple_file = triples_dir / f"{rid}.json"
            if triple_file.exists():
                with open(triple_file, "r", encoding="utf-8") as f:
                    triples = json.load(f)
                record_map[rid] = (text, triples)
        
        html_files = visualizer.batch_render(record_map, viz_dir, group_by=group_by)
        console.print(f"\n[bold green]✓ Created {len(html_files)} extraction visualizations[/bold green]")
        console.print(f"Output: {viz_dir}")
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Knowledge graph generation framework."""
    if ctx.invoked_subcommand is None:
        # When called via Typer with no subcommand, show help
        raise typer.Exit()


# =============================================================================
# KGB ASCII Banner & Interactive Shell
# =============================================================================

KGB_BANNER = r"""
    ╔═══════════════════════════════════════════════════════╗
    ║                                                       ║
    ║      ◉         ◉    ◉━━━━◉━━━━◉    ◉━━━━◉━━━━◉        ║
    ║      ┃        ╱     ┃              ┃          ╲       ║
    ║      ◉       ◉      ◉              ◉           ◉      ║
    ║      ┃      ╱       ┃              ┃          ╱       ║
    ║      ◉━━━━━◉        ◉    ◉━━━━◉    ◉━━━━◉━━━━◉        ║
    ║      ┃      ╲       ┃         ┃    ┃          ╲       ║
    ║      ◉       ◉      ◉         ◉    ◉           ◉      ║
    ║      ┃        ╲     ┃         ┃    ┃          ╱       ║
    ║      ◉         ◉    ◉━━━━◉━━━━◉    ◉━━━━◉━━━━◉        ║
    ║                                                       ║
    ║     K n o w l e d g e   G r a p h   B u i l d e r     ║
    ║                                                       ║
    ╚═══════════════════════════════════════════════════════╝
"""


_HISTORY_FILE = Path.home() / ".kgb_history"
_HISTORY_LENGTH = 1000

# Top-level commands and subcommands for tab completion
_COMPLETIONS = [
    "extract", "augment", "connectivity", "convert",
    "visualize", "network", "extraction",
    "list", "domains", "clients", "pipelines", "run-pipeline",
    "help", "exit", "quit",
    "--input", "--input-file", "--output", "--output-dir",
    "--domain", "--client", "--model", "--mode",
    "--limit", "--temp", "--workers", "--timeout",
    "--no-progress", "--api-key", "--base-url",
    "--text-field", "--id-field", "--record-ids",
    "--prompt", "--dark-mode", "--layout",
    "--extract", "--augment", "--convert", "--visualize",
    "--config", "-f",
    "--max-disconnected", "--max-iterations",
    "--triples", "--speed", "--group-by",
    "-i", "-o", "-d", "-c", "-m", "-l", "-t",
]


def _setup_readline():
    """Configure readline for history, line editing, and tab completion."""
    if readline is None:
        return
    # Load persisted history
    try:
        readline.read_history_file(_HISTORY_FILE)
    except FileNotFoundError:
        pass
    readline.set_history_length(_HISTORY_LENGTH)
    atexit.register(readline.write_history_file, str(_HISTORY_FILE))

    # Tab completion
    def completer(text: str, state: int) -> str | None:
        buf = readline.get_line_buffer().lstrip()
        # If cursor is on the first token, complete command names only
        if " " not in buf:
            options = [c for c in _COMPLETIONS if c.startswith(text) and not c.startswith("-")]
        else:
            options = [c for c in _COMPLETIONS if c.startswith(text)]
        return options[state] if state < len(options) else None

    readline.set_completer(completer)
    readline.set_completer_delims(" \t")
    # macOS ships libedit instead of GNU readline; bind syntax differs
    if "libedit" in (readline.__doc__ or ""):
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


def interactive_shell():
    """Run the interactive KGB shell with REPL."""
    _setup_readline()

    console.print(KGB_BANNER, style="bold green")
    console.print(f"  Knowledge Graph Builder v{__version__}", style="bold white")
    console.print("  Type [bold]help[/bold] for commands, [bold]exit[/bold] to quit.\n")

    while True:
        try:
            line = input("KGB> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!", style="bold green")
            break

        if not line:
            continue

        if line.lower() in ("exit", "quit"):
            console.print("Goodbye!", style="bold green")
            break

        if line.lower() == "help":
            line = "--help"

        try:
            args = shlex.split(line)
        except ValueError as e:
            console.print(f"[red]Parse error:[/red] {e}")
            continue

        try:
            app(args, standalone_mode=False)
        except SystemExit:
            # Typer/Click raises SystemExit on --help and errors; absorb it
            pass
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")


def main_entry():
    """Console-scripts entry point.

    - With arguments: one-shot mode (delegates to Typer app).
    - Without arguments: launches the interactive KGB shell.
    """
    if len(sys.argv) > 1:
        app()
    else:
        interactive_shell()


if __name__ == "__main__":
    main_entry()
