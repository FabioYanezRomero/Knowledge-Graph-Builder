"""Client configuration for LLM clients."""

from __future__ import annotations

from dataclasses import dataclass

ClientType = str


@dataclass
class ClientConfig:
    """Configuration for creating an LLM client.

    This is a pure data container that holds all parameters needed to
    instantiate any supported LLM client type. No provider-specific
    logic is applied here - each client's from_config() method handles
    its own defaults.

    This design makes the config suitable for CLI integration (e.g., Typer)
    where user input should be preserved as-is.
    """

    # Client type selection
    client_type: ClientType = "gemini"

    # Common parameters (all clients)
    model_id: str | None = None  # None = use client's default
    temperature: float = 0.0
    max_workers: int | None = None  # None = use client's default
    max_char_buffer: int = 8000
    show_progress: bool = True

    # Langextract-specific parameters
    extraction_passes: int = 1  # Number of extraction passes (higher = better recall)
    batch_length: int | None = None  # None = use client's default

    # API-based clients (Gemini)
    api_key: str | None = None

    # Local server-based clients (Ollama, LM Studio)
    base_url: str | None = None
    timeout: int = 120

    # Ollama generation controls (ignored by other providers)
    think: bool | None = None       # False disables the model's "thinking" phase
    options: dict | None = None     # extra Ollama options: num_ctx, top_p, num_predict, ...
    reset_every: int | None = None  # unload the model every N extractions (low-parallelism wedge guard)


__all__ = ["ClientConfig", "ClientType"]
