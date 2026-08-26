"""LLM client abstraction layer for knowledge graph extraction."""

from __future__ import annotations

from . import chunk_safety
from .base import BaseLLMClient, LLMClientError
from .config import ClientConfig, ClientType
from .factory import ClientFactory, client
from .providers import GeminiClient, OllamaClient, LMStudioClient

# Every backend extracts through langextract, and langextract lets one bad chunk
# abort the whole document. Install the bound before any client can be built.
chunk_safety.install()

__all__ = [
    "BaseLLMClient",
    "LLMClientError",
    "ClientConfig",
    "ClientType",
    "ClientFactory",
    "GeminiClient",
    "OllamaClient",
    "LMStudioClient",
]
