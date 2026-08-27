"""Client factory for creating LLM clients based on configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, TypeVar

from .base import BaseLLMClient, LLMClientError
from .config import ClientConfig, ClientType

if TYPE_CHECKING:
    pass

T = TypeVar("T", bound=BaseLLMClient)


def client(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a client class with the factory.

    Usage:
        @client("gemini")
        class GeminiClient(BaseLLMClient):
            pass

    Args:
        name: The client type name to register under.

    Returns:
        A decorator that registers the class and returns it unchanged.
    """
    def decorator(cls: type[T]) -> type[T]:
        ClientFactory.register(name, cls)
        return cls
    return decorator


class ClientFactory:
    """Factory class for creating LLM clients.

    This class provides a unified interface for instantiating different
    LLM client types based on configuration. It uses a registry pattern
    where each client class registers itself with its type identifier.

    The registry is populated via the @client decorator on each provider class.

    Supported client types are dynamically determined by the registry.

    Examples:
        >>> config = ClientConfig(client_type="gemini", api_key="your-key")
        >>> client = ClientFactory.create(config)

        >>> # Create with kwargs directly
        >>> client = ClientFactory.from_config(
        ...     client_type="ollama",
        ...     model_id="llama3.1",
        ...     base_url="http://localhost:11434"
        ... )

        >>> # Check available clients
        >>> ClientFactory.get_available_clients()
        ['gemini', 'ollama', 'lmstudio']
    """

    # Registry of available client classes - populated by @client decorator
    _client_registry: dict[ClientType, type[BaseLLMClient]] = {}

    @classmethod
    def register(cls, client_type: ClientType, client_class: type[BaseLLMClient]) -> None:
        """Register a client class for a given type identifier.

        Args:
            client_type: The type identifier (e.g., 'gemini', 'ollama')
            client_class: The client class to register
        """
        cls._client_registry[client_type] = client_class

    @classmethod
    def create(cls, config: ClientConfig) -> BaseLLMClient:
        """Create an LLM client based on configuration.

        Uses the registry to look up the appropriate client class,
        then delegates instantiation to that class's from_config method.

        Args:
            config: Client configuration specifying type and parameters

        Returns:
            Instantiated client matching the requested type

        Raises:
            LLMClientError: If client type is not registered
        """
        client_class = cls._client_registry.get(config.client_type)
        if client_class is None:
            available = ", ".join(cls._client_registry.keys()) or "none"
            raise LLMClientError(
                f"Unsupported client type: '{config.client_type}'. "
                f"Available types: {available}"
            )
        return client_class.from_config(config)

    @classmethod
    def from_config(cls, **kwargs) -> BaseLLMClient:
        """Create a client directly from configuration parameters.

        Convenience method that creates a ClientConfig and then
        instantiates the appropriate client.

        Args:
            **kwargs: Configuration parameters passed to ClientConfig

        Returns:
            Instantiated LLM client

        Examples:
            >>> client = ClientFactory.from_config(
            ...     client_type="gemini",
            ...     model_id="gemini-flash-latest",
            ...     api_key="your-key"
            ... )
        """
        config = ClientConfig(**kwargs)
        return cls.create(config)

    @classmethod
    def get_available_clients(cls) -> list[ClientType]:
        """Get list of registered client types.

        Returns:
            List of supported client type identifiers
        """
        return list(cls._client_registry.keys())

    @classmethod
    def is_registered(cls, client_type: ClientType) -> bool:
        """Check if a client type is registered.

        Args:
            client_type: The type identifier to check

        Returns:
            True if the client type is registered
        """
        return client_type in cls._client_registry


__all__ = ["ClientFactory", "client"]
