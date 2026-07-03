"""Registry and filesystem discovery for knowledge domains.

A domain is a directory of resources (prompts, examples, schema) — no code
required. `get_domain(name)` resolves, in order:

1. A class registered via @domain(name) / register_domain (for domains that
   need custom Python behavior).
2. A directory `kgb/domains/<name>/` shipped with the package.
3. A directory `<root>/<name>/` for each root in the KGB_DOMAINS_PATH
   environment variable (os.pathsep-separated), so use-case domains can live
   outside this repository.
4. `name` itself as a path to a domain directory (e.g. ./my_domains/finance).

A directory qualifies as a domain if it contains an `extraction/` subfolder.

Usage:
    my_domain = get_domain("legal", extraction_mode="open")
    my_domain = get_domain("./path/to/my_usecase")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, TypeVar

if TYPE_CHECKING:
    from .base import KnowledgeDomain


_DOMAIN_REGISTRY: dict[str, type[KnowledgeDomain]] = {}

_PACKAGE_DOMAINS_DIR = Path(__file__).parent


def _external_domain_roots() -> list[Path]:
    """Directories listed in KGB_DOMAINS_PATH (os.pathsep-separated)."""
    raw = os.environ.get("KGB_DOMAINS_PATH", "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def _is_domain_dir(path: Path) -> bool:
    return (path / "extraction").is_dir()


def _find_domain_dir(name: str) -> Path | None:
    """Locate a domain directory by name or path."""
    candidates = [_PACKAGE_DOMAINS_DIR / name]
    candidates += [root / name for root in _external_domain_roots()]
    candidates.append(Path(name).expanduser())
    for candidate in candidates:
        if _is_domain_dir(candidate):
            return candidate
    return None

T = TypeVar("T", bound="KnowledgeDomain")


def domain(name: str) -> Callable[[type[T]], type[T]]:
    """Decorator to register a domain class.
    
    Usage:
        @domain("legal")
        class LegalDomain(KnowledgeDomain):
            pass
    
    Args:
        name: The name to register the domain under.
        
    Returns:
        A decorator that registers the class and returns it unchanged.
    """
    def decorator(cls: type[T]) -> type[T]:
        register_domain(name, cls)
        return cls
    return decorator


def register_domain(name: str, domain_class: type[KnowledgeDomain]) -> None:
    """Register a new knowledge domain.
    
    Args:
        name: The name to register the domain under.
        domain_class: The domain class to register.
    """
    _DOMAIN_REGISTRY[name] = domain_class


def get_domain(name: str, **kwargs) -> KnowledgeDomain:
    """Get a domain instance by name or directory path.

    Args:
        name: Domain name (e.g., "legal") or path to a domain directory
        **kwargs: Arguments to pass to the domain constructor (e.g., extraction_mode)

    Returns:
        KnowledgeDomain instance

    Raises:
        ValueError: If no registered class or domain directory matches
    """
    if name in _DOMAIN_REGISTRY:
        return _DOMAIN_REGISTRY[name](**kwargs)

    domain_dir = _find_domain_dir(name)
    if domain_dir is not None:
        from .base import KnowledgeDomain
        return KnowledgeDomain(root_dir=domain_dir, **kwargs)

    available = ", ".join(list_available_domains())
    raise ValueError(
        f"Unknown domain '{name}'. Available: {available or 'none'}. "
        f"A domain is a directory with an extraction/ subfolder; pass its "
        f"path directly or add its parent directory to KGB_DOMAINS_PATH."
    )


def list_available_domains() -> list[str]:
    """List all domains: registered classes plus discovered directories."""
    names = set(_DOMAIN_REGISTRY)
    for root in (_PACKAGE_DOMAINS_DIR, *_external_domain_roots()):
        if root.is_dir():
            names.update(d.name for d in root.iterdir() if _is_domain_dir(d))
    return sorted(names)


__all__ = ["domain", "register_domain", "get_domain", "list_available_domains"]
