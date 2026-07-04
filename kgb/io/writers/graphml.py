"""Convert JSON triples to GraphML format.

This module provides functions to convert extracted knowledge graph triples
from JSON format to NetworkX GraphML format for visualization and analysis.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import networkx as nx


def normalize_entity_name(name: str) -> str:
    """Normalize entity name by stripping whitespace.

    Args:
        name: Original entity name

    Returns:
        Normalized entity name
    """
    if not name:
        return ""
    return name.strip()


def get_canonical_name(name: str, entity_map: dict[str, str]) -> str:
    """Get canonical form of entity name using case-insensitive matching.

    Args:
        name: Entity name to normalize
        entity_map: Dictionary mapping lowercase names to canonical forms

    Returns:
        Canonical entity name
    """
    normalized = normalize_entity_name(name)
    if not normalized:
        return ""

    key = normalized.lower()

    if key in entity_map:
        return entity_map[key]

    entity_map[key] = normalized
    return normalized


from ...domains import Triple


def json_to_graphml(
    triples: list[Triple] | list[dict[str, Any]],
    output_path: Path | str | None = None
) -> nx.DiGraph:
    """Convert a list of triples to a NetworkX DiGraph.

    Normalizes entity names to avoid disconnected components due to
    case/whitespace variations.

    Args:
        triples: List of extracted triples (Triple objects or dicts)
        output_path: Optional path to save GraphML file

    Returns:
        NetworkX directed graph
    """
    G = nx.DiGraph()
    entity_map: dict[str, str] = {}

    # Ensure we have Triple objects
    validated_triples: list[Triple] = []
    for t in triples:
        if isinstance(t, Triple):
            validated_triples.append(t)
        else:
            try:
                validated_triples.append(Triple(**t))
            except Exception:
                continue

    for t in validated_triples:
        head = get_canonical_name(t.head, entity_map)
        tail = get_canonical_name(t.tail, entity_map)

        if not head or not tail:
            continue

        # Add nodes
        if head not in G.nodes():
            G.add_node(head)
        if tail not in G.nodes():
            G.add_node(tail)

        # Build edge attributes
        edge_attrs = {
            "relation": t.relation,
            "inference": t.inference.value
        }

        G.add_edge(head, tail, **edge_attrs)

    # Save if path provided
    if output_path:
        nx.write_graphml(G, str(output_path))

    return G


def convert_json_directory(
    input_dir: Path | str,
    output_dir: Path | str
) -> list[Path]:
    """Convert all JSON files in a directory to GraphML format.

    Args:
        input_dir: Directory containing JSON files
        output_dir: Directory to save GraphML files

    Returns:
        List of paths to created GraphML files
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graphml_files = []

    for json_file in input_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {json_file}: Invalid JSON")
                continue

        if not isinstance(data, list):
            print(f"Skipping {json_file}: Not a list of triples")
            continue

        output_path = output_dir / f"{json_file.stem}.graphml"
        G = json_to_graphml(data, output_path)

        print(f"Converted {json_file.name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        graphml_files.append(output_path)

    return graphml_files


def merge_json_directories(
    input_dir: Path | str,
    output_path: Path | str,
) -> nx.DiGraph:
    """Merge all JSON triple files in a directory into a single GraphML file.

    Entity names are canonicalized across documents, so records mentioning
    the same entity share a node in the merged graph.

    Args:
        input_dir: Directory containing per-record JSON triple files
        output_path: Path of the merged GraphML file to write

    Returns:
        The merged NetworkX directed graph
    """
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_triples: list[dict[str, Any]] = []
    for json_file in sorted(input_dir.glob("*.json")):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping {json_file}: Invalid JSON")
                continue
        if not isinstance(data, list):
            print(f"Skipping {json_file}: Not a list of triples")
            continue
        all_triples.extend(data)

    G = json_to_graphml(all_triples, output_path)
    print(f"Merged {len(all_triples)} triples into {output_path.name}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


__all__ = ["json_to_graphml", "convert_json_directory", "merge_json_directories", "normalize_entity_name", "get_canonical_name"]
