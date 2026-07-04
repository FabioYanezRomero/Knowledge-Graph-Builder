"""Interactive text visualization of extracted triples using langextract."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import langextract as lx
from langextract import data


from ..domains import Triple, InferenceType


def _iter_head_tail(triples: list[Triple] | list[dict[str, Any]]):
    """Yield (head, tail) for each triple, accepting Triple objects or dicts."""
    for t in triples:
        if isinstance(t, Triple):
            yield t.head, t.tail
        elif isinstance(t, dict):
            yield t.get("head", ""), t.get("tail", "")


class TextVisualizer:
    """Create interactive HTML visualizations of extracted triples in source text.

    This class converts knowledge graph extraction results into langextract's
    AnnotatedDocument format and generates animated HTML visualizations that
    highlight entities in the original text.
    """

    def __init__(
        self,
        animation_speed: float = 1.0,
        show_legend: bool = True,
        gif_optimized: bool = False
    ) -> None:
        """Initialize the visualizer.

        Args:
            animation_speed: Animation speed in seconds between entity highlights
            show_legend: Whether to show color legend for entity types
            gif_optimized: Apply GIF-optimized styling (larger fonts, better contrast)
        """
        self.animation_speed = animation_speed
        self.show_legend = show_legend
        self.gif_optimized = gif_optimized

    def _create_extraction(
        self,
        entity_text: str,
        entity_type: str,
        text: str,
        extraction_index: int,
        relation: str | None = None,
        role: str | None = None,
        inference: InferenceType | str | None = None
    ) -> data.Extraction | None:
        """Create an Extraction object for an entity."""
        # Find the entity in the text (case-sensitive)
        start_pos = text.find(entity_text)

        if start_pos == -1:
            # Try case-insensitive search
            text_lower = text.lower()
            entity_lower = entity_text.lower()
            start_pos = text_lower.find(entity_lower)

            if start_pos == -1:
                return None

        end_pos = start_pos + len(entity_text)
        char_interval = data.CharInterval(start_pos=start_pos, end_pos=end_pos)

        # Build attributes dict
        attributes = {}
        if relation:
            attributes["relation"] = relation
        if role:
            attributes["role"] = role
        if inference:
            attributes["inference"] = str(inference)

        # Build description
        description_parts = []
        if role and relation:
            description_parts.append(f"{role.capitalize()} of relation: '{relation}'")
        
        inf_str = str(inference) if inference else ""
        if inf_str:
            description_parts.append(f"Inference: {inf_str}")
            
        description = " | ".join(description_parts) if description_parts else None

        return data.Extraction(
            extraction_class=entity_type,
            extraction_text=entity_text,
            char_interval=char_interval,
            extraction_index=extraction_index,
            description=description,
            attributes=attributes if attributes else None
        )

    def render_triples_in_text(
        self,
        text: str,
        triples: list[Triple] | list[dict[str, Any]],
        document_id: str | None = None,
        group_by: str = "entity_type"  # "entity_type" or "relation"
    ) -> str:
        """Render an interactive HTML visualization from extracted triples.

        Args:
            text: The original text that was analyzed
            triples: List of extracted triples (Triples or dicts)
            document_id: Optional identifier for this document
            group_by: How to group entities - "entity_type" or "relation"

        Returns:
            HTML string with interactive visualization
        """
        if not text or not triples:
            return "<p>No text or triples to visualize</p>"

        # Convert to Triple objects if needed
        validated_triples: list[Triple] = []
        for t in triples:
            if isinstance(t, Triple):
                validated_triples.append(t)
            else:
                try:
                    validated_triples.append(Triple(**t))
                except Exception:
                    continue

        extractions = []
        extraction_index = 0
        seen_entities = set()

        for t in validated_triples:
            # Determine entity types
            if group_by == "relation":
                head_type = f"{t.relation} (source)"
                tail_type = f"{t.relation} (target)"
            else:
                # Distinguish augmented triples in the class name for CSS styling
                suffix = " (Augmented)" if t.inference == InferenceType.CONTEXTUAL else ""
                head_type = f"Head Entity{suffix}"
                tail_type = f"Tail Entity{suffix}"

            # Extract head
            if t.head and t.head not in seen_entities:
                extraction = self._create_extraction(
                    entity_text=t.head,
                    entity_type=head_type,
                    text=text,
                    extraction_index=extraction_index,
                    relation=t.relation,
                    role="head",
                    inference=t.inference
                )
                if extraction:
                    extractions.append(extraction)
                    extraction_index += 1
                    seen_entities.add(t.head)

            # Extract tail
            if t.tail and t.tail not in seen_entities:
                extraction = self._create_extraction(
                    entity_text=t.tail,
                    entity_type=tail_type,
                    text=text,
                    extraction_index=extraction_index,
                    relation=t.relation,
                    role="tail",
                    inference=t.inference
                )
                if extraction:
                    extractions.append(extraction)
                    extraction_index += 1
                    seen_entities.add(t.tail)

        if not extractions:
            return "<p>No entities found in the text</p>"

        annotated_doc = data.AnnotatedDocument(
            text=text,
            extractions=extractions,
            document_id=document_id
        )

        html = lx.visualize(
            annotated_doc,
            animation_speed=self.animation_speed,
            show_legend=self.show_legend,
            gif_optimized=self.gif_optimized
        )

        if hasattr(html, '_repr_html_'):
            html = html._repr_html_()
        return str(html)

    def save_html(
        self,
        text: str,
        triples: list[Triple] | list[dict[str, Any]],
        output_path: Path | str,
        document_id: str | None = None,
        group_by: str = "entity_type"
    ) -> Path:
        """Render and save the HTML visualization to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        html = self.render_triples_in_text(
            text=text,
            triples=triples,
            document_id=document_id,
            group_by=group_by
        )

        # Wrap in a premium HTML document
        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KG Visualization - {document_id or 'Untitled'}</title>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --card-bg: rgba(255, 255, 255, 0.8);
            --text-color: #1e293b;
            --muted-text: #64748b;
            --accent-color: #3b82f6;
            --glass-border: rgba(255, 255, 255, 0.3);
            --shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        }}

        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-color: #0f172a;
                --card-bg: rgba(30, 41, 59, 0.7);
                --text-color: #f1f5f9;
                --muted-text: #94a3b8;
                --glass-border: rgba(255, 255, 255, 0.1);
                --shadow: 0 10px 15px -3px rgb(0 0 0 / 0.5);
            }}
        }}

        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.5;
            margin: 0;
            padding: 40px 20px;
            transition: background-color 0.3s ease;
        }}

        .container {{
            max-width: 1000px;
            margin: 0 auto;
        }}

        .header {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 32px;
            border-radius: 16px;
            margin-bottom: 32px;
            box-shadow: var(--shadow);
        }}

        .header h1 {{
            margin: 0 0 16px 0;
            font-size: 24px;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}

        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 16px;
            font-size: 14px;
            color: var(--muted-text);
        }}

        .stat-item b {{
            color: var(--text-color);
            display: block;
            font-size: 18px;
        }}

        .content {{
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--glass-border);
            padding: 40px;
            border-radius: 16px;
            box-shadow: var(--shadow);
        }}

        /* Style for augmented highlights if possible through langextract CSS injection or override */
        mark[data-extraction-class*="Augmented"] {{
            border: 1px dashed currentColor;
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Knowledge Graph Extraction</h1>
            <div class="stats">
                <div class="stat-item">
                    ID
                    <b>{document_id or 'N/A'}</b>
                </div>
                <div class="stat-item">
                    Entities
                    <b>{len({name for h, t in _iter_head_tail(triples) for name in (h, t)})}</b>
                </div>
                <div class="stat-item">
                    Triples
                    <b>{len(triples)}</b>
                </div>
            </div>
        </div>
        <div class="content">
            {html}
        </div>
    </div>
</body>
</html>"""

        output_path.write_text(full_html, encoding="utf-8")
        return output_path

    def batch_render(
        self,
        records: dict[str, tuple[str, list[Triple] | list[dict[str, Any]]]],
        output_dir: Path | str,
        group_by: str = "entity_type"
    ) -> list[Path]:
        """Render visualizations for multiple records."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        created_files = []
        for record_id, (text, triples) in records.items():
            output_path = output_dir / f"{record_id}.html"

            try:
                self.save_html(
                    text=text,
                    triples=triples,
                    output_path=output_path,
                    document_id=record_id,
                    group_by=group_by
                )
                created_files.append(output_path)
            except Exception as e:
                print(f"Error creating visualization for {record_id}: {e}")

        return created_files



__all__ = [
    "TextVisualizer",
]
