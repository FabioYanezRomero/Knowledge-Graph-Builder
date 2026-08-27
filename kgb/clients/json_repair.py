"""JSON repair shared by every backend's augment() path.

``augment()`` bypasses langextract in all three providers, so each one parses raw
model output on its own. That left ollama with salvage and the other two raising
on the first bad character — one stray comma cost the whole consolidation. The
repair is not provider-specific, so it lives here and they all get it.
"""

from __future__ import annotations


def _json_loads_ok(text: str) -> bool:
    import json
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _salvage_json_objects(text: str) -> str | None:
    """Rebuild a clean JSON array from the top-level ``{...}`` objects in ``text``,
    keeping only the ones that parse. Handles the malformations a strict parser
    chokes on but that our lighter fixes miss: doubled commas, unquoted keys in a
    stray object, and — importantly — a response truncated mid-object because the
    model hit max_tokens (the incomplete trailing object is simply dropped)."""
    import json
    import re

    objs: list = []
    depth = 0
    start: int | None = None
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    frag = text[start : i + 1]
                    try:
                        objs.append(json.loads(frag))
                    except Exception:
                        try:  # one more try after dropping a trailing comma
                            objs.append(json.loads(re.sub(r",(\s*})", r"\1", frag)))
                        except Exception:
                            pass
                    start = None
    return json.dumps(objs) if objs else None


def _repair_json_text(text: str) -> str:
    """Make local-model JSON parseable by the native provider's strict parser.

    Applies cheap fixes first (strip control chars, drop trailing commas). If the
    result still doesn't parse, salvages the individual valid objects so a single
    malformed or truncated object doesn't lose the whole extraction. Guarantees
    the returned text is valid JSON whenever any object could be recovered.
    """
    import re

    fixed = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)  # control chars
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)                # trailing commas

    # Fast path: if it (or its fenced body) already parses, keep it as-is so the
    # common healthy case is untouched.
    body = fixed
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", fixed)
    if m:
        body = m.group(1)
    if _json_loads_ok(body):
        return fixed

    salvaged = _salvage_json_objects(fixed)
    return salvaged if salvaged is not None else fixed
