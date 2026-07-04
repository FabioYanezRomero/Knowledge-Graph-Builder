---
name: add-domain
description: Manage knowledge domains (e.g., Medical, Finance). Covers adding new domains, updating prompts, and adding few-shot examples.
---

# Adding a Knowledge Domain

This skill documents how to add a new knowledge domain. A domain is a **directory of resources — no Python code required**.

## Overview

Domains are bundled resource sets containing prompts and few-shot examples for extraction and augmentation. The system provides:
- Filesystem discovery: any directory with an `extraction/` subfolder is a domain
- External domains via `KGB_DOMAINS_PATH` or a direct path to `--domain` (preferred for use-case-specific domains — no fork/PR needed)
- Optional `@domain()` decorator registry, only for domains that need custom Python behavior
- Strategy-based augmentation folders
- Optional schema constraints (entity types + relation types)

## Architecture

```
                          Domains Module
    ┌───────────────────────────────────────────────────────────┐
    │                                                           │
    │  registry.py                base.py                       │
    │  ├─ @domain()               ├─ KnowledgeDomain (ABC)      │
    │  ├─ register_domain()       ├─ DomainComponent            │
    │  ├─ get_domain()            ├─ DomainLike (Protocol)      │
    │  └─ list_available_domains()└─ DomainResourceError        │
    │                                                           │
    │  models.py                                                │
    │  ├─ Triple, InferenceType, ExtractionMode                 │
    │  ├─ Extraction, ExtractionExample                         │
    │  ├─ AugmentationExample, DomainSchema                     │
    │  └─ DomainExamples                                        │
    │                                                           │
    │  legal/                     default/                      │
    │  ├─ extraction/             ├─ extraction/                │
    │  ├─ augmentation/           └─ augmentation/              │
    │  └─ schema.json                                           │
    │                                                           │
    └───────────────────────────────────────────────────────────┘

Resolution order in get_domain(name):
  1. _DOMAIN_REGISTRY (classes registered via @domain — optional)
  2. kgb/domains/<name>/            (packaged domains)
  3. <root>/<name>/ for each root in KGB_DOMAINS_PATH
  4. <name> as a direct path to a domain directory
```

## Dependencies

| Component | Library | Purpose |
|-----------|---------|---------|
| Schema validation | `pydantic>=2.0` | Triple validation |
| Extraction | `langextract>=0.1` | Prompt framework |
| Resource loading | `pathlib` (stdlib) | File operations |

## Directory Structure

The directory can live **anywhere** — inside `kgb/domains/` (for domains shipped with the package) or in your own project (resolved via `KGB_DOMAINS_PATH` or a direct path):

```text
<domain_name>/
├── extraction/
│   ├── prompt_open.md          # Open extraction prompt
│   ├── prompt_constrained.md   # Type-constrained extraction prompt
│   └── examples.json           # Few-shot extraction examples
├── consolidation/              # Strategies that merge/clean (add no knowledge)
│   └── entity_resolution/
│       ├── prompt.md
│       └── examples.json
├── augmentation/               # Strategies that add new triples
│   └── connectivity/           # Strategy folder (one per strategy)
│       ├── prompt.md           # Strategy-specific prompt
│       └── examples.json       # Few-shot examples
└── schema.json                 # Optional: entity/relation type constraints
```

Strategy folders are looked up under both `consolidation/` and
`augmentation/` — the split documents whether a strategy adds knowledge
(augment) or merges existing knowledge (consolidate).

> **File extensions**: Prompts use `.md` (markdown). The base class resolves `prompt_open.md` or `prompt_constrained.md` based on `extraction_mode`, and `prompt.md` for augmentation strategies.

## Step 1: Create Resource Files

### Extraction Prompts

Create `extraction/prompt_open.md`:

```markdown
Extract all knowledge graph triples from the following biomedical text.
Focus on explicit relationships between biomedical entities.

For each relationship identified, extract:
- **head**: The source entity
- **relation**: The relationship type
- **tail**: The target entity

{{schema_constraints}}
```

> **Important:** Do NOT include output format instructions. The `langextract` framework generates format instructions from examples. You can include `{{schema_constraints}}` to inject entity/relation type guidance.

Create `extraction/prompt_constrained.md` (for `--mode constrained`):

```markdown
Extract knowledge graph triples from the following biomedical text.
Only extract entities and relations that match the provided schema types.

{{schema_constraints}}
```

### Extraction Examples (`extraction/examples.json`)

```json
[
  {
    "text": "Aspirin is used to treat headaches and reduce fever.",
    "extractions": [
      {
        "extraction_class": "Triple",
        "extraction_text": "Aspirin is used to treat headaches",
        "char_start": 0,
        "char_end": 35,
        "attributes": {
          "head": "Aspirin",
          "relation": "treats",
          "tail": "headaches",
          "inference": "explicit"
        }
      },
      {
        "extraction_class": "Triple",
        "extraction_text": "Aspirin is used to reduce fever",
        "char_start": 0,
        "char_end": 50,
        "attributes": {
          "head": "Aspirin",
          "relation": "reduces",
          "tail": "fever",
          "inference": "explicit"
        }
      }
    ]
  }
]
```

> **Key fields**: `char_start`/`char_end` must be valid character positions in the `text`. `extraction_text` is the span that justifies the extraction. `inference` must be `"explicit"` for extraction examples.

### Augmentation Prompt (`augmentation/connectivity/prompt.md`)

```markdown
You are a biomedical knowledge graph expert.

Given the following text and a partially extracted knowledge graph with disconnected components,
generate new triples that bridge the disconnected components.

## Source Text
{{text}}

## Current Triples
{{current_triples}}

## Disconnected Components
{{disconnected_components}}

{{schema_constraints}}

Generate bridging triples as a JSON array. Each triple must have:
- head, relation, tail, inference ("contextual"), justification
```

### Augmentation Examples (`augmentation/connectivity/examples.json`)

```json
[
  {
    "input": {
      "text": "Aspirin treats headaches. Ibuprofen is an NSAID.",
      "components": [
        {"entities": ["Aspirin", "headaches"]},
        {"entities": ["Ibuprofen", "NSAID"]}
      ]
    },
    "output": [
      {
        "head": "Aspirin",
        "relation": "is_a",
        "tail": "NSAID",
        "inference": "contextual",
        "justification": "Aspirin is also classified as an NSAID, bridging the two components."
      }
    ]
  }
]
```

## Step 2: Create Schema (Optional)

Create `schema.json`:

```json
{
  "entity_types": ["Drug", "Disease", "Symptom", "Gene", "Protein"],
  "relation_types": ["treats", "causes", "indicates", "inhibits", "binds_to"]
}
```

When present, schema constraints are:
- Injected into prompts via `{{schema_constraints}}`
- Used for validation warnings (not hard enforcement by default)
- Accessible via `domain.schema.entity_types` and `domain.schema.relation_types`

## Step 3: Make the Domain Discoverable

**No code needed.** Pick one:

1. **External domain (preferred for use-case-specific domains):** keep the directory in your own project and either pass its path directly (`--domain ./my_domains/biomedical`) or add its parent to `KGB_DOMAINS_PATH`:

   ```bash
   export KGB_DOMAINS_PATH=~/my-kg-domains   # contains biomedical/
   kgb extract --input data.jsonl --domain biomedical
   ```

2. **Packaged domain:** place the directory at `kgb/domains/biomedical/`. It is discovered automatically — no `__init__.py`, no registry import. If you add a packaged domain, also confirm its data files match the `"kgb.domains"` globs in `[tool.setuptools.package-data]` (pyproject.toml).

3. **Custom behavior only:** if the domain needs Python logic (overriding resource loading, etc.), subclass `KnowledgeDomain` and register with `@domain("biomedical")`. The base class resolves resources relative to the file defining the subclass (`inspect.getfile`); override with `root_dir=` for testing.

## Step 4: Verify

### Check Discovery

```bash
python -c "from kgb.domains import list_available_domains; print(list_available_domains())"
# Output: ['biomedical', 'default', 'legal']  (with KGB_DOMAINS_PATH set, if external)
```

### Unit Tests

```python
import pytest
from kgb.domains import get_domain, list_available_domains, DomainResourceError


def test_domain_registered():
    assert "biomedical" in list_available_domains()


def test_extraction_prompt_loads():
    domain = get_domain("biomedical")
    assert len(domain.extraction.prompt) > 50


def test_extraction_examples_valid():
    domain = get_domain("biomedical")
    examples = domain.extraction.examples
    assert isinstance(examples, list)
    assert len(examples) > 0
    assert "text" in examples[0]
    assert "extractions" in examples[0]


def test_augmentation_strategy_exists():
    domain = get_domain("biomedical")
    assert "connectivity" in domain.list_augmentation_strategies()

    conn = domain.get_augmentation("connectivity")
    assert len(conn.prompt) > 0
    assert isinstance(conn.examples, list)


def test_schema_loads():
    domain = get_domain("biomedical")
    assert "Drug" in domain.schema.entity_types
    assert "treats" in domain.schema.relation_types


def test_constrained_mode():
    domain = get_domain("biomedical", extraction_mode="constrained")
    assert "constrained" in domain.extraction._prompt_path.name


def test_missing_strategy():
    domain = get_domain("biomedical")
    with pytest.raises(DomainResourceError):
        domain.get_augmentation("nonexistent")
```

## CLI Usage

```bash
# Extract with your domain
kgb extract --input data.jsonl --domain biomedical

# Constrained mode (uses prompt_constrained.md)
kgb extract --input data.jsonl --domain biomedical --mode constrained

# Augment with connectivity strategy
kgb augment connectivity --input data.jsonl --domain biomedical

# List available domains
kgb list domains
```

## Troubleshooting

### "DomainResourceError: Resource not found"
- Verify file exists: `ls kgb/domains/biomedical/extraction/`
- Check filename matches exactly: `prompt_open.md` (not `.txt`)
- Augmentation prompts must be `prompt.md` inside strategy folders

### "ValueError: Unknown domain 'biomedical'"
- Verify the directory contains an `extraction/` subfolder (that's what marks it as a domain)
- If external: check `KGB_DOMAINS_PATH` points to the **parent** directory, or pass the full path to `--domain`

### "ValidationError: examples[0]..."
- Check `examples.json` matches the ExtractionExample schema
- Validate JSON: `python -m json.tool examples.json`
- Ensure `char_start`/`char_end` are valid integers

## Error Handling

```python
from kgb.domains import get_domain, DomainResourceError

try:
    domain = get_domain("biomedical")
    prompt = domain.extraction.prompt
except DomainResourceError as e:
    print(f"Resource error: {e} (file: {e.resource_path})")
except ValueError as e:
    print(f"Domain not found: {e}")
```

## Files to Create

All under `<domain_dir>/` (your project or `kgb/domains/biomedical/`):

| File | Action |
|------|--------|
| `extraction/prompt_open.md` | Create — open extraction prompt |
| `extraction/prompt_constrained.md` | Create — constrained prompt |
| `extraction/examples.json` | Create — few-shot examples |
| `augmentation/connectivity/prompt.md` | Create — augmentation prompt |
| `augmentation/connectivity/examples.json` | Create — augmentation examples |
| `schema.json` | Create — entity/relation types (optional) |

No Python files and no registry edits needed.

## Verification Checklist

- [ ] Directory structure matches layout above (`.md` extensions for prompts)
- [ ] Extraction prompts do NOT include format instructions (langextract handles that)
- [ ] `examples.json` includes `char_start`/`char_end` and `extraction_text`
- [ ] Augmentation folder per strategy (at least `connectivity/`)
- [ ] Domain appears in `kgb list domains` (set `KGB_DOMAINS_PATH` first if external)
- [ ] Optional `schema.json` with entity_types and relation_types
- [ ] Tests pass for discovery, resource loading, and schema
