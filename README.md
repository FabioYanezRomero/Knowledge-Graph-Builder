# Knowledge Graph Builder (KGB)

A modular system for extracting knowledge graphs from text using multiple LLM backends. Supports **Gemini API**, **Ollama**, and **LM Studio** with a unified client abstraction layer.

## Features

- **Multi-Backend LLM Support** — Gemini API (cloud), Ollama (local), LM Studio (local)
- **Knowledge Graph Extraction** — Structured triples (head-relation-tail) with source grounding via [langextract](https://github.com/langextract/langextract); entities with no stated relation are kept as isolated nodes rather than fabricated edges
- **Graph Consolidation** — Merge/clean operations that add no knowledge (entity and relation resolution), layered so that fuzzy matching proposes and deterministic guards decide
- **Graph Augmentation** — Strategies that add new contextual triples (connectivity bridging)
- **Origin Tracking** — Every triple tagged as `explicit` (extracted) or `contextual` (augmented)
- **Interactive Visualizations** — Cytoscape.js interactive network graphs with node dragging, search/filter, and context menus; entity text highlighting
- **Domain System** — Customizable prompts, examples, and schema constraints per knowledge domain
- **Pipeline Orchestration** — YAML-driven or flag-based multi-step pipelines
- **Multiple I/O Formats** — JSONL, JSON, CSV input; GraphML output

---

## Quick Start

### Installation

```bash
# Using Makefile (recommended)
make install

# Or manually
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires **Python 3.11+**.

`make install` auto-detects the first available interpreter that is already
`Python 3.11+` among common commands such as `python3.13`, `python3.12`,
`python3.11`, `python3`, and `python`.
If your supported interpreter lives under a different name or path, use
`make install PYTHON=/path/to/python3.11`.

### Prerequisites

| Backend | Requirement |
|---------|-------------|
| Gemini | API key from [Google AI Studio](https://aistudio.google.com/app/apikey) — set `GOOGLE_API_KEY` in `.env` |
| Ollama | [Ollama](https://ollama.ai/) running locally (`ollama serve`) |
| LM Studio | [LM Studio](https://lmstudio.ai/) with server enabled and model loaded |

### Basic Usage

```bash
# Interactive mode (REPL)
kgb

# One-shot extraction
kgb extract --input data.jsonl --domain legal --client gemini

# Full pipeline via script
bash scripts/test_single_extraction_gemini.sh
```

---

## Pipeline Steps

```
Text → Extract → Consolidate → Augment → Convert → Visualize
         ↓            ↓            ↓         ↓          ↓
       JSON      JSON (merged)   JSON+    GraphML     HTML
     (explicit)  (adds nothing) (contextual)
```

Each LLM stage has **one** job, and its prompt must not do another stage's:

| Stage | Job | Must never |
|---|---|---|
| `extract` | ground what the text states | invent a relation to connect a lonely entity |
| `consolidate` | merge name variants | add knowledge or introduce a new entity |
| `augment` | add contextual edges | merge entities, or fabricate to force connectivity |

Overloading one stage with another's job is the most expensive bug class in this
project — it cost 34–89% noisy edges before it was found. **Read
[BEST_PRACTICES.md](BEST_PRACTICES.md) before writing prompts for a new domain.**

### Step 1: Extract Triples

Extracts source-grounded triples using langextract. Each triple carries the exact
source span (`extraction_text`) and its character offsets in the original text.

A salient entity with **no relation stated in the text** is emitted as a
standalone entity and becomes an isolated node — extraction never invents a
relation just to connect it.

```bash
kgb extract \
  --input data/legal/legal_background.jsonl \
  --domain legal \
  --client gemini \
  --output-dir outputs/run
```

### Step 2: Consolidate (merge variants, add no knowledge)

Merges surface variants of the same entity (`"Salvino"`, `"Defendant Salvino"` →
`"Michael J. Salvino"`). Layered so that no single fallible component makes an
irreversible merge: deterministic sieves → fuzzy *candidates* → LLM judgment →
deterministic veto + closed-set guard. A canonical name that does not already
exist in the graph is rejected.

```bash
kgb consolidate entity_resolution \
  --input data/legal/legal_background.jsonl \
  --domain legal \
  --client gemini \
  --output-dir outputs/run
```

`relation_resolution` (normalizing relation labels) is currently available
through the YAML pipeline only.

### Step 3: Augment Connectivity

Generates bridging triples (tagged `contextual`) to connect disconnected graph components.

```bash
kgb augment connectivity \
  --input data/legal/legal_background.jsonl \
  --domain legal \
  --client gemini \
  --output-dir outputs/run \
  --max-disconnected 3 \
  --max-iterations 3
```

> **Do not set `--max-disconnected 1`.** A document holds several independent
> facts; forcing one connected component leaves the model no legal move except
> to invent hub/filler edges. Measured cost of forcing it: 34% noisy edges in
> pathology, 89% in legal. Leaving a graph honestly fragmented is the correct
> outcome.

### Step 4: Convert to GraphML

```bash
kgb convert --input outputs/run/extracted_json --output outputs/run/graphml
```

### Step 5: Visualize

```bash
# Network topology (nodes colored by origin: Extracted/Augmented/Both)
kgb visualize network --input outputs/run/graphml --output outputs/run/network_viz

# Entity highlighting in source text
kgb visualize extraction \
  --input data/legal/legal_background.jsonl \
  --triples outputs/run/extracted_json \
  --output outputs/run/extraction_viz
```

### Full Pipeline (YAML)

```bash
kgb run-pipeline --config kgb/pipeline/configs/legal_ollama.yaml
```

Or with flags:

```bash
kgb run-pipeline --input data.jsonl --domain legal --client ollama \
  --extract --augment --convert --visualize
```

In YAML mode, each `extract`/`augment` step can override the default client —
use a cheap model for high-volume extraction and a stronger one where
hallucination risk matters:

```yaml
client:            # default for all steps
  type: ollama
  model: gemma3:27b

steps:
  - extract:
      client:
        model: gemma3:1b   # only the listed fields are overridden
  - augment
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `kgb extract` | Extract knowledge graph triples from text |
| `kgb augment connectivity` | Bridge disconnected graph components |
| `kgb consolidate entity_resolution` | Merge entity name variants (adds no triples) |
| `kgb convert` | Convert JSON triples to GraphML (`--merge` for one cross-document graph) |
| `kgb visualize network` | Interactive network graph (Cytoscape.js) |
| `kgb visualize extraction` | Entity highlights in source text (langextract) |
| `kgb run-pipeline` | Run multi-step pipeline (YAML or flags) |
| `kgb list domains` | List available knowledge domains |
| `kgb list clients` | List registered LLM clients |
| `kgb list pipelines` | List built-in YAML pipeline configs |
| `kgb domain lint <name\|path>` | Validate a domain's prompts, examples, and schema |

Common options (most commands):

| Option | Description |
|--------|-------------|
| `--input, -i` | Input file (JSONL, JSON, CSV, TXT) or a directory of `.txt` files |
| `--output-dir, -o` | Output directory |
| `--domain, -d` | Knowledge domain (`default`, `legal`, `pathology`, or a path) |
| `--client, -c` | LLM backend (`gemini`, `ollama`, `lmstudio`) |
| `--model` | Model identifier (uses provider default if omitted) |
| `--mode, -m` | Extraction mode (`open`, `constrained`) |
| `--record-ids` | Filter specific record IDs |
| `--temp` | LLM temperature (default: 0.0) |
| `--workers` | Max parallel workers |
| `--timeout` | Request timeout in seconds |

---

## Output Structure

Each run produces a timestamped directory:

```
test_outputs/single_extraction_20260318_101048/
├── metadata.json          # Run configuration and timestamp
├── extracted_json/        # JSON triples (explicit + contextual)
│   └── UKSC-2009-0143.json
├── graphml/               # NetworkX-compatible GraphML
│   └── UKSC-2009-0143.graphml
├── network_viz/           # Interactive Cytoscape.js HTML
│   └── UKSC-2009-0143.html
└── extraction_viz/        # Entity highlighting HTML
    └── UKSC-2009-0143.html
```

### Triple Format

```json
[
  {
    "head": "Sigma Finance Corporation",
    "relation": "entered",
    "tail": "receivership",
    "inference": "explicit",
    "justification": null,
    "extraction_text": "Sigma Finance Corporation entered receivership",
    "char_start": 1240,
    "char_end": 1286
  },
  {
    "head": "financial markets",
    "relation": "impacted",
    "tail": "Sigma Finance Corporation",
    "inference": "contextual",
    "justification": "The text states the impact on financial markets...",
    "extraction_text": null,
    "char_start": null,
    "char_end": null
  },
  {
    "head": "structured investment vehicle",
    "relation": "",
    "tail": ""
  }
]
```

- `inference: "explicit"` — Directly extracted from text with source grounding
- `inference: "contextual"` — Inferred during augmentation to bridge components
- **Provenance is the trust signal.** A grounded triple carries `extraction_text`
  and document offsets; `null` offsets mean the triple was inferred, not read.
  Treat inferred triples as low-trust.
- **Empty `relation`/`tail` is a standalone entity**, not a broken triple — an
  entity the text mentions without relating to anything. It becomes an isolated
  node in the GraphML.

---

## Architecture

```
kgb/
├── __main__.py              # Typer CLI + interactive REPL
├── builder/                 # Graph construction logic
│   ├── extraction.py        # Triple extraction (uses langextract)
│   ├── strategies.py        # Strategy registry (@register_strategy, augment/consolidate)
│   ├── augmentation.py      # connectivity strategy (adds triples)
│   ├── consolidation/       # Merge/clean strategies (add no triples)
│   │   ├── entity_resolution.py
│   │   ├── relation_resolution.py
│   │   └── layers/          # Reusable: sieves, schwartz_hearst, fuzzy, veto, guard
│   └── validation.py        # Schema validation + prompt rendering
├── clients/                 # LLM client abstraction
│   ├── base.py              # BaseLLMClient (extract + augment interface)
│   ├── config.py            # ClientConfig dataclass
│   ├── factory.py           # ClientFactory + @client() decorator
│   ├── defaults.py          # Provider defaults loader
│   ├── configs/             # Provider default JSON files
│   └── providers/           # Implementations
│       ├── gemini.py        # Google Gemini (native SDK)
│       ├── ollama.py        # Ollama (OpenAI-compatible)
│       └── lmstudio.py      # LM Studio (OpenAI-compatible)
├── domains/                 # Knowledge domain resources
│   ├── base.py              # KnowledgeDomain + DomainComponent
│   ├── registry.py          # @domain() decorator + registry
│   ├── models.py            # Triple, InferenceType, DomainSchema
│   ├── lint.py              # Domain validator (kgb domain lint)
│   ├── default/             # Generic domain
│   ├── legal/               # Legal domain (prompts, examples, schema)
│   └── pathology/           # Pathology/clinical domain (prompts, examples, schema)
├── eval/                    # Offline consolidation benchmark (FP/FN, no LLM)
├── io/                      # Input/output handling
│   ├── readers/             # JSONL, JSON, CSV loaders
│   └── writers/             # GraphML converter
├── visualization/           # HTML visualization engines
│   ├── graph_viz.py         # Cytoscape.js network graphs (origin coloring)
│   └── text_viz.py          # langextract entity highlighting
└── pipeline/                # Pipeline orchestration
    ├── runner.py            # PipelineRunner
    ├── context.py           # PipelineContext
    ├── config.py            # YAML config loader
    ├── steps/               # Pipeline step implementations
    └── configs/             # Built-in YAML pipeline configs
```

### Full System Architecture

![KGB Architecture](images/Architecture.png)

The system is orchestrated by the **Pipeline**, driven by the **CLI**. I/O Readers load input data, the **Client** communicates with LLM backends, the **Builder** manages extraction/augmentation logic, and the **Domain** provides prompts, examples, and schema constraints. I/O Writers produce GraphML output and the **Visualization** module generates interactive HTML views.

### Builder Architecture

![KGB Builder](images/Builder.png)

The Builder module coordinates extraction and augmentation. **Extract** uses langextract for source-grounded triples; **Augment** generates bridging triples via direct LLM inference. Both rely on the **Client** abstraction (BaseLLMClient / ClientFactory) and the **Domain** system (KnowledgeDomain / DomainRegistry) for prompts, few-shot examples, and entity/relation type constraints. The **Validation** subsystem normalizes and validates triples against schema constraints.

### Extensibility

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                       KGB Extension Points                          │
  │                                                                     │
  │  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐  │
  │  │  I/O Readers  │  │  I/O Writers  │  │     Visualization       │  │
  │  │  add-dataset- │  │ add-converter │  │   add-visualization     │  │
  │  │  format       │  │               │  │                         │  │
  │  └───────┬───────┘  └───────┬───────┘  └────────────┬────────────┘  │
  │          │                  │                        │               │
  │          ▼                  ▼                        ▼               │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │                    Pipeline Runner                            │   │
  │  │            load → build → convert → visualize                │   │
  │  └─────────────────────┬────────────────────────────────────────┘   │
  │                        │                                            │
  │                        ▼                                            │
  │  ┌──────────────────────────────────────────────────────────────┐   │
  │  │                        Builder                               │   │
  │  │          extract_triples() → augment_triples()               │   │
  │  └───────┬────────────────┬───────────────────────┬─────────────┘   │
  │          │                │                       │                 │
  │          ▼                ▼                       ▼                 │
  │  ┌──────────────┐  ┌─────────────┐  ┌──────────────────────────┐   │
  │  │  LLM Clients │  │   Domains   │  │  Augmentation Strategies │   │
  │  │add-llm-client│  │ add-domain  │  │  add-augmentation-       │   │
  │  │              │  │             │  │  strategy                 │   │
  │  └──────────────┘  └─────────────┘  └──────────────────────────┘   │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

| Pattern | Where | Mechanism |
|---------|-------|-----------|
| **Factory + Registry** | Clients | `@client("name")` decorator → `ClientFactory.create(config)` |
| **Registry** | Domains | `@domain("name")` decorator → `get_domain("name")` |
| **Strategy + Registry** | Augmentation | `@register_strategy("name")` → `augment_triples(strategy="name")` |
| **Component** | Domains | `DomainComponent` lazy-loads prompt + examples per activity |

### Client Interface

All LLM backends implement two core methods:

| Method | Purpose | Source grounding | Used by |
|--------|---------|-----------------|---------|
| `extract()` | Extract triples from text | Yes (char positions) | `builder/extraction.py` |
| `augment()` | Generate bridging triples | No | `builder/augmentation.py` |

```python
from kgb.clients import ClientFactory, ClientConfig

config = ClientConfig(client_type="ollama", model_id="gemma3:1b")
client = ClientFactory.create(config)

# Source-grounded extraction
triples = client.extract(text="...", prompt_description="...")

# Inference-based augmentation (no char positions)
bridges = client.augment(text="...", prompt_description="...", format_type=Triple)
```

### Domain System

A domain is just a **directory of resources** — no Python code required. The
mechanics are below; for what to actually *write* in those prompts, see
[BEST_PRACTICES.md](BEST_PRACTICES.md) — it exists so a new domain doesn't repeat
mistakes already paid for. Validate with `kgb domain lint <name|path>`.

Any directory with this layout is a valid domain:

```
my_usecase/
├── extraction/
│   ├── prompt_open.md           # Open extraction prompt
│   ├── prompt_constrained.md    # Constrained extraction prompt
│   └── examples.json            # Few-shot extraction examples
├── consolidation/               # Merge/clean strategies (add no knowledge)
│   └── entity_resolution/
│       ├── prompt.md
│       └── examples.json
├── augmentation/                # Strategies that add new triples
│   └── connectivity/
│       ├── prompt.md
│       └── examples.json
└── schema.json                  # Entity/relation type constraints
```

Strategy folders are resolved under both `consolidation/` and `augmentation/`,
so the split is taxonomy, not a loading contract — existing domains that keep
everything under `augmentation/` continue to work.

Domains are resolved in this order:

1. Classes registered via `@domain("name")` (only needed for custom Python behavior)
2. Directories shipped with the package (`kgb/domains/<name>/`)
3. Directories under any root listed in the `KGB_DOMAINS_PATH` environment variable — keep your use-case domains in your own project, no fork or PR needed
4. A direct path: `--domain ./my_domains/finance`

```bash
# Use your own domains without touching this repo
export KGB_DOMAINS_PATH=~/my-kg-domains
kgb extract --input data.jsonl --domain finance --client ollama

# Or point at a domain directory directly
kgb extract --input data.jsonl --domain ./domains/finance --client ollama
```

```python
from kgb.domains import get_domain, list_available_domains

print(list_available_domains())  # ['default', 'legal', ...]

domain = get_domain("legal", extraction_mode="open")
prompt = domain.extraction.prompt
examples = domain.extraction.examples
schema = domain.schema  # DomainSchema with entity_types, relation_types
```

---

## LLM Clients

| Client | Type | Default Model | Setup |
|--------|------|---------------|-------|
| `gemini` | Cloud API | gemini-flash-latest | Set `GOOGLE_API_KEY` in `.env` |
| `ollama` | Local | llama3.1 | `ollama serve` + `ollama pull <model>` |
| `lmstudio` | Local | (loaded model) | Start LM Studio server on port 1234 |

### Environment Variables

```bash
# .env file (auto-loaded)
GOOGLE_API_KEY=your-gemini-api-key

# Or export directly
export GOOGLE_API_KEY="your-key"
```

---

## Visualization

### Network Graph

- Nodes colored by origin: **blue** (Extracted), **amber** (Augmented), **violet** (Both)
- Augmented edges rendered with dashed lines
- Layouts: cose (force-directed), circle, dagre (hierarchical) — switchable in-browser
- Node dragging, search/filter bar, right-click context menus
- Path finder, export (PNG/SVG/JSON)
- Dark mode support
- Hover tooltips with node degree, origin, and edge attributes

### Entity Highlighting

- Source text with color-coded entity spans
- Animated highlight transitions
- Grouping by entity type or relation
- Augmented entities visually distinguished

---

## Extensibility

The main extension points are in the codebase itself:

- Add providers under `kgb/clients/providers/` with defaults in `kgb/clients/configs/`
- Add domains as plain directories — in your own project via `KGB_DOMAINS_PATH` or `--domain <path>` (preferred), or under `kgb/domains/<name>/` for domains shipped with the package
- Add augmentation strategies in `kgb/builder/augmentation.py`
- Add readers under `kgb/io/readers/`
- Add writers under `kgb/io/writers/`
- Add visualizations under `kgb/visualization/`

---

## Testing

### With Scripts

```bash
# Gemini (requires API key)
bash scripts/test_single_extraction_gemini.sh

# Ollama (requires local server)
bash scripts/test_single_extraction_ollama.sh

# LM Studio (requires local server)
bash scripts/test_single_extraction_lmstudio.sh
```

Configure hyperparameters at the top of each script (model, temperature, record IDs, etc.).

### Quick Validation

```bash
kgb list domains
kgb list clients
kgb extract --input data/legal/legal_background.jsonl --domain legal --client ollama --model gemma3:1b --record-ids UKSC-2009-0143
```

---

## Docker

```bash
# Build
make docker-build

# Smoke test the installed CLI in the image
docker run --rm kg-constructor list clients

# Interactive session
make docker-start

# Background dev container
make docker-dev
make docker-stop
```

The image installs the packaged CLI. Repository helper scripts and local datasets are
meant to be run from a mounted checkout, for example via `make docker-start`.

---

## License

See LICENSE file for details.
