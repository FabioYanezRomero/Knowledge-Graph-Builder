# Best Practices

Practices for producing a *correct* knowledge graph, not just a connected one.
Every claim here is backed by a measured experiment on this repo's corpora.

If you read one thing, read this:

> **One stage, one job. Do not ask a single prompt to extract, normalize and
> connect at the same time.**

---

## The rule

The pipeline has three LLM stages with three distinct, non-overlapping jobs:

```
extract      text → triples        GROUND only   (say what the text says)
consolidate  triples → triples     MERGE only    (add no knowledge)
augment      triples → triples     ADD only      (never merge)
```

Each stage's prompt should be *unable* to do another stage's job. Not
"discouraged from" — unable, because the instruction simply is not there.

## Why: three times we broke this rule, and what it cost

The single most expensive class of bug in this project was a stage quietly doing
a second job. It happened three times, in three different stages, and each time
the symptom appeared *downstream* of the real cause.

### 1. Extraction that also typed entities

The extraction prompt tried to be helpful: rather than leave a salient entity
dangling, it invented a typing triple (`X is_type SomeCategory`) to attach it.

- **Symptom:** edges with `char_start: None` whose relation and category text
  appeared nowhere in the source. Augmentation then faithfully *amplified* the
  pattern into ungrounded hypernym edges across the whole graph.
- **Where the blame landed:** on augment. Augment was innocent — it was copying
  a pattern extraction had seeded.
- **Fix:** emit a triple **only** when the relation is named in the text. A
  salient entity with no stated relation becomes a **standalone entity**
  (`{"head": "<entity>", "relation": "", "tail": ""}`) and travels through the
  pipeline as an **isolated node**.

An isolated node is a truthful statement: *"this entity was mentioned but not
related to anything here."* A fabricated edge is a lie, and every later stage
will amplify it.

### 2. Consolidation that also invented entities

Consolidation is defined as "merge variants, add no knowledge". Asked to resolve
`J. Smith`, the LLM happily returned the canonical `John Smith Jr.` — a name
that **did not exist anywhere in the graph**. A merge step had silently become
an entity-creation step.

- **Fix:** the **closed-set guard** (`kgb/builder/consolidation/layers/guard.py`).
  A rewrite whose canonical target is not an existing entity is rejected, because
  it would introduce a node with no extraction grounding.
- The invariant is enforced in code, not requested in the prompt. Prompts are
  advisory; guards are not.

This is the general shape of the fix: **the LLM proposes, deterministic code
decides.** Consolidation layers run in this order, and never collapse into "ask
the LLM to merge":

```
deterministic MERGE  (sieves: exact/acronym)         high precision, first
  → fuzzy PROPOSE     (look-alikes as CANDIDATES)     high recall, never merges
    → LLM DECIDE       (semantic judgment)             precision
      → deterministic FILTER (veto + closed-set guard) bounds the errors
```

Fuzzy matching **never merges** — it only surfaces candidates for the LLM to
judge, so a false fuzzy match becomes a question rather than a bad merge.

### 3. Augmentation that also forced connectivity

The connectivity prompt listed vague fillers as *"useful relations"*
(`related_to`, `associated_with`, `documented_in`) and licensed *"hub nodes"*
(`Pathology Report`, `The Legal Case`). It ran with `max_disconnected: 1`, which
**forces** every document into one connected component.

Together, those leave the model no legal move except to attach every orphan to a
document-level hub through a vacuous edge. The prompt was not failing to prevent
the noise — it was **requesting** it.

Measured (augment-only A/B on the same consolidated graphs, so the prompt is the
only variable):

| Domain | Old prompt (`max_disconnected: 1`) | New prompt (`max_disconnected: 3`) |
|---|---:|---:|
| Pathology (15 reports) | 34% noise | **4%** |
| Legal (15 records) | 89% noise | **24%**, hub-attachment eliminated (0) |

**Fix:** forbid the fillers and hub-attachment by name; reframe the goal from
"connect everything" to *"connect only when the text supports it"*; allow an
empty array as a correct answer; raise `max_disconnected` above 1.

A knowledge graph that is honestly fragmented is worth more than one that is
densely wrong: every fabricated edge is a false fact a downstream consumer will
trust.

---

## Principles

### 1 — Extraction is grounding ONLY

Emit `(head, relation, tail)` **only when the text states a relationship**. The
relation label may be a normalized form of what the text says ("of the bladder"
→ `located_in`), but never invent a relationship the text does not state.

- Standalone entities are isolated nodes, not fabricated triples.
- Do **not** put coreference, normalization or connectivity rules in the
  extraction prompt. Those are downstream stages.
- Preserve provenance: grounded triples carry `extraction_text` + document
  offsets. Their absence is the signal that a triple was inferred, not read.

Prompt rule worth including verbatim in any domain:

> Emit a (head, relation, tail) triple ONLY when the text states a relationship
> between the two entities (the label may be a normalized form of what the text
> says); never invent a relationship the text does not state. If an entity has no
> stated relationship, emit it standalone with empty relation/tail.

Anti-pattern: listing relation types the model should "use" invites fabrication
of exactly those types. List them as *permitted when grounded*, never as a menu
to fill.

### 2 — Consolidation is layered; fuzzy proposes, the LLM decides

See case 2 above for the layer order. Also:

- `entity_resolution` uses all five layers; `relation_resolution` uses fuzzy +
  LLM + guard (relation labels have no source spans, so sieves don't apply).
- Default is **"link, don't merge"** — preserve distinctions unless there is
  positive evidence that two mentions share a referent.

### 3 — Augmentation must not fabricate to force connectivity

- Set `max_disconnected` to a small number **greater than 1** (e.g. 3).
- Forbid vague filler predicates and hub-attachment explicitly, by name.
- Add direction rules (a part is `located_in` the whole, not the reverse) and
  forbid synonymy edges — that is consolidation's job.
- Treat augment output as low-trust: `inference: contextual`, no provenance.

### 4 — Cross-document consolidation needs entity TYPING

Do **not** consolidate across documents on surface similarity. Entities split
into two kinds that surface form cannot tell apart:

- **Instance-level** (`patient`, `specimen`, `41 grams`) — bound to one document.
  In one corpus `patient` matched exactly across 14 of 15 reports; merging would
  collapse 14 different people into one meaningless super-node.
- **Type-level** (`cystoscopy`, `adenocarcinoma`, `PSA`) — genuinely the same
  concept across documents; merging helps.

Until you have an entity-typing signal, keep consolidation **per document**.

---

## Two limits worth knowing before you start

**The prompt-obedience ceiling.** Even a rewritten prompt leaves residual noise
(~4% pathology, ~24% legal) because the model evades the forbid-list through
morphological variants: forbid `related_to` and it emits `relates to`; forbid
`subject_of` and it emits `subject of`. String-for-string prohibition cannot
close the lexical neighbourhood around a term. Past that ceiling the levers are
deterministic filtering and typed vocabularies — not a longer prompt.

**Harden your metric before you trust it.** An early measurement reported the new
connectivity prompt at 0% noise. The prompt was not perfect; the *detector* was
lenient — it matched `related_to` but missed `relates to`, `are_type_of`,
`subject of`. Broadening its vocabulary corrected the figure to 24%. A lenient
metric will happily certify a noisy graph as clean.

---

## Checklist for a new domain

- [ ] Extraction prompt says "ground only; standalone if no relation; never invent"
- [ ] Extraction prompt contains **no** normalization / coreference / connectivity rules
- [ ] Relation list (if any) framed as "permitted when grounded", not as a menu
- [ ] Connectivity prompt forbids fillers + hub nodes by name, and allows an empty array
- [ ] `max_disconnected` > 1
- [ ] No cross-document consolidation without an entity-typing signal
- [ ] `kgb domain lint <name>` passes
