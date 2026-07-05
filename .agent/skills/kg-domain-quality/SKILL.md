---
name: kg-domain-quality
description: Quality practices for building kgb KG domains — grounding-only extraction, layered consolidation, connectivity that doesn't fabricate, and why cross-document merging needs entity typing. Use when authoring or tuning extraction/consolidation/augmentation prompts for any domain.
---

# KG Domain Quality Practices

Hard-won practices for producing a *correct* knowledge graph, not just a connected
one. `add-domain` covers the mechanical wiring (files, discovery); this skill
covers **what to put in the prompts and why**, so a new domain doesn't repeat
mistakes we already paid for. Each principle is backed by measured evidence in
`reports/`.

The pipeline has three distinct stages with three distinct jobs. Keep them
separate — the most common failure is overloading one stage with another's job.

```
extract      text → triples        GROUND only (say what the text says)
consolidate  triples → triples     MERGE variants (add no knowledge)
augment      triples → triples     ADD context edges (never merge)
```

## Principle 1 — Extraction is grounding ONLY

Extraction emits a `(head, relation, tail)` triple **only when the text states a
relationship** between two entities. The relation label may be a normalized form
of what the text says (e.g. "of the bladder" → `located_in`), but never invent a
relationship the text does not state.

- **Standalone entities are isolated nodes, not fabricated triples.** If an entity
  is salient but has no stated relation, emit it standalone
  (`{"head": "<entity>", "relation": "", "tail": ""}`) so it becomes an isolated
  node. Do NOT connect it with a made-up relation.
- **Never fabricate `is_type`/`located_in`/etc.** just to have an edge. If it
  isn't in the text, it belongs downstream (augment/ontology), not here.
- **Do NOT** put coreference, normalization, or connectivity instructions in the
  extraction prompt. Those are downstream stages.
- **Preserve provenance.** Grounded triples carry `extraction_text` + document
  offsets; keep them (they distinguish grounded from inferred edges).

Prompt rule to include verbatim (domain-agnostic):
> Emit a (head, relation, tail) triple ONLY when the text states a relationship
> between the two entities (the label may be a normalized form of what the text
> says); never invent a relationship the text does not state. If an entity has no
> stated relationship, emit it standalone with empty relation/tail.

Anti-pattern: a prompt that lists relation types the model should "use" invites
fabrication of those types. List them as *permitted when grounded*, not as a
menu to fill.

## Principle 2 — Consolidation is layered; fuzzy proposes, the LLM decides

A `consolidate` step composes reusable layers (`kgb/builder/consolidation/layers/`)
in this order — never collapse them into "ask the LLM to merge":

```
deterministic MERGE  (sieves: exact/acronym, closed-set)   high precision, first
  → fuzzy PROPOSE     (difflib look-alikes as CANDIDATES)    high recall, no merge
    → LLM DECIDE       (semantic judgment on the candidates)  precision
      → deterministic FILTER (veto numeric/staging, closed-set guard)  bounds errors
```

- **Fuzzy never merges** — it only surfaces candidate pairs for the LLM. A false
  fuzzy match becomes a question, not a bad merge, so a permissive threshold is
  safe.
- **The LLM is bracketed by deterministic layers**: sieves shrink its work, veto +
  closed-set guard bound its errors (it can never introduce an entity absent from
  the graph).
- `entity_resolution` uses all five layers; `relation_resolution` uses fuzzy + LLM
  + guard (relation labels have no source spans for sieves).
- Default is **"link, don't merge"** — preserve distinctions unless there is
  positive evidence two mentions are the same referent.

## Principle 3 — Augmentation must not fabricate to force connectivity

The connectivity strategy adds context edges, but its objective ("reduce
disconnected components") is dangerous if pushed too hard. A single document holds
several **independent** facts; it is not one connected thing.

Measured (`reports/` augment review, 15 pathology reports): the old prompt +
`max_disconnected: 1` produced **34%** vague/hub-attachment noise. Rewriting the
prompt + `max_disconnected: 3` cut it to **4%** with 0 failures.

Do:
- Set `max_disconnected` to a small number > 1 (e.g. 3). Do **not** force 1.
- In the connectivity prompt: "connect two entities only if the text supports a
  real relation; it is BETTER to leave components disconnected than to invent
  one." Allow an **empty array**.
- Forbid vague filler predicates (`related_to`, `associated_with`,
  `documented_in`, `has_finding`) and hub-attachment (`X documented_in
  "Pathology Report"`). These carry no knowledge.
- Add direction rules (a part `located_in` the whole, not reversed) and forbid
  synonymy edges (that's consolidation's job).

Don't: list vague relations or "allowed hub nodes" in the prompt — that is a
license to fabricate. (This was the exact bug in the old pathology prompt.)

## Principle 4 — Cross-document consolidation needs entity TYPING

Do **not** run consolidation across documents on surface similarity. Entities
split into two kinds, and surface form (exact/fuzzy) cannot tell them apart:

- **Instance-level** (`patient`, `specimen`, `this biopsy`, `41 grams`) — bound to
  one document. Merging across documents asserts false identity. In one corpus
  `patient` appeared in 14/15 reports as an exact match → merging collapses 14
  people into one meaningless super-node.
- **Type/concept-level** (`cystoscopy`, `adenocarcinoma`, `PSA`) — recur across
  documents as the same concept; merging is useful.

Safe cross-document merging requires an entity-typing signal (ontology / semantic
types), not similarity. Until you have that, keep consolidation **per-document**.
Full analysis: `reports/cross_document_consolidation.md`.

## Prompt-writing checklist (any domain)

- [ ] Extraction prompt says "ground only; standalone if no relation; never invent"
- [ ] Extraction prompt does NOT contain normalization/coref/connectivity rules
- [ ] Extraction relation list (if any) framed as "permitted when grounded", not a menu
- [ ] Connectivity prompt forbids fillers + hub nodes, allows empty array, "leave disconnected rather than invent"
- [ ] `max_disconnected` > 1
- [ ] No cross-document consolidation without an entity-typing signal
- [ ] Treat augment output as low-trust (`inference: contextual`, no provenance)
```
