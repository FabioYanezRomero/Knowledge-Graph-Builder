# Cross-Document Consolidation: When It Helps, When It Corrupts

**Date:** 2026-07-05
**Domain:** Pathology | **Corpus:** 15 urology/pathology reports (distinct patients & procedures)
**Model:** gemma4:12b-mlx (Ollama, native, `think:false`) | **Mode:** Open | **Temperature:** 0.0

---

## Question

The consolidation stage merges entity/relation variants using a layered pipeline
(deterministic → fuzzy → LLM → filter). Today it runs **per document**. Should we
extend it to run **across documents** — and does that even make sense when the
documents are neither from the same patient nor the same topic?

**Short answer: not in general.** Blanket cross-document consolidation corrupts
the graph on its most common nodes. It only makes sense on a *subset* of
entities, and telling that subset apart safely needs something the current
fuzzy+LLM machinery does not have: entity typing.

---

## Background: the consolidation architecture

A `consolidate` pipeline step is a thin dispatcher; the real work lives in a
*strategy* that composes reusable *layers* (`kgb/builder/consolidation/layers/`):

| Layer | Role | Runs |
|-------|------|------|
| sieves (+ schwartz_hearst) | deterministic MERGE (exact/acronym, closed-set) | first |
| fuzzy | PROPOSE look-alike candidates (blocking, never merges) | — |
| LLM (`client.augment`) | DECIDE which candidates are the same referent | — |
| veto | deterministic FILTER (numeric/staging siblings) | last |
| guard | closed-set invariant (no invented entity survives) | last |

- `entity_resolution` = all five layers.
- `relation_resolution` = fuzzy + LLM + guard (relation labels have no source
  spans for sieves; the fuzzy veto already declines what veto would catch).

This design is sound: fuzzy is high-recall blocking, the LLM supplies semantic
precision, and the deterministic layers bound the damage. The problem discussed
here is **not the design — it is the scope.**

---

## Per-document scope has little to do

Within a single short report there are few name variants to merge. On report
`01_turbt`: extraction produced 35 nodes; entity_resolution merged exactly **1**;
relation_resolution merged **0**. The fuzzy+LLM machinery is mostly idle
per-document. The apparent opportunity is cross-document.

## The cross-document opportunity — and the trap

Across the 15 reports there are **329** unique entities (case-insensitive) and
**13** cross-document fuzzy-similar pairs (difflib ≥ 0.82, endpoints in different
documents):

**Genuinely the same concept (safe to merge):**
- `moderate trabeculation` ↔ `moderate trabeculations` (plural)
- `pathology report` ↔ `patholoogy report` (typo)
- `transurethral resection of prostate` ↔ `... of the prostate`
- `ct of the abdomen and pelvis` ↔ `ct scan of abdomen and pelvis`

**Surface-similar but must NOT merge (the fuzzy trap):**
- `41 grams` ↔ `45 grams` (different measurement)
- `no papillary tumors` ↔ `papillary tumor` (negation)
- `moderately differentiated` ↔ `poorly differentiated` (opposite grade)
- `left apex of prostate` ↔ `left lobe of prostate` (different anatomy)

The veto catches the numeric case; the LLM is needed for negation/anatomy/grade.
So far this argues *for* the layered design. But the 13-pair view understates the
real hazard, which is **exact matches**, not fuzzy ones.

## The real hazard: shared instance entities

Entities that appear in many of the (unrelated) reports:

| Appears in | Entity | Kind |
|:---:|--------|------|
| **14 / 15** | `patient` | instance |
| 7 | `cystoscopy` | type |
| 7 | `bladder` | type |
| 6 | `prostate` | type |
| 5 | `urethra`, `tumor` | type |
| 5 | `pathology report` | type (generic) |
| 4 | `adenocarcinoma` | type |
| 3 | `radical prostatectomy`, `psa`, `seminal vesicles`, `negative`, `hematuria` | mixed |

`patient` occurs in **14 of 15** reports. It is an **exact match** — the
deterministic sieve merges it with zero LLM involvement. Merging it
cross-document collapses 14 different people into **one super-patient node**
wired to every finding in the corpus. It would become the single most central
node in the graph, and it would be meaningless. The naive cross-document pass
does its **greatest damage on the most frequent entity**, silently, via the
highest-precision layer.

---

## The distinction that actually matters: instance vs. type

Cross-document consolidation is not one operation. Entities fall into two kinds:

- **Instance-level** — `patient`, `specimen`, `this biopsy`, `41 grams`,
  `negative`. Bound to a single document/patient. Merging across documents
  asserts a false identity.
- **Type / concept-level** — `cystoscopy`, `prostate`, `adenocarcinoma`, `PSA`,
  `radical prostatectomy`. Recur across patients as the same concept. Merging
  builds a useful concept/cohort graph ("how many reports show high-grade PIN?").

The layers that consolidate — exact-match sieve and fuzzy blocking — operate on
**surface form**, which does not carry this distinction. `patient` ↔ `patient` is
a perfect string match regardless of it being an instance. Surface similarity
cannot separate a recurring concept from a recurring instance word.

---

## Decision: gate cross-document consolidation on purpose and typing

**Whether to consolidate across documents depends on what the KG is for:**

| KG purpose | Cross-document consolidation |
|------------|------------------------------|
| Per-patient / clinical graph (each report = its own graph) | **No.** Keep per-document consolidation. Correct default for reports of different patients. |
| Corpus / cohort concept graph | **Only on type-level entities**, never on instances. |

**And safe cross-document consolidation requires entity typing (instance vs.
type), which cannot be derived from surface similarity.** That is ontology /
semantic-type territory (e.g. UMLS/SNOMED semantic types, or a typed NER pass) —
the same capability the augmentation stage is waiting on. Without it, a blanket
cross-document pass is **net-negative**: it corrupts the most common nodes
(`patient`, `specimen`) while fixing a handful of concept typos.

### Recommendation

1. **Do not build a generic cross-document consolidation pass now.** It is not a
   fuzzy+LLM problem; it is an entity-typing problem.
2. **Keep consolidation per-document.** Within a document, `patient` has one
   referent and merging is safe.
3. **When a corpus concept graph is genuinely needed**, gate cross-document
   merging on entity type (concept-only), sourced from an ontology — not from
   string or embedding similarity. This slots into the same future work as
   ontology-backed augmentation.

This is consistent with the project's standing "link, don't merge" default:
preserve distinctions unless there is positive evidence (here, a *type*
signal, not a *similarity* signal) that two mentions are the same thing.
