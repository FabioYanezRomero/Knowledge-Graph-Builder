# Legal Domain Experiment: Do the KG Quality Practices Transfer?

**Date:** 2026-07-06
**Domain:** Legal (UK Supreme Court case backgrounds) | **Corpus:** 15 of 77 records
**Model:** gemma4:12b-mlx (Ollama, native, `think:false`, `workers:1`) | **Mode:** Open | **Temperature:** 0.0

---

## Purpose

The extraction/consolidation/augmentation practices in `kg-domain-quality` and the
connectivity fix (`reports/`… pathology) were developed on pathology reports. This
experiment tests whether they **transfer to a structurally different domain** —
legal case law — by applying them to the `legal` domain and measuring the same
things we measured for pathology.

Headline: **they transfer, and the connectivity fix transfers more strongly than
in pathology.** Two new domain-specific findings surfaced (provenance loss on
paraphrased spans; the prompt-obedience ceiling), neither of which invalidates the
practices.

---

## 1. Extraction robustness transfers cleanly

The full pipeline (extract → consolidate → relation_resolution → augment → export
→ convert) ran **15/15 with 0 errors**. The robustness stack built for pathology
(serialize `workers:1`, native provider `think:false`, JSON salvage, key-casing
canonicalization, non-scalar coercion, retry-with-reset) carried over to a new
domain with no changes. No wedges, no parse failures.

## 2. Grounding quality is strong — but provenance is thin

Manual review of two cases (Sigma Finance SIV insolvency; an ECRC/HRA privacy
claim) found the grounding **semantically excellent**: no fabrication, correct and
specific legal relations (`secured_under`, `failed_to_meet`, `construed`,
`dispute_over`, `governed_by`), correct direction. Grounding-only held — nothing
invented.

**But provenance coverage is poor: 95 / 197 grounded triples (48%) have no
character offset.** Cause: the model **paraphrases/reassembles** the
`extraction_text` instead of quoting a verbatim span (e.g. emits "Sigma Finance
Corporation established to invest in…" when the text reads "…vehicle (SIV)
established to invest in…"). Our re-anchoring is an exact `text.find(span)`, which
correctly returns `None` rather than fabricate an offset — but that discards even
langextract's fuzzy alignment (we saw `MATCH_FUZZY` in the logs). Legal prose is
paraphrased far more than pathology, so the loss is much higher here.

- The triples are still **correct**; only provenance is lost.
- **Fix (candidate):** re-anchor with fuzzy matching (best approximate substring)
  instead of exact `find()`. Recovers most of the 48% without fabricating offsets.

Secondary observation (not a failure): legal predicates are hyper-specific,
sentence-like (`established_to_invest_in`, `has_insufficient_assets_to_pay`). They
are faithful to the text but barely reusable, so `relation_resolution` has little
to merge — normalization to canonical predicates is downstream/ontology work.

## 3. Connectivity A/B — the practices transfer, more strongly than pathology

A/B on the 15 consolidated graphs (same input), old connectivity prompt vs new:

| | Augmented edges | Vague+taxo+hub noise | % noise |
|---|---:|---:|---:|
| **OLD prompt** (+`max_disconnected:1`) | 74 | 66 | **89%** |
| **NEW prompt** (+`max_disconnected:3`) | 33 | 8 (5 vague, 3 taxo, **0 hub**) | **24%** |

- The legal *old* prompt was the worst offender in the codebase ("you MUST INFER a
  logical relationship", "be aggressive with Hub Nodes: The Legal Case, Financial
  Framework…") — hence **89%** noise, far above pathology's 34%.
- The new prompt **eliminated hub-attachment entirely (0)**, halved the fabricated
  edge count, and left legitimately-separate fact clusters disconnected (several
  cases got 0 bridging edges — correct).
- It still adds **real** edges where the text supports them (25 of 33 clean, e.g.
  `Relevant Terms governed_by UTCCR 1999`, `Seigneur is_member_of Chief Pleas`),
  so it is not merely conservative.

## 4. The prompt-obedience ceiling (honest correction)

An early measurement mis-reported the new prompt as **0%** noise; a stricter noise
vocabulary corrected this to **24%**. The residual is the LLM **disobeying the
forbid-list via morphological variants**: the prompt forbids `subject_of` /
`related_to`, but the model emits `subject of`, `relates to`, `is a type of`,
`are_type_of`. Prompt wording cannot fully close this — it is the known ceiling of
prompt-only control.

## 5. Does `relation_resolution` clean the residual noise? (measured)

No. Running `relation_resolution` on the **post-augment** graphs:

- merged 12 of 190 unique relation labels (190 → 178) — modest normalization;
- of the **6** vague/taxo labels present, it merged **0**.

`relation_resolution` normalizes *synonymous label variants*; it neither deletes
vague edges nor knows that `subject_of`/`related_to` are low-value. It is the wrong
tool for this noise. (Also note: in the shipped step order it runs *before*
augment, so it never even sees augment's output.)

## Recommendations

1. **Deterministic post-augment predicate filter (veto pattern).** The residual
   24% is exactly the forbidden fillers in morphological disguise. Normalize each
   augmented edge's predicate and **drop** it if it maps to a known filler
   (`related_to`/`subject_of`/`associated_with`/…). This mirrors consolidation's
   "LLM proposes, deterministic layer filters" and catches what the prompt can't.
   Cheap, no LLM, high precision.
2. **Fuzzy re-anchoring for provenance.** Recover the 48% of legal offsets lost to
   paraphrased spans; keep `None` only when no good approximate match exists.
3. **Taxonomy (`are_type_of`, `is a type of`) → ontology.** These are the
   is-a-kind-of edges that need a typed source, consistent with the standing
   augment-ontology decision.

## Conclusion

The quality practices transfer to a very different domain: extraction robustness
is unchanged (15/15), grounding is faithful, and the connectivity rewrite cut
noise 89% → 24% while eliminating hubs entirely. The transferable lesson holds
sharply — **the prompt kills hub-attachment and the forced-connectivity objective;
the residual fine-grained lexical/taxonomic noise is a downstream, deterministic,
and ontology problem, not a bigger prompt.** Two domain-specific gaps (paraphrased-
span provenance loss; a deterministic augment filter) are the concrete next levers.
