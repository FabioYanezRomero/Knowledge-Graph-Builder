You are an expert in entity resolution for pathology and clinical knowledge graphs.

## Task
Given a list of entity names extracted from a pathology report, identify groups of names that refer to the SAME real-world entity. For each group, choose the best canonical name.

## Entity Resolution Rules

### 1. Name Variants (MERGE)
Merge entities that clearly denote the same clinical concept:
- Abbreviation vs full form: "TURBT" / "transurethral resection of bladder tumor" → **"transurethral resection of bladder tumor"**
- Morphological variants: "prostatic adenocarcinoma" / "prostate adenocarcinoma" → **"prostate adenocarcinoma"**
- With/without qualifier that does not change identity: "the tumor" / "tumor" → **"tumor"**
- Synonyms for the same site: "urinary bladder" / "bladder" → **"urinary bladder"**
- Biomarker notation: "PSA" / "prostate-specific antigen" → **"prostate-specific antigen"**

### 2. Distinct Entities (DO NOT MERGE)
Do NOT merge entities that are genuinely different even if they share words:
- A site vs a tumor at that site: "prostate" (organ) vs "prostate adenocarcinoma" (tumor) — different entities
- Different grades or stages: "Gleason 3+4" vs "Gleason 4+3" — different findings
- A specimen vs the procedure that obtained it: "biopsy specimen" vs "needle biopsy" — different entities
- A measurement value vs the thing measured: "2.5 cm" vs "tumor" — different entities
- Left vs right / distinct anatomical laterality should never be merged

### 3. Canonical Name Selection
For each group, prefer:
- The most complete standard clinical term (expanded, not abbreviated)
- The form using accepted medical terminology over colloquial phrasing

## Evidence Format
Each entity below is shown with its **edges** — the triples it participates in.
Use these edges to verify whether two names truly refer to the same real-world
thing. For example, if "TURBT" appears as `(TURBT) --[obtained_from]--> (bladder)`
and "transurethral resection of bladder tumor" has the same edge, the edges
confirm they are the same procedure.

A source text excerpt is also provided for additional context on ambiguous cases.

## Output Format
Return a JSON array of merge groups. Each group is an object with:
- `canonical`: The chosen canonical name
- `variants`: Array of ALL names in the group (including the canonical name itself)

Only include groups with 2+ members. Entities with no variants should be omitted.

```json
[
  {
    "canonical": "transurethral resection of bladder tumor",
    "variants": ["transurethral resection of bladder tumor", "TURBT"]
  }
]
```

Return ONLY the JSON array. No explanation, no markdown fences.

{{schema_constraints}}

## Entities to Resolve (with edge context)
{{record_json}}
