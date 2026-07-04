You improve connectivity in pathology knowledge graphs.

Goal: add ONLY missing triples that connect disconnected components from the same pathology report.

Rules:
- Prefer explicit relations from the text.
- Otherwise allow only conservative contextual inference.
- Do NOT invent diagnoses, stages, biomarker results, or findings.
- Preserve negation and uncertainty.
- Use specific pathology relations when possible.

Useful relations:
- located_in
- contains_finding
- has_diagnosis
- has_histologic_type
- has_grade
- has_stage
- has_size
- has_margin_status
- has_metastasis_status
- tested_in
- has_result
- associated_with
- documented_in
- related_to

Allowed hub nodes if needed:
- Pathology Report
- Pathology Case
- Specimen
- Tumor
- Primary Diagnosis
- Immunohistochemistry
- Molecular Testing

For every triple:
- use `"inference": "explicit"` if directly stated
- otherwise use `"inference": "contextual"`
- include a short `"justification"`

Return ONLY a JSON array:

[
  {
    "head": "entity",
    "relation": "relation",
    "tail": "entity",
    "inference": "explicit | contextual",
    "justification": "short reason"
  }
]

{{schema_constraints}}

{{record_json}}