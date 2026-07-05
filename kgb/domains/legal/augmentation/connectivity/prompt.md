You improve connectivity in a legal-case knowledge graph.

Goal: add a bridging triple ONLY when the text actually supports a real relation
between two entities that are currently in different components. It is BETTER to
leave components disconnected than to invent a relation to connect them. A case
background legitimately contains several independent facts (parties, instruments,
events, holdings); not everything is related. Returning an EMPTY array is a valid,
common answer.

Hard rules:
- Only connect two entities if the text states or clearly implies a specific
  relation between them. If it does not, leave them disconnected.
- NEVER attach an entity to a generic hub node (e.g. "The Legal Case", "Financial
  Framework", "Corporate Structure") just to reduce components. Anchoring
  everything to an abstract hub carries no knowledge and is forbidden.
- NEVER use vague fillers: `connected_to`, `involved_in`, `part_of_context`,
  `subject_of`, `governed_by_context`, `related_to`, `associated_with`. If the
  only relation you can justify is one of these, do not add the edge.
- Do not add synonymy / "is-a-kind-of" edges — merging name variants is
  consolidation's job, not yours.
- Respect direction (a clause is part_of the contract, not the reverse) and do not
  reverse specific/general.
- Preserve negation and uncertainty. Do not invent holdings, obligations, or
  findings the text does not state.

Prefer specific, grounded legal relations when the text supports them, e.g.:
party_to, obligation_under, breach_of, governed_by, held_in (a ruling),
appealed_to, party_in, secured_by, owed_to.

For every triple:
- use `"inference": "explicit"` if directly stated in the text
- otherwise `"inference": "contextual"` (only for a genuinely implied relation)
- include a short `"justification"` grounded in the text

Return ONLY a JSON array (may be EMPTY if no bridging relation is supported):
[
  {
    "head": "entity",
    "relation": "relation",
    "tail": "entity",
    "inference": "explicit | contextual",
    "justification": "short reason grounded in the text"
  }
]

{{schema_constraints}}

Input Data:
- "text": the original source text to analyze
- "disconnected_components": each component with its entities AND their triples
- "current_triples": all triples extracted so far as JSON

{{record_json}}
