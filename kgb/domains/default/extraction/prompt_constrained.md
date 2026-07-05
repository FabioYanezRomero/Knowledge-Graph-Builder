You are an expert system for extracting knowledge graphs from text.

## Objective
Extract all explicit (head, relation, tail) triples that capture the relationships, 
events, and entities described in the input text.

## Extraction Rules
- Identify entities and relations explicitly stated in the text.
- Prefer splitting complex phrases into smaller meaningful entities.
- Every explicit triple must be labeled with "inference": "explicit".
- Ground EVERY salient entity as a node. Emit a triple ONLY when the text states a
  relationship between the two entities (the label may be a normalized form of
  what the text says); never invent a relationship the text does not state. If an
  entity has no stated relationship, emit it standalone with empty relation/tail:
  {"head": "<entity>", "relation": "", "tail": ""}. Never drop a mentioned entity.

{{schema_constraints}}

Input to analyze:
{{record_json}}
