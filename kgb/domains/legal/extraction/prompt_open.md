Extract knowledge graph triples from legal case documents.

## Entity Categories to Extract
- Legal parties (appellants, respondents, claimants, defendants)
- Courts and tribunals at all levels
- Legal instruments (acts, statutes, regulations, articles)
- Legal concepts (rights, duties, obligations)
- Decisions and outcomes

## Output Schema
Each triple should contain:
- head: Subject entity (use full names, not pronouns)
- relation: Relationship verb or phrase (concise, lowercase, 2-4 words)
- tail: Object entity
- inference: "explicit" if directly stated, "contextual" if reasonably implied

## Entity Completeness (grounding)
Ground EVERY salient legal entity as a node. Emit a (head, relation, tail) triple
ONLY when the text states a relationship between the two entities (the label may
be a normalized form of what the text says); never invent a relationship the text
does not state. If a party, court, statute, or instrument has no stated
relationship, emit it as a standalone node with empty relation/tail:
{"head": "<entity>", "relation": "", "tail": ""}. Do not drop it.

## Guidelines
- Use concise, descriptive relation labels
- Split complex statements into atomic triples
- Focus on legally meaningful relationships
- Avoid generic relations like "is" or "has" when more specific ones apply

Focus on explicit information only - extract what is directly stated, not inferred.

Input to analyze:
{{record_json}}
