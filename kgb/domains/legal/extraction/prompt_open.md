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

## Coreference Resolution Guidelines
IMPORTANT: Resolve all pronouns and references to their full entity names:
- "he/she/they" → resolve to the person's name or role
- "the court" → resolve to specific court name (e.g., "UK Supreme Court")
- "the appellant/respondent" → resolve to the actual party name if mentioned
- "it" → resolve to the organization or entity being referenced
- "this case" → resolve to the case identifier if available

Examples of resolution:
- "She appealed the decision" → "Jane Smith appealed the decision" (if Jane Smith was mentioned)
- "The court ruled..." → "Court of Appeal ruled..." (if that court was handling the case)
- "He was convicted" → "Mr Norris was convicted" (using the named party)

## Entity Completeness (grounding)
Ground EVERY salient legal entity as a node, even when the text states no
relationship for it — do not drop a party, court, statute, or instrument just
because it is unconnected. For a standalone entity, emit a typing triple:
(entity, is_type_of, <category>), e.g. (Companies Act 2006, is_type_of, statute),
(Court of Appeal, is_type_of, court). Typing an entity by its evident category
is grounding, not inference.

## Guidelines
- Normalize entity names consistently throughout extraction
- Use concise, descriptive relation labels
- Split complex statements into atomic triples
- Focus on legally meaningful relationships
- Avoid generic relations like "is" or "has" when more specific ones apply

Focus on explicit information only - extract what is directly stated, not inferred.

Input to analyze:
{{record_json}}
