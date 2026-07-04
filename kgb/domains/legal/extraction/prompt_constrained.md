Extract knowledge graph triples from the background section of legal case summaries.

Focus on the background facts for the legal decision. Extract all explicit (head, relation, tail) triples that capture the relationships, events, and entities described.

Extraction Guidelines:
- Extract entities and relations explicitly stated in the text
- Use exact text spans for extractions - do not paraphrase
- Split complex phrases into smaller meaningful entities when appropriate
- Focus ONLY on explicit information (not inferred relationships)
- Ground EVERY salient entity as a node, even with no stated relation. For a
  standalone entity emit a typing triple: (entity, is_type_of, <category>), e.g.
  (Companies Act 2006, is_type_of, statute). Never drop a mentioned party,
  court, statute, or instrument for lack of a relation.

Entity Types:
- Legal parties: plaintiffs, defendants, appellants, respondents
- Organizations: companies, government bodies, courts
- Legal instruments: contracts, deeds, agreements, statutes
- Events and actions: filing, signing, breaching, etc.
- Dates, monetary amounts, locations
- Legal concepts: liability, damages, jurisdiction

Relation Types:
- Hierarchical: is_type_of, part_of, belongs_to
- Legal: filed_against, represented_by, governed_by, breached
- Temporal: occurred_on, preceded, followed
- Causal: caused, resulted_in, led_to
- Participation: involved_in, party_to, signed_by

{{schema_constraints}}

Input to analyze:
{{record_json}}
