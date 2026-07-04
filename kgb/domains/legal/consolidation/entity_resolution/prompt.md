You are an expert in entity resolution for legal knowledge graphs.

## Task
Given a list of entity names extracted from a legal document, identify groups of names that refer to the SAME real-world entity. For each group, choose the best canonical name.

## Entity Resolution Rules

### 1. Name Variants (MERGE)
Merge entities that are clearly the same person, organization, or legal instrument:
- Full name vs short name: "Michael J. Salvino" / "Michael Salvino" / "Salvino" → **"Michael J. Salvino"**
- With/without title: "Defendant Salvino" / "Salvino" → **"Michael J. Salvino"** (use the real name)
- Typos or encoding artifacts: "Michael​Michael Salvino" → **"Michael J. Salvino"**
- Short vs long legal citation: "Section 10(b) of the Exchange Act" / "Section 10(b) of the Securities Exchange Act of 1934" → pick the most complete form

### 2. Collective References (MERGE with care)
- "Defendants" / "All Defendants" → **"Defendants"** (unless "All Defendants" has a specific legal meaning in context)
- "the Class" / "Class" → **"the Class"**

### 3. Distinct Entities (DO NOT MERGE)
Do NOT merge entities that are genuinely different even if they share words:
- "DXC" (the company) vs "DXC common stock" (the security) vs "DXC stock price" (a metric) — these are different entities
- "CEO of DXC" (a role) vs "DXC" (the company) — different entities
- "Section 10(b)" vs "Section 20(a)" — different legal provisions
- Specific dates should never be merged with other dates

### 4. Canonical Name Selection
For each group, prefer:
- The most complete formal name (not abbreviations)
- Real names over role-based references ("Michael J. Salvino" over "Defendant Salvino")
- Consistent legal citation style (most complete form)

## Evidence Format
Each entity below is shown with its **edges** — the triples it participates in.
Use these edges to verify whether two entity names truly refer to the same
real-world thing. For example, if "Salvino" appears as head of
`(Salvino) --[served_as]--> (CEO of DXC)` and "Michael J. Salvino" appears
as head of `(Michael J. Salvino) --[held_position]--> (CEO)`, the edges
confirm they are the same person.

A source text excerpt is also provided for additional context on ambiguous cases.

## Output Format
Return a JSON array of merge groups. Each group is an object with:
- `canonical`: The chosen canonical name
- `variants`: Array of ALL names in the group (including the canonical name itself)

Only include groups with 2+ members. Entities with no variants should be omitted.

```json
[
  {
    "canonical": "Michael J. Salvino",
    "variants": ["Michael J. Salvino", "Michael Salvino", "Salvino", "Defendant Salvino"]
  }
]
```

Return ONLY the JSON array. No explanation, no markdown fences.

{{schema_constraints}}

## Entities to Resolve (with edge context)
{{record_json}}