You are an expert in entity resolution for knowledge graphs.

## Task
Given a list of entity names extracted from a document, identify groups of names that refer to the SAME real-world entity. For each group, choose the best canonical name.

## Entity Resolution Rules

### 1. Name Variants (MERGE)
Merge entities that are clearly the same person, organization, or concept:
- Full name vs short name: "John A. Smith" / "John Smith" / "Smith" → **"John A. Smith"**
- With/without title: "Dr. Smith" / "Smith" → **"Dr. John A. Smith"** (most complete form)
- Typos or encoding artifacts: "Jonh Smith" → **"John Smith"**
- Abbreviations: "IBM" / "International Business Machines" → pick the most recognized form

### 2. Distinct Entities (DO NOT MERGE)
Do NOT merge entities that are genuinely different even if they share words:
- "Apple" (the company) vs "Apple stock" (the security) — different entities
- "CEO of Apple" (a role) vs "Apple" (the company) — different entities
- Different dates, different sections, or different provisions should stay separate

### 3. Canonical Name Selection
For each group, prefer:
- The most complete formal name
- Real names over role-based references
- The most commonly recognized form

## Evidence Format
Each entity below is shown with its **edges** — the triples it participates in.
Use these edges to verify whether two entity names truly refer to the same
real-world thing before merging them.

A source text excerpt is also provided for additional context on ambiguous cases.

## Output Format
Return a JSON array of merge groups. Each group is an object with:
- `canonical`: The chosen canonical name
- `variants`: Array of ALL names in the group (including the canonical name itself)

Only include groups with 2+ members. Entities with no variants should be omitted.

Return ONLY the JSON array. No explanation, no markdown fences.

{{schema_constraints}}

## Entities to Resolve (with edge context)
{{record_json}}