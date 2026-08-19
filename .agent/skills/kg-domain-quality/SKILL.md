---
name: kg-domain-quality
description: Quality practices for building kgb KG domains — grounding-only extraction, layered consolidation, connectivity that doesn't fabricate, and why cross-document merging needs entity typing. Use when authoring or tuning extraction/consolidation/augmentation prompts for any domain.
---

# KG Domain Quality Practices

**Read `BEST_PRACTICES.md` in the repository root.** It is the canonical version
of these practices, kept there so repo users see it too. This file is a pointer
so the two cannot drift apart.

`add-domain` covers the mechanical wiring (files, discovery); `BEST_PRACTICES.md`
covers **what to put in the prompts and why**.

The one rule that generates the rest:

```
extract      text → triples        GROUND only   (say what the text says)
consolidate  triples → triples     MERGE only    (add no knowledge)
augment      triples → triples     ADD only      (never merge)
```

The most common and most expensive failure is overloading one stage with
another's job. `BEST_PRACTICES.md` documents the three times we did it, what it
cost, and the checklist that prevents it.
