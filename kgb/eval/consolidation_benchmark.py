"""Offline benchmark for the deterministic consolidation guards.

Measures the discriminative veto and Schwartz-Hearst against labeled pairs
WITHOUT an LLM and without hand-labeling the bulk:
- hard negatives are generated programmatically by perturbing the discriminative
  token of a real entity (guaranteed negatives by construction, à la HELEA);
- positives come from Schwartz-Hearst's own acronym definitions plus a small
  seed of semantic variants.

Error semantics (for a merge decision):
  FP = a merge that should NOT happen but is allowed  -> corrupts the graph.
  FN = a merge that SHOULD happen but is blocked       -> leaves duplicates.

A component cannot grade itself: SH positives grade the LLM, not SH; the veto is
graded here against independently-constructed pairs.

Run:  python -m kgb.eval.consolidation_benchmark
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..builder.consolidation import (
    extract_abbreviation_pairs,
    merge_allowed,
    run_sieves,
    resolve_chains,
    acronym_sieve,
    DEFAULT_SIEVES,
)


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    should_merge: bool
    category: str
    text: str = ""  # source context (needed for the acronym sieve)


# Small hand-seeded set: the semantic cases no deterministic rule can auto-label.
SEED_PAIRS: list[Pair] = [
    # positives the veto must NOT block
    Pair("prostatic adenocarcinoma", "prostate adenocarcinoma", True, "morphological"),
    Pair("PSA", "prostate-specific antigen", True, "acronym"),
    Pair("TURBT", "transurethral resection of bladder tumor", True, "acronym"),
    Pair("IL-6", "interleukin-6", True, "acronym-numeric"),  # number travels with expansion
    # negatives the veto should block (numeric / staging) — language-general
    Pair("Gleason 3+4", "Gleason 4+3", False, "numeric-order"),
    Pair("IL-6", "IL-10", False, "numeric-diff"),
    Pair("stage II", "stage III", False, "roman-staging"),
    Pair("Section 10(b)", "Section 20(a)", False, "numeric-code"),
    # negatives the current veto does NOT cover (no number/roman) — the known gap
    Pair("left kidney", "right kidney", False, "laterality"),
    Pair("positive margin", "negative margin", False, "polarity"),
]

# Surface-layer positives the sieve bag should catch WITHOUT the LLM. Split by
# which sieve is responsible, to show the improvement from adding exact_match.
SIEVE_POSITIVES: list[Pair] = [
    Pair("Bladder", "bladder", True, "case"),
    Pair("T-cell", "T cell", True, "punctuation"),
    Pair("prostate  gland", "prostate gland", True, "whitespace"),
    Pair("PSA", "prostate-specific antigen", True, "acronym",
         text="prostate-specific antigen (PSA)"),
    Pair("TURBT", "transurethral resection of bladder tumor", True, "acronym",
         text="transurethral resection of bladder tumor (TURBT)"),
]

# Negatives the sieve bag must NOT merge (precision = no regression).
SIEVE_NEGATIVES: list[Pair] = [
    Pair("Gleason 3+4", "Gleason 4+3", False, "numeric-order"),
    Pair("IL-6", "IL-10", False, "numeric-diff"),
    Pair("stage II", "stage III", False, "roman-staging"),
    Pair("prostate", "prostate adenocarcinoma", False, "distinct-specificity"),
]

# Entities used to auto-generate hard negatives by perturbing a discriminative token.
_HARD_NEGATIVE_SEEDS = ["Gleason 3+4", "stage II", "IL-6", "10 mg", "Section 10(b)"]


def _bump_roman(name: str) -> str | None:
    order = ["II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
    for i, r in enumerate(order[:-1]):
        if re.search(rf"\b{r}\b", name):
            return re.sub(rf"\b{r}\b", order[i + 1], name, count=1)
    return None


def generate_hard_negatives() -> list[Pair]:
    """Perturb a discriminative token to manufacture guaranteed-negative pairs.

    Same surface, one discriminative token changed -> must not merge, by
    construction. No annotation needed.
    """
    negatives: list[Pair] = []
    for name in _HARD_NEGATIVE_SEEDS:
        numbers = re.findall(r"\d+", name)
        if len(numbers) >= 2:
            # swap the first two numbers (e.g. 3+4 -> 4+3)
            perturbed = name.replace(numbers[0], "\0", 1).replace(numbers[1], numbers[0], 1).replace("\0", numbers[1], 1)
            negatives.append(Pair(name, perturbed, False, "gen-numeric-swap"))
        elif numbers:
            # increment the single number (e.g. IL-6 -> IL-7)
            perturbed = name.replace(numbers[0], str(int(numbers[0]) + 1), 1)
            negatives.append(Pair(name, perturbed, False, "gen-numeric-inc"))
        bumped = _bump_roman(name)
        if bumped:
            negatives.append(Pair(name, bumped, False, "gen-roman-bump"))
    return negatives


def evaluate_veto(pairs: list[Pair]) -> dict:
    """Confusion matrix for the veto's merge_allowed decision vs ground truth."""
    tp = fp = fn = tn = 0
    misses: list[Pair] = []   # negatives the veto let through (potential FP source)
    false_vetoes: list[Pair] = []  # positives the veto wrongly blocked (FN)
    for p in pairs:
        allowed = merge_allowed(p.a, p.b)
        if p.should_merge and allowed:
            tp += 1
        elif p.should_merge and not allowed:
            fn += 1
            false_vetoes.append(p)
        elif not p.should_merge and allowed:
            fp += 1
            misses.append(p)
        else:
            tn += 1
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "misses": misses, "false_vetoes": false_vetoes,
    }


def _bag_merges(pair: Pair, sieves) -> bool:
    """Does the sieve bag merge this pair's two entities into one canonical?"""
    mapping = resolve_chains(run_sieves(pair.text, [pair.a, pair.b], sieves))
    return mapping.get(pair.a, pair.a) == mapping.get(pair.b, pair.b)


def evaluate_sieve_bag(pairs: list[Pair], sieves) -> dict:
    """Precision/recall of a sieve configuration on labeled pairs.

    Precision < 1.0 would be a regression (the sieves merged something they
    should not). Recall shows coverage of surface positives.
    """
    tp = fp = fn = tn = 0
    false_merges: list[Pair] = []
    for p in pairs:
        merged = _bag_merges(p, sieves)
        if p.should_merge and merged:
            tp += 1
        elif p.should_merge and not merged:
            fn += 1
        elif not p.should_merge and merged:
            fp += 1
            false_merges.append(p)
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "false_merges": false_merges}


def evaluate_sh(definitions: list[tuple[str, str, str]]) -> dict:
    """Recall of Schwartz-Hearst on sentences that define an acronym.

    Each case: (sentence, expected_short, expected_long).
    """
    found = 0
    missed: list[str] = []
    for sentence, short, long in definitions:
        pairs = extract_abbreviation_pairs(sentence)
        if (short, long) in pairs:
            found += 1
        else:
            missed.append(short)
    return {"found": found, "total": len(definitions), "missed": missed}


_SH_DEFINITIONS = [
    ("Prostate-specific antigen (PSA) was elevated.", "PSA", "Prostate-specific antigen"),
    ("He underwent transurethral resection of bladder tumor (TURBT).", "TURBT", "transurethral resection of bladder tumor"),
    ("The deoxyribonucleic acid (DNA) sample was clean.", "DNA", "deoxyribonucleic acid"),
    ("Magnetic resonance imaging (MRI) confirmed the mass.", "MRI", "Magnetic resonance imaging"),
]


def run() -> dict:
    pairs = SEED_PAIRS + generate_hard_negatives()
    veto = evaluate_veto(pairs)
    sh = evaluate_sh(_SH_DEFINITIONS)

    # No-regression / improvement: run the sieve bag on surface positives +
    # negatives, comparing Schwartz-Hearst alone vs the full bag.
    sieve_pairs = SIEVE_POSITIVES + SIEVE_NEGATIVES
    sh_only = evaluate_sieve_bag(sieve_pairs, [acronym_sieve])
    full_bag = evaluate_sieve_bag(sieve_pairs, DEFAULT_SIEVES)

    return {
        "veto": veto,
        "sh": sh,
        "n_pairs": len(pairs),
        "sieve_sh_only": sh_only,
        "sieve_full_bag": full_bag,
    }


def format_report(results: dict) -> str:
    v = results["veto"]
    sh = results["sh"]
    lines = [
        "=== Consolidation benchmark (deterministic components) ===",
        f"Pairs evaluated: {results['n_pairs']}",
        "",
        "Discriminative veto (merge decision vs ground truth):",
        f"  TP={v['tp']}  TN={v['tn']}  FP={v['fp']}  FN={v['fn']}",
        f"  Correctly blocked negatives (TN): {v['tn']}",
        f"  Wrongly blocked positives (FN):   {v['fn']}",
        f"  Negatives let through (FP):        {v['fp']}",
    ]
    if v["false_vetoes"]:
        lines.append("  FN detail (positives wrongly blocked):")
        lines += [f"    - {p.a}  ~  {p.b}  [{p.category}]" for p in v["false_vetoes"]]
    if v["misses"]:
        lines.append("  FP detail (negatives the veto does NOT cover):")
        lines += [f"    - {p.a}  vs  {p.b}  [{p.category}]" for p in v["misses"]]
    lines += [
        "",
        f"Schwartz-Hearst recall on acronym definitions: {sh['found']}/{sh['total']}",
    ]
    if sh["missed"]:
        lines.append(f"  missed: {', '.join(sh['missed'])}")

    sh_only = results["sieve_sh_only"]
    full = results["sieve_full_bag"]
    lines += [
        "",
        "Sieve bag (no-regression = precision 1.0; improvement = recall up):",
        f"  SH-only:   precision={sh_only['precision']:.2f}  recall={sh_only['recall']:.2f}  (TP={sh_only['tp']} FP={sh_only['fp']} FN={sh_only['fn']})",
        f"  Full bag:  precision={full['precision']:.2f}  recall={full['recall']:.2f}  (TP={full['tp']} FP={full['fp']} FN={full['fn']})",
    ]
    if full["false_merges"]:
        lines.append("  REGRESSION — false merges by the full bag:")
        lines += [f"    - {p.a}  vs  {p.b}  [{p.category}]" for p in full["false_merges"]]
    else:
        lines.append("  No false merges by the full bag (precision preserved).")
    return "\n".join(lines)


def main() -> None:
    print(format_report(run()))


if __name__ == "__main__":
    main()
