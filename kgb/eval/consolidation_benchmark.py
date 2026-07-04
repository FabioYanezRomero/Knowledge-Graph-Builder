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

from ..builder.consolidation import extract_abbreviation_pairs, merge_allowed


@dataclass(frozen=True)
class Pair:
    a: str
    b: str
    should_merge: bool
    category: str


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
    return {"veto": veto, "sh": sh, "n_pairs": len(pairs)}


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
    return "\n".join(lines)


def main() -> None:
    print(format_report(run()))


if __name__ == "__main__":
    main()
