"""Discriminative-signature veto.

Blocks merges between names that are surface-similar but conceptually distinct
because they differ in a discriminative token — the case where an LLM (or fuzzy
matching) is most likely to merge wrongly. Examples it stops:
  "Gleason 3+4" vs "4+3"   (numeric order)
  "IL-6" vs "IL-10"        (different number)
  "stage II" vs "stage III" (Roman numeral)
  "10 mg" vs "20 mg", "Section 10(b)" vs "20(a)"

The signature uses only LANGUAGE-GENERAL signals — Arabic numbers and Roman
staging numerals — so there is no hardcoded per-language word list. Polarity /
laterality markers (positive/negative, left/right) are a separate concern and,
if added, should be domain-declarable data (like DomainSchema.type_relations),
not hardcoded here.

Merge property: an acronym and its expansion keep the same numeric signature
("IL-6" -> "interleukin-6", both {6}), so this veto never blocks a true
Schwartz-Hearst expansion while still blocking false siblings ("IL-6"/"IL-10").
"""

from __future__ import annotations

import re

# Roman staging numerals as whole tokens. "I" is excluded (too ambiguous as a
# standalone letter/pronoun); II-XII cover clinical/legal staging in practice.
_ROMAN_STAGES = frozenset({
    "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII",
})


def discriminative_signature(name: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return the (numbers, roman_numerals) that distinguish a name, in order.

    Order matters: "3+4" -> (('3','4'), ()) differs from "4+3" -> (('4','3'), ()).
    """
    numbers = tuple(re.findall(r"\d+", name))
    romans = tuple(
        tok for tok in re.findall(r"[A-Za-z]+", name.upper()) if tok in _ROMAN_STAGES
    )
    return numbers, romans


def merge_allowed(a: str, b: str) -> bool:
    """Two names may merge only if their discriminative signatures match."""
    return discriminative_signature(a) == discriminative_signature(b)


def apply_discriminative_veto(
    mapping: dict[str, str],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Drop merges whose variant and canonical differ in discriminative signature.

    Returns (kept, vetoed) where vetoed holds the (variant, canonical) pairs
    blocked as surface-similar-but-conceptually-different.
    """
    kept: dict[str, str] = {}
    vetoed: list[tuple[str, str]] = []
    for variant, canonical in mapping.items():
        if merge_allowed(variant, canonical):
            kept[variant] = canonical
        else:
            vetoed.append((variant, canonical))
    return kept, vetoed


__all__ = ["discriminative_signature", "merge_allowed", "apply_discriminative_veto"]
