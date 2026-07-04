"""Schwartz-Hearst abbreviation extraction.

Deterministic extraction of (short_form, long_form) pairs defined
parenthetically in text — e.g. "transurethral resection of bladder tumor
(TURBT)" or "TURBT (transurethral resection of bladder tumor)". Based on
Schwartz & Hearst (2003), "A simple algorithm for identifying abbreviation
definitions in biomedical text".

No model, no training: it reads the source text to confirm acronym/expansion
pairs. Consolidation applies these only to entities already in the graph
(closed-set), so it never introduces ungrounded nodes.
"""

from __future__ import annotations

import re

_PAREN = re.compile(r"\(([^()]+)\)")


def _is_short_form(candidate: str) -> bool:
    """A short form is 2-10 chars, at most 2 words, starts alphanumeric, has a letter."""
    if not (2 <= len(candidate) <= 10):
        return False
    if len(candidate.split()) > 2:
        return False
    if not candidate[0].isalnum():
        return False
    return any(c.isalpha() for c in candidate)


def _find_best_long_form(short: str, long: str) -> str | None:
    """Schwartz-Hearst matcher: shrink `long` to the shortest suffix whose
    characters cover `short` right-to-left, with short's first char aligned to
    a word start. Returns the matched long form, or None."""
    s_index = len(short) - 1
    l_index = len(long) - 1
    while s_index >= 0:
        curr = short[s_index].lower()
        if not curr.isalnum():
            s_index -= 1
            continue
        while (
            (l_index >= 0 and long[l_index].lower() != curr)
            or (s_index == 0 and l_index > 0 and long[l_index - 1].isalnum())
        ):
            l_index -= 1
        if l_index < 0:
            return None
        l_index -= 1
        s_index -= 1
    start = long.rfind(" ", 0, l_index + 1) + 1
    return long[start:].strip()


def _valid(short: str, long: str) -> bool:
    if not long or len(long) <= len(short):
        return False
    # Long form shouldn't sprawl far beyond the number of short-form chars.
    return len(long.split()) <= len(short) + 5


def _match(short: str, long_window: str) -> str | None:
    if not long_window:
        return None
    long = _find_best_long_form(short, long_window)
    return long if long and _valid(short, long) else None


def extract_abbreviation_pairs(text: str) -> list[tuple[str, str]]:
    """Extract (short_form, long_form) pairs defined parenthetically in text.

    Handles both "long form (SF)" and "SF (long form)". Returns unique pairs in
    order of appearance. The short form is always returned first regardless of
    which side of the parenthesis it was on.
    """
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for m in _PAREN.finditer(text):
        inner = m.group(1).strip()

        if _is_short_form(inner):
            # "long form (SF)": long candidate is the words before "(".
            before = text[: m.start()].rstrip()
            words = before.split()
            if not words:
                continue
            n = min(len(inner) + 5, len(inner) * 2)
            long = _match(inner, " ".join(words[-n:]))
            pair = (inner, long) if long else None
        else:
            # "SF (long form)": the token just before "(" is the short form,
            # the parenthetical is its long form.
            before = text[: m.start()].rstrip()
            words = before.split()
            if not words:
                continue
            short = words[-1].strip(".,;:")
            if not _is_short_form(short):
                continue
            long = _match(short, inner)
            pair = (short, long) if long else None

        if pair and pair not in seen:
            seen.add(pair)
            pairs.append(pair)

    return pairs


__all__ = ["extract_abbreviation_pairs"]
