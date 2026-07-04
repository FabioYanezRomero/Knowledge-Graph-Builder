"""Schwartz-Hearst abbreviation extraction + the closed-set acronym pre-pass."""

from kgb.builder.consolidation import extract_abbreviation_pairs, acronym_mapping


def test_long_form_then_short():
    pairs = extract_abbreviation_pairs(
        "The patient underwent transurethral resection of bladder tumor (TURBT)."
    )
    assert ("TURBT", "transurethral resection of bladder tumor") in pairs


def test_short_form_then_long():
    pairs = extract_abbreviation_pairs(
        "We measured TURBT (transurethral resection of bladder tumor) outcomes."
    )
    assert ("TURBT", "transurethral resection of bladder tumor") in pairs


def test_rejects_non_matching_parenthetical():
    # "a big firm" is not a valid expansion of "Company" -> no pair
    assert extract_abbreviation_pairs("He works at the Company (a big firm).") == []


def test_no_parenthetical():
    assert extract_abbreviation_pairs("No abbreviations defined here.") == []


def test_acronym_prepass_closed_to_existing_entities():
    text = "Prostate-specific antigen (PSA) was elevated after the procedure."
    # both forms exist as entities -> merge short into long
    entities = ["PSA", "Prostate-specific antigen", "procedure"]
    assert acronym_mapping(text, entities) == {"PSA": "Prostate-specific antigen"}


def test_acronym_prepass_skips_when_expansion_not_an_entity():
    text = "Prostate-specific antigen (PSA) was elevated."
    # expansion never got extracted -> nothing to merge (closed-set)
    assert acronym_mapping(text, ["PSA", "procedure"]) == {}
