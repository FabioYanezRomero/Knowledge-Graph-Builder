"""Discriminative-signature veto: block surface-similar-but-distinct merges."""

from kgb.builder.consolidation import (
    discriminative_signature,
    merge_allowed,
    apply_discriminative_veto,
)


def test_numeric_order_blocks_gleason():
    assert not merge_allowed("Gleason 3+4", "Gleason 4+3")


def test_different_number_blocks_interleukin():
    assert not merge_allowed("IL-6", "IL-10")


def test_roman_staging_blocks():
    assert not merge_allowed("stage II", "stage III")


def test_same_number_allows_acronym_expansion():
    # the number travels with the expansion -> veto must NOT block it
    assert merge_allowed("IL-6", "interleukin-6")


def test_no_discriminative_token_allows_merge():
    assert merge_allowed("prostatic adenocarcinoma", "prostate adenocarcinoma")
    assert merge_allowed("PSA", "prostate-specific antigen")
    assert merge_allowed("TURBT", "transurethral resection of bladder tumor")


def test_signature_shape():
    assert discriminative_signature("Gleason 3+4") == (("3", "4"), ())
    assert discriminative_signature("stage III") == ((), ("III",))
    assert discriminative_signature("prostate") == ((), ())


def test_apply_veto_partitions_mapping():
    mapping = {
        "Gleason 4+3": "Gleason 3+4",   # vetoed (numeric order)
        "prostatic adenocarcinoma": "prostate adenocarcinoma",  # kept
        "IL-10": "IL-6",                # vetoed (different number)
    }
    kept, vetoed = apply_discriminative_veto(mapping)
    assert kept == {"prostatic adenocarcinoma": "prostate adenocarcinoma"}
    assert set(vetoed) == {("Gleason 4+3", "Gleason 3+4"), ("IL-10", "IL-6")}
