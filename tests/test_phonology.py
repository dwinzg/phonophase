"""
Tests for phonophase.phonology.

This module exists because the seven category map covered 41 of 387 phone
symbols, discarding aspiration, breathy voice, nasalisation, length and
retroflexion. Those are the contrasts that distinguish languages, so a cross
linguistic claim built on the old map would have been measuring English.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phonophase.phonology import (  # noqa: E402
    FEATURE_NAMES,
    coverage,
    feature_matrix,
    features,
    is_parseable,
)

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


def test_feature_vector_shape_and_values():
    v = features("m")
    assert v is not None
    assert len(v) == 24, f"expected 24 features, got {len(v)}"
    assert set(v) <= {-1, 0, 1}, f"unexpected values {sorted(set(v))}"


def test_sonorant_obstruent_split_matches_the_hypothesis():
    # The pre-registered grouping and the feature decomposition must be the same
    # claim, not two similar ones. Sonorants are [+son], obstruents [-son].
    for phone in ("m", "n", "l", "r", "a", "i", "w"):
        assert features(phone)[F["son"]] == 1, f"{phone} should be +son"
    for phone in ("p", "t", "k", "s", "z", "f"):
        assert features(phone)[F["son"]] == -1, f"{phone} should be -son"


def test_the_contrasts_the_old_map_discarded_are_now_represented():
    # Each of these was thrown away by the seven category map, and each marks a
    # distinction that carries across languages.
    assert features("bʰ")[F["sg"]] == 1, "breathy voice should be +sg"
    assert features("b")[F["sg"]] == -1, "plain voiced stop should be -sg"
    assert features("ã")[F["nas"]] == 1, "nasalised vowel should be +nas"
    assert features("a")[F["nas"]] == -1, "oral vowel should be -nas"
    assert features("aː")[F["long"]] == 1, "long vowel should be +long"
    assert features("a")[F["long"]] == -1, "short vowel should be -long"


def test_voicing_is_recovered():
    assert features("b")[F["voi"]] == 1
    assert features("p")[F["voi"]] == -1
    assert features("z")[F["voi"]] == 1
    assert features("s")[F["voi"]] == -1


def test_nasals_are_sonorant_and_nasal_but_not_continuant():
    for phone in ("m", "n", "ŋ"):
        v = features(phone)
        assert v[F["son"]] == 1 and v[F["nas"]] == 1 and v[F["cont"]] == -1, phone


def test_non_ipa_entries_are_rejected_not_guessed():
    for junk in ("<pad>", "<unk>", "??", "a1", "i.4", "", "5"):
        assert features(junk) is None, f"{junk!r} should not parse"
        assert not is_parseable(junk)


def test_feature_matrix_drops_unparseable_and_reports_what_it_kept():
    phones = ["m", "??", "s", "a1", "a"]
    matrix, kept = feature_matrix(phones)

    assert kept == ["m", "s", "a"], kept
    assert matrix.shape == (3, 24), matrix.shape
    # The caller must use the returned list, since the input order no longer maps.
    assert matrix[kept.index("m")][F["nas"]] == 1


def test_feature_matrix_handles_nothing_parseable():
    matrix, kept = feature_matrix(["??", "a1", "<pad>"])
    assert kept == [] and matrix.shape == (0, 24)


def test_coverage_reports_a_real_fraction():
    inventory = ["m", "n", "s", "p", "bʰ", "ã", "aː", "??", "a1", "<pad>"]
    c = coverage(inventory)

    assert c["total"] == 8, "special tokens should not count toward the total"
    assert c["parsed"] == 7, c["parsed"]
    assert c["unparsed"] == ["a1"], c["unparsed"]
    assert 0.8 < c["fraction"] <= 1.0


def test_lookup_is_cached_and_stable():
    # Called once per slice over hundreds of thousands of rows, so repeated
    # lookups must be both cheap and identical.
    assert features("m") is features("m")


def test_multi_segment_strings_average_rather_than_drop():
    # The aligner occasionally emits affricates as one token. Averaging keeps
    # them contributing instead of silently discarding them.
    v = features("tʃ")
    assert v is not None and len(v) == 24
    assert np.all(np.isfinite(np.asarray(v, dtype=float)))


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_phonology")
