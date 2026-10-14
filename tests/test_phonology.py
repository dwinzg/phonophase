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
    parse_tone,
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
    # `a1` and `i.4` moved out of this list on 2026-09-19. They are tone marked
    # phones, not junk, and they now decode to their base segment. A bare tone
    # marker such as `5` stays here, because stripping its digit leaves no
    # segment at all.
    for junk in ("<pad>", "<unk>", "??", "", "5"):
        assert features(junk) is None, f"{junk!r} should not parse"
        assert not is_parseable(junk)


def test_feature_matrix_drops_unparseable_and_reports_what_it_kept():
    phones = ["m", "??", "s", "a1", "a"]
    matrix, kept = feature_matrix(phones)

    # `a1` is kept from 2026-09-19, as the vowel /a/ carrying tone 1. `??` is
    # still dropped, which is what this test is about.
    assert kept == ["m", "s", "a1", "a"], kept
    assert matrix.shape == (4, 24), matrix.shape
    # The caller must use the returned list, since the input order no longer maps.
    assert matrix[kept.index("m")][F["nas"]] == 1


def test_feature_matrix_handles_nothing_parseable():
    matrix, kept = feature_matrix(["??", "<unk>", "<pad>"])
    assert kept == [] and matrix.shape == (0, 24)


def test_coverage_reports_a_real_fraction():
    inventory = ["m", "n", "s", "p", "bʰ", "ã", "aː", "??", "a1", "5", "<pad>"]
    c = coverage(inventory)

    assert c["total"] == 9, "special tokens should not count toward the total"
    # `a1` parses from 2026-09-19. `5` is a bare tone marker and does not.
    assert c["parsed"] == 8, c["parsed"]
    assert c["unparsed"] == ["5"], c["unparsed"]
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




# --------------------------------------------------------------------------
# tone, added 2026-09-19
#
# The aligner emits 85 tone marked symbols out of 393 and they were rejected
# outright, which excluded every tonal language. That was acceptable while the
# language set was European. It stopped being acceptable when the only way to
# separate sonority from voicing turned out to need a language with voiceless
# sonorants, and the reachable candidates include tonal ones.
# --------------------------------------------------------------------------

def test_a_tone_digit_is_stripped_and_returned_separately():
    assert parse_tone("a1") == ("a", 1)
    assert parse_tone("iɛ5") == ("iɛ", 5)
    assert parse_tone("onɡ5") == ("onɡ", 5)


def test_a_phone_with_no_tone_is_returned_unchanged():
    assert parse_tone("a") == ("a", None)
    assert parse_tone("m") == ("m", None)


def test_a_bare_tone_marker_has_no_segment_and_is_rejected():
    """
    `1` through `5` appear in the vocabulary as tone markers with no phone
    attached. Stripping the digit leaves nothing, and inventing a vowel for it
    would be fabricating a segment that was never emitted.
    """
    for marker in ["1", "2", "3", "4", "5"]:
        assert parse_tone(marker) == (None, int(marker))
        assert features(marker) is None


def test_a_tone_marked_vowel_now_has_the_features_of_its_base():
    """
    The whole point. `a1` is the vowel /a/ carrying tone 1, so it must decode to
    exactly the feature vector of /a/, not to nothing.
    """
    assert features("a1") is not None
    assert features("a1") == features("a")
    assert features("iɛ5") == features("iɛ")


def test_tone_is_not_folded_into_the_feature_vector():
    """
    Tone is suprasegmental. PanPhon's hitone and hireg describe register on a
    segment, not lexical tone, so writing tone into them would put two different
    things in one dimension and make cross linguistic comparison meaningless.
    """
    assert features("a1") == features("a5"), (
        "two tones on the same vowel must give the same segmental features"
    )


def test_non_ipa_junk_is_still_rejected_after_the_change():
    for junk in ["<pad>", "??", "", "<unk>"]:
        assert features(junk) is None


def test_tone_marked_symbols_raise_coverage_substantially():
    """
    Guards the change from silently regressing. The espeak inventory is the real
    one the aligner emits.
    """
    inventory = ["a", "m", "a1", "iɛ5", "onɡ5", "ɑu2", "u5", "??", "1"]
    c = coverage(inventory)

    # `??` is a special token and coverage excludes it from the total by
    # design, so the only thing that should fail to parse here is the bare
    # tone marker, which has no segment attached.
    assert c["total"] == 8, c
    assert c["parsed"] == 7, f"expected the tone marked vowels to parse, got {c}"
    assert c["unparsed"] == ["1"], c["unparsed"]


def test_coverage_reports_symbols_panphon_only_partly_consumed():
    """
    PanPhon silently drops characters it does not recognise, so `dZ`, which is
    X-SAMPA for the affricate, comes back with the features of a plain /d/ and
    nothing says so. That is a trap rather than a live error, because no such
    symbol appears in the English eval data, but it would bite the moment a
    corpus transcribed in X-SAMPA arrives.

    So `features` rejects it and coverage says what was consumed.
    """
    c = coverage(["m", "a", "dZ", "aɪɚ"])

    assert "dZ" in dict(c["partial"]), c["partial"]
    assert dict(c["partial"])["dZ"] == "d", "the symbol decoded to its prefix"
    assert "m" not in dict(c["partial"])
    assert features("dZ") is None and "dZ" in c["unparsed"]
    assert features("aɪ") is not None, "a fully consumed diphthong still parses"


def test_a_fully_consumed_inventory_reports_no_partials():
    assert coverage(["m", "n", "a", "i", "s"])["partial"] == []


def test_a_broken_panphon_fails_loudly_rather_than_parsing_nothing():
    import phonophase.phonology as ph

    def broken():
        raise ImportError("panphon unavailable")

    original = ph._feature_table
    ph._feature_table = broken
    ph.features.cache_clear()
    try:
        ph.features("a")
    except ImportError:
        pass
    else:
        raise AssertionError("a missing PanPhon must raise, not return None")
    finally:
        ph._feature_table = original
        ph.features.cache_clear()
    assert ph.features("a") is not None


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_phonology")
