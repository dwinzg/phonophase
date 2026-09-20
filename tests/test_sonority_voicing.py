"""
Tests for scripts/sonority_voicing.py.

The script exists to decide between two readings of the confirmed finding, so
its failure mode is not a crash, it is returning a confident verdict that the
data does not support. Every test here is built around a dataset whose answer
is known by construction, so a wrong verdict is detectable.

Three properties are load bearing and each has a test that fails without it.

    The three cells come from the feature vector, not from a hand written phone
    list, or the analysis is English shaped and cannot cross a language border.

    The bootstrap resamples utterances once per draw and reuses that draw for
    every cell, or the interval on a difference between cells ignores the
    correlation between them and is wrong in an unknown direction.

    Undecided and rejected are different outcomes. A cell with too few
    utterances must widen the interval and return undecided, never a rejection.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

from sonority_voicing import (  # noqa: E402
    SON_VOI, OBS_VOI, OBS_UNVOI, SON_UNVOI,
    analyse,
    assign_cells,
    bonafide_label,
    cluster_groups,
    contrast_label,
    drop_quietest,
    subset_rows,
    cell_of,
    folded_from_auc,
    verdict,
)

BOOT = 300


# --------------------------------------------------------------------------
# the partition
# --------------------------------------------------------------------------

def test_english_sonorants_land_in_the_voiced_sonorant_cell():
    for phone in ["m", "n", "ŋ", "l", "ɹ", "w", "j", "i", "ɑ", "u"]:
        assert cell_of(phone) == SON_VOI, f"{phone} should be [+son, +voi]"


def test_english_voiced_obstruents_land_in_their_own_cell():
    for phone in ["b", "d", "ɡ", "v", "z", "ð", "ʒ"]:
        assert cell_of(phone) == OBS_VOI, f"{phone} should be [-son, +voi]"


def test_english_voiceless_obstruents_land_in_the_third_cell():
    for phone in ["p", "t", "k", "f", "s", "θ", "ʃ", "tʃ"]:
        assert cell_of(phone) == OBS_UNVOI, f"{phone} should be [-son, -voi]"


def test_glottals_populate_the_fourth_cell_which_is_what_crosses_the_two_factors():
    """
    PanPhon, following Hayes, treats the glottals as [+son] because they have no
    oral constriction. That puts /h/ and /ʔ/ at [+son, -voi], which is the cell
    that turns a three step ladder into a crossed design. It is a convention
    rather than a fact, so the script reports it as such and offers the
    sensitivity run that folds these phones in with the obstruents.
    """
    assert cell_of("h") == SON_UNVOI
    assert cell_of("ʔ") == SON_UNVOI


def test_the_partition_comes_from_features_not_from_a_phone_list():
    """
    A phone absent from English still has to be classified, or the analysis
    cannot be repeated in another language. These are the cases that a hand
    written English list would miss.
    """
    assert cell_of("ɲ") == SON_VOI, "palatal nasal, in French and Polish"
    assert cell_of("ʐ") == OBS_VOI, "voiced retroflex fricative, in Polish"
    assert cell_of("ɕ") == OBS_UNVOI, "voiceless alveolo-palatal fricative, in Polish"
    assert cell_of("m̥") == SON_UNVOI, "voiceless nasal, the fourth cell"


def test_unparseable_phones_are_rejected_rather_than_guessed_at():
    for phone in ["<pad>", "??", "", "1"]:
        assert cell_of(phone) is None, f"{phone!r} must not be assigned a cell"


def test_assign_cells_reports_what_it_dropped_instead_of_dropping_silently():
    phones = np.array(["a", "b", "p", "??", "<pad>", "m"])
    cells, dropped = assign_cells(phones)

    assert list(cells) == [SON_VOI, OBS_VOI, OBS_UNVOI, None, None, SON_VOI]
    assert dropped == {"??": 1, "<pad>": 1}, "dropped phones must be counted by symbol"


# --------------------------------------------------------------------------
# which label is bonafide, which is the sign of everything downstream
# --------------------------------------------------------------------------

def test_the_bonafide_label_is_read_from_the_attack_column_not_assumed():
    """
    In the ASVspoof tables label 1 is SPOOF and the bonafide rows carry attack
    '-'. Assume it the other way round and every direction in the output
    inverts while every magnitude stays right, which is the kind of error that
    survives review. So it is read from the data and checked.
    """
    y = np.array([0, 0, 1, 1])
    attack = np.array(["-", "-", "A07", "A08"])
    assert bonafide_label({"y": y, "attack": attack}) == 0

    flipped = {"y": np.array([1, 1, 0, 0]), "attack": attack}
    assert bonafide_label(flipped) == 1


def test_a_table_whose_attacks_contradict_its_labels_fails_loudly():
    contradictory = {
        "y": np.array([0, 1, 0, 1]),
        "attack": np.array(["-", "-", "A07", "A08"]),
    }
    try:
        bonafide_label(contradictory)
    except ValueError as exc:
        assert "attack" in str(exc).lower()
    else:
        raise AssertionError("a table where attack and label disagree must raise")


def test_a_table_with_no_attack_column_says_so_rather_than_guessing():
    try:
        bonafide_label({"y": np.array([0, 1]), "attack": np.array(["", ""])})
    except ValueError as exc:
        assert "attack" in str(exc).lower()
    else:
        raise AssertionError("an unmarked table must raise rather than pick a label")


# --------------------------------------------------------------------------
# the silence sensitivity, which is a published concern and not a hypothetical
# --------------------------------------------------------------------------

def test_dropping_the_quietest_slices_removes_exactly_that_fraction():
    """
    Zhang et al. 2023 report that the proportion of silence differs between
    genuine and TTS speech, so a near silent slice can separate the classes for
    a reason that has nothing to do with phase. Cell D is carried entirely by
    /h/, which is the quietest segment in the inventory, so this matters here
    more than anywhere.
    """
    data = {
        "y": np.arange(100) % 2,
        "u": np.array([f"U{i:03d}" for i in range(100)]),
        "phone": np.array(["a"] * 100),
        "x": np.arange(100, dtype=float),
        "energy": np.arange(100, dtype=float),
        "cell": np.array([SON_VOI] * 100, dtype=object),
        "bonafide_label": 0,
    }

    kept, report = drop_quietest(data, 0.20)

    assert kept["y"].size == 80
    assert kept["energy"].min() == 20.0, "the quietest fifth should be the rows removed"
    assert report["n_dropped"] == 20


def test_dropping_nothing_is_the_identity():
    data = {
        "y": np.array([0, 1, 0, 1]),
        "u": np.array(["A", "B", "C", "D"]),
        "phone": np.array(["a"] * 4),
        "x": np.array([1.0, 2.0, 3.0, 4.0]),
        "energy": np.array([1.0, 2.0, 3.0, 4.0]),
        "cell": np.array([SON_VOI] * 4, dtype=object),
        "bonafide_label": 0,
    }
    kept, report = drop_quietest(data, 0.0)

    assert kept["y"].size == 4
    assert report["n_dropped"] == 0


def test_the_drop_is_reported_per_class_because_it_can_be_lopsided():
    """
    A global energy threshold can remove far more of one class than the other,
    and if it does, the sensitivity run has changed the comparison rather than
    cleaned it. The counts have to be visible.
    """
    data = {
        # bonafide is label 0 here, and is the quiet class by construction
        "y": np.array([0] * 50 + [1] * 50),
        "u": np.array([f"U{i:03d}" for i in range(100)]),
        "phone": np.array(["a"] * 100),
        "x": np.arange(100, dtype=float),
        "energy": np.concatenate([np.arange(50.0), np.arange(1000.0, 1050.0)]),
        "cell": np.array([SON_VOI] * 100, dtype=object),
        "bonafide_label": 0,
    }

    _, report = drop_quietest(data, 0.20)

    assert report["n_dropped_bonafide"] == 20
    assert report["n_dropped_spoof"] == 0
    assert report["lopsided"] is True, (
        "a drop falling entirely on one class must be flagged, not reported as a "
        "clean sensitivity check"
    )


# --------------------------------------------------------------------------
# what the intervals generalise to, which the audit showed was the real issue
# --------------------------------------------------------------------------

def test_system_clustering_resamples_utterances_for_real_and_systems_for_fake():
    """
    The spoof side of the eval table is 1,495 utterances from 13 synthesis
    systems. Clustering it by utterance makes the interval a statement about
    more utterances from those same 13 systems. For a claim about speech, or
    about synthesis in general, the system is the exchangeable unit and the
    effective n is 13.

    Bonafide has no system, so it stays clustered by utterance either way.
    """
    data = {
        "y": np.array([0, 0, 1, 1]),
        "u": np.array(["b1", "b2", "s1", "s2"]),
        "attack": np.array(["-", "-", "A07", "A08"]),
        "bonafide_label": 0,
    }

    assert list(cluster_groups(data, "utterance")) == ["b1", "b2", "s1", "s2"]
    assert list(cluster_groups(data, "system")) == ["b1", "b2", "A07", "A08"]


def test_system_clustering_widens_the_interval():
    """
    Fewer clusters on the spoof side must mean a wider interval. If it does not,
    the option is not doing anything.
    """
    rng = np.random.default_rng(3)
    n_utt, per = 160, 6
    # The dependence must sit at the SYSTEM level, so every utterance from one
    # system shares its offset. That is the real structure: a vocoder leaves the
    # same signature on everything it produces.
    system_shock = {f"A{k:02d}": rng.normal(0, 1.2) for k in range(4)}
    y, u, attack, cell, x = [], [], [], [], []
    for i in range(n_utt):
        label = i % 2
        system = f"A{i % 4:02d}" if label == 1 else "-"
        shock = system_shock[system] if label == 1 else 0.0
        for _ in range(per):
            y.append(label); u.append(f"U{i:04d}"); attack.append(system)
            cell.append(SON_VOI)
            x.append(shock + rng.normal(0, 1) + (0.8 if label == 0 else 0.0))
    data = {"y": np.asarray(y), "u": np.asarray(u), "attack": np.asarray(attack),
            "cell": np.asarray(cell, dtype=object), "x": np.asarray(x),
            "phone": np.asarray(["a"] * len(y)), "bonafide_label": 0}

    narrow = analyse(data, n_boot=BOOT, seed=0, cluster="utterance")["cells"][SON_VOI]
    wide = analyse(data, n_boot=BOOT, seed=0, cluster="system")["cells"][SON_VOI]

    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"]), (
        "clustering the spoof side by system must widen the interval, because "
        "four systems carry less information than eighty utterances"
    )


# --------------------------------------------------------------------------
# defects found by the external audit, 2026-09-19
# --------------------------------------------------------------------------

def test_subsetting_keeps_every_array_aligned_including_ones_added_later():
    """
    The CLI used to subset a hand written list of seven array names. `energy`
    was added to load() afterwards and not to that list, so combining
    --drop-phone with --drop-quietest compared a full length mask against short
    labels and crashed. A hand written list drifts from the loader; deriving the
    list from the data cannot.
    """
    data = {
        "y": np.array([0, 1, 0, 1]),
        "u": np.array(["A", "B", "C", "D"]),
        "phone": np.array(["a", "b", "c", "d"]),
        "x": np.array([1.0, 2.0, 3.0, 4.0]),
        "energy": np.array([5.0, 6.0, 7.0, 8.0]),
        "cell": np.array([SON_VOI] * 4, dtype=object),
        "bonafide_label": 0,
        "dropped": {"??": 3},
    }
    keep = np.array([True, False, True, False])
    out = subset_rows(data, keep)

    for key in ("y", "u", "phone", "x", "energy", "cell"):
        assert out[key].size == 2, f"{key} was not subset"
    assert list(out["energy"]) == [5.0, 7.0]
    # Non-array entries must survive untouched.
    assert out["bonafide_label"] == 0
    assert out["dropped"] == {"??": 3}


def test_an_interval_excluding_zero_is_never_labelled_as_spanning_it():
    """
    `clear` requires the interval AND the q. The printed column showed only the
    interval, so a contrast with an interval of [+0.0011, +0.0400] and q = 0.06
    printed as "spans zero", which contradicts itself on the page.
    """
    survives = {"ci_low": 0.01, "ci_high": 0.04, "q": 0.001, "clear": True}
    fails_fdr = {"ci_low": 0.0011, "ci_high": 0.0400, "q": 0.06, "clear": False}
    spans = {"ci_low": -0.01, "ci_high": 0.04, "q": 0.5, "clear": False}

    assert contrast_label(survives) == "clear of zero"
    assert contrast_label(fails_fdr) == "not clear after correction"
    assert contrast_label(spans) == "spans zero"
    assert "spans zero" not in contrast_label(fails_fdr)


def test_the_correction_family_holds_only_independent_contrasts():
    """
    A-C is A-B plus B-C exactly, so three contrasts carry two degrees of
    freedom. Correcting across all three treats a derived quantity as an
    independent test.
    """
    data = _dataset({SON_VOI: 1.6, OBS_VOI: 0.8, OBS_UNVOI: 0.0}, n_utt=120, seed=31)
    contrasts = analyse(data, n_boot=BOOT, seed=0)["contrasts"]

    assert abs((contrasts["A-B"]["delta"] + contrasts["B-C"]["delta"])
               - contrasts["A-C"]["delta"]) < 1e-9, "A-C must be the sum, by construction"
    assert contrasts["A-C"]["q"] is None, "a derived contrast carries no q of its own"
    assert contrasts["A-B"]["q"] is not None and contrasts["B-C"]["q"] is not None


# --------------------------------------------------------------------------
# folding
# --------------------------------------------------------------------------

def test_folding_treats_both_directions_alike():
    assert folded_from_auc(0.7) == folded_from_auc(0.3)
    assert folded_from_auc(0.5) == 0.5


# --------------------------------------------------------------------------
# synthetic datasets with a known answer
# --------------------------------------------------------------------------

def _dataset(effects, n_utt=120, per_cell=4, seed=0, cells=None):
    """
    Build a slice table whose per cell separation is known by construction.

    effects maps a cell to the mean shift applied to bonafide slices in that
    cell. Every utterance contributes slices to every cell, which is what the
    real table looks like, and it is what makes the cells correlated.
    """
    cells = cells or [SON_VOI, OBS_VOI, OBS_UNVOI]
    rng = np.random.default_rng(seed)
    y, u, cell, x = [], [], [], []
    for i in range(n_utt):
        label = i % 2  # 1 bonafide, 0 spoof
        utt = f"U{i:04d}"
        # A per utterance offset, so slices inside one utterance are correlated
        # exactly as they are in the real table.
        offset = rng.normal(0, 1.0)
        for c in cells:
            for _ in range(per_cell):
                y.append(label)
                u.append(utt)
                cell.append(c)
                x.append(offset + rng.normal(0, 1.0) + (effects[c] if label == 1 else 0.0))
    return {
        "y": np.asarray(y, dtype=np.int64),
        "u": np.asarray(u),
        "cell": np.asarray(cell),
        "x": np.asarray(x, dtype=np.float64),
        "phone": np.asarray(["x"] * len(y)),
        # These fixtures plant the effect on label 1, the opposite of the
        # ASVspoof tables, which is exactly why this is carried rather than
        # assumed.
        "bonafide_label": 1,
    }


def test_recovers_a_separation_that_is_there_and_a_null_that_is_not():
    data = _dataset({SON_VOI: 2.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, seed=1)
    r = analyse(data, n_boot=BOOT, seed=0)

    strong = r["cells"][SON_VOI]
    null = r["cells"][OBS_UNVOI]

    assert strong["auc"] > 0.75, f"planted separation not recovered, got {strong['auc']}"
    assert not strong["crosses_chance"]
    assert null["auc"] < 0.60, f"null cell should sit near chance, got {null['auc']}"


def test_a_difference_that_is_absent_by_construction_has_an_interval_spanning_zero():
    data = _dataset({SON_VOI: 1.0, OBS_VOI: 1.0, OBS_UNVOI: 1.0}, seed=2)
    r = analyse(data, n_boot=BOOT, seed=0)

    for contrast in r["contrasts"].values():
        assert contrast["ci_low"] <= 0 <= contrast["ci_high"], (
            f"identical cells produced a difference clear of zero: {contrast}"
        )


def test_the_draw_is_shared_across_cells_so_the_paired_interval_is_narrower():
    """
    The cells are correlated because they are drawn from the same utterances.
    Pairing the draw removes that shared variance from the difference, so the
    paired interval must be narrower than one built from independent draws.
    Without the shared draw this test fails.
    """
    data = _dataset({SON_VOI: 1.2, OBS_VOI: 0.6, OBS_UNVOI: 0.0}, seed=3)
    r = analyse(data, n_boot=BOOT, seed=0)

    paired = r["contrasts"]["A-C"]
    paired_width = paired["ci_high"] - paired["ci_low"]

    a = r["cells"][SON_VOI]
    c = r["cells"][OBS_UNVOI]
    independent_width = ((a["ci_high"] - a["ci_low"]) ** 2
                         + (c["ci_high"] - c["ci_low"]) ** 2) ** 0.5

    assert paired_width < independent_width, (
        "the paired interval is not narrower than the independent combination, "
        "which means the draws are not shared across cells"
    )


def test_clustering_is_by_utterance_not_by_slice():
    """
    Same number of slices, fewer source utterances. The interval has to widen,
    because the information is in the utterances. This is the pseudoreplication
    that made the pilot's intervals too narrow.
    """
    many = _dataset({SON_VOI: 1.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=120, per_cell=4, seed=4)
    few = _dataset({SON_VOI: 1.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=20, per_cell=24, seed=4)

    wide = analyse(few, n_boot=BOOT, seed=0)["cells"][SON_VOI]
    narrow = analyse(many, n_boot=BOOT, seed=0)["cells"][SON_VOI]

    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"]), (
        "twenty utterances must give a wider interval than a hundred and twenty "
        "at the same slice count"
    )


# --------------------------------------------------------------------------
# the verdict, which is the part that can mislead
# --------------------------------------------------------------------------

def test_sonority_pattern_is_called_sonority():
    """A far above B, B level with C. Voiced obstruents pattern with voiceless."""
    data = _dataset({SON_VOI: 1.6, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=200, seed=5)
    assert verdict(analyse(data, n_boot=BOOT, seed=0))["reading"] == "sonority"


def test_voicing_pattern_is_called_voicing():
    """A level with B, both far above C. Voiced obstruents pattern with sonorants."""
    data = _dataset({SON_VOI: 1.6, OBS_VOI: 1.6, OBS_UNVOI: 0.0}, n_utt=200, seed=6)
    assert verdict(analyse(data, n_boot=BOOT, seed=0))["reading"] == "voicing"


def test_a_monotone_ladder_is_called_graded_rather_than_either_label():
    data = _dataset({SON_VOI: 2.2, OBS_VOI: 1.1, OBS_UNVOI: 0.0}, n_utt=200, seed=7)
    assert verdict(analyse(data, n_boot=BOOT, seed=0))["reading"] == "graded"


def test_no_structure_is_called_indistinguishable():
    data = _dataset({SON_VOI: 0.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=200, seed=8)
    assert verdict(analyse(data, n_boot=BOOT, seed=0))["reading"] == "indistinguishable"


def test_voiced_obstruents_leading_is_reported_as_the_surprise_it_would_be():
    data = _dataset({SON_VOI: 0.0, OBS_VOI: 1.6, OBS_UNVOI: 0.0}, n_utt=200, seed=9)
    r = verdict(analyse(data, n_boot=BOOT, seed=0))
    assert r["reading"] == "voicing-not-sonority"
    assert r["surprise"] is True


def _result(ab, bc, ac, usable=True, steps=None):
    """
    A result dict with the contrasts set by hand.

    The decision rule is tested directly here rather than through a sampled
    dataset, so a rule change is caught by a deterministic test instead of by
    an effect size that happens to land on one side of a threshold.

    `steps` is the interval on (A-B) minus (B-C), the quantity that says whether
    the two ladder steps differ from each other at all. None means the analysis
    did not produce it, which is what a table with an unusable ladder gives.
    """
    def contrast(lo, hi):
        return {"delta": (lo + hi) / 2, "ci_low": lo, "ci_high": hi,
                "q": 0.001 if (lo > 0 or hi < 0) else 0.9,
                "clear": lo > 0 or hi < 0}

    cell = {"usable": usable, "n_utterances_bonafide": 100, "n_utterances_spoof": 100,
            "n_slices": 400, "why_unusable": None}
    out = {
        "cells": {SON_VOI: dict(cell), OBS_VOI: dict(cell), OBS_UNVOI: dict(cell)},
        "contrasts": {"A-B": contrast(*ab), "B-C": contrast(*bc), "A-C": contrast(*ac)},
        "step_difference": None,
    }
    if steps is not None:
        lo, hi = steps
        out["step_difference"] = {
            "delta": (lo + hi) / 2, "ci_low": lo, "ci_high": hi,
            "p": 0.001 if (lo > 0 or hi < 0) else 0.9,
            "clear": lo > 0 or hi < 0,
        }
    return out


# --------------------------------------------------------------------------
# naming a factor requires the two steps to differ, not one of them to fail
# --------------------------------------------------------------------------

def test_one_step_clearing_alone_does_not_name_a_factor():
    """
    The pattern the eval table actually shows under system clustering. B-C
    clears, A-B does not, and the two steps are not distinguishable from each
    other.

    The steps were +0.0282 and +0.0319, a ratio of 0.88, while their standard
    errors differed by a factor of 1.85 and their difference was -0.0037 with p
    about 0.9. Under utterance clustering both cleared and the reading was
    graded; under system clustering only one cleared and the reading became
    voicing, on identical point estimates. So the label was decided by power
    alone, which is a failure to reject dressed as acceptance of a null.
    """
    r = verdict(_result(ab=(-0.014, 0.065), bc=(0.010, 0.053), ac=(0.020, 0.090),
                        steps=(-0.050, 0.045)))

    assert r["reading"] != "voicing", (
        "one step failing to clear is not evidence that the other is the factor"
    )
    assert r["reading"] == "steps-indistinguishable"
    assert "power" in r["why"].lower()


def test_a_factor_is_named_when_the_two_steps_genuinely_differ():
    """
    The rule must still be able to name voicing. A real voicing pattern has A
    level with B and both far above C, so the step difference is clearly
    negative rather than merely unproven.
    """
    r = verdict(_result(ab=(-0.010, 0.012), bc=(0.030, 0.070), ac=(0.030, 0.080),
                        steps=(-0.070, -0.020)))

    assert r["reading"] == "voicing"


def test_sonority_is_named_on_the_mirror_image_and_not_otherwise():
    named = verdict(_result(ab=(0.030, 0.070), bc=(-0.010, 0.012), ac=(0.030, 0.080),
                            steps=(0.020, 0.070)))
    assert named["reading"] == "sonority"

    unnamed = verdict(_result(ab=(0.010, 0.053), bc=(-0.014, 0.065), ac=(0.020, 0.090),
                              steps=(-0.045, 0.050)))
    assert unnamed["reading"] == "steps-indistinguishable"


def test_a_missing_step_difference_cannot_license_a_label():
    """
    If the analysis did not produce the step difference, no factor is named. The
    absence of a check is not the check passing.
    """
    r = verdict(_result(ab=(-0.014, 0.065), bc=(0.010, 0.053), ac=(0.020, 0.090)))
    assert r["reading"] == "steps-indistinguishable"


def test_the_step_difference_comes_out_of_the_same_draws_as_the_steps():
    """
    (A-B) minus (B-C) has to be computed inside the shared draw, or its interval
    would not carry the correlation between the two steps, which is the whole
    reason a paired bootstrap is used here.
    """
    data = _dataset({SON_VOI: 2.2, OBS_VOI: 1.1, OBS_UNVOI: 0.0}, n_utt=200, seed=7)
    result = analyse(data, n_boot=BOOT, seed=0)

    step = result["step_difference"]
    ab = result["contrasts"]["A-B"]["delta"]
    bc = result["contrasts"]["B-C"]["delta"]

    assert step is not None
    assert abs(step["delta"] - (ab - bc)) < 1e-12, (
        "the point estimate must be the difference of the two step estimates"
    )
    assert step["ci_low"] < step["delta"] < step["ci_high"]


def test_an_ordered_ladder_whose_steps_are_individually_undetermined_is_not_called_flat():
    """
    A clears C, but neither single step clears zero. Calling that
    indistinguishable would report no structure where there is structure, and
    calling it graded would claim two steps the data did not show.
    """
    r = verdict(_result(ab=(-0.005, 0.030), bc=(-0.005, 0.030), ac=(0.010, 0.045)))

    assert r["reading"] == "graded-weak"
    assert "A-C" in r["why"]


def test_voiceless_obstruents_above_voiced_ones_is_flagged_as_a_surprise():
    r = verdict(_result(ab=(0.010, 0.040), bc=(-0.040, -0.010), ac=(-0.005, 0.030)))

    assert r["surprise"] is True
    assert "B-C" in r["why"]


def test_an_unusable_cell_blocks_the_verdict_even_when_the_contrasts_look_clear():
    r = _result(ab=(0.010, 0.040), bc=(-0.005, 0.030), ac=(0.020, 0.060))
    r["cells"][OBS_VOI]["usable"] = False
    r["cells"][OBS_VOI]["why_unusable"] = "12 bonafide utterances, floor is 10"

    out = verdict(r)
    assert out["reading"] == "undecided"
    assert OBS_VOI in out["why"]


def test_too_few_utterances_returns_undecided_and_never_a_rejection():
    data = _dataset({SON_VOI: 1.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=6, per_cell=40, seed=10)
    r = verdict(analyse(data, n_boot=BOOT, seed=0))

    assert r["reading"] == "undecided"
    assert "underpowered" in r["why"].lower()


def test_a_missing_cell_is_undecided_rather_than_a_crash():
    """Some languages have no voiced obstruents at all. That is not an error."""
    data = _dataset({SON_VOI: 1.0, OBS_UNVOI: 0.0}, n_utt=100,
                    cells=[SON_VOI, OBS_UNVOI], seed=11)
    r = verdict(analyse(data, n_boot=BOOT, seed=0))

    assert r["reading"] == "undecided"
    assert OBS_VOI in r["why"], "the report must name the cell that was missing"


# --------------------------------------------------------------------------
# reporting obligations, which exist because of earlier mistakes
# --------------------------------------------------------------------------

def test_counts_are_reported_in_utterances_as_well_as_slices():
    data = _dataset({SON_VOI: 1.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=100, seed=12)
    cell = analyse(data, n_boot=BOOT, seed=0)["cells"][SON_VOI]

    assert cell["n_utterances_bonafide"] == 50
    assert cell["n_utterances_spoof"] == 50
    assert cell["n_slices"] == 400


def test_slice_duration_is_reported_per_cell_because_it_is_the_obvious_confound():
    """
    A longer slice estimates its own statistic more precisely, so a cell whose
    slices are longer could separate better for that reason alone. The cell
    table has to carry the number that lets a reader rule it out, or accept it.
    """
    data = _dataset({SON_VOI: 1.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=40, seed=15)
    data["duration"] = np.where(data["cell"] == SON_VOI, 0.10, 0.06)

    cells = analyse(data, n_boot=BOOT, seed=0)["cells"]
    assert abs(cells[SON_VOI]["duration_median"] - 0.10) < 1e-9
    assert abs(cells[OBS_VOI]["duration_median"] - 0.06) < 1e-9


def test_per_phone_counts_are_reported_so_a_cell_resting_on_one_phone_is_visible():
    """
    T3 put `lat` at the top of a 24 feature table on the strength of a single
    phone. A cell carried by one segment has to be visible in the output.
    """
    data = _dataset({SON_VOI: 1.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=40, seed=13)
    data["phone"] = np.where(data["cell"] == OBS_VOI, "z", "a")

    phones = analyse(data, n_boot=BOOT, seed=0)["cells"][OBS_VOI]["phones"]
    assert list(phones) == [("z", 160)], f"expected one phone carrying the cell, got {phones}"


# --------------------------------------------------------------------------
# the crossed design, which is what actually separates the two factors
# --------------------------------------------------------------------------

def _crossed(effects, n_utt=200, per_cell=4, seed=0):
    return _dataset(effects, n_utt=n_utt, per_cell=per_cell, seed=seed,
                    cells=[SON_VOI, OBS_VOI, OBS_UNVOI, SON_UNVOI])


def test_an_effect_that_tracks_voicing_only_shows_up_as_a_voicing_main_effect():
    """
    Planted so that the two voiced cells separate and the two voiceless ones do
    not, with sonority crossed against it. A design that cannot recover this is
    not answering the question it was built for.
    """
    data = _crossed({SON_VOI: 1.6, OBS_VOI: 1.6, OBS_UNVOI: 0.0, SON_UNVOI: 0.0}, seed=20)
    m = analyse(data, n_boot=BOOT, seed=0)["main_effects"]

    assert m["voi"]["ci_low"] > 0, f"voicing effect not recovered: {m['voi']}"
    assert m["son"]["ci_low"] <= 0 <= m["son"]["ci_high"], (
        f"a sonority effect appeared where none was planted: {m['son']}"
    )


def test_an_effect_that_tracks_sonority_only_shows_up_as_a_sonority_main_effect():
    data = _crossed({SON_VOI: 1.6, OBS_VOI: 0.0, OBS_UNVOI: 0.0, SON_UNVOI: 1.6}, seed=21)
    m = analyse(data, n_boot=BOOT, seed=0)["main_effects"]

    assert m["son"]["ci_low"] > 0, f"sonority effect not recovered: {m['son']}"
    assert m["voi"]["ci_low"] <= 0 <= m["voi"]["ci_high"], (
        f"a voicing effect appeared where none was planted: {m['voi']}"
    )


def test_an_effect_confined_to_one_cell_shows_up_as_an_interaction():
    data = _crossed({SON_VOI: 2.0, OBS_VOI: 0.0, OBS_UNVOI: 0.0, SON_UNVOI: 0.0}, seed=22)
    m = analyse(data, n_boot=BOOT, seed=0)["main_effects"]

    assert m["interaction"]["ci_low"] > 0, (
        f"an effect in one cell only must show an interaction: {m['interaction']}"
    )


def test_main_effects_are_absent_when_the_fourth_cell_is_missing():
    """A language with no voiceless sonorants cannot support the crossed test."""
    data = _dataset({SON_VOI: 1.0, OBS_VOI: 0.5, OBS_UNVOI: 0.0}, n_utt=100, seed=23)
    assert analyse(data, n_boot=BOOT, seed=0)["main_effects"] is None


def test_flipping_the_label_convention_flips_direction_and_leaves_separation_alone():
    """
    The same audio, the same separation, the other labelling convention. If the
    convention leaked into the magnitudes rather than only into the direction,
    the two runs would disagree on how much the cells separate.
    """
    data = _dataset({SON_VOI: 1.5, OBS_VOI: 0.0, OBS_UNVOI: 0.0}, n_utt=120, seed=30)
    flipped = dict(data, y=1 - data["y"], bonafide_label=0)

    straight = analyse(data, n_boot=BOOT, seed=0)["cells"][SON_VOI]
    inverted = analyse(flipped, n_boot=BOOT, seed=0)["cells"][SON_VOI]

    assert straight["direction"] == inverted["direction"] == "bonafide-high"
    assert abs(straight["auc"] - inverted["auc"]) < 1e-9
    assert straight["n_utterances_bonafide"] == inverted["n_utterances_bonafide"]


def test_direction_is_reported_so_a_sign_flip_cannot_hide_behind_folding():
    """
    Folding makes a statistic running spoof-high count the same as one running
    bonafide-high. That is right for measuring separation and wrong for
    reading a mechanism, so the direction has to survive into the output.
    """
    data = _dataset({SON_VOI: -1.5, OBS_VOI: 1.5, OBS_UNVOI: 0.0}, n_utt=150, seed=14)
    cells = analyse(data, n_boot=BOOT, seed=0)["cells"]

    assert cells[SON_VOI]["direction"] == "spoof-high"
    assert cells[OBS_VOI]["direction"] == "bonafide-high"
    assert cells[SON_VOI]["auc"] > 0.6, "folding should make both cells separate"


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_sonority_voicing")
