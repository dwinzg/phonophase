"""
Tests for scripts/attack_analysis.py.

The claim this script exists to check is that the sonorant effect reverses on
four of thirteen attacks, which was recorded in the research log with no command
beside it and has been treated as a serious threat to the mechanism ever since.

So the first obligation is that the quantity being called "the sonorant
advantage" is the one the pre-registration defines, and the second is that the
difference between two different quantities, whether phase beats magnitude at
all and whether it beats it MORE in sonorants, cannot be silently conflated.
There is a test for exactly that, because that conflation is the likeliest way
this whole line of reasoning goes wrong.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

from attack_analysis import (  # noqa: E402
    OBSTRUENTS,
    SONORANTS,
    apply_fdr,
    pooled_asymmetries,
    check_coverage,
    exact_permutation_p,
    group_means,
    load_taxonomy,
    per_attack,
)

BOOT = 200
TAXONOMY = os.path.join(os.path.dirname(__file__), "..", "configs",
                        "asvspoof2019_attacks.yaml")


# --------------------------------------------------------------------------
# the taxonomy, which is external evidence and has to stay that way
# --------------------------------------------------------------------------

def test_every_eval_attack_has_a_recorded_waveform_generator():
    taxonomy = load_taxonomy(TAXONOMY)
    evaluation = [a for a, r in taxonomy.items() if r["partition"] == "eval"]

    assert len(evaluation) == 13, f"eval holds A07 to A19, got {sorted(evaluation)}"
    for attack in evaluation:
        assert taxonomy[attack]["waveform_generator"], f"{attack} has no waveform generator"


def test_the_derived_grouping_admits_a_third_value_for_the_case_the_paper_leaves_open():
    """
    A13 filters a TTS voice and the paper does not name the TTS that produced
    it, so whether its phase originates in a human recording is not decidable
    from the source. Recording that as unknown rather than as false is the whole
    difference between evidence and a guess.
    """
    taxonomy = load_taxonomy(TAXONOMY)
    values = {r["natural_phase_origin"] for r in taxonomy.values()}

    assert values == {True, False, None}
    assert taxonomy["A13"]["natural_phase_origin"] is None
    assert taxonomy["A16"]["natural_phase_origin"] is True
    assert taxonomy["A11"]["natural_phase_origin"] is False


def test_an_attack_missing_from_the_taxonomy_stops_the_analysis():
    taxonomy = load_taxonomy(TAXONOMY)
    try:
        check_coverage(["A07", "A99"], taxonomy)
    except ValueError as exc:
        assert "A99" in str(exc)
    else:
        raise AssertionError("an attack with no taxonomy row must stop the run")


# --------------------------------------------------------------------------
# synthetic tables with a known answer
# --------------------------------------------------------------------------

def _table(plan, n_utt=60, per_category=3, seed=0):
    """
    Build a slice table whose per attack answer is known.

    plan maps an attack id to a dict with four shifts, the mean offset applied
    to that attack's SPOOF slices in the phase and magnitude statistics,
    separately for sonorant and obstruent categories.

    The shift goes on the spoof side rather than the bonafide side because the
    bonafide pool is shared by every attack, so a shift there would move all
    thirteen at once. Separation is measured folded, so planting the difference
    on either side gives the same magnitude.
    """
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in ("y", "u", "attack", "category", "phase", "magnitude")}

    def emit(utterance, label, attack, shifts):
        offset = rng.normal(0, 0.7)
        for group, categories in (("son", SONORANTS), ("obs", OBSTRUENTS)):
            for category in categories:
                for _ in range(per_category):
                    rows["y"].append(label)
                    rows["u"].append(utterance)
                    rows["attack"].append(attack)
                    rows["category"].append(category)
                    rows["phase"].append(
                        offset + rng.normal(0, 1) + shifts[f"{group}_phase"])
                    rows["magnitude"].append(
                        offset + rng.normal(0, 1) + shifts[f"{group}_magnitude"])

    zero = {"son_phase": 0.0, "son_magnitude": 0.0,
            "obs_phase": 0.0, "obs_magnitude": 0.0}
    for i in range(n_utt):
        emit(f"B{i:04d}", 0, "-", zero)
    for attack, shifts in plan.items():
        for i in range(n_utt):
            emit(f"{attack}_{i:04d}", 1, attack, {**zero, **shifts})

    data = {k: np.asarray(v) for k, v in rows.items()}
    data["y"] = data["y"].astype(np.int64)
    data["phase"] = data["phase"].astype(np.float64)
    data["magnitude"] = data["magnitude"].astype(np.float64)
    data["bonafide_label"] = 0
    return data


def test_recovers_a_planted_per_attack_asymmetry():
    data = _table({
        "A07": {"son_phase": 1.2},          # phase leads magnitude in sonorants
        "A08": {"obs_phase": 1.2},          # and in obstruents instead
    }, seed=1)

    result = per_attack(data, n_boot=BOOT, seed=0)

    assert result["A07"]["asymmetry"] > 0.10, result["A07"]
    assert result["A08"]["asymmetry"] < -0.10, result["A08"]


def test_an_attack_with_no_structure_gives_an_asymmetry_interval_spanning_zero():
    data = _table({"A07": {}}, seed=2)
    a07 = per_attack(data, n_boot=BOOT, seed=0)["A07"]

    assert a07["asymmetry_ci_low"] <= 0 <= a07["asymmetry_ci_high"], a07


def test_the_level_and_the_asymmetry_are_different_quantities():
    """
    The load bearing test. An attack where magnitude beats phase everywhere, but
    beats it by LESS in sonorants, has a negative level and a positive
    asymmetry. Reporting only the level would call that a reversal of the
    sonorant hypothesis when the hypothesis is about the asymmetry and holds.

    This is the shape the research log recorded as four reversals.
    """
    data = _table({"A08": {"son_magnitude": 1.0, "obs_magnitude": 1.8}}, n_utt=90, seed=3)
    a08 = per_attack(data, n_boot=BOOT, seed=0)["A08"]

    assert a08["sonorant_advantage"] < 0, "planted so phase loses in sonorants too"
    assert a08["asymmetry"] > 0, "but it loses less there, so the asymmetry holds"
    assert a08["asymmetry_ci_low"] > 0, "and that is not noise, by construction"


def test_clustering_is_by_utterance():
    tight = _table({"A07": {"son_phase": 1.0}}, n_utt=90, per_category=3, seed=4)
    loose = _table({"A07": {"son_phase": 1.0}}, n_utt=12, per_category=22, seed=4)

    wide = per_attack(loose, n_boot=BOOT, seed=0)["A07"]
    narrow = per_attack(tight, n_boot=BOOT, seed=0)["A07"]

    wide_width = wide["asymmetry_ci_high"] - wide["asymmetry_ci_low"]
    narrow_width = narrow["asymmetry_ci_high"] - narrow["asymmetry_ci_low"]
    assert wide_width > narrow_width, (
        "twelve utterances must give a wider interval than ninety at a similar "
        "slice count, or the bootstrap is resampling slices"
    )


def test_each_attack_is_scored_against_the_shared_bonafide_pool():
    """
    An attack contributes only spoof utterances, so its separation is measured
    against the bonafide pool. The counts have to show that, because it is the
    reason the thirteen attack statistics are correlated and the reason the
    permutation test below is approximate.
    """
    data = _table({"A07": {"son_phase": 1.0}, "A08": {}}, n_utt=50, seed=5)
    result = per_attack(data, n_boot=BOOT, seed=0)

    assert result["A07"]["n_utterances_bonafide"] == 50
    assert result["A07"]["n_utterances_spoof"] == 50
    assert result["A08"]["n_utterances_bonafide"] == 50


def test_pooling_and_averaging_categories_are_different_definitions():
    """
    The registered definition averages the per category advantages. Pooling the
    slices instead weights each category by how many slices it has. On
    unbalanced categories the two can disagree in sign, which is exactly how the
    original four reversals claim arose, so both are computable here and the
    report prints both.
    """
    data = _table({"A07": {"son_phase": 0.0}}, n_utt=60, per_category=3, seed=40)

    # Make one sonorant category dominate, and give it the opposite sign.
    big = data["category"] == SONORANTS[0]
    spoof = data["y"] == 1
    data["phase"] = np.where(big & spoof, data["phase"] - 2.0, data["phase"])

    by_category = per_attack(data, n_boot=60, seed=0)["A07"]["asymmetry"]
    pooled = pooled_asymmetries(data)["A07"]

    assert abs(by_category - pooled) > 0.01, (
        f"the two definitions should differ on unbalanced categories, "
        f"got {by_category:.4f} and {pooled:.4f}"
    )


# --------------------------------------------------------------------------
# correction across attacks, because thirteen intervals is thirteen chances
# --------------------------------------------------------------------------

def test_fdr_is_applied_across_attacks_rather_than_each_one_alone():
    """
    One attack at raw p = 0.04 among twelve at 0.9 does not survive correction
    across thirteen tests. Reporting it as clear would be counting one of
    thirteen chances as a finding, which is how "8 of 13" and "12 of 13" get
    built out of noise.
    """
    results = {}
    for i, p in enumerate([0.04] + [0.9] * 12):
        results[f"A{i + 7:02d}"] = {
            "asymmetry": 0.03, "asymmetry_p": p,
            "asymmetry_ci_low": 0.001, "asymmetry_ci_high": 0.06,
        }

    apply_fdr(results)

    assert results["A07"]["asymmetry_q"] > 0.05
    assert results["A07"]["asymmetry_clear"] is False, (
        "raw p below alpha is not the same as surviving thirteen tests"
    )


def test_an_interval_spanning_zero_is_never_clear_however_small_its_p():
    results = {"A07": {"asymmetry": 0.0, "asymmetry_p": 0.0001,
                       "asymmetry_ci_low": -0.02, "asymmetry_ci_high": 0.02}}
    apply_fdr(results)
    assert results["A07"]["asymmetry_clear"] is False


def test_a_clearly_negative_attack_survives_correction_and_keeps_its_sign():
    results = {}
    for i, p in enumerate([0.0001] + [0.0002] * 12):
        results[f"A{i + 7:02d}"] = {
            "asymmetry": -0.08 if i == 0 else 0.05, "asymmetry_p": p,
            "asymmetry_ci_low": -0.13 if i == 0 else 0.02,
            "asymmetry_ci_high": -0.02 if i == 0 else 0.09,
        }

    apply_fdr(results)

    assert results["A07"]["asymmetry_clear"] is True
    assert results["A07"]["asymmetry"] < 0


# --------------------------------------------------------------------------
# the group test
# --------------------------------------------------------------------------

def test_exact_permutation_enumerates_rather_than_samples():
    """
    Thirteen attacks split three against ten is 286 assignments, so the null is
    enumerable and the p value is exact. A sampled approximation would return a
    different number on a different seed, which for a table this small is a
    needless source of noise.
    """
    values = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    in_group = np.array([True, True, True, False, False, False])

    r = exact_permutation_p(values, in_group)

    assert r["n_assignments"] == 20, "6 choose 3"
    # The observed split and its mirror are the only two assignments this
    # extreme, so the two sided p is exactly 2/20.
    assert abs(r["p_value"] - 0.1) < 1e-12
    assert abs(r["difference"] - 1.0) < 1e-12


def test_the_permutation_p_does_not_depend_on_which_group_is_called_first():
    values = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])
    in_group = np.array([True, False, True, False, False, True, False])

    forward = exact_permutation_p(values, in_group)
    backward = exact_permutation_p(values, ~in_group)

    assert abs(forward["p_value"] - backward["p_value"]) < 1e-12
    assert abs(forward["difference"] + backward["difference"]) < 1e-12


def test_a_group_of_one_side_only_is_refused_rather_than_scored():
    values = np.array([1.0, 2.0, 3.0])
    try:
        exact_permutation_p(values, np.array([True, True, True]))
    except ValueError as exc:
        assert "group" in str(exc).lower()
    else:
        raise AssertionError("a test with nothing to compare against must raise")


def test_group_means_splits_on_the_taxonomy_and_excludes_the_unknown_case():
    taxonomy = load_taxonomy(TAXONOMY)
    statistics = {a: 1.0 if taxonomy[a]["natural_phase_origin"] is True else 0.0
                  for a in taxonomy if taxonomy[a]["partition"] == "eval"}

    split = group_means(statistics, taxonomy, "natural_phase_origin")

    assert split["excluded"] == ["A13"], "the unknown case must not be forced into a group"
    assert sorted(split["true"]) == ["A16", "A17", "A19"]
    assert len(split["false"]) == 9
    assert abs(split["mean_true"] - 1.0) < 1e-12
    assert abs(split["mean_false"] - 0.0) < 1e-12


def test_a_split_too_small_to_test_says_so_instead_of_returning_a_p_value():
    """
    Three attacks against nine is 220 assignments, so the smallest attainable
    two sided p is about 0.009. Fine. Two against one is not, and the script has
    to refuse rather than report a p value that could never have been small.
    """
    values = np.array([1.0, 0.0, 0.0])
    r = exact_permutation_p(values, np.array([True, False, False]))

    assert r["smallest_attainable_p"] > 0.5
    assert r["underpowered"] is True


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_attack_analysis")
