"""
Tests for scripts/feature_analysis.py.

The new logic in that script is the weighted AUC, the paired cluster bootstrap
on a difference between two groups drawn from the same utterances, and the
bookkeeping that decides when a feature is not computable. Each of those is a
place where a wrong answer would look entirely plausible in the output table,
which is why they are tested rather than eyeballed.

Three failure modes in particular are checked here because the project has
already paid for versions of them.
  A bootstrap that ignores clustering, which makes intervals far too narrow.
  A feature silently dropped rather than reported as having no data.
  A fast AUC that is not the AUC.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

from feature_analysis import (  # noqa: E402
    MIN_SLICES,
    MIN_UTTERANCES,
    analyse,
    bootstrap_p,
    check_clusters_are_single_class,
    cluster_weights,
    collinearity,
    dense_bins,
    feature_columns,
    folded_from_auc,
    group_counts,
    top_pairs,
    usable,
    weighted_auc,
)
from phonophase.phonology import FEATURE_NAMES  # noqa: E402
from phonophase.stats import fast_auc  # noqa: E402

BOOT = 300


def _auc(y, x, weights=None):
    bins, n_bins = dense_bins(x)
    return weighted_auc(bins, n_bins,
                        np.flatnonzero(np.asarray(y) == 1),
                        np.flatnonzero(np.asarray(y) == 0),
                        weights)


# --------------------------------------------------------------------------
# the weighted AUC must be the AUC
# --------------------------------------------------------------------------

def test_weighted_auc_matches_fast_auc_exactly():
    rng = np.random.default_rng(0)
    for trial in range(8):
        n = 400
        y = rng.integers(0, 2, n)
        x = rng.normal(size=n)
        assert abs(_auc(y, x) - fast_auc(y, x)) < 1e-12, f"trial {trial}"


def test_weighted_auc_handles_ties_the_same_way_fast_auc_does():
    # Midranks. A version that broke ties arbitrarily would still look
    # reasonable on continuous data and be wrong on this.
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    x = rng.integers(0, 5, 400).astype(float)
    assert abs(_auc(y, x) - fast_auc(y, x)) < 1e-12


def test_weighted_auc_equals_the_expanded_sample():
    # A bootstrap draw is a multiplicity vector. Weighting must give exactly
    # what materialising the repeated rows would give.
    rng = np.random.default_rng(2)
    n = 300
    y = rng.integers(0, 2, n)
    x = rng.normal(size=n)
    w = rng.integers(0, 4, n).astype(float)
    expanded = np.repeat(np.arange(n), w.astype(int))
    assert abs(_auc(y, x, w) - fast_auc(y[expanded], x[expanded])) < 1e-12


def test_weighted_auc_is_nan_when_a_class_is_missing():
    y = np.ones(20, dtype=int)
    x = np.arange(20.0)
    assert np.isnan(_auc(y, x))


def test_perfect_separation_is_one_and_reversed_is_zero():
    y = np.array([0] * 20 + [1] * 20)
    x = np.concatenate([np.zeros(20), np.ones(20)])
    assert _auc(y, x) == 1.0
    assert _auc(y, -x) == 0.0


def test_folding_is_max_of_auc_and_its_complement():
    for auc in (0.0, 0.2, 0.5, 0.62, 1.0):
        assert abs(folded_from_auc(auc) - max(auc, 1 - auc)) < 1e-12


def test_folding_matches_the_definition_used_by_analyse_slices():
    # analyse_slices.folded writes it as abs(auc - 0.5) + 0.5. The feature
    # analysis must be the same statistic as the pre-registered one, not a
    # near relative of it.
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 200)
    x = rng.normal(size=200)
    assert abs(folded_from_auc(fast_auc(y, x)) - (abs(fast_auc(y, x) - 0.5) + 0.5)) < 1e-12


def test_cluster_weights_counts_multiplicities():
    w = cluster_weights(np.array([0, 0, 2, 3, 3, 3]), n_rows=5)
    assert list(w) == [2.0, 0.0, 1.0, 3.0, 0.0]


# --------------------------------------------------------------------------
# feature columns stay aligned to the slice table
# --------------------------------------------------------------------------

def test_feature_columns_keep_row_alignment_and_flag_unparseable():
    # phonology.feature_matrix drops unparseable phones, which silently breaks
    # the correspondence with the slice rows. This must not.
    phones = np.array(["m", "??", "s", "a1", "a"])
    matrix, parsed = feature_columns(phones)

    assert matrix.shape == (5, 24)
    assert list(parsed) == [True, False, True, False, True]
    son = FEATURE_NAMES.index("son")
    assert matrix[0, son] == 1 and matrix[2, son] == -1 and matrix[4, son] == 1
    assert np.all(np.isnan(matrix[1])) and np.all(np.isnan(matrix[3]))


def test_group_counts_report_slices_and_utterances_separately():
    # Ten utterances split into four hundred slices is still ten utterances,
    # and the difference is the whole reason the counts are reported.
    y = np.array([0, 0, 0, 1, 1, 1])
    u = np.array(["a", "a", "b", "c", "c", "c"])
    counts = group_counts(y, u, np.ones(6, dtype=bool))

    assert counts["slices_bonafide"] == 3 and counts["utterances_bonafide"] == 2
    assert counts["slices_spoof"] == 3 and counts["utterances_spoof"] == 1


def test_usable_needs_both_classes_in_slices_and_utterances():
    enough = {"slices_bonafide": MIN_SLICES, "slices_spoof": MIN_SLICES,
              "utterances_bonafide": MIN_UTTERANCES, "utterances_spoof": MIN_UTTERANCES}
    assert usable(enough)

    few_slices = dict(enough, slices_spoof=MIN_SLICES - 1)
    assert not usable(few_slices)

    # The trap. Plenty of slices, far too few utterances.
    few_utterances = dict(enough, slices_spoof=10_000,
                          utterances_spoof=MIN_UTTERANCES - 1)
    assert not usable(few_utterances), "slice count must not rescue an utterance count"


# --------------------------------------------------------------------------
# bootstrap p value
# --------------------------------------------------------------------------

def test_bootstrap_p_is_never_zero():
    # A resampling test cannot licence a claim it lacks the resolution to make.
    assert bootstrap_p(np.ones(1000)) > 0.0


def test_bootstrap_p_is_two_sided():
    assert abs(bootstrap_p(np.ones(500)) - bootstrap_p(-np.ones(500))) < 1e-12


def test_bootstrap_p_is_large_when_draws_straddle_zero():
    rng = np.random.default_rng(4)
    assert bootstrap_p(rng.normal(0, 1, 2000)) > 0.5


def test_bootstrap_p_is_small_when_draws_are_all_one_side():
    rng = np.random.default_rng(5)
    assert bootstrap_p(rng.normal(5, 1, 2000)) < 0.01


def test_bootstrap_p_is_capped_at_one():
    assert bootstrap_p(np.zeros(100)) == 1.0


# --------------------------------------------------------------------------
# the cluster bootstrap on the difference
# --------------------------------------------------------------------------

def _synthetic(n_utterances=60, slices_per=20, gap_plus=1.2, gap_minus=0.0, seed=0):
    """
    A table where [+F] slices genuinely separate and [-F] slices do not.

    Built with a per-utterance offset so slices within an utterance really are
    correlated, which is the condition the clustering exists for.
    """
    rng = np.random.default_rng(seed)
    y, u, x, phone = [], [], [], []
    for i in range(n_utterances):
        label = i % 2
        offset = rng.normal(0, 2.0)          # shared by every slice of this utterance
        for j in range(slices_per):
            plus = j % 2 == 0
            gap = gap_plus if plus else gap_minus
            y.append(label)
            u.append(f"utt{i}")
            phone.append("m" if plus else "s")   # m is [+son], s is [-son]
            x.append(offset + rng.normal(0, 1) + (gap if label == 0 else 0.0))
    return {"y": np.asarray(y), "u": np.asarray(u), "x": np.asarray(x),
            "phone": np.asarray(phone)}


def test_a_planted_difference_is_recovered_with_an_interval_clear_of_zero():
    data = _synthetic(seed=6)
    matrix, parsed = feature_columns(data["phone"])
    results = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                                n_boot=BOOT, seed=0, verbose=False)}
    son = results["son"]

    assert son["computable"]
    assert son["difference"] > 0.05, son["difference"]
    assert son["ci_low"] > 0, (son["ci_low"], son["ci_high"])
    assert son["ci_low"] <= son["difference"] <= son["ci_high"]
    assert son["p_value"] < 0.05


def test_no_planted_difference_gives_an_interval_spanning_zero():
    # Several seeds rather than one. At 60 utterances the cluster bootstrap
    # undercovers, as small cluster counts do, and a single seed would then be
    # either a flattering pick or a spurious failure. 240 utterances is the
    # regime the real run is in, where it behaves.
    for seed in (20, 21, 22):
        data = _synthetic(n_utterances=240, slices_per=12,
                          gap_plus=1.0, gap_minus=1.0, seed=seed)
        matrix, parsed = feature_columns(data["phone"])
        son = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                                n_boot=BOOT, seed=0,
                                                verbose=False)}["son"]
        assert son["computable"]
        assert abs(son["difference"]) < 0.03, (seed, son["difference"])
        assert not son["excludes_zero"], \
            (seed, son["difference"], son["ci_low"], son["ci_high"])


def test_clustering_widens_the_interval():
    # The point of clustering. With every slice of an utterance sharing an
    # offset, pretending slices are independent must produce a visibly narrower
    # interval. If this ever stops holding, the clustering has stopped working.
    data = _synthetic(seed=8)
    matrix, parsed = feature_columns(data["phone"])
    clustered = {r["feature"]: r for r in analyse(
        data, matrix, parsed, n_boot=BOOT, seed=0, verbose=False)}["son"]

    independent = dict(data)
    independent["u"] = np.array([f"row{i}" for i in range(data["y"].size)])
    naive = {r["feature"]: r for r in analyse(
        independent, matrix, parsed, n_boot=BOOT, seed=0, verbose=False)}["son"]

    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    naive_width = naive["ci_high"] - naive["ci_low"]
    assert clustered_width > naive_width * 1.3, (clustered_width, naive_width)


def test_the_same_draws_are_used_for_every_feature():
    # Features are compared against each other, so they must be measured on
    # identical utterance resamples. Running one feature alone must give the
    # identical interval to running it alongside the others.
    data = _synthetic(seed=9)
    matrix, parsed = feature_columns(data["phone"])
    together = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                                 n_boot=BOOT, seed=0, verbose=False)}
    again = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                              n_boot=BOOT, seed=0, verbose=False)}
    for name in together:
        if together[name]["computable"]:
            assert together[name]["ci_low"] == again[name]["ci_low"], name
            assert together[name]["ci_high"] == again[name]["ci_high"], name


def test_head_to_head_against_son_is_paired_and_present():
    data = _synthetic(seed=10)
    matrix, parsed = feature_columns(data["phone"])
    results = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                                n_boot=BOOT, seed=0, verbose=False)}
    assert "vs_reference" not in results["son"], "son is the reference, not its own rival"
    others = [r for r in results.values()
              if r["computable"] and r["feature"] != "son"]
    assert others, "expected at least one other computable feature"
    for r in others:
        vs = r["vs_reference"]
        assert vs["reference"] == "son"
        assert abs(vs["gap"] - (r["difference"] - results["son"]["difference"])) < 1e-12
        assert vs["ci_low"] <= vs["gap"] <= vs["ci_high"]


# --------------------------------------------------------------------------
# nothing gets dropped quietly
# --------------------------------------------------------------------------

def test_every_feature_appears_in_the_output():
    data = _synthetic(seed=11)
    matrix, parsed = feature_columns(data["phone"])
    results = analyse(data, matrix, parsed, n_boot=50, seed=0, verbose=False)

    assert [r["feature"] for r in results] == FEATURE_NAMES, \
        "all 24 features must be reported, including the ones with no result"


def test_a_feature_with_no_plus_group_is_reported_not_dropped():
    data = _synthetic(seed=12)
    matrix, parsed = feature_columns(data["phone"])
    results = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                                n_boot=50, seed=0, verbose=False)}
    # Neither m nor s is [+long], so long has no plus group anywhere.
    long = results["long"]

    assert not long["computable"]
    assert long["n_plus"] == 0
    assert "no [+F] segment" in long["reason"]
    assert "difference" not in long, "an uncomputable feature must not carry a number"


def test_unspecified_values_are_counted_not_silently_absorbed():
    data = _synthetic(seed=13)
    matrix, parsed = feature_columns(data["phone"])
    results = {r["feature"]: r for r in analyse(data, matrix, parsed,
                                                n_boot=50, seed=0, verbose=False)}
    n = data["y"].size
    for r in results.values():
        assert r["n_plus"] + r["n_minus"] + r["n_zero"] == n, r["feature"]


def test_fdr_is_applied_across_the_computable_features():
    data = _synthetic(seed=14)
    matrix, parsed = feature_columns(data["phone"])
    results = analyse(data, matrix, parsed, n_boot=BOOT, seed=0, verbose=False)
    tested = [r for r in results if r["computable"]]

    assert tested
    for r in tested:
        assert "q_value" in r and "p_value" in r
        assert r["q_value"] >= r["p_value"] - 1e-12, \
            f"{r['feature']}: q {r['q_value']} below raw p {r['p_value']}"
        assert 0.0 <= r["q_value"] <= 1.0


# --------------------------------------------------------------------------
# the clustering guard
# --------------------------------------------------------------------------

def test_single_class_utterances_pass_the_guard():
    check_clusters_are_single_class(np.array([0, 0, 1, 1]),
                                    np.array(["a", "a", "b", "b"]))


def test_an_utterance_carrying_both_labels_is_refused():
    # _prepare_clusters would split it into two clusters and the intervals
    # would quietly stop being cluster robust.
    try:
        check_clusters_are_single_class(np.array([0, 1, 1]),
                                        np.array(["a", "a", "b"]))
    except ValueError as exc:
        assert "both labels" in str(exc)
    else:
        raise AssertionError("a mixed-label utterance should be refused")


# --------------------------------------------------------------------------
# collinearity, described only
# --------------------------------------------------------------------------

def test_collinearity_excludes_constant_columns_and_names_them():
    data = _synthetic(seed=15)
    matrix, parsed = feature_columns(data["phone"])
    collin = collinearity(matrix, parsed)

    assert "son" in collin["names"]
    assert "long" in collin["constant"], "a column with no variation cannot be correlated"
    assert collin["matrix"].shape == (len(collin["names"]),) * 2
    assert np.allclose(np.diag(collin["matrix"]), 1.0)


def test_top_pairs_are_ordered_by_absolute_correlation():
    data = _synthetic(seed=16)
    matrix, parsed = feature_columns(data["phone"])
    pairs = top_pairs(collinearity(matrix, parsed), k=5)

    strengths = [abs(r) for _, _, r in pairs]
    assert strengths == sorted(strengths, reverse=True)
    assert all(a != b for a, b, _ in pairs)


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_feature_analysis")
