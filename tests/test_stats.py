"""
Tests for src/phonophase/stats.py.

These exist because the first probe run reported bare AUC values from as few as
35 samples across roughly 49 tests with no interval, no null and no correction.
The machinery that fixes that has to be correct, or it replaces one false
confidence with another.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phonophase.stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_auc_ci,
    paired_bootstrap_delta,
    permutation_test_auc,
)

# Bootstrap draws are kept low here so the suite stays fast. Real runs use more.
BOOT = 400
PERM = 400


def _separable(n=200, gap=3.0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    s = np.concatenate([rng.normal(0, 1, n), rng.normal(gap, 1, n)])
    return y, s


def test_perfect_separation_gives_tight_interval_at_one():
    y = np.concatenate([np.zeros(50, int), np.ones(50, int)])
    s = np.concatenate([np.zeros(50), np.ones(50)])
    r = bootstrap_auc_ci(y, s, n_boot=BOOT, seed=0)

    assert r["auc"] == 1.0
    assert r["ci_low"] == 1.0 and r["ci_high"] == 1.0
    assert not r["crosses_chance"]
    assert r["n_pos"] == 50 and r["n_neg"] == 50


def test_uninformative_scores_give_an_interval_spanning_chance():
    rng = np.random.default_rng(1)
    y = np.concatenate([np.zeros(150, int), np.ones(150, int)])
    s = rng.normal(size=300)
    r = bootstrap_auc_ci(y, s, n_boot=BOOT, seed=0)

    assert r["crosses_chance"], f"interval {r['ci_low']:.3f} to {r['ci_high']:.3f} should span 0.5"


def test_small_samples_give_wider_intervals_than_large_ones():
    # The point of the interval. A 0.70 from 35 samples must not read the same
    # as a 0.70 from 2000.
    wide = bootstrap_auc_ci(*_separable(n=25, gap=1.0, seed=2), n_boot=BOOT, seed=0)
    narrow = bootstrap_auc_ci(*_separable(n=1000, gap=1.0, seed=2), n_boot=BOOT, seed=0)

    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"]) * 2


def test_interval_brackets_the_observed_value():
    r = bootstrap_auc_ci(*_separable(n=300, gap=1.5), n_boot=BOOT, seed=0)
    assert r["ci_low"] <= r["auc"] <= r["ci_high"]


def test_bootstrap_needs_both_classes():
    try:
        bootstrap_auc_ci(np.ones(20, int), np.arange(20.0), n_boot=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError when one class is missing")


def test_permutation_detects_real_separation():
    y, s = _separable(n=150, gap=2.0, seed=3)
    r = permutation_test_auc(y, s, n_perm=PERM, seed=0)
    assert r["p_value"] < 0.01, f"clear separation should give a small p, got {r['p_value']}"


def test_permutation_does_not_flag_noise():
    rng = np.random.default_rng(4)
    y = np.concatenate([np.zeros(120, int), np.ones(120, int)])
    r = permutation_test_auc(y, rng.normal(size=240), n_perm=PERM, seed=0)
    assert r["p_value"] > 0.05, f"noise should not be flagged, got {r['p_value']}"


def test_permutation_p_is_never_zero():
    # A permutation test cannot licence p equals zero, so the estimator adds one
    # to numerator and denominator.
    y = np.concatenate([np.zeros(60, int), np.ones(60, int)])
    s = np.concatenate([np.zeros(60), np.ones(60)])
    r = permutation_test_auc(y, s, n_perm=PERM, seed=0)
    assert r["p_value"] > 0, "p must stay strictly positive"
    assert r["p_value"] <= 1.0 / (PERM + 1) + 1e-12


def test_permutation_is_two_sided():
    # A statistic running the wrong way is still separation, and must register.
    y, s = _separable(n=150, gap=2.0, seed=5)
    forward = permutation_test_auc(y, s, n_perm=PERM, seed=0)
    reversed_ = permutation_test_auc(y, -s, n_perm=PERM, seed=0)
    assert reversed_["auc"] < 0.5 < forward["auc"]
    assert reversed_["p_value"] < 0.01


def test_paired_delta_detects_a_genuinely_better_statistic():
    rng = np.random.default_rng(6)
    y = np.concatenate([np.zeros(300, int), np.ones(300, int)])
    strong = np.concatenate([rng.normal(0, 1, 300), rng.normal(2.0, 1, 300)])
    weak = np.concatenate([rng.normal(0, 1, 300), rng.normal(0.2, 1, 300)])

    r = paired_bootstrap_delta(y, strong, weak, n_boot=BOOT, seed=0)
    assert r["delta"] > 0
    assert r["significant"], f"interval {r['ci_low']:.3f} to {r['ci_high']:.3f} should exclude 0"
    assert r["ci_low"] > 0


def test_paired_delta_finds_no_difference_between_equals():
    # Same statistic against itself. The difference is exactly zero and the
    # interval must not claim a winner.
    y, s = _separable(n=250, gap=1.2, seed=7)
    r = paired_bootstrap_delta(y, s, s, n_boot=BOOT, seed=0)

    assert abs(r["delta"]) < 1e-12
    assert not r["significant"]
    assert r["ci_low"] <= 0 <= r["ci_high"]


def test_paired_delta_is_antisymmetric():
    y, s = _separable(n=200, gap=1.0, seed=8)
    rng = np.random.default_rng(9)
    other = s + rng.normal(0, 2, s.size)

    ab = paired_bootstrap_delta(y, s, other, n_boot=BOOT, seed=0)
    ba = paired_bootstrap_delta(y, other, s, n_boot=BOOT, seed=0)
    assert abs(ab["delta"] + ba["delta"]) < 1e-12


def test_cluster_bootstrap_widens_intervals_on_clustered_data():
    # The reason this exists. Rows drawn from a few sources are not independent.
    # Here 20 clusters produce 10 near identical rows each, so the effective
    # sample size is 20 and not 200. A row level bootstrap will understate the
    # uncertainty badly.
    rng = np.random.default_rng(101)
    y, s_, g = [], [], []
    for cluster in range(40):
        label = cluster % 2
        # One offset per cluster, so rows within a cluster carry almost no
        # independent information.
        centre = rng.normal(label * 0.8, 1.0)
        for _ in range(10):
            y.append(label)
            s_.append(centre + rng.normal(0, 0.02))
            g.append(f"utt{cluster}")
    y = np.array(y); s_ = np.array(s_); g = np.array(g)

    naive = bootstrap_auc_ci(y, s_, n_boot=BOOT, seed=0)
    clustered = bootstrap_auc_ci(y, s_, n_boot=BOOT, seed=0, groups=g)

    naive_width = naive["ci_high"] - naive["ci_low"]
    clustered_width = clustered["ci_high"] - clustered["ci_low"]

    assert clustered_width > naive_width * 1.5, (
        f"cluster bootstrap must be wider, got {clustered_width:.3f} "
        f"against {naive_width:.3f}"
    )
    assert clustered["auc"] == naive["auc"], "the point estimate must not change"
    assert clustered["n_clusters_pos"] == 20
    assert clustered["n_clusters_neg"] == 20


def test_cluster_bootstrap_matches_naive_when_each_row_is_its_own_cluster():
    # Degenerate case. One row per cluster means the two procedures are the
    # same thing, so they should broadly agree.
    y, s_ = _separable(n=120, gap=1.2, seed=102)
    g = np.arange(y.size).astype(str)

    naive = bootstrap_auc_ci(y, s_, n_boot=BOOT, seed=0)
    clustered = bootstrap_auc_ci(y, s_, n_boot=BOOT, seed=0, groups=g)

    naive_width = naive["ci_high"] - naive["ci_low"]
    clustered_width = clustered["ci_high"] - clustered["ci_low"]
    assert abs(clustered_width - naive_width) < 0.06, \
        f"widths should be close, {clustered_width:.3f} against {naive_width:.3f}"


def test_paired_delta_accepts_clusters():
    rng = np.random.default_rng(103)
    y, s_, g = [], [], []
    for cluster in range(30):
        label = cluster % 2
        centre = rng.normal(label * 1.0, 1.0)
        for _ in range(8):
            y.append(label); s_.append(centre + rng.normal(0, 0.05)); g.append(f"u{cluster}")
    y = np.array(y); strong = np.array(s_); g = np.array(g)
    weak = strong + rng.normal(0, 3.0, strong.size)

    naive = paired_bootstrap_delta(y, strong, weak, n_boot=BOOT, seed=0)
    clustered = paired_bootstrap_delta(y, strong, weak, n_boot=BOOT, seed=0, groups=g)

    assert abs(clustered["delta"] - naive["delta"]) < 1e-12
    assert (clustered["ci_high"] - clustered["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


def test_benjamini_hochberg_on_the_original_worked_example():
    # The fifteen hypothesis example from Benjamini and Hochberg 1995. The
    # step up procedure rejects the first four. Verified by hand against
    # p_(k) <= k/n * alpha.
    p = np.array([0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298,
                  0.0344, 0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000])
    r = benjamini_hochberg(p, alpha=0.05)
    assert r["rejected"].sum() == 4, f"expected 4 rejections, got {int(r['rejected'].sum())}"
    assert r["rejected"][:4].all()
    assert not r["rejected"][4:].any()


def test_benjamini_hochberg_is_a_step_up_not_a_per_test_threshold():
    # This set has five values under an uncorrected 0.05 but only two survive
    # the step up, because p_(3) of 0.039 exceeds its threshold of 0.015. A
    # naive implementation that thresholds each p independently would wrongly
    # reject five, so this pins the procedure rather than the outcome.
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    r = benjamini_hochberg(p, alpha=0.05)

    assert int((p < 0.05).sum()) == 5, "sanity check on the fixture"
    assert r["rejected"].sum() == 2, f"step up should reject 2, got {int(r['rejected'].sum())}"
    assert r["rejected"][:2].all()


def test_benjamini_hochberg_controls_false_discoveries_under_the_null():
    # Forty nine tests is roughly what the probe runs. Uncorrected, about two
    # would pass at 0.05 by chance alone, which is enough to invent a pattern.
    rng = np.random.default_rng(11)
    p = rng.uniform(size=49)
    r = benjamini_hochberg(p, alpha=0.05)
    assert r["rejected"].sum() <= 2, f"too many rejections under the null, {int(r['rejected'].sum())}"


def test_q_values_are_monotone_and_bounded():
    rng = np.random.default_rng(12)
    p = np.sort(rng.uniform(size=30))
    q = benjamini_hochberg(p)["q_values"]

    assert np.all(q >= 0) and np.all(q <= 1)
    assert np.all(np.diff(q) >= -1e-12), "q values must not decrease with p"
    assert np.all(q >= p - 1e-12), "adjusted values cannot fall below raw p values"


def test_benjamini_hochberg_handles_empty_input():
    r = benjamini_hochberg([])
    assert r["rejected"].size == 0


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_stats")
