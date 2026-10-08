"""
Tests for verdicts that must not depend on the random seed, and for the
speaker level draws.

The defect these tests exist to prevent
    A percentile bootstrap bound is an order statistic of the draws, so it moves
    from seed to seed. On the eval table, clustered by system at 2,000 draws,
    the lower bound of the LEVEL landed at +0.003, -0.001, +0.001 and +0.001 on
    seeds 0 to 3, and the first falsification condition printed PASS, FAIL, PASS
    and PASS on identical data. A verdict that the seed decides is not a verdict.

What is asserted
    That the Monte Carlo band on a bound covers the true quantile at about its
    nominal rate, that a bound inside that band of the null is reported as
    undecided and one well clear of it is not, and that the three state verdict
    never turns undecided into a pass or a fail.

    For the speaker draws: that a paired draw carries one speaker's rows into
    both classes at once, which is the only thing that distinguishes it from a
    stratified draw over the same groups.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import analyse_slices  # noqa: E402
from analyse_slices import UNDECIDED, at_least, require_cluster  # noqa: E402
from phonophase.stats import (  # noqa: E402
    bootstrap_auc_ci,
    cluster_groups,
    percentile_interval,
    shared_cluster_draws,
)


# --------------------------------------------------------------------------
# the Monte Carlo band on a percentile bound
# --------------------------------------------------------------------------

def test_the_band_covers_the_true_quantile_at_about_its_nominal_rate():
    """
    Draws from a standard normal, whose 2.5 and 97.5 percent quantiles are
    known. The band is built from binomial order statistics at the 99 percent
    level and is slightly conservative, so coverage must not fall below nominal
    by more than the sampling error of 400 repetitions.
    """
    rng = np.random.default_rng(0)
    covered_low = covered_high = 0
    reps = 400
    for _ in range(reps):
        r = percentile_interval(rng.normal(size=2000))
        covered_low += r["low_band"][0] <= -1.959964 <= r["low_band"][1]
        covered_high += r["high_band"][0] <= 1.959964 <= r["high_band"][1]
    assert covered_low / reps >= 0.97, covered_low / reps
    assert covered_high / reps >= 0.97, covered_high / reps


def test_the_band_is_not_so_wide_that_everything_is_undecided():
    """A band that always contains the null would also pass the coverage test."""
    r = percentile_interval(np.random.default_rng(1).normal(size=2000))
    assert r["low_band"][1] - r["low_band"][0] < 0.5
    assert r["low_band"][0] <= r["ci_low"] <= r["low_band"][1]
    assert r["high_band"][0] <= r["ci_high"] <= r["high_band"][1]


def test_a_bound_sitting_on_the_null_is_undecided_whichever_side_it_lands():
    """
    The case that printed PASS on three seeds and FAIL on one. The true lower
    bound is placed exactly on the null, so the percentile bound falls either
    side of it by chance. The band is a 99 percent one, so nearly every seed
    must report undecided, and the few that do not are its stated error rate.
    """
    signs, undecided, seeds = set(), 0, 200
    for seed in range(seeds):
        draws = np.random.default_rng(seed).normal(loc=1.959964, size=2000)
        r = percentile_interval(draws)
        signs.add(r["ci_low"] > 0)
        undecided += r["undecided"]
    assert signs == {True, False}, "the fixture must actually flip sign across seeds"
    assert undecided / seeds >= 0.97, undecided / seeds


def test_a_bound_well_clear_of_the_null_is_decided_and_needs_no_more_draws():
    draws = np.random.default_rng(0).normal(loc=5.0, size=2000)
    r = percentile_interval(draws)
    assert not r["undecided"] and r["excludes_null"]
    assert r["draws_to_settle"] is None

    straddling = percentile_interval(np.random.default_rng(0).normal(size=2000))
    assert not straddling["undecided"] and not straddling["excludes_null"]


def test_an_undecided_bound_says_how_many_draws_would_settle_it():
    """More than were used, since the band has to shrink to clear the null."""
    draws = np.random.default_rng(3).normal(loc=1.959964 + 0.02, size=2000)
    r = percentile_interval(draws)
    assert r["undecided"]
    assert r["draws_to_settle"] is None or r["draws_to_settle"] > 2000


def test_the_null_is_a_parameter_because_an_auc_is_tested_against_a_half():
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 300)
    strong = bootstrap_auc_ci(y, y + rng.normal(0, 0.5, y.size), n_boot=400)
    assert not strong["crosses_chance"] and not strong["chance_undecided"]


# --------------------------------------------------------------------------
# the three state verdict
# --------------------------------------------------------------------------

def test_undecided_cells_never_become_a_pass_or_a_fail():
    assert at_least(2, clear=2, unsettled=0) is True
    assert at_least(2, clear=3, unsettled=0) is True
    assert at_least(2, clear=2, unsettled=1) is True, "already decided without it"
    assert at_least(2, clear=1, unsettled=1) is UNDECIDED
    assert at_least(2, clear=0, unsettled=2) is UNDECIDED
    assert at_least(2, clear=1, unsettled=0) is False
    assert at_least(2, clear=0, unsettled=1) is False, "cannot reach 2 either way"


def test_the_registered_interval_functions_report_whether_they_are_settled():
    data = _speaker_table()
    level = analyse_slices.advantage_ci(data, analyse_slices.SONORANTS, 200, 0)
    asym = analyse_slices.asymmetry_ci(data, 200, 0)
    for r in (level, asym):
        assert {"undecided", "draws_to_settle", "excludes_zero"} <= set(r)


# --------------------------------------------------------------------------
# speaker level draws
# --------------------------------------------------------------------------

def _speaker_table(n_speakers=12, utterances_per_speaker=4, per_utt=3, seed=0):
    """
    Six categories. Every speaker has bonafide utterances, and the first eight
    speakers have spoofed ones as well, which is the shape of the eval
    partition: the spoofed speakers are a subset of the bonafide ones.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for category in analyse_slices.SONORANTS + analyse_slices.OBSTRUENTS:
        cols = {"y": [], "u": [], "attack": [], "speaker": [],
                analyse_slices.PHASE: [], analyse_slices.MAGNITUDE: []}
        for s in range(n_speakers):
            for label in ((0, 1) if s < 8 else (0,)):
                for k in range(utterances_per_speaker):
                    for _ in range(per_utt):
                        cols["y"].append(label)
                        cols["u"].append(f"S{s:02d}_{label}_{k}")
                        cols["attack"].append("-" if label == 0 else "A07")
                        cols["speaker"].append(f"S{s:02d}")
                        cols[analyse_slices.PHASE].append(label + rng.normal())
                        cols[analyse_slices.MAGNITUDE].append(rng.normal())
        data[category] = {k: np.asarray(v) for k, v in cols.items()}
        data[category]["bonafide_label"] = 0
    return data


def test_speaker_clustering_groups_rows_by_speaker_on_both_sides():
    d = _speaker_table()["NASALS"]
    assert (cluster_groups(d, "speaker") == d["speaker"]).all()
    assert (cluster_groups(d, "speaker-paired") == d["speaker"]).all()
    try:
        cluster_groups(d, "phone")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown cluster level must be refused")


def test_a_paired_draw_carries_a_speaker_into_both_classes_at_once():
    """
    In a paired draw the bonafide rows and the spoofed rows come from the same
    picked speakers. In a stratified draw the two classes are picked separately,
    so they disagree on most draws. Only speakers with both classes can show the
    difference, hence the restriction to the first eight.
    """
    d = _speaker_table()["NASALS"]
    both = np.isin(d["speaker"], [f"S{s:02d}" for s in range(8)])
    y, speaker = d["y"][both], d["speaker"][both]

    def picks_agree(stratify):
        agree = []
        for idx in shared_cluster_draws(y, speaker, 50, 0, stratify=stratify):
            bona = set(speaker[idx][y[idx] == 0].tolist())
            spoof = set(speaker[idx][y[idx] == 1].tolist())
            agree.append(bona == spoof)
        return np.mean(agree)

    assert picks_agree(stratify=False) == 1.0
    assert picks_agree(stratify=True) < 0.2


def test_a_picked_speaker_brings_every_one_of_its_rows():
    d = _speaker_table()["NASALS"]
    size = {s: int((d["speaker"] == s).sum()) for s in np.unique(d["speaker"])}
    for idx in shared_cluster_draws(d["y"], d["speaker"], 20, 1, stratify=False):
        seen, counts = np.unique(d["speaker"][idx], return_counts=True)
        for s, c in zip(seen.tolist(), counts.tolist()):
            assert c % size[s] == 0, "a speaker was drawn in part"


def test_speaker_clustering_is_refused_when_a_row_has_no_speaker():
    data = _speaker_table()
    require_cluster(data, "speaker")
    unplaced = data["NASALS"]["speaker"].astype(object)
    unplaced[0] = analyse_slices.UNKNOWN_SPEAKER
    data["NASALS"]["speaker"] = unplaced
    for level in ("speaker", "speaker-paired"):
        try:
            require_cluster(data, level)
        except SystemExit as exc:
            assert "speaker" in str(exc)
        else:
            raise AssertionError("a table with an unplaced row must be refused")


def test_the_shipped_table_gets_its_speakers_from_the_protocol():
    """
    results/eval_slices.csv.gz predates the speaker column. Every row must
    still resolve, to the 67 speakers of the eval partition.
    """
    root = os.path.join(os.path.dirname(__file__), "..")
    table = os.path.join(root, "results", "eval_slices.csv.gz")
    speakers = analyse_slices.read_speaker_map(analyse_slices.DEFAULT_PROTOCOL)
    data = analyse_slices.load(table, "run", speakers=speakers)
    require_cluster(data, "speaker")
    seen = set().union(*(set(d["speaker"].tolist()) for d in data.values()))
    assert len(seen) == 67, len(seen)
    spoofed = set().union(*(set(d["speaker"][d["y"] == 1].tolist())
                            for d in data.values()))
    assert len(spoofed) == 48 and spoofed < seen


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_verdict_stability")
