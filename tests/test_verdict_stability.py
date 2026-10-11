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
from analyse_slices import (  # noqa: E402
    UNDECIDED,
    at_least,
    condition_1,
    overall,
    require_cluster,
)
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


def test_a_bound_well_clear_of_the_null_is_settled_on_the_side_it_is_on():
    above = percentile_interval(np.random.default_rng(0).normal(loc=5.0, size=2000))
    assert above["above_null"] is True and above["below_null"] is False
    assert above["excludes_null"] and not above["undecided"]

    straddling = percentile_interval(np.random.default_rng(0).normal(size=2000))
    assert straddling["above_null"] is False and straddling["below_null"] is False
    assert not straddling["excludes_null"] and not straddling["undecided"]

    below = percentile_interval(np.random.default_rng(0).normal(loc=-5.0, size=2000))
    assert below["above_null"] is False and below["below_null"] is True
    assert below["excludes_null"] and not below["undecided"]


def test_too_few_draws_to_bracket_a_bound_settle_nothing():
    """
    With a handful of draws the ranks the band needs do not exist. Clipping them
    to the smallest and largest draw gave a band that covered the true quantile
    22 percent of the time at ten draws, under a label that says 99.
    """
    for n in (3, 10, 100):
        r = percentile_interval(np.random.default_rng(0).normal(loc=5.0, size=n))
        assert r["low_band"] is None and r["above_null"] is None, n
        assert r["undecided"], f"{n} draws were reported as settling a verdict"
    assert not percentile_interval(
        np.random.default_rng(0).normal(loc=5.0, size=1000))["undecided"]


def test_a_draw_that_is_not_a_number_is_refused_rather_than_read_as_a_fail():
    for bad in ([], [0.1, float("nan"), 0.2]):
        try:
            percentile_interval(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be refused")


def test_an_auc_is_tested_against_a_half_not_against_zero():
    """
    Pure noise must cross chance and must be settled about it. Tested against
    zero instead, an AUC interval near 0.5 would read as excluding the null.
    """
    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 300)
    noise = bootstrap_auc_ci(y, rng.normal(size=y.size), n_boot=1000)
    assert noise["crosses_chance"] and not noise["chance_undecided"]
    strong = bootstrap_auc_ci(y, y + rng.normal(0, 0.5, y.size), n_boot=1000)
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


def _level(low_band, positive):
    return {"advantage": 0.5 * (low_band[0] + low_band[1]) + 0.04,
            "ci_low": low_band[0], "ci_high": low_band[0] + 0.08,
            "positive": positive, "undecided": positive is None, "n_draws": 2000}


def test_condition_1_is_undecided_only_when_the_lower_bound_is_near_zero():
    assert condition_1(None) is None
    assert condition_1(_level((0.02, 0.03), True)) is True
    assert condition_1(_level((-0.002, 0.003), None)) is UNDECIDED
    assert condition_1(_level((-0.05, -0.04), False)) is False
    below_with_its_top_on_zero = {**_level((-0.05, -0.04), False), "undecided": True}
    assert condition_1(below_with_its_top_on_zero) is False


def test_an_interval_wholly_below_zero_fails_condition_1_however_close_its_top():
    """
    The condition is "excludes zero and is positive". An interval of
    [-0.037, -0.002] has its UPPER bound within Monte Carlo error of zero, so
    whether it excludes zero is unsettled, but the condition fails either way.
    Reporting it as undecided would let a decided failure print as
    "not a rejection".
    """
    draws = np.random.default_rng(0).normal(loc=-0.02, scale=0.02 / 1.96, size=2000)
    interval = percentile_interval(draws)
    assert interval["undecided"], "the fixture's upper bound must sit on zero"
    assert interval["above_null"] is False

    data = _speaker_table()
    data = {c: {**d, analyse_slices.PHASE: d[analyse_slices.MAGNITUDE] * 1.0,
                analyse_slices.MAGNITUDE: d["y"] + d[analyse_slices.MAGNITUDE]}
            for c, d in data.items()}
    level = analyse_slices.advantage_ci(data, analyse_slices.SONORANTS, 600, 0)
    assert level["ci_high"] < 0, "magnitude separates and phase does not, by construction"
    assert condition_1(level) is False


def test_the_overall_verdict_over_every_combination_of_states():
    """
    A decided failure is a failure. An undecided condition is neither a pass
    nor a fail and must never print as either. A condition that could not be
    evaluated is not a pass.
    """
    T, F, U = True, False, UNDECIDED
    for primary in (T, F, U, None):
        for c1 in (T, F, U, None):
            for c2 in (T, F):
                for c3 in (T, F, U):
                    states = (primary, c1, c2, c3)
                    got = overall(list(states))
                    if any(s is F for s in states):
                        assert got == "failed", states
                    elif any(s is U for s in states):
                        assert got == "undecided", states
                    elif any(s is None for s in states):
                        assert got == "failed", states
                    else:
                        assert got == "supported", states


def test_the_registered_interval_functions_report_whether_they_are_settled():
    data = _speaker_table()
    level = analyse_slices.advantage_ci(data, analyse_slices.SONORANTS, 600, 0)
    asym = analyse_slices.asymmetry_ci(data, 600, 0)
    for r in (level, asym):
        assert {"undecided", "positive", "excludes_zero"} <= set(r)
    assert level["positive"] is True and not level["undecided"], (
        "phase separates by a full standard deviation in this fixture")


def _run_main(table, *extra):
    """The script's printed output for one command line."""
    import contextlib
    import io
    argv, out = sys.argv, io.StringIO()
    pair = analyse_slices.PHASE, analyse_slices.MAGNITUDE
    sys.argv = ["analyse_slices.py", "--table", table, *extra]
    try:
        with contextlib.redirect_stdout(out):
            analyse_slices.main()
    finally:
        sys.argv = argv
        analyse_slices.PHASE, analyse_slices.MAGNITUDE = pair
    return out.getvalue()


def _write_table(path, data, flat_obstruents=False):
    import csv
    rng = np.random.default_rng(1)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["utterance_id", "speaker", "label", "attack", "category",
                    "seg", analyse_slices.PHASE, analyse_slices.MAGNITUDE])
        for category, d in data.items():
            for i in range(d["y"].size):
                phase = d[analyse_slices.PHASE][i]
                if flat_obstruents and category in analyse_slices.OBSTRUENTS:
                    phase = rng.normal()
                w.writerow([d["u"][i], d["speaker"][i], d["y"][i], d["attack"][i],
                            category, "run", phase, d[analyse_slices.MAGNITUDE][i]])


def test_the_printed_verdict_follows_the_states_end_to_end():
    """
    The pieces above are tested alone. This runs the script itself, because a
    verdict that is computed correctly and printed wrongly is still wrong.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        table = os.path.join(tmp, "t.csv")
        _write_table(table, _speaker_table(n_speakers=30), flat_obstruents=True)

        settled = _run_main(table, "--n-boot", "600")
        assert "PRE-REGISTERED TEST" in settled
        assert "[PASS] sonorant advantage exceeds obstruent" in settled
        assert "[PASS] 1." in settled
        assert "H1 SUPPORTED" in settled

        # Too few draws for any bound to be settled, so nothing may print as a
        # pass or as a rejection.
        unsettled = _run_main(table, "--n-boot", "40")
        assert "[UNDECIDED] sonorant advantage exceeds obstruent" in unsettled
        assert "[UNDECIDED] 1." in unsettled
        assert "[UNDECIDED] 3." in unsettled
        assert "UNDECIDED at this draw count" in unsettled
        assert "SUPPORTED" not in unsettled and "REJECTED" not in unsettled

        other = _run_main(table, "--n-boot", "600", "--cluster", "speaker-paired")
        assert "NOT THE REGISTERED TEST" in other
        assert "no verdict on H1" in other and "H1 SUPPORTED" not in other


def test_the_per_language_path_reports_the_same_three_states():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        table = os.path.join(tmp, "t.csv")
        _write_table(table, _speaker_table(n_speakers=30), flat_obstruents=True)
        settled = _run_main(table, "--by-language", "--n-boot", "600")
        assert any(line.endswith("  holds") for line in settled.splitlines()), settled
        unsettled = _run_main(table, "--by-language", "--n-boot", "40")
        assert "undecided at this draw count" in unsettled
        assert "  holds" not in unsettled
        other = _run_main(table, "--by-language", "--n-boot", "600",
                          "--phase", analyse_slices.MAGNITUDE,
                          "--magnitude", analyse_slices.PHASE)
        assert "NOT THE REGISTERED TEST" in other


def test_a_clear_level_does_not_carry_h1_when_the_difference_is_not_clear():
    """
    H1's primary test is the difference between sonorants and obstruents. With
    phase separating equally in both groups the level passes and the
    difference does not, and the script must not print SUPPORTED.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        table = os.path.join(tmp, "t.csv")
        _write_table(table, _speaker_table(n_speakers=30))
        out = _run_main(table, "--n-boot", "600")
        assert "[PASS] 1." in out
        assert "[FAIL] sonorant advantage exceeds obstruent" in out
        assert "H1 SUPPORTED" not in out


def test_the_script_asks_for_the_paired_draw_when_told_to():
    """Otherwise the speaker-paired row is the speaker row under another name."""
    data = _speaker_table()
    paired = analyse_slices.advantage_ci(data, analyse_slices.SONORANTS, 300, 0,
                                         cluster="speaker-paired")
    apart = analyse_slices.advantage_ci(data, analyse_slices.SONORANTS, 300, 0,
                                        cluster="speaker")
    assert (paired["ci_low"], paired["ci_high"]) != (apart["ci_low"], apart["ci_high"])
    paired = analyse_slices.asymmetry_ci(data, 300, 0, cluster="speaker-paired")
    apart = analyse_slices.asymmetry_ci(data, 300, 0, cluster="speaker")
    assert (paired["ci_low"], paired["ci_high"]) != (apart["ci_low"], apart["ci_high"])


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


def test_speaker_clustering_is_refused_on_a_handful_of_speakers():
    """
    A speaker column holding one constant value gave an interval of zero width
    and printed a pass.
    """
    data = _speaker_table()
    for d in data.values():
        d["speaker"] = np.full(d["speaker"].size, "everyone")
    for level in ("speaker", "speaker-paired"):
        try:
            require_cluster(data, level)
        except SystemExit as exc:
            assert "1 speakers" in str(exc)
        else:
            raise AssertionError("one speaker must not be resampled")


def test_the_paired_draw_reaches_the_per_category_interval_too():
    """
    The per category table and condition 3 come from bootstrap_auc_ci. Under
    speaker-paired it must make the same paired draw as the LEVEL does, or one
    run mixes two designs under one heading.
    """
    d = _speaker_table()["NASALS"]
    args = dict(n_boot=300, seed=0, groups=d["speaker"])
    stratified = bootstrap_auc_ci(d["y"], d[analyse_slices.PHASE], **args)
    paired = bootstrap_auc_ci(d["y"], d[analyse_slices.PHASE], stratify=False, **args)
    assert (stratified["ci_low"], stratified["ci_high"]) != (
        paired["ci_low"], paired["ci_high"])
    assert paired["ci_low"] < paired["auc"] < paired["ci_high"]


def test_a_statistic_the_table_lacks_or_one_compared_with_itself_is_refused():
    root = os.path.join(os.path.dirname(__file__), "..")
    table = os.path.join(root, "results", "eval_slices.csv.gz")
    analyse_slices.check_columns(table, "mgd_hf_std", "hf_lf_tilt")
    for phase, magnitude in (("mgd_hf_std", "no_such_column"),
                             ("mgd_hf_std", "mgd_hf_std")):
        try:
            analyse_slices.check_columns(table, phase, magnitude)
        except SystemExit:
            continue
        raise AssertionError((phase, magnitude))


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
