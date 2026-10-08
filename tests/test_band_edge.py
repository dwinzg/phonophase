"""
Tests for scripts/band_edge.py.

What has to be true for its numbers to mean anything
    A treatment must change the audio and nothing else, so it may not change
    the length or shift the signal in time, or the cached alignment would cut
    different speech in the two arms and any difference would be a
    segmentation difference.

    The descriptors must find a band edge that was put there on purpose.

    The interval machinery must be the registered estimator when the arms are
    different recordings, and must take both versions of an utterance together
    when the arms are the same recordings.
"""

import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import analyse_slices as A  # noqa: E402
import band_edge as B  # noqa: E402
import slice_and_probe as S  # noqa: E402
from phonophase.probe import slice_statistics  # noqa: E402


def noise(seconds=2.0, seed=0):
    return np.random.default_rng(seed).standard_normal(int(seconds * B.SR)) * 0.1


def lag_of(a, b, span=40):
    """The shift of b against a that maximises their correlation, in samples."""
    lags = range(-span, span + 1)
    core = slice(span, a.size - span)
    return max(lags, key=lambda k: float(np.dot(a[core], np.roll(b, -k)[core])))


# --------------------------------------------------------------------------
# treatments
# --------------------------------------------------------------------------

def test_every_round_trip_keeps_the_length_and_the_timing():
    for n in (16000, 16001, 23456, 62725):
        y = noise(seed=n)[:n]
        for name in B.RESAMPLERS:
            for rate in (22050, 32000):
                out = B.round_trip(y, rate, name)
                assert out.size == y.size, (name, rate, n)
                assert lag_of(y, out) == 0, (name, rate, n)


def test_a_round_trip_leaves_the_band_below_the_edge_alone():
    t = np.arange(B.SR) / B.SR
    tone = np.sin(2 * np.pi * 3000 * t)
    for name in B.RESAMPLERS:
        out = B.round_trip(tone, 22050, name)
        core = slice(2000, -2000)
        assert np.max(np.abs(out[core] - tone[core])) < 0.02, name


def test_the_low_pass_removes_what_is_above_the_cutoff_without_delay():
    y = noise()
    out = B.low_pass(y, 6000)
    assert out.size == y.size
    freqs, db = B.ltas_db(out)
    _, db_in = B.ltas_db(y)
    assert np.all(db[freqs > 6400] < db_in[freqs > 6400] - 60)
    assert np.max(np.abs(db[freqs < 5600] - db_in[freqs < 5600])) < 0.5
    assert lag_of(y, out) == 0


def test_treat_parses_each_spelling_and_refuses_the_rest():
    y = noise(0.5)
    assert B.treat(y, "none") is y
    assert np.array_equal(B.treat(y, "rt:poly:22050"), B.round_trip(y, 22050, "poly"))
    assert np.array_equal(B.treat(y, "lp:6000"), B.low_pass(y, 6000))
    quantised = B.treat(y, "pcm16") * 32768
    assert np.array_equal(quantised, np.round(quantised))
    assert np.max(np.abs(B.treat(y, "pcm16") - y)) <= 0.5 / 32768 + 1e-12
    assert np.array_equal(B.treat(y, "rt:poly:22050>lp:6000>pcm16"),
                          B.pcm16(B.low_pass(B.round_trip(y, 22050, "poly"), 6000)))
    for bad in ("nonsense", "none+pcm16", "none:1", "rt:poly:22050>"):
        try:
            B.treat(y, bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must be refused")


# --------------------------------------------------------------------------
# descriptors
# --------------------------------------------------------------------------

def test_the_descriptors_find_an_edge_that_was_put_there():
    y = noise(4.0)
    flat, cut = B.describe(y), B.describe(B.low_pass(y, 6000))
    assert flat["bw30_hz"] > 7900 and flat["bw40_hz"] > 7900
    assert abs(cut["bw30_hz"] - 6000) < 150
    assert abs(cut["bw40_hz"] - 6000) < 150
    assert abs(flat["edge_ratio_db"]) < 1.0
    assert cut["edge_ratio_db"] < -40
    assert flat["edge_slope_db_per_khz"] > -6
    assert cut["edge_slope_db_per_khz"] < -100
    assert abs(cut["edge_slope_hz"] - 6000) < 300


def test_the_edge_slope_reaches_an_edge_just_under_nyquist():
    cut = B.describe(B.low_pass(noise(4.0), 7750))
    assert cut["edge_slope_db_per_khz"] < -60
    assert cut["edge_slope_hz"] >= 7700


def test_the_edge_slope_does_not_mistake_a_gentle_tilt_for_a_filter():
    from scipy.signal import lfilter
    tilted = lfilter([1.0], [1.0, -0.9], noise(4.0))
    assert B.describe(tilted)["edge_slope_db_per_khz"] > -15


# --------------------------------------------------------------------------
# slicing at a cached alignment
# --------------------------------------------------------------------------

def _write_utterance(tmp, y):
    import soundfile as sf
    path = os.path.join(tmp, "u1.flac")
    sf.write(path, y, B.SR, subtype="PCM_16")
    return (path, "u1", "-", "en", "test", "bonafide", "-")


def test_slices_match_the_statistics_of_the_same_samples_cut_by_hand():
    import tempfile
    runs = [("m", 10, 11), ("s", 20, 23), ("?", 30, 31), ("a", 40, 41)]
    with tempfile.TemporaryDirectory() as tmp:
        item = _write_utterance(tmp, noise(2.0))
        speech = B.read_audio(item[0])
        rows = B.slice_utterance(item, runs, "none", ["run", "window"])
        assert [r["phoneme"] for r in rows] == ["m", "m", "s", "s", "a", "a"]
        for row, (_, _, seg, first, last, a, b) in zip(
                rows, S.iter_slices(speech.size, runs, ["run", "window"])):
            assert row["seg"] == seg and row["start_frame"] == first
            assert row["n_frames"] == last - first
            expected = slice_statistics(speech[a:b])
            assert row[A.PHASE] == expected[A.PHASE]
            assert row[A.MAGNITUDE] == expected[A.MAGNITUDE]
        assert set(rows[0]) == set(S.FIELDS)


def test_a_treatment_changes_the_statistics_and_not_the_cuts():
    import tempfile
    runs = [("m", 10, 11), ("s", 20, 23), ("a", 40, 41)]
    with tempfile.TemporaryDirectory() as tmp:
        item = _write_utterance(tmp, noise(2.0))
        plain = B.slice_utterance(item, runs, "none", ["run"])
        treated = B.slice_utterance(item, runs, "rt:poly:22050", ["run"])
    cut = ("phoneme", "seg", "start_frame", "n_frames", "duration_s", "category")
    assert [[r[k] for k in cut] for r in plain] == [[r[k] for k in cut] for r in treated]
    assert all(p[A.PHASE] != t[A.PHASE] for p, t in zip(plain, treated))


def test_the_band_column_is_the_registered_statistic_when_the_band_is_the_upper_half():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        item = _write_utterance(tmp, noise(2.0))
        rows = B.slice_utterance(item, [("m", 10, 13)], "none", ["run"],
                                 band_hz=(4000, 8001))
        narrow = B.slice_utterance(item, [("m", 10, 13)], "none", ["run"],
                                   band_hz=(4000, 6500))
    assert abs(rows[0][B.BAND_COLUMN] - rows[0][A.PHASE]) < 1e-12
    assert narrow[0][B.BAND_COLUMN] != narrow[0][A.PHASE]
    assert B.band_rows((4000, 8001)).tolist() == list(range(128, 257))


def test_the_alignment_cache_round_trips_and_keeps_empty_utterances():
    import csv
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "align.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=B.ALIGN_FIELDS)
            w.writeheader()
            w.writerow({"utterance_id": "u1", "n_samples": 9, "phoneme": "m",
                        "first": 3, "last": 5})
            w.writerow({"utterance_id": "u2", "n_samples": 9, "phoneme": "",
                        "first": "", "last": ""})
        assert B.read_alignment(path) == {"u1": [("m", 3, 5)], "u2": []}
        assert S.already_done(path) == {"u1", "u2"}


# --------------------------------------------------------------------------
# intervals
# --------------------------------------------------------------------------

def synthetic_rows(n_utt=80, per=6, shift=0.5, seed=0, attack="A07"):
    """Two classes of different utterances, with a per utterance shock."""
    rng = np.random.default_rng(seed)
    rows = []
    for label in (0, 1):
        for u in range(n_utt):
            shock = rng.standard_normal()
            for cat in A.SONORANTS + A.OBSTRUENTS:
                boost = shift * label * (cat in A.SONORANTS)
                for _ in range(per):
                    rows.append({
                        "utterance_id": f"{label}_{u}", "label": str(label),
                        "attack": attack if label else "-", "category": cat,
                        "seg": "run",
                        A.PHASE: str(shock + boost + rng.standard_normal()),
                        A.MAGNITUDE: str(shock + rng.standard_normal())})
    return rows


def test_the_unpaired_interval_is_the_registered_estimator():
    data = B.to_data(synthetic_rows())
    mine = B.level_asymmetry(data, 200, seed=3)
    registered = A.asymmetry_ci(data, 200, seed=3)
    assert mine["asymmetry"]["value"] == registered["asymmetry"]
    assert mine["asymmetry"]["ci_low"] == registered["ci_low"]
    assert mine["asymmetry"]["ci_high"] == registered["ci_high"]
    level = A.advantage_ci(data, A.SONORANTS, 200, seed=3)
    assert abs(mine["level"]["value"] - level["advantage"]) < 1e-12
    assert mine["asymmetry"]["value"] > 0


def paired_rows(n_utt=80, per=6, seed=0, effect=0.0):
    """One set of utterances, and the same set again as a second arm."""
    rng = np.random.default_rng(seed)
    base, treated = [], []
    for u in range(n_utt):
        shock = 3.0 * rng.standard_normal()
        for cat in A.SONORANTS + A.OBSTRUENTS:
            for _ in range(per):
                p = shock + rng.standard_normal()
                m = shock + rng.standard_normal()
                common = {"utterance_id": f"u{u}", "label": "0", "attack": "-",
                          "category": cat, "seg": "run"}
                base.append({**common, A.PHASE: str(p), A.MAGNITUDE: str(m)})
                bump = effect * (cat in A.SONORANTS)
                treated.append({**common, A.PHASE: str(p + bump + 0.05 * rng.standard_normal()),
                                A.MAGNITUDE: str(m + 0.05 * rng.standard_normal())})
    return base, treated


def test_two_arm_rows_relabels_the_second_arm_and_keeps_the_utterance_ids():
    base, treated = paired_rows(n_utt=12)
    rows = B.two_arm_rows(base, treated, "treated")
    assert len(rows) == len(base) + len(treated)
    data = B.to_data(rows)
    d = data["NASALS"]
    assert d["bonafide_label"] == 0
    assert set(d["u"][d["y"] == 0].tolist()) == set(d["u"][d["y"] == 1].tolist())
    assert set(d["attack"][d["y"] == 1].tolist()) == {"treated"}


def test_each_arm_carries_its_own_table_and_the_treated_one_is_label_1():
    """
    Arm 1 built from the base table would make every contrast a comparison of
    a recording with itself, and swapped arms would reverse every signed AUC.
    """
    base, treated = paired_rows(n_utt=12, effect=5.0)
    rows = B.two_arm_rows(base, treated, "treated")
    half = len(rows) // 2
    assert [r[A.PHASE] for r in rows[:half]] == [r[A.PHASE] for r in base]
    assert [r[A.PHASE] for r in rows[half:]] == [r[A.PHASE] for r in treated]
    assert {r["attack"] for r in rows[:half]} == {"-"}
    assert {r["attack"] for r in rows[half:]} == {"treated"}
    data = B.to_data(rows)
    nasals = data["NASALS"]
    assert nasals["bonafide_label"] == 0
    assert (nasals[A.PHASE][nasals["y"] == 1].mean()
            > nasals[A.PHASE][nasals["y"] == 0].mean() + 3.0)


def test_two_arm_rows_refuses_arms_that_hold_different_slices():
    base, treated = paired_rows(n_utt=12)
    try:
        B.two_arm_rows(base, treated[:-1], "treated")
    except ValueError:
        return
    raise AssertionError("arms of different slices must be refused")


def test_a_sliced_row_is_label_0_for_bonafide_and_1_for_spoof():
    """Label 1 is SPOOF. The other convention inverts every direction."""
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        item = _write_utterance(tmp, noise(2.0))
        for label_str, attack, want in (("bonafide", "-", 0), ("spoof", "A07", 1)):
            labelled = (item[0], item[1], attack, item[3], item[4], label_str, item[6])
            rows = B.slice_utterance(labelled, [("m", 10, 13)], "none", ["run"])
            assert rows and {r["label"] for r in rows} == {want}
            assert {r["attack"] for r in rows} == {attack}


def test_two_arm_rows_drops_spoofed_rows_from_both_arms():
    rows = synthetic_rows(n_utt=12)
    arms = B.two_arm_rows(rows, rows, "treated")
    assert all(r["utterance_id"].startswith("0_") for r in arms)
    assert {r["label"] for r in arms[:len(arms) // 2]} == {"0"}
    assert {r["label"] for r in arms[len(arms) // 2:]} == {1}


def test_a_joint_draw_takes_both_versions_of_every_utterance_it_picks():
    base, treated = paired_rows(n_utt=20)
    data = B.to_data(B.two_arm_rows(base, treated, "treated"))
    y, groups, *_ = A.pooled(data, A.SONORANTS + A.OBSTRUENTS, "utterance")
    idx = next(iter(B.shared_cluster_draws(y, groups, 1, 0, stratify=False)))
    assert idx.size == y.size
    for u in np.unique(groups[idx]):
        picked = idx[groups[idx] == u]
        assert (y[picked] == 0).sum() == (y[picked] == 1).sum()
    # An unpaired draw, by contrast, picks the two arms' utterances separately.
    idx = next(iter(B.shared_cluster_draws(y, groups, 1, 0)))
    assert (set(groups[idx][y[idx] == 0].tolist())
            != set(groups[idx][y[idx] == 1].tolist()))


def test_the_joint_interval_is_narrower_when_the_arms_share_utterances():
    base, treated = paired_rows(effect=0.3)
    data = B.to_data(B.two_arm_rows(base, treated, "treated"))
    joint = B.level_asymmetry(data, 300, seed=0, joint=True)
    unpaired = B.level_asymmetry(data, 300, seed=0, joint=False)
    assert joint["asymmetry"]["value"] == unpaired["asymmetry"]["value"]
    width = lambda r: r["asymmetry"]["ci_high"] - r["asymmetry"]["ci_low"]  # noqa: E731
    assert width(joint) < 0.7 * width(unpaired)


def test_identical_arms_give_no_asymmetry_under_a_joint_draw():
    base, _ = paired_rows()
    data = B.to_data(B.two_arm_rows(base, base, "treated"))
    r = B.level_asymmetry(data, 200, seed=0, joint=True)
    assert r["asymmetry"]["value"] == 0.0
    assert r["asymmetry"]["ci_low"] <= 0.0 <= r["asymmetry"]["ci_high"]


def _sliced(rows):
    return [{**r, "phoneme": "m", "start_frame": i, "n_frames": 1}
            for i, r in enumerate(rows)]


def test_change_is_zero_for_one_table_and_finds_a_removed_asymmetry():
    before = _sliced(synthetic_rows(shift=1.5))
    same = B.change(before, before, 100, seed=0)
    assert same["asymmetry"] == {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}

    after = _sliced(synthetic_rows(shift=0.0))
    moved = B.change(before, after, 100, seed=0)
    assert moved["asymmetry"]["value"] < -0.1
    assert moved["asymmetry"]["ci_high"] < 0
    separate = B.level_asymmetry(B.to_data(before), 100, 0)["asymmetry"]
    assert (moved["asymmetry"]["ci_high"] - moved["asymmetry"]["ci_low"]
            < 2 * (separate["ci_high"] - separate["ci_low"]))


def test_change_refuses_tables_that_do_not_hold_the_same_slices():
    before = _sliced(synthetic_rows(n_utt=12))
    try:
        B.change(before, before[1:], 10, seed=0)
    except ValueError:
        return
    raise AssertionError("mismatched tables must be refused")


def test_against_keeps_bonafide_and_the_named_attacks_only():
    rows = synthetic_rows(n_utt=12, attack="A07") + [
        r for r in synthetic_rows(n_utt=12, attack="A09") if r["label"] == "1"]
    kept = B.against(rows, ["A09"])
    assert {r["attack"] for r in kept} == {"-", "A09"}
    assert B.against(rows, None) is rows


def test_to_data_files_a_substitute_phase_column_only_when_asked():
    rows = [{**r, B.BAND_COLUMN: "7.0"} for r in synthetic_rows(n_utt=12)]
    assert B.to_data(rows)["NASALS"][A.PHASE][0] != 7.0
    assert B.to_data(rows, phase=B.BAND_COLUMN)["NASALS"][A.PHASE][0] == 7.0


def test_descriptor_aucs_are_signed_with_spoof_positive():
    n = 60
    desc = {"attack": np.asarray(["-"] * n + ["A07"] * n + ["A08"] * n),
            "y": np.asarray([0] * n + [1] * (2 * n))}
    rng = np.random.default_rng(0)
    for name in B.DESCRIPTORS:
        desc[name] = np.concatenate([rng.standard_normal(n),
                                     rng.standard_normal(n) + 3,
                                     rng.standard_normal(n) - 3])
    aucs = B.descriptor_aucs(desc, n_boot=50, seed=0)
    assert sorted(aucs) == ["A07", "A08"]
    assert aucs["A07"]["bw30_hz"]["auc"] > 0.95
    assert aucs["A08"]["bw30_hz"]["auc"] < 0.05


def test_spearman_finds_a_monotone_relation_and_reports_the_detectable_rho():
    x = np.arange(13.0)
    rho, p, critical = B.spearman(x, x ** 3, n_perm=4000)
    assert abs(rho - 1.0) < 1e-12 and p < 0.01
    assert 0.5 < critical < 0.62
    rho, p, _ = B.spearman(x, np.random.default_rng(1).permutation(13), n_perm=4000)
    assert p > 0.05


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_band_edge")
