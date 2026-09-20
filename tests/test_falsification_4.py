"""
Tests for scripts/falsification_4.py.

That script evaluates the fourth falsification condition of
docs/PREREGISTRATION.md, which no script in this repository evaluated before:

    The effect is present on train attacks A01 to A06 but absent on eval, which
    would make it a property of those six synthesis systems rather than of
    speech.

The failure modes worth a test here are all quiet ones. None of them crashes.

    The attack filter silently deletes the bonafide side. Bonafide rows carry
    attack '-', so `attack in {"A01", ..., "A06"}` keeps only spoof and leaves a
    single class table that every downstream AUC reads as chance. The condition
    would then read as absent on both partitions for a reason that has nothing
    to do with speech.

    Label 1 is SPOOF in these tables. Assuming the other convention inverts
    every direction while leaving every magnitude intact, so the output still
    looks plausible.

    The LEVEL and the ASYMMETRY get merged. The sonorant advantage on its own
    and the sonorant-minus-obstruent difference are different numbers and this
    project has already shipped two claims that confused them. The condition
    fires or not per quantity, so both are carried separately end to end.

    Presence is read off the point estimate rather than the interval, which
    turns "small" into "absent".

    An unmeasurable partition is reported as absent. Absent is a finding;
    unmeasurable is not. Condition 4 turns on "absent on eval", so a partition
    that could not be measured has to come out UNRESOLVED rather than firing
    the condition.

    The bootstrap resamples slices. Many slices come from one source utterance
    sharing a speaker, a channel and a synthesis system, so slice resampling
    understates the uncertainty and manufactures presence.
"""

import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import falsification_4  # noqa: E402
from falsification_4 import (  # noqa: E402
    DOES_NOT_FIRE,
    FIRES,
    TRAIN_ATTACKS,
    UNRESOLVED,
    condition_4,
    load_attacks,
    measure,
    presence,
)
from analyse_slices import MAGNITUDE, PHASE  # noqa: E402

BOOT = 300
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

HEADER = ("utterance_id,label,attack,language,corpus,category,phoneme,seg,"
          "start_frame,n_frames,duration_s,mgd_std,mgd_iqr,mgd_frame_spread,"
          "mgd_hf_std,mfcc_std,mfcc_iqr,mfcc_frame_spread,hf_energy,hf_lf_tilt")


def _row(utt, label, attack, category, phase, magnitude, seg="run"):
    return (f"{utt},{label},{attack},en,ASV,{category},a,{seg},"
            f"0,10,0.1,0,0,0,{phase},0,0,{magnitude},0,0")


def _table_text(attacks=("A01", "A02", "A03"), n_utt=12, per_utt=4,
                spoof_label=1):
    """
    A small well formed table. Bonafide carries attack '-' and label
    `1 - spoof_label`, so the convention can be inverted for one test.
    """
    bona_label = 1 - spoof_label
    lines = [HEADER]
    for cat in ("NASALS", "APPROXIMANTS", "VOWELS_STEADY",
                "FRICATIVES", "PLOSIVES", "AFFRICATES"):
        for i in range(n_utt):
            for k in range(per_utt):
                lines.append(_row(f"B{cat[:3]}{i:03d}", bona_label, "-", cat,
                                  1.0 + 0.01 * (i * per_utt + k), 0.5))
            for a_i, attack in enumerate(attacks):
                for k in range(per_utt):
                    lines.append(_row(f"S{cat[:3]}{attack}{i:03d}", spoof_label,
                                      attack, cat,
                                      2.0 + 0.01 * (i * per_utt + k) + a_i, 0.6))
    return "\n".join(lines) + "\n"


def _write(tmp, name, text, packed=False):
    path = os.path.join(tmp, name)
    opener = gzip.open if packed else open
    with opener(path, "wt") as fh:
        fh.write(text)
    return path


def _synthetic(son_gap, obs_gap, n_utt=14, per_utt=5, seed=0, unique_utt=False):
    """
    Dict-of-arrays in the shape analyse_slices.load returns.

    `son_gap` and `obs_gap` set how far the phase statistic separates the
    classes in the sonorant and obstruent categories. The magnitude statistic
    is pure noise in both, so the advantage tracks the gap.
    """
    rng = np.random.default_rng(seed)
    data = {}
    for cat, gap in (("NASALS", son_gap), ("APPROXIMANTS", son_gap),
                     ("VOWELS_STEADY", son_gap), ("FRICATIVES", obs_gap),
                     ("PLOSIVES", obs_gap), ("AFFRICATES", obs_gap)):
        y, u, attack, phase, mag = [], [], [], [], []
        for label in (0, 1):
            for i in range(n_utt):
                centre = rng.normal(gap if label == 1 else 0.0, 0.3)
                for k in range(per_utt):
                    y.append(label)
                    tag = f"{cat}{label}{i:03d}"
                    u.append(f"{tag}_{k}" if unique_utt else tag)
                    attack.append("-" if label == 0 else "A01")
                    phase.append(centre + rng.normal(0, 0.05))
                    mag.append(rng.normal(0, 1.0))
        data[cat] = {
            "y": np.asarray(y), "u": np.asarray(u),
            "attack": np.asarray(attack),
            PHASE: np.asarray(phase), MAGNITUDE: np.asarray(mag),
            "bonafide_label": 0,
        }
    return data


# --------------------------------------------------------------------------
# the attack filter
# --------------------------------------------------------------------------

def test_the_attack_filter_keeps_every_bonafide_row():
    """
    Bonafide rows carry attack '-', so the obvious filter deletes the entire
    genuine side and leaves a one class table. Nothing downstream raises: every
    AUC just collapses towards chance, and the condition reads as absent on both
    partitions for a reason that is not about speech at all.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "t.csv", _table_text(attacks=("A01", "A02", "A03")))
        both = load_attacks(path, "run", None)
        filtered = load_attacks(path, "run", ["A01", "A02"])

    for cat in ("NASALS", "FRICATIVES"):
        bona_all = int((both[cat]["y"] == 0).sum())
        bona_kept = int((filtered[cat]["y"] == 0).sum())
        assert bona_kept == bona_all > 0, (cat, bona_kept, bona_all)
        assert set(np.unique(filtered[cat]["y"]).tolist()) == {0, 1}, cat


def test_the_attack_filter_drops_the_attacks_it_was_not_given():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "t.csv", _table_text(attacks=("A01", "A02", "A03")))
        filtered = load_attacks(path, "run", ["A01", "A02"])

    seen = set()
    for d in filtered.values():
        seen |= set(np.unique(d["attack"]).tolist())
    assert seen == {"-", "A01", "A02"}, seen


def test_the_filter_reads_gzip_because_both_tables_are_gzipped():
    """
    results/eval_slices.csv.gz and results/dev_slices.csv.gz are both
    compressed, and opening one as text is what made the registered analysis
    die with UnicodeDecodeError before it read a row.
    """
    import tempfile

    text = _table_text(attacks=("A01",))
    with tempfile.TemporaryDirectory() as tmp:
        plain = _write(tmp, "t.csv", text)
        packed = _write(tmp, "t.csv.gz", text, packed=True)
        a = load_attacks(plain, "run", ["A01"])
        b = load_attacks(packed, "run", ["A01"])

    assert set(a) == set(b)
    for cat in a:
        for key in ("y", "u", "attack", PHASE, MAGNITUDE):
            assert np.array_equal(a[cat][key], b[cat][key]), (cat, key)


def test_the_train_attack_set_is_the_six_the_registration_names():
    assert list(TRAIN_ATTACKS) == ["A01", "A02", "A03", "A04", "A05", "A06"]


def test_the_label_convention_is_read_from_the_attack_column_not_assumed():
    """
    Label 1 is SPOOF in the ASVspoof tables. Assuming it means bonafide inverts
    every direction while leaving every magnitude correct, so the output stays
    plausible and the verdict silently flips.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        normal = _write(tmp, "n.csv", _table_text(spoof_label=1))
        inverted = _write(tmp, "i.csv", _table_text(spoof_label=0))
        a = load_attacks(normal, "run", ["A01"])
        b = load_attacks(inverted, "run", ["A01"])

    assert a["NASALS"]["bonafide_label"] == 0
    assert b["NASALS"]["bonafide_label"] == 1


# --------------------------------------------------------------------------
# the two quantities, kept apart
# --------------------------------------------------------------------------

def test_the_level_and_the_asymmetry_are_reported_as_separate_quantities():
    """
    "The sonorant advantage" names two things. The LEVEL is the sonorant
    phase-minus-magnitude advantage on its own. The ASYMMETRY is that minus the
    obstruent advantage. On data where obstruents carry an advantage of their
    own the two differ, and only one of them is what H1 claims.
    """
    data = _synthetic(son_gap=1.2, obs_gap=0.8)
    r = measure(data, n_boot=BOOT, seed=0)

    assert set(r) >= {"level", "asymmetry", "obstruent_level"}
    for name in ("level", "asymmetry", "obstruent_level"):
        assert set(r[name]) >= {"estimate", "ci_low", "ci_high", "excludes_zero"}

    assert r["obstruent_level"]["estimate"] > 0.05, (
        "the fixture is meant to give obstruents a real advantage too, "
        "otherwise the two quantities cannot be told apart")
    assert abs(r["level"]["estimate"] - r["asymmetry"]["estimate"]) > 0.01, (
        "level and asymmetry came out interchangeable on a fixture built to "
        "separate them")


def test_the_asymmetry_is_the_level_minus_the_obstruent_advantage():
    """Pins the arithmetic, so neither can drift into standing for the other."""
    data = _synthetic(son_gap=1.2, obs_gap=0.6)
    r = measure(data, n_boot=BOOT, seed=0)
    assert np.isclose(
        r["asymmetry"]["estimate"],
        r["level"]["estimate"] - r["obstruent_level"]["estimate"],
        atol=1e-12)


# --------------------------------------------------------------------------
# presence, absence, and the difference between absent and unmeasured
# --------------------------------------------------------------------------

def test_presence_is_decided_by_the_interval_not_by_the_point_estimate():
    """
    A small estimate whose interval clears zero is present. A large estimate
    whose interval straddles zero is not. Reading the estimate alone turns
    "small" into "absent", which is the substitution this condition is most
    exposed to.
    """
    small_but_clear = {"estimate": 0.004, "ci_low": 0.001, "ci_high": 0.007,
                       "excludes_zero": True}
    large_but_noisy = {"estimate": 0.40, "ci_low": -0.10, "ci_high": 0.90,
                       "excludes_zero": False}
    assert presence(small_but_clear) is True
    assert presence(large_but_noisy) is False


def test_a_negative_effect_is_not_presence():
    """
    The condition is about the effect being present, and H1 states its
    direction. An interval that excludes zero from below is a reversal, not a
    presence.
    """
    reversed_ = {"estimate": -0.05, "ci_low": -0.09, "ci_high": -0.01,
                 "excludes_zero": True}
    assert presence(reversed_) is False


def test_an_unmeasurable_quantity_is_undecided_rather_than_absent():
    assert presence(None) is None
    assert presence({"estimate": float("nan"), "ci_low": float("nan"),
                     "ci_high": float("nan"), "excludes_zero": False}) is None


def test_condition_4_fires_only_on_present_train_attacks_and_absent_eval():
    """The condition as worded, as a truth table."""
    assert condition_4(True, False) == FIRES
    assert condition_4(True, True) == DOES_NOT_FIRE
    assert condition_4(False, False) == DOES_NOT_FIRE
    assert condition_4(False, True) == DOES_NOT_FIRE


def test_an_unmeasured_partition_never_fires_and_never_clears_the_condition():
    """
    Undecided is not rejected, and it is not a pass either. A run with too
    little data on either side has refuted nothing and confirmed nothing.
    """
    for train in (True, False, None):
        for eval_ in (True, False, None):
            if train is None or eval_ is None:
                assert condition_4(train, eval_) == UNRESOLVED, (train, eval_)


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------

def test_the_bootstrap_resamples_source_utterances_not_slices():
    """
    Slices from one utterance share a speaker, a channel and, for spoof, one
    synthesis system. The registered setting resamples utterances. Treating each
    slice as independent narrows the interval by roughly the square root of
    slices per utterance and manufactures presence out of pseudoreplication.

    Same rows both times. Only the grouping changes.
    """
    clustered = measure(_synthetic(son_gap=0.5, obs_gap=0.5, per_utt=6),
                        n_boot=BOOT, seed=0)
    as_if_independent = measure(
        _synthetic(son_gap=0.5, obs_gap=0.5, per_utt=6, unique_utt=True),
        n_boot=BOOT, seed=0)

    wide = clustered["level"]["ci_high"] - clustered["level"]["ci_low"]
    narrow = (as_if_independent["level"]["ci_high"]
              - as_if_independent["level"]["ci_low"])
    assert wide > narrow * 1.2, (wide, narrow)


def test_the_usage_block_says_how_to_obtain_a_table_it_does_not_track():
    """
    results/dev_slices.csv.gz is a data artifact this branch does not carry, and
    a documented command naming a file that is not there is the defect that made
    the registered analysis uncallable. So the docstring has to name both tables
    and show how to get the missing one.
    """
    doc = falsification_4.__doc__
    assert "results/dev_slices.csv.gz" in doc
    assert "results/eval_slices.csv.gz" in doc
    assert "git show" in doc or "git checkout" in doc, (
        "the usage block names a table that is not tracked here without saying "
        "how to retrieve it")


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_falsification_4")
