"""
Tests for scripts/analyse_slices.py.

This script executes the pre-registered decision procedure, so its failure mode
is not a crash but a verdict the data does not support, printed under a heading
that says the verdict was fixed in advance.

Four properties are load bearing and each has a test that fails without it.

    The table opens whatever compression it arrives in. The only table artifact
    in the repository is gzipped and the documented command named a plain .csv
    that is not there, so the registered analysis crashed before reading a row.

    Utterance clustering resamples exactly what it used to resample. That is the
    registered setting and every published number comes from it, so the option
    added here must be incapable of moving it.

    System clustering keeps bonafide clustered by utterance. Collapsing genuine
    speech into one group would shrink the effective sample to two and the
    resulting interval would be meaningless in a way nothing in the output
    would reveal.

    The two clusterings are allowed to disagree, and when they do it is the
    system clustered answer that is about unseen synthesis systems. A dataset
    whose effect is carried by a single system is built here so that the
    disagreement has a known direction.

The LEVEL, the sonorant advantage on its own, and the ASYMMETRY, sonorant
advantage minus obstruent advantage, are separate quantities. This project has
shipped two claims that confused them, so one test asserts the arithmetic that
tells them apart.
"""

import gzip
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))

import analyse_slices  # noqa: E402
from analyse_slices import (  # noqa: E402
    MAGNITUDE,
    PHASE,
    SONORANTS,
    adequately_powered,
    advantage_ci,
    asymmetry_ci,
    bonafide_label,
    cluster_groups,
    load,
    open_table,
    require_cluster,
    with_bonafide_label,
)
from phonophase.stats import bootstrap_auc_ci  # noqa: E402
import phonophase.stats as stats  # noqa: E402

BOOT = 300
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
EVAL_TABLE = os.path.join(ROOT, "results", "eval_slices.csv.gz")

HEADER = ("utterance_id,label,attack,language,corpus,category,phoneme,seg,"
          "start_frame,n_frames,duration_s,mgd_std,mgd_iqr,mgd_frame_spread,"
          "mgd_hf_std,mfcc_std,mfcc_iqr,mfcc_frame_spread,hf_energy,hf_lf_tilt")


def _row(utt, label, attack, category, phase, magnitude, seg="run"):
    return (f"{utt},{label},{attack},en,ASV,{category},a,{seg},"
            f"0,10,0.1,0,0,0,{phase},0,0,{magnitude},0,0")


def _table_text(n=40):
    """A tiny but well formed slice table, both classes, both segmentations."""
    lines = [HEADER]
    for i in range(n):
        lines.append(_row(f"B{i:03d}", 0, "-", "NASALS", 1.0 + i * 0.01, 0.5))
        lines.append(_row(f"S{i:03d}", 1, f"A{7 + i % 3:02d}", "NASALS",
                          2.0 + i * 0.01, 0.6))
        lines.append(_row(f"B{i:03d}", 0, "-", "NASALS", 1.0 + i * 0.01, 0.5,
                          seg="window"))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# the defect: the only table in the repository could not be opened
# --------------------------------------------------------------------------

def test_load_reads_gzip_and_plain_text_to_the_same_arrays():
    """
    The documented command crashed with UnicodeDecodeError on byte 0x8b, which
    is the second byte of the gzip magic number, because load() opened the
    table as text and the only table in the repository is compressed.

    Round trip: the same bytes compressed and uncompressed must load identically.
    """
    import tempfile

    text = _table_text()
    with tempfile.TemporaryDirectory() as tmp:
        plain = os.path.join(tmp, "t.csv")
        packed = os.path.join(tmp, "t.csv.gz")
        with open(plain, "w") as fh:
            fh.write(text)
        with gzip.open(packed, "wt") as fh:
            fh.write(text)

        a = load(plain, "run")
        b = load(packed, "run")

    assert set(a) == set(b) == {"NASALS"}
    for key in ("y", "u", "attack", PHASE, MAGNITUDE):
        assert np.array_equal(a["NASALS"][key], b["NASALS"][key]), key
    assert a["NASALS"]["bonafide_label"] == b["NASALS"]["bonafide_label"] == 0
    assert a["NASALS"]["y"].size == 80


def test_open_table_decides_by_magic_bytes_not_by_suffix():
    """
    A table is routinely renamed on the way out of a notebook or a bundle. The
    bytes are what the reader has to cope with, so a gzip stream called .csv
    must still open.
    """
    import tempfile

    text = _table_text(n=12)
    with tempfile.TemporaryDirectory() as tmp:
        lying = os.path.join(tmp, "not_really_plain.csv")
        with gzip.open(lying, "wt") as fh:
            fh.write(text)
        with open_table(lying) as fh:
            first = fh.readline().strip()
    assert first == HEADER


def test_every_documented_command_names_a_table_that_exists():
    """
    The usage block said results/eval_slices.csv for months while the only file
    present was results/eval_slices.csv.gz. Documentation that names a missing
    file is a defect with the same consequence as a bug, and this is the cheap
    way to stop it coming back.
    """
    words = analyse_slices.__doc__.split()
    tables = [words[i + 1] for i, w in enumerate(words) if w == "--table"]
    assert tables, "the usage block should show at least one --table example"
    for table in tables:
        assert os.path.exists(os.path.join(ROOT, table)), (
            f"the module docstring documents --table {table}, which is not in "
            f"the repository"
        )


# --------------------------------------------------------------------------
# one definition of the clustering, shared by both scripts
# --------------------------------------------------------------------------

def test_both_scripts_use_the_same_cluster_groups_object():
    """
    Two copies of this function would be two definitions of what every interval
    in the project means, free to drift apart one commit at a time. They must be
    the same object, not merely the same behaviour.
    """
    import sonority_voicing

    assert cluster_groups is stats.cluster_groups
    assert sonority_voicing.cluster_groups is stats.cluster_groups


def test_utterance_clustering_hands_back_the_utterance_column_unchanged():
    """
    Every published number comes from the utterance clustered path. The option
    added here must be incapable of moving it, and the strongest available
    statement of that is that the groups are the utterance column itself.
    """
    data = {
        "y": np.array([0, 0, 1, 1]),
        "u": np.array(["b1", "b2", "s1", "s2"]),
        "attack": np.array(["-", "-", "A07", "A08"]),
        "bonafide_label": 0,
    }
    groups = cluster_groups(data, "utterance")
    assert np.array_equal(groups, data["u"])


def test_bonafide_rows_are_not_collapsed_into_one_cluster_under_system():
    """
    Genuine speech has no synthesis system. Its attack column is a single
    marker, so clustering the bonafide side by attack would reduce it from
    hundreds of utterances to one group, and the interval would be a statement
    about nothing. Bonafide stays clustered by utterance under both settings.
    """
    n = 25
    data = {
        "y": np.array([0] * n + [1] * n),
        "u": np.array([f"b{i}" for i in range(n)] + [f"s{i}" for i in range(n)]),
        "attack": np.array(["-"] * n + [f"A{7 + i % 4:02d}" for i in range(n)]),
        "bonafide_label": 0,
    }
    groups = cluster_groups(data, "system")
    is_bona = data["y"] == 0

    assert len(set(groups[is_bona].tolist())) == n, (
        "bonafide must keep one cluster per utterance under system clustering"
    )
    assert "-" not in set(groups[is_bona].tolist())
    assert set(groups[~is_bona].tolist()) == {"A07", "A08", "A09", "A10"}, (
        "the spoof side must cluster by attack, four systems here, not 25 "
        "utterances"
    )


def test_bonafide_label_is_read_from_the_data_and_agrees_with_sonority_voicing():
    """
    Assuming label 1 means bonafide would invert cluster_groups silently: the
    genuine side would collapse into one attack shaped group and the spoof side
    would spread over utterances. Both scripts read the answer from the attack
    column, and they must read the same answer.
    """
    import sonority_voicing

    y = np.array([0, 0, 1, 1, 1])
    attack = np.array(["-", "-", "A07", "A08", "A09"])
    assert bonafide_label(y, attack) == 0
    assert sonority_voicing.bonafide_label({"y": y, "attack": attack}) == 0

    inverted_y = np.array([1, 1, 0, 0, 0])
    assert bonafide_label(inverted_y, attack) == 1
    assert sonority_voicing.bonafide_label(
        {"y": inverted_y, "attack": attack}) == 1


def test_system_clustering_refuses_a_table_it_cannot_read_the_convention_from():
    """
    Quietly falling back to utterance clustering would print a narrow interval
    under a heading that says system. Refuse instead.
    """
    data = with_bonafide_label({"NASALS": {
        "y": np.array([0, 0, 1, 1]),
        "u": np.array(["b1", "b2", "s1", "s2"]),
        "attack": np.array(["", "", "", ""]),
    }})
    assert data["NASALS"]["bonafide_label"] is None

    require_cluster(data, "utterance")  # the registered path is unaffected
    try:
        require_cluster(data, "system")
    except SystemExit as exc:
        assert "attack column" in str(exc)
    else:
        raise AssertionError("system clustering should refuse this table")


# --------------------------------------------------------------------------
# what the two clusterings do to the interval
# --------------------------------------------------------------------------

def _system_structured_category(rng, n_systems=4, utt_per_system=25, per_utt=5,
                                shift=0.8, system_sd=1.0):
    """
    One category whose spoof side carries a per system offset.

    The dependence has to sit at the system level, because that is the structure
    the option exists to respect: a vocoder leaves the same signature on
    everything it produces.
    """
    shocks = {f"A{7 + k:02d}": rng.normal(0, system_sd) for k in range(n_systems)}
    y, u, attack, phase, magnitude = [], [], [], [], []
    for k in range(n_systems):
        system = f"A{7 + k:02d}"
        for i in range(utt_per_system):
            for _ in range(per_utt):
                y.append(1)
                u.append(f"S{k}_{i:03d}")
                attack.append(system)
                phase.append(shocks[system] + rng.normal(0, 1))
                magnitude.append(rng.normal(0, 1))
    for i in range(n_systems * utt_per_system):
        for _ in range(per_utt):
            y.append(0)
            u.append(f"B{i:04d}")
            attack.append("-")
            phase.append(shift + rng.normal(0, 1))
            magnitude.append(rng.normal(0, 1))
    return {
        "y": np.asarray(y),
        "u": np.asarray(u),
        "attack": np.asarray(attack),
        PHASE: np.asarray(phase),
        MAGNITUDE: np.asarray(magnitude),
        "bonafide_label": 0,
    }


def test_system_clustering_gives_strictly_wider_intervals_than_utterance():
    """
    Four systems carry less information than a hundred utterances. If the
    interval does not widen, the option is doing nothing and the wider claim it
    licenses is unearned.

    Checked at both places the script forms an interval: the per category AUC
    and the group level advantage.
    """
    rng = np.random.default_rng(7)
    data = {c: _system_structured_category(rng)
            for c in ("NASALS", "APPROXIMANTS", "VOWELS_STEADY",
                      "FRICATIVES", "PLOSIVES", "AFFRICATES")}

    for cat in data:
        narrow = bootstrap_auc_ci(data[cat]["y"], data[cat][PHASE], n_boot=BOOT,
                                  seed=0, groups=cluster_groups(data[cat], "utterance"))
        wide = bootstrap_auc_ci(data[cat]["y"], data[cat][PHASE], n_boot=BOOT,
                                seed=0, groups=cluster_groups(data[cat], "system"))
        assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"]), (
            f"{cat}: clustering the spoof side by system must widen the interval"
        )
        assert wide["n_clusters_pos"] == 4
        assert narrow["n_clusters_pos"] == 100

    narrow = advantage_ci(data, ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"],
                          BOOT, 0, cluster="utterance")
    wide = advantage_ci(data, ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"],
                        BOOT, 0, cluster="system")
    assert narrow["advantage"] == wide["advantage"], (
        "the clustering changes the interval only. A moved point estimate means "
        "the resampling is being applied to the observed value, which is a bug"
    )
    assert (wide["ci_high"] - wide["ci_low"]) > (narrow["ci_high"] - narrow["ci_low"])


def test_the_two_clusterings_disagree_in_a_known_direction():
    """
    A dataset whose whole effect is carried by one synthesis system out of
    thirteen. The separation is real in this sample and survives resampling
    utterances, because the proportion of that system's utterances barely moves.
    It must not survive resampling systems: a bootstrap over thirteen systems
    leaves that one out about a third of the time, and those draws have no
    effect at all.

    The registered clustering therefore says the statistic clears chance and the
    system clustering says it has not been shown to. That is not a bug in either
    one. They answer different questions, and the second is the one a claim
    about unseen synthesis systems needs.
    """
    rng = np.random.default_rng(11)
    n_systems, utt_per_system, per_utt = 13, 80, 3
    y, u, attack, x = [], [], [], []
    for k in range(n_systems):
        system = f"A{7 + k:02d}"
        # Only A07 differs from genuine speech. The other twelve are drawn from
        # exactly the bonafide distribution.
        offset = 6.0 if k == 0 else 0.0
        for i in range(utt_per_system):
            for _ in range(per_utt):
                y.append(1)
                u.append(f"S{k}_{i:03d}")
                attack.append(system)
                x.append(offset + rng.normal(0, 1))
    for i in range(n_systems * utt_per_system):
        for _ in range(per_utt):
            y.append(0)
            u.append(f"B{i:04d}")
            attack.append("-")
            x.append(rng.normal(0, 1))

    data = {"y": np.asarray(y), "u": np.asarray(u), "attack": np.asarray(attack),
            "bonafide_label": 0}
    x = np.asarray(x)

    by_utterance = bootstrap_auc_ci(data["y"], x, n_boot=2000, seed=0,
                                    groups=cluster_groups(data, "utterance"))
    by_system = bootstrap_auc_ci(data["y"], x, n_boot=2000, seed=0,
                                 groups=cluster_groups(data, "system"))

    assert by_utterance["auc"] == by_system["auc"], "same sample, same AUC"
    assert not by_utterance["crosses_chance"], (
        "resampling utterances keeps the proportion of A07 nearly fixed, so the "
        "effect looks solid"
    )
    assert by_system["crosses_chance"], (
        "resampling systems drops A07 entirely in about a third of draws, so the "
        "effect must not be shown to clear chance"
    )
    assert by_system["ci_low"] < by_utterance["ci_low"]


# --------------------------------------------------------------------------
# the LEVEL and the ASYMMETRY are different numbers
# --------------------------------------------------------------------------

def test_asymmetry_is_the_level_minus_the_obstruent_advantage():
    """
    The LEVEL is the sonorant advantage. The ASYMMETRY is the sonorant advantage
    minus the obstruent advantage. H1 is about the second. Two shipped claims in
    this project reported one and named the other, so the arithmetic that keeps
    them apart is asserted rather than assumed.
    """
    rng = np.random.default_rng(5)
    data = {c: _system_structured_category(rng, shift=0.9 if c in
                                           ("NASALS", "APPROXIMANTS", "VOWELS_STEADY")
                                           else 0.1)
            for c in ("NASALS", "APPROXIMANTS", "VOWELS_STEADY",
                      "FRICATIVES", "PLOSIVES", "AFFRICATES")}

    son = advantage_ci(data, ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"], BOOT, 0)
    obs = advantage_ci(data, ["FRICATIVES", "PLOSIVES", "AFFRICATES"], BOOT, 1)
    asym = asymmetry_ci(data, BOOT, 0)

    assert abs(asym["asymmetry"] - (son["advantage"] - obs["advantage"])) < 1e-12
    assert asym["asymmetry"] != son["advantage"], (
        "if these ever coincide the test has stopped distinguishing them"
    )
    assert asym["ci_low"] < asym["asymmetry"] < asym["ci_high"]


# --------------------------------------------------------------------------
# the real table
# --------------------------------------------------------------------------

def test_the_eval_table_has_thirteen_spoof_clusters_under_system_clustering():
    """
    The claim this option exists to support is that the eval partition's 1,495
    spoof utterances come from 13 synthesis systems, so the effective sample
    size on the spoof side is 13 and not 1,495. Read off the real table rather
    than asserted in a docstring.

    Thirteen is few for a cluster bootstrap, where the usual guidance is thirty
    to fifty. The interval it produces is reported with that caveat, not
    silently traded for a different estimator.
    """
    if not os.path.exists(EVAL_TABLE):
        return  # the table is a downloaded artifact, absent on a fresh clone

    data = load(EVAL_TABLE, "run")
    for cat, d in data.items():
        groups = cluster_groups(d, "system")
        is_bona = d["y"] == d["bonafide_label"]
        assert d["bonafide_label"] == 0
        assert len(set(groups[~is_bona].tolist())) == 13, cat
        assert len(set(groups[is_bona].tolist())) == len(set(d["u"][is_bona].tolist()))
        assert np.array_equal(cluster_groups(d, "utterance"), d["u"])


# --------------------------------------------------------------------------
# the power precondition, which is counted in bonafide utterances
# --------------------------------------------------------------------------

def test_the_power_check_counts_bonafide_utterances_and_not_both_classes():
    """
    docs/PREREGISTRATION.md requires "at least 200 bonafide source utterances
    per phoneme category" and states that the binding constraint is bonafide
    utterances. The check read min(bonafide, spoof), so it also demanded 200
    spoofed utterances under a label that says bonafide.

    That is stricter than the registered requirement. A category with 1,344
    bonafide utterances and 150 spoofed ones meets the registration and was
    reported as underpowered, which on a table with a small spoof side would
    demote a result the registration says is adequately powered.
    """
    per_category = {
        "NASALS": {"utterances": (1344, 150)},
        "APPROXIMANTS": {"utterances": (1211, 1296)},
        "VOWELS_STEADY": {"utterances": (1493, 1484)},
    }
    assert adequately_powered(per_category, SONORANTS) == list(SONORANTS)


def test_the_power_check_still_rejects_too_few_bonafide_utterances():
    """The requirement is 200, and 199 is not 200."""
    per_category = {
        "NASALS": {"utterances": (199, 9999)},
        "APPROXIMANTS": {"utterances": (200, 9999)},
        "VOWELS_STEADY": {"utterances": (1493, 1484)},
    }
    powered = adequately_powered(per_category, SONORANTS)
    assert "NASALS" not in powered
    assert powered == ["APPROXIMANTS", "VOWELS_STEADY"]


def test_a_category_that_never_got_measured_is_not_powered():
    """
    A category dropped by `usable` never reaches per_category at all, and a
    missing count must not read as a satisfied requirement.
    """
    assert adequately_powered({}, SONORANTS) == []


def test_the_power_check_is_unchanged_on_the_published_eval_counts():
    """
    The correction must not move the published verdict. The eval run reported
    1344 / 1211 / 1493 bonafide utterances against spoof counts all above 200,
    so it passed before and passes now.
    """
    per_category = {
        "NASALS": {"utterances": (1344, 1383)},
        "APPROXIMANTS": {"utterances": (1211, 1296)},
        "VOWELS_STEADY": {"utterances": (1493, 1484)},
    }
    assert len(adequately_powered(per_category, SONORANTS)) == 3


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_analyse_slices")
