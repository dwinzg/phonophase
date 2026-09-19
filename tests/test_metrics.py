"""Tests for src/metrics.py."""

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phonophase.metrics import (  # noqa: E402
    ASVSPOOF2019_COST_MODEL,
    compute_eer,
    compute_min_tdcf,
    det_curve,
    evaluate_by_attack,
    evaluate_by_group,
    evaluate_by_phoneme,
    evaluate_scores,
    format_attack_table,
    format_group_table,
    format_report,
    load_asv_scores,
    tdcf_from_labelled_scores,
)

# The ASV scores ship with the corpus. Tests that need them skip when the
# corpus is absent, so CI stays green on a runner with no data.
ASV_EVAL = os.path.join(
    os.path.dirname(__file__), "..", "data", "LA",
    "ASVspoof2019_LA_asv_scores", "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt",
)


def _synthetic_asv(seed=0, n=4000):
    """Stand in ASV scores with the usual ordering, target above nontarget."""
    rng = np.random.default_rng(seed)
    return {
        "target": rng.normal(10.0, 4.0, size=n),
        "nontarget": rng.normal(-10.0, 4.0, size=n),
        # Spoofing attacks land near target scores, which is what makes them
        # dangerous and what t-DCF is built to express.
        "spoof": rng.normal(6.0, 5.0, size=n),
    }


def test_perfect_separation_gives_zero_eer():
    # Scores that separate cleanly. brentq based EER, which is what
    # 05_train_old.py used, raises on exactly this input.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    eer, _ = compute_eer(y_true, y_scores, pos_label=1)
    assert eer == 0.0, f"expected 0.0, got {eer}"


def test_uninformative_scores_give_half_eer():
    # Identical scores everywhere carry no information at all.
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.5, 0.5, 0.5, 0.5])
    eer, _ = compute_eer(y_true, y_scores, pos_label=1)
    assert abs(eer - 0.5) < 1e-9, f"expected 0.5, got {eer}"


def test_eer_is_symmetric_under_score_and_label_flip():
    # Negating scores and flipping which class is positive must leave EER alone.
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=400)
    y_scores = y_true + rng.normal(0, 1.0, size=400)

    eer_a, _ = compute_eer(y_true, y_scores, pos_label=1)
    eer_b, _ = compute_eer(1 - y_true, -y_scores, pos_label=1)
    assert abs(eer_a - eer_b) < 1e-9, f"{eer_a} vs {eer_b}"


def test_inverted_scores_warn():
    # An AUC below chance is the signature of a flipped score convention. The
    # project's zero shot Bengali run reported AUC 40.9 percent, so this guard
    # exists to make that case impossible to report by accident.
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = evaluate_scores(y_true, y_scores, pos_label=1)

    assert result["auc"] < 0.5, f"expected sub chance AUC, got {result['auc']}"
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        "expected a RuntimeWarning about inverted scores"


def test_metric_block_is_internally_consistent():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 2, size=1000)
    y_scores = y_true * 1.5 + rng.normal(0, 1.0, size=1000)

    m = evaluate_scores(y_true, y_scores, pos_label=1)
    c = m["counts"]

    assert c["tp"] + c["fp"] + c["fn"] + c["tn"] == 1000
    expected_precision = c["tp"] / (c["tp"] + c["fp"])
    expected_recall = c["tp"] / (c["tp"] + c["fn"])
    assert abs(m["precision"] - expected_precision) < 1e-12
    assert abs(m["recall"] - expected_recall) < 1e-12

    expected_f1 = 2 * expected_precision * expected_recall / (expected_precision + expected_recall)
    assert abs(m["f1"] - expected_f1) < 1e-12

    for key in ("accuracy", "precision", "recall", "f1", "eer", "auc"):
        assert 0.0 <= m[key] <= 1.0, f"{key} out of range at {m[key]}"


def test_explicit_threshold_is_respected():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.6, 0.9])
    m = evaluate_scores(y_true, y_scores, pos_label=1, threshold=0.5)
    assert m["threshold"] == 0.5
    assert m["accuracy"] == 1.0


def test_single_class_input_is_rejected():
    try:
        compute_eer(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3]))
    except ValueError:
        return
    raise AssertionError("expected ValueError when only one class is present")


def test_report_renders_every_line():
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.1, 0.4, 0.6, 0.9])
    text = format_report(evaluate_scores(y_true, y_scores), title="TEST")
    for label in ("Threshold", "Accuracy", "Precision", "Recall", "F1", "Equal Err", "AUC"):
        assert label in text, f"missing {label} in report"


def test_det_curve_is_monotonic_and_bounded():
    rng = np.random.default_rng(3)
    frr, far, thr = det_curve(rng.normal(2, 1, 500), rng.normal(-2, 1, 500))

    assert len(frr) == len(far) == len(thr)
    assert np.all(np.diff(frr) >= -1e-12), "false rejection must not decrease"
    assert np.all(np.diff(far) <= 1e-12), "false acceptance must not increase"
    assert frr.min() >= 0 and frr.max() <= 1
    assert far.min() >= 0 and far.max() <= 1


def test_perfect_countermeasure_gives_near_zero_tdcf():
    # Bonafide scored far above spoof, so the countermeasure separates cleanly.
    asv = _synthetic_asv()
    bonafide = np.full(500, 10.0)
    spoof = np.full(500, -10.0)

    tdcf, _ = compute_min_tdcf(bonafide, spoof, asv)
    assert tdcf < 0.01, f"perfect countermeasure should approach 0, got {tdcf}"


def test_uninformative_countermeasure_gives_tdcf_near_one():
    # A normalised t-DCF of 1 is the cost of running the ASV system with no
    # protection at all, so a countermeasure carrying no information sits there.
    asv = _synthetic_asv()
    rng = np.random.default_rng(11)
    bonafide = rng.normal(0, 1, 2000)
    spoof = rng.normal(0, 1, 2000)

    tdcf, _ = compute_min_tdcf(bonafide, spoof, asv)
    assert 0.9 < tdcf <= 1.01, f"uninformative countermeasure should sit near 1, got {tdcf}"


def test_better_countermeasure_scores_lower_tdcf():
    asv = _synthetic_asv()
    rng = np.random.default_rng(5)

    weak, _ = compute_min_tdcf(rng.normal(0.3, 1, 2000), rng.normal(0, 1, 2000), asv)
    strong, _ = compute_min_tdcf(rng.normal(3.0, 1, 2000), rng.normal(0, 1, 2000), asv)
    assert strong < weak, f"stronger separation must lower t-DCF, {strong} vs {weak}"


def test_tdcf_wrapper_matches_the_explicit_form():
    # The wrapper flips orientation, since the rest of this module scores
    # higher as more spoof while t-DCF scores higher as more bonafide.
    asv = _synthetic_asv()
    rng = np.random.default_rng(9)

    bonafide = rng.normal(2.0, 1.0, 800)
    spoof = rng.normal(-1.0, 1.0, 800)

    direct, _ = compute_min_tdcf(bonafide, spoof, asv)

    y_true = np.concatenate([np.zeros(800, dtype=int), np.ones(800, dtype=int)])
    y_scores = np.concatenate([-bonafide, -spoof])
    wrapped, _ = tdcf_from_labelled_scores(y_true, y_scores, asv, pos_label=1)

    assert abs(direct - wrapped) < 1e-12, f"{direct} vs {wrapped}"


def test_cost_model_priors_sum_correctly():
    c = ASVSPOOF2019_COST_MODEL
    total = c["Pspoof"] + c["Ptar"] + c["Pnon"]
    assert abs(total - 1.0) < 1e-12, f"priors must sum to 1, got {total}"


def test_real_asv_scores_load_when_present():
    if not os.path.exists(ASV_EVAL):
        return  # corpus not on this machine, nothing to check
    asv = load_asv_scores(ASV_EVAL)

    assert set(asv) == {"target", "nontarget", "spoof"}
    assert asv["target"].size == 5370
    assert asv["nontarget"].size == 33327
    assert asv["spoof"].size == 63882
    for key in asv:
        assert np.all(np.isfinite(asv[key])), f"{key} holds non finite scores"

    # A real ASV system ranks genuine targets above impostors.
    assert asv["target"].mean() > asv["nontarget"].mean()


def test_tdcf_against_real_asv_scores():
    if not os.path.exists(ASV_EVAL):
        return
    asv = load_asv_scores(ASV_EVAL)

    perfect, _ = compute_min_tdcf(np.full(500, 5.0), np.full(500, -5.0), asv)
    assert perfect < 0.01, f"perfect countermeasure on real ASV scores gave {perfect}"

    rng = np.random.default_rng(17)
    useless, _ = compute_min_tdcf(rng.normal(size=3000), rng.normal(size=3000), asv)
    assert 0.9 < useless <= 1.01, f"uninformative countermeasure gave {useless}"


def test_per_attack_breakdown():
    rng = np.random.default_rng(23)
    n_bona = 400

    # Three attacks of increasing difficulty. A03 overlaps bonafide most, so it
    # must come out with the highest EER.
    parts, labels, attacks = [rng.normal(-2.0, 1.0, n_bona)], [np.zeros(n_bona, int)], [np.array(["-"] * n_bona)]
    for name, mean in (("A01", 3.0), ("A02", 1.0), ("A03", -1.8)):
        parts.append(rng.normal(mean, 1.0, 300))
        labels.append(np.ones(300, int))
        attacks.append(np.array([name] * 300))

    y_scores = np.concatenate(parts)
    y_true = np.concatenate(labels)
    attack_ids = np.concatenate(attacks)

    by_attack = evaluate_by_attack(y_true, y_scores, attack_ids, pos_label=1)

    assert set(by_attack) == {"A01", "A02", "A03"}
    assert all(v["n_spoof"] == 300 for v in by_attack.values())
    assert by_attack["A03"]["eer"] > by_attack["A02"]["eer"] > by_attack["A01"]["eer"], \
        "harder attacks must show higher EER"

    table = format_attack_table(by_attack)
    # Worst first, so the hardest attack heads the table.
    assert table.index("A03") < table.index("A01")


def test_grouping_works_for_phoneme_classes():
    # The measurement behind the slicing hypothesis. If the artifact were
    # uniform across speech every class would land at the same EER and slicing
    # would buy nothing. A spread is what would justify it.
    rng = np.random.default_rng(31)
    n_bona = 400

    parts = [rng.normal(-2.0, 1.0, n_bona)]
    labels = [np.zeros(n_bona, int)]
    classes = [np.array(["-"] * n_bona)]

    for name, mean in (("FRICATIVES", 2.5), ("PLOSIVES", 0.5), ("VOWELS_STEADY", -1.9)):
        parts.append(rng.normal(mean, 1.0, 250))
        labels.append(np.ones(250, int))
        classes.append(np.array([name] * 250))

    by_phoneme = evaluate_by_phoneme(
        np.concatenate(labels), np.concatenate(parts), np.concatenate(classes)
    )

    assert set(by_phoneme) == {"FRICATIVES", "PLOSIVES", "VOWELS_STEADY"}
    assert by_phoneme["VOWELS_STEADY"]["eer"] > by_phoneme["FRICATIVES"]["eer"], \
        "classes that overlap bonafide must show higher EER"

    table = format_group_table(by_phoneme, "PHONEME")
    assert "PHONEME" in table
    # Worst first.
    assert table.index("VOWELS_STEADY") < table.index("FRICATIVES")


def test_attack_grouping_is_the_generic_grouping():
    # evaluate_by_attack is a name over evaluate_by_group, so they must agree.
    rng = np.random.default_rng(37)
    y_true = np.concatenate([np.zeros(200, int), np.ones(200, int)])
    y_scores = np.concatenate([rng.normal(0, 1, 200), rng.normal(1.5, 1, 200)])
    groups = np.concatenate([np.array(["-"] * 200), np.array(["A07", "A08"] * 100)])

    a = evaluate_by_attack(y_true, y_scores, groups)
    b = evaluate_by_group(y_true, y_scores, groups)
    assert a == b


def test_group_table_widens_for_long_labels():
    # Phoneme class names are far longer than attack ids, so the column must
    # not truncate them.
    rng = np.random.default_rng(41)
    y_true = np.concatenate([np.zeros(100, int), np.ones(100, int)])
    y_scores = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
    groups = np.concatenate([np.array(["-"] * 100), np.array(["VOWELS_MOVING"] * 100)])

    table = format_group_table(evaluate_by_group(y_true, y_scores, groups), "PHONEME CLASS")
    assert "VOWELS_MOVING" in table
    assert "PHONEME CLASS" in table


def test_group_table_handles_empty_input():
    assert "no group labels" in format_group_table({})


def test_attack_table_handles_empty_input():
    assert "no attack labels" in format_attack_table({})


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_metrics")
