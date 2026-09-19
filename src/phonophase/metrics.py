"""
Evaluation metrics for PADS.

Why this file exists
    05_train.py reports loss and accuracy only. Accuracy on ASVspoof 2019 LA is
    close to meaningless, because the train split is 2580 bonafide against
    22800 spoof, so a model that answers spoof every time already scores 89.8
    percent. Equal Error Rate is the metric the anti spoofing literature is
    written in, and it is the number quoted in the project's own research
    update, but nothing in this repository computed it. Now something does.

    This module is also the way to re verify scores that come back from the
    HPC. Save per utterance scores to disk there, load them here, and the
    numbers can be checked independently of the training code that produced
    them.

Score orientation
    Every function takes scores where a HIGHER value means MORE likely to be
    pos_label. Get this backwards and EER silently reports its complement.
    evaluate_scores warns when AUC falls below 0.5, which is the signature of
    an inverted score convention. That warning is worth taking seriously. The
    project's zero shot Bengali run reported AUC 40.9 percent, and an AUC that
    far below chance on a large test set is more often a sign flip than it is
    genuine anti learning.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def compute_eer(y_true, y_scores, pos_label: int = 1) -> tuple[float, float]:
    """
    Return (eer, threshold) with eer as a fraction in [0, 1].

    Found by locating where the false negative rate crosses the false positive
    rate along the ROC curve. This is preferred over the brentq root finding
    used in 05_train_old.py, which raises when the two curves do not bracket a
    sign change. That happens on perfectly separable scores, exactly the case
    a smoke test is most likely to hit.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    if len(np.unique(y_true)) < 2:
        raise ValueError("compute_eer needs both classes present in y_true")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
    fnr = 1.0 - tpr

    # The crossing rarely lands exactly on a sampled ROC point, so take the
    # closest one and average the two rates there.
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    return eer, float(thresholds[idx])


def evaluate_scores(
    y_true,
    y_scores,
    pos_label: int = 1,
    threshold: float | None = None,
) -> dict:
    """
    Produce the full metric block.

    threshold selects the operating point for the accuracy style metrics. Left
    as None it uses Youden's J statistic, the point maximising tpr minus fpr.
    Pass a value explicitly to score at a fixed operating point, which is what
    you want when comparing a deployed threshold across corpora.

    Returned keys
        threshold, eer_threshold, accuracy, precision, recall, f1, eer, auc
    All rates are fractions in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    auc = float(roc_auc_score((y_true == pos_label).astype(int), y_scores))
    if auc < 0.5:
        warnings.warn(
            f"AUC is {auc:.3f}, below chance. Scores are probably inverted "
            f"relative to pos_label={pos_label}. Negate the scores or flip "
            "pos_label and re check before reporting these numbers.",
            RuntimeWarning,
            stacklevel=2,
        )

    eer, eer_threshold = compute_eer(y_true, y_scores, pos_label=pos_label)

    if threshold is None:
        fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=pos_label)
        threshold = float(thresholds[int(np.argmax(tpr - fpr))])

    predicted_pos = y_scores >= threshold
    actual_pos = y_true == pos_label

    tp = int(np.sum(predicted_pos & actual_pos))
    fp = int(np.sum(predicted_pos & ~actual_pos))
    fn = int(np.sum(~predicted_pos & actual_pos))
    tn = int(np.sum(~predicted_pos & ~actual_pos))

    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "threshold": float(threshold),
        "eer_threshold": eer_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "eer": eer,
        "auc": auc,
        "counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def format_report(metrics: dict, title: str = "FINAL EVALUATION RESULTS") -> str:
    """
    Render the metric block in the layout used in the April 2026 research
    update, so old and new results can be compared line by line.
    """
    bar = "=" * 50
    return "\n".join(
        [
            bar,
            title.center(50),
            bar,
            f"Threshold:        {metrics['threshold']:.4f}",
            f"Accuracy (Acc):   {metrics['accuracy'] * 100:.1f}%",
            f"Precision (Prec): {metrics['precision'] * 100:.1f}%",
            f"Recall (Rec):     {metrics['recall'] * 100:.1f}%",
            f"F1 Score (F1):    {metrics['f1'] * 100:.1f}%",
            f"Equal Err (EER):  {metrics['eer'] * 100:.1f}%",
            f"AUC:              {metrics['auc'] * 100:.1f}%",
            bar,
        ]
    )


# ---------------------------------------------------------------------------
# t-DCF
#
# The other metric ASVspoof 2019 is scored on, and the one reviewers expect
# next to EER. It is a tandem metric, so it needs the ASV scores that ship with
# the corpus, at data/LA/ASVspoof2019_LA_asv_scores/.
#
# The formulation below follows Kinnunen et al. 2018 and was checked line by
# line against the reference implementation rather than written from memory,
# because a wrong t-DCF is worse than an absent one.
#
#     C1 = Ptar (Cmiss_cm - Cmiss_asv Pmiss_asv) - Pnon Cfa_asv Pfa_asv
#     C2 = Cfa_cm Pspoof (1 - Pmiss_spoof_asv)
#     tDCF(s) = C1 Pmiss_cm(s) + C2 Pfa_cm(s)
#     normalised by min(C1, C2), then minimised over s
# ---------------------------------------------------------------------------

# The ASVspoof 2019 cost model. Pspoof is the prior on a spoofing attack, Ptar
# and Pnon split the remainder between genuine target and genuine nontarget
# trials, and the C terms are the costs of each error type.
ASVSPOOF2019_COST_MODEL = {
    "Pspoof": 0.05,
    "Ptar": (1 - 0.05) * 0.99,
    "Pnon": (1 - 0.05) * 0.01,
    "Cmiss_asv": 1.0,
    "Cfa_asv": 10.0,
    "Cmiss_cm": 1.0,
    "Cfa_cm": 10.0,
}


def load_asv_scores(path: str) -> dict:
    """
    Read an ASVspoof 2019 ASV score file.

    Three whitespace separated columns, the attack id or the word bonafide,
    the trial type, and the score. Returns target, nontarget and spoof arrays.
    """
    buckets: dict[str, list] = {"target": [], "nontarget": [], "spoof": []}
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            _, trial, score = parts
            if trial in buckets:
                buckets[trial].append(float(score))
    return {k: np.asarray(v, dtype=np.float64) for k, v in buckets.items()}


def det_curve(target_scores, nontarget_scores):
    """
    Return (false rejection rate, false acceptance rate, thresholds).

    Scores are oriented so that a HIGHER value means more target. This is the
    standard detection curve used throughout the speaker verification
    literature and it is what t-DCF is defined over.
    """
    target_scores = np.asarray(target_scores, dtype=np.float64)
    nontarget_scores = np.asarray(nontarget_scores, dtype=np.float64)

    n = target_scores.size + nontarget_scores.size
    all_scores = np.concatenate((target_scores, nontarget_scores))
    labels = np.concatenate((np.ones(target_scores.size), np.zeros(nontarget_scores.size)))

    order = np.argsort(all_scores, kind="mergesort")
    labels = labels[order]

    target_sums = np.cumsum(labels)
    nontarget_sums = nontarget_scores.size - (np.arange(1, n + 1) - target_sums)

    frr = np.concatenate((np.atleast_1d(0), target_sums / target_scores.size))
    far = np.concatenate((np.atleast_1d(1), nontarget_sums / nontarget_scores.size))
    thresholds = np.concatenate(
        (np.atleast_1d(all_scores[order[0]] - 1e-3), all_scores[order])
    )
    return frr, far, thresholds


def asv_error_rates(asv: dict, threshold: float) -> tuple[float, float, float]:
    """Return (Pfa_asv, Pmiss_asv, Pmiss_spoof_asv) at a fixed ASV threshold."""
    return (
        float(np.mean(asv["nontarget"] >= threshold)),
        float(np.mean(asv["target"] < threshold)),
        float(np.mean(asv["spoof"] < threshold)),
    )


def compute_min_tdcf(
    bonafide_scores,
    spoof_scores,
    asv: dict,
    cost_model: dict | None = None,
) -> tuple[float, float]:
    """
    Return (min normalised t-DCF, the CM threshold achieving it).

    Score orientation here is the ASVspoof convention, HIGHER means more
    BONAFIDE, which is the opposite of the pos_label=1 convention the rest of
    this module uses. Pass the two classes as separate arrays so there is no
    way to get it wrong silently. Use tdcf_from_labelled_scores if you are
    starting from a labels and scores pair.

    A perfect countermeasure gives 0. A countermeasure carrying no information
    gives approximately 1, since 1 is the cost of the ASV system operating with
    no protection at all.
    """
    cost_model = cost_model or ASVSPOOF2019_COST_MODEL

    # The ASV system is pinned at its own EER operating point, which is the
    # convention the challenge uses.
    frr_asv, far_asv, thr_asv = det_curve(asv["target"], asv["nontarget"])
    asv_threshold = float(thr_asv[int(np.nanargmin(np.abs(frr_asv - far_asv)))])

    Pfa_asv, Pmiss_asv, Pmiss_spoof_asv = asv_error_rates(asv, asv_threshold)

    # Pmiss_cm rejects bonafide, Pfa_cm accepts spoof.
    Pmiss_cm, Pfa_cm, thresholds = det_curve(bonafide_scores, spoof_scores)

    C1 = (
        cost_model["Ptar"] * (cost_model["Cmiss_cm"] - cost_model["Cmiss_asv"] * Pmiss_asv)
        - cost_model["Pnon"] * cost_model["Cfa_asv"] * Pfa_asv
    )
    C2 = cost_model["Cfa_cm"] * cost_model["Pspoof"] * (1 - Pmiss_spoof_asv)

    if min(C1, C2) <= 0:
        raise ValueError(
            f"degenerate cost model, C1={C1} C2={C2}. Check that the ASV scores "
            "match the partition being evaluated."
        )

    tdcf = C1 * Pmiss_cm + C2 * Pfa_cm
    tdcf_norm = tdcf / min(C1, C2)

    best = int(np.argmin(tdcf_norm))
    return float(tdcf_norm[best]), float(thresholds[best])


def tdcf_from_labelled_scores(y_true, y_scores, asv: dict, pos_label: int = 1):
    """
    Convenience wrapper for the convention used elsewhere in this module, where
    labels are bonafide 0 and spoof 1 and a higher score means more spoof.

    The scores are negated on the way through, because t-DCF is defined with
    higher meaning more bonafide.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    spoof_mask = y_true == pos_label
    return compute_min_tdcf(-y_scores[~spoof_mask], -y_scores[spoof_mask], asv)


def evaluate_by_group(y_true, y_scores, groups, pos_label: int = 1) -> dict:
    """
    EER per group, each measured against the full bonafide pool.

    The grouping is whatever you need it to be. Attack id answers which
    synthesis methods evade the detector. Phoneme class answers whether the
    artifact is concentrated in particular sounds, which is the question the
    slicing branch exists to ask. Language answers where transfer breaks.

    Each group is scored against every bonafide sample rather than against its
    own bonafide subset, so the numbers are comparable across groups.
    """
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=np.float64)
    groups = np.asarray(groups)

    bonafide = y_true != pos_label
    results = {}

    for group in sorted(set(groups[~bonafide].tolist())):
        selected = bonafide | (groups == group)
        try:
            eer, threshold = compute_eer(y_true[selected], y_scores[selected], pos_label)
        except ValueError:
            continue
        results[group] = {
            "eer": eer,
            "threshold": threshold,
            "n_spoof": int(np.sum(groups == group)),
        }
    return results


def evaluate_by_attack(y_true, y_scores, attacks, pos_label: int = 1) -> dict:
    """
    EER per attack type. A thin name over evaluate_by_group, kept because a
    single number over A07 to A19 hides which synthesis methods evade the
    detector and that breakdown deserves an obvious entry point. A17 is a known
    hard case on ASVspoof 2019 LA.
    """
    return evaluate_by_group(y_true, y_scores, attacks, pos_label)


def evaluate_by_phoneme(y_true, y_scores, phoneme_classes, pos_label: int = 1) -> dict:
    """
    EER per phoneme class.

    This is the measurement behind the slicing hypothesis. If the artifact were
    uniform across speech then every class would land at the same EER, and
    slicing would buy nothing. A spread across classes is what would justify it.

    Note that a spread alone is not a novel finding. That fricatives, plosives
    and diphthongs carry the strongest cues is already published, see
    configs/literature.yaml. The open question is whether PHASE behaves
    differently across classes than magnitude does.
    """
    return evaluate_by_group(y_true, y_scores, phoneme_classes, pos_label)


def format_group_table(by_group: dict, label: str = "GROUP") -> str:
    """Render a per group breakdown, worst first."""
    if not by_group:
        return "no group labels available"

    rows = sorted(by_group.items(), key=lambda kv: -kv[1]["eer"])
    width = max(len(label), max(len(str(k)) for k in by_group)) + 1
    lines = [f"{label:<{width}} {'N':>7} {'EER':>8}", "-" * (width + 17)]
    lines += [
        f"{str(group):<{width}} {info['n_spoof']:>7} {info['eer'] * 100:>7.2f}%"
        for group, info in rows
    ]
    return "\n".join(lines)


def format_attack_table(by_attack: dict) -> str:
    """Render the per attack breakdown, worst first."""
    if not by_attack:
        return "no attack labels available"

    rows = sorted(by_attack.items(), key=lambda kv: -kv[1]["eer"])
    lines = [f"{'ATTACK':<8} {'N':>7} {'EER':>8}", "-" * 25]
    lines += [
        f"{attack:<8} {info['n_spoof']:>7} {info['eer'] * 100:>7.2f}%"
        for attack, info in rows
    ]
    return "\n".join(lines)
