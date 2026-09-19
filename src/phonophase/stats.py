"""
Statistical machinery for the probes.

Why this file exists
    The first probe run reported bare AUC values from as few as 35 samples per
    class, across roughly 49 category and statistic combinations, with no
    confidence interval, no null and no correction. Numbers like that are not
    interpretable, and the temptation to believe them is exactly the failure
    mode an unsupervised project should be guarding against.

    Everything here is corpus agnostic and has no dependency on the rest of the
    package, so other people can lift it.

What each piece answers
    bootstrap_auc_ci        how uncertain is this AUC
    permutation_test_auc    could this separation have arisen by chance
    paired_bootstrap_delta  is A genuinely better than B on the same samples
    benjamini_hochberg      how many of these many tests survive correction
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


def fast_auc(y_true, scores) -> float:
    """
    AUC via the Mann-Whitney U statistic.

    Numerically identical to sklearn's roc_auc_score, ties included, but without
    the per-call validation and label binarisation that dominate when the
    function runs tens of thousands of times inside a bootstrap. On the eval run
    that difference was the gap between an analysis taking seconds and one
    taking over ten minutes.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)

    positive = y == 1
    n_pos = int(positive.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = rankdata(s)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _stratified_resample(rng, pos_idx, neg_idx):
    """
    Resample within each class.

    An unstratified bootstrap can draw a sample containing only one class, and
    AUC is undefined there. Stratifying also keeps the class ratio fixed, which
    matters because these corpora are heavily imbalanced.
    """
    return np.concatenate([
        rng.choice(pos_idx, size=pos_idx.size, replace=True),
        rng.choice(neg_idx, size=neg_idx.size, replace=True),
    ])


def _prepare_clusters(y_true, groups, label):
    """
    Precompute a ragged index for one class, as (sorted_indices, starts, counts).

    Rows belonging to a cluster are laid out contiguously so a bootstrap draw
    becomes a vectorised gather instead of a Python loop. The naive version
    concatenated a few thousand small arrays per draw, which on the eval run
    meant an analysis that should take seconds took over ten minutes.
    """
    mask = np.asarray(y_true) == label
    idx = np.flatnonzero(mask)
    g = np.asarray(groups)[mask]

    order = np.argsort(g, kind="stable")
    idx_sorted = idx[order]
    _, starts, counts = np.unique(g[order], return_index=True, return_counts=True)
    return idx_sorted, starts, counts


def _cluster_draw(rng, prepared):
    """
    Resample whole clusters with replacement and return the row indices.

    Needed because rows are not independent. In this project a source utterance
    yields many phoneme slices sharing a speaker, a channel and, for spoofed
    audio, the same synthesis system. Treating them as independent is
    pseudoreplication and makes intervals far too narrow. On the pilot, 129
    fricative slices came from 23 utterances.
    """
    parts = []
    for idx_sorted, starts, counts in prepared:
        picked = rng.integers(0, starts.size, size=starts.size)
        lens = counts[picked]
        total = int(lens.sum())
        if total == 0:
            continue
        # Expand each picked cluster into its rows without a Python loop.
        which = np.repeat(np.arange(picked.size), lens)
        out_starts = np.zeros(picked.size, dtype=np.int64)
        np.cumsum(lens[:-1], out=out_starts[1:])
        within = np.arange(total) - out_starts[which]
        parts.append(idx_sorted[starts[picked][which] + within])
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def _cluster_index(y_true, groups):
    """Prepare both classes for the cluster bootstrap."""
    prepared = [_prepare_clusters(y_true, groups, label) for label in (1, 0)]
    for idx_sorted, starts, _ in prepared:
        if starts.size == 0:
            raise ValueError("cluster bootstrap needs both classes present")
    return prepared


def bootstrap_auc_ci(
    y_true,
    scores,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    groups=None,
) -> dict:
    """
    Return the observed AUC with a percentile bootstrap confidence interval.

    A wide interval is the useful output when n is small. It is the thing that
    stops a 0.70 from 35 samples being read as a finding.

    Pass groups when rows are not independent, for example a source utterance
    id shared by many phoneme slices. The bootstrap then resamples whole
    clusters rather than rows, which is the only way the interval reflects the
    real amount of information present.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("bootstrap_auc_ci needs both classes present")

    observed = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(seed)

    clusters = _cluster_index(y_true, groups) if groups is not None else None

    draws = []
    for _ in range(n_boot):
        idx = (_cluster_draw(rng, clusters) if clusters
               else _stratified_resample(rng, pos_idx, neg_idx))
        # A cluster draw can land on one class only. Skip rather than crash.
        if idx.size == 0 or np.unique(y_true[idx]).size < 2:
            continue
        draws.append(fast_auc(y_true[idx], scores[idx]))
    draws = np.asarray(draws, dtype=np.float64)

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "auc": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_pos": int(pos_idx.size),
        "n_neg": int(neg_idx.size),
        "n_clusters_pos": int(clusters[0][1].size) if clusters else None,
        "n_clusters_neg": int(clusters[1][1].size) if clusters else None,
        # An interval spanning 0.5 means the statistic has not been shown to
        # separate the classes at all.
        "crosses_chance": bool(lo <= 0.5 <= hi),
    }


def permutation_test_auc(
    y_true,
    scores,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict:
    """
    Two sided permutation test against the null that scores carry no class
    information. Labels are shuffled, which preserves the score distribution
    exactly and so needs no distributional assumption.

    Deviation is measured as |AUC - 0.5| so a statistic that runs in the
    unexpected direction still counts as separation.
    """
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)

    observed = float(roc_auc_score(y_true, scores))
    observed_dev = abs(observed - 0.5)

    rng = np.random.default_rng(seed)
    shuffled = y_true.copy()
    at_least_as_extreme = 0
    for _ in range(n_perm):
        rng.shuffle(shuffled)
        if abs(roc_auc_score(shuffled, scores) - 0.5) >= observed_dev:
            at_least_as_extreme += 1

    # Add one to numerator and denominator. This keeps p strictly positive,
    # since a permutation test cannot licence a claim of p equals zero.
    return {
        "auc": observed,
        "p_value": (at_least_as_extreme + 1) / (n_perm + 1),
        "n_perm": n_perm,
    }


def paired_bootstrap_delta(
    y_true,
    scores_a,
    scores_b,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    groups=None,
) -> dict:
    """
    Confidence interval on AUC(a) minus AUC(b), both measured on the SAME
    samples in each resample.

    This is the right test for the phase against magnitude comparison. Two
    independent intervals that happen to overlap do not tell you whether one
    statistic beats the other, because both are measured on the same slices and
    are therefore correlated. Resampling the slices and taking the difference
    inside each draw accounts for that.
    """
    y_true = np.asarray(y_true)
    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)

    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    if pos_idx.size == 0 or neg_idx.size == 0:
        raise ValueError("paired_bootstrap_delta needs both classes present")

    observed = float(roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b))

    rng = np.random.default_rng(seed)
    clusters = _cluster_index(y_true, groups) if groups is not None else None

    draws = []
    for _ in range(n_boot):
        idx = (_cluster_draw(rng, clusters) if clusters
               else _stratified_resample(rng, pos_idx, neg_idx))
        yt = y_true[idx]
        if idx.size == 0 or np.unique(yt).size < 2:
            continue
        draws.append(fast_auc(yt, scores_a[idx]) - fast_auc(yt, scores_b[idx]))
    draws = np.asarray(draws, dtype=np.float64)

    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "delta": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        # The only claim worth making. An interval excluding zero means one
        # statistic beat the other on these samples.
        "significant": bool(lo > 0 or hi < 0),
    }


def benjamini_hochberg(p_values, alpha: float = 0.05) -> dict:
    """
    Benjamini and Hochberg false discovery rate control.

    The probe runs one test per category and statistic combination, which was
    roughly 49 tests on the first pass. At an uncorrected 0.05 that is about
    two expected false positives, which is enough to manufacture a pattern out
    of noise. FDR is preferred over Bonferroni here because the goal is to find
    directions worth pursuing rather than to make one confirmatory claim.

    Returns the rejection mask, the adjusted q values, and the threshold used.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return {"rejected": np.array([], dtype=bool), "q_values": p.copy(), "threshold": 0.0}

    order = np.argsort(p)
    ranked = p[order]

    # Step up. Find the largest k with p_(k) <= k/n * alpha.
    thresholds = (np.arange(1, n + 1) / n) * alpha
    passing = np.flatnonzero(ranked <= thresholds)
    cutoff = ranked[passing[-1]] if passing.size else 0.0

    # Adjusted values, enforced monotone from the largest p downward.
    q_ranked = ranked * n / np.arange(1, n + 1)
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q = np.empty_like(q_ranked)
    q[order] = q_ranked

    return {"rejected": p <= cutoff, "q_values": q, "threshold": float(cutoff)}
