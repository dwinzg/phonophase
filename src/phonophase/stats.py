"""
Statistical machinery for the probes.

Why this file exists
    The first probe run reported bare AUC values from as few as 35 samples per
    class, across roughly 49 category and statistic combinations, with no
    confidence interval, no null and no correction. Numbers like that are not
    interpretable, and the temptation to believe them is exactly the failure
    mode an unsupervised project should be guarding against.

    Everything here has no dependency on the rest of the package, so other
    people can lift it. Two pieces are not corpus agnostic and say so:
    cluster_groups and the bonafide convention below know what an attack column
    is. They sit here anyway, because the alternative was a copy in each script
    and the copies had already drifted apart.

What each piece answers
    cluster_groups          which unit the bootstrap resamples, and so what
                            the interval is a statement about
    shared_cluster_draws    one resample per draw, shared by every statistic
                            computed inside it
    bonafide_label          which label means genuine speech, read from the data
    percentile_interval     the interval, and whether the draw count settles
                            which side of the null its bounds fall on
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


def percentile_interval(draws, alpha: float = 0.05, null: float = 0.0,
                        z: float = 2.576) -> dict:
    """
    The percentile interval, with the Monte Carlo error of each bound.

    A percentile bound is an order statistic of the draws, so it is itself an
    estimate, and rerunning with another seed moves it. When a bound sits close
    to the null that movement decides whether the interval excludes it: on the
    eval table, clustered by system at 2,000 draws, three seeds printed PASS on
    the first falsification condition and one printed FAIL, on identical data.

    The band on each bound is the distribution free interval for a quantile.
    The number of draws below the true q quantile is binomial(n, q), so the
    order statistics at ranks n q -+ z sqrt(n q (1-q)) bracket it with no
    assumption about the shape of the bootstrap distribution. The default z is
    the 99 percent one. A band is None when there are too few draws for those
    ranks to exist, a few hundred at the default settings, and nothing is then
    reported as settled.

    Three answers come back, each True, False, or None for not settled at this
    draw count:

        above_null      the lower bound is above `null`
        below_null      the upper bound is below `null`
        `undecided`     is True when "the interval excludes `null`" is not
                        settled

    This is about Monte Carlo error only. A settled bound can still belong to
    an interval that covers poorly, for instance one over 13 clusters. And it
    stops one seed printing a pass where another prints a fail; a bound near
    the edge of its band can still read as settled on one seed and not settled
    on the next.
    """
    x = np.sort(np.asarray(draws, dtype=np.float64))
    n = x.size
    if n == 0 or not np.isfinite(x).all():
        raise ValueError("percentile_interval needs at least one draw, all finite")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    def band(q):
        half = z * np.sqrt(n * q * (1 - q))
        low_rank = int(np.floor(n * q - half)) - 1
        high_rank = int(np.ceil(n * q + half))
        if low_rank < 0 or high_rank > n - 1:
            return None
        return float(x[low_rank]), float(x[high_rank])

    low_band, high_band = band(alpha / 2), band(1 - alpha / 2)
    above = (None if low_band is None else True if low_band[0] > null
             else False if low_band[1] <= null else None)
    below = (None if high_band is None else True if high_band[1] < null
             else False if high_band[0] >= null else None)
    settled = (above is True or below is True
               or (above is False and below is False))

    return {
        "ci_low": float(lo),
        "ci_high": float(hi),
        "low_band": low_band,
        "high_band": high_band,
        "excludes_null": bool(lo > null or hi < null),
        "above_null": above,
        "below_null": below,
        "undecided": not settled,
    }


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


def _cluster_index(y_true, groups, stratify: bool = True):
    """
    Prepare the clusters for the bootstrap, within each class or, with
    `stratify=False`, once across both. See shared_cluster_draws.
    """
    if not stratify:
        everything = np.zeros(np.asarray(groups).size, dtype=np.int64)
        return [_prepare_clusters(everything, groups, 0)]
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
    stratify: bool = True,
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

    clusters = (_cluster_index(y_true, groups, stratify)
                if groups is not None else None)

    draws = []
    for _ in range(n_boot):
        idx = (_cluster_draw(rng, clusters) if clusters
               else _stratified_resample(rng, pos_idx, neg_idx))
        # A cluster draw can land on one class only. Skip rather than crash.
        if idx.size == 0 or np.unique(y_true[idx]).size < 2:
            continue
        draws.append(fast_auc(y_true[idx], scores[idx]))
    draws = np.asarray(draws, dtype=np.float64)

    interval = percentile_interval(draws, alpha, null=0.5)
    lo, hi = interval["ci_low"], interval["ci_high"]
    return {
        "auc": observed,
        "ci_low": lo,
        "ci_high": hi,
        "n_pos": int(pos_idx.size),
        "n_neg": int(neg_idx.size),
        "n_clusters_pos": int(clusters[0][1].size) if clusters else None,
        "n_clusters_neg": int(clusters[-1][1].size) if clusters else None,
        # An interval spanning 0.5 means the statistic has not been shown to
        # separate the classes at all.
        "crosses_chance": bool(lo <= 0.5 <= hi),
        # Whether that answer would survive another seed at this draw count.
        "chance_undecided": interval["undecided"],
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


def shared_cluster_draws(y_true, groups, n_boot: int, seed: int = 0,
                         stratify: bool = True):
    """
    Yield the row indices for each bootstrap draw, ONE cluster resample per draw.

    Why this is the only draw design in the package
        Two designs used to live here and they disagreed. The pre-registered
        analysis resampled each phoneme category independently and took the mean
        of the per category advantages; `scripts/sonority_voicing.py` resampled
        once and recomputed every cell inside that one draw. The registered path
        was the naive one.

        One source utterance yields slices in every category it contains, so the
        per category advantages are not independent. Drawing them independently
        discards the dependence they share and shrinks the variance of their
        mean, by about half on the eval table, so the interval on a mean over
        three categories came out 40 to 55 percent narrower than it should.

    What a shared draw buys, and what it costs
        A mean over categories gets WIDER, because the shared utterance effect
        no longer averages away. For a difference between two groups of
        categories two effects compete: each group mean gets wider for the same
        reason, and the two group means become positively correlated, which
        cancels in the difference. When the shared effect dominates, the
        difference gets narrower; on the eval table the two nearly offset and
        the interval barely moves. So state the quantity before quoting an
        interval, and measure rather than assume which way it went.

    Pass the POOLED labels and groups, over every category the caller will
    compute inside the draw. The caller then splits the returned rows by
    category. A category the draw missed entirely, or landed on with one class
    only, must be skipped rather than counted as zero.

    `stratify=False` resamples clusters once across both classes instead of
    within each. That is the draw for a cluster present on both sides, such as
    a speaker with bonafide and spoofed utterances: one pick carries the
    speaker's rows into both classes, so whatever the speaker contributes to
    both is kept paired. The class ratio then varies from draw to draw.
    """
    clusters = _cluster_index(y_true, groups, stratify)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield _cluster_draw(rng, clusters)


# A bonafide row carries no attack, and the spellings below are the ones the
# registered corpora use. One definition, because two scripts reading the
# convention differently is the same class of defect as two bootstrap designs.
BONAFIDE_MARKERS = {"-", "", "bonafide", "none", "nan", "real"}


def _read_bonafide_label(labels, attacks):
    """
    Return (label, reason). `label` is None when the table cannot answer, and
    `reason` says why.

    Two callers need different failure behaviour, which is why the reason is
    carried rather than raised here. The registered analysis defaults to
    utterance clustering and never consults the answer, so a table without a
    usable attack column must still analyse. `sonority_voicing` cannot run at
    all without knowing the direction, so it refuses loudly.
    """
    y = np.asarray(labels)
    attack = np.asarray(attacks if attacks is not None else [])
    if attack.size != y.size:
        return None, ("cannot tell which label is bonafide without an attack "
                      "column of the same length as the labels")
    if y.size == 0:
        return None, "the table has no rows, so the bonafide label cannot be read"

    marked = np.isin(np.char.lower(attack.astype(str)), sorted(BONAFIDE_MARKERS))
    if not marked.any() or marked.all():
        return None, ("the attack column marks either no rows or every row as "
                      "bonafide, so the bonafide label cannot be read from it")

    seen = np.unique(y[marked])
    if seen.size != 1:
        return None, (f"attack column and label column disagree: rows with no "
                      f"attack carry labels {seen.tolist()}")
    label = int(seen[0])
    if bool((y[~marked] == label).any()):
        return None, (f"attack column and label column disagree: attacked rows "
                      f"also carry label {label}, which the bonafide rows carry")
    return label, None


def bonafide_label(labels, attacks):
    """
    Which value of the label column means bonafide, read from the attack column.

    In the ASVspoof tables label 1 is SPOOF and bonafide rows carry attack '-'.
    Assuming the other convention inverts every direction in the output while
    leaving every magnitude correct, which is an error that survives review. It
    happened once in this project and was caught only because a direction column
    contradicted a confirmed finding. So the convention is read from the data and
    cross checked rather than assumed.

    None when the table cannot answer. Use require_bonafide_label where an
    analysis cannot proceed without it.
    """
    return _read_bonafide_label(labels, attacks)[0]


def require_bonafide_label(labels, attacks) -> int:
    """bonafide_label, refusing with the reason rather than returning None."""
    label, reason = _read_bonafide_label(labels, attacks)
    if label is None:
        raise ValueError(reason)
    return label


CLUSTERS = ("utterance", "system", "speaker", "speaker-paired")


def cluster_groups(data, cluster="utterance"):
    """
    The grouping the bootstrap resamples, and therefore what its interval means.

    `utterance` resamples source utterances on both sides. The interval is then
    a statement about more utterances drawn from the same synthesis systems.

    `system` keeps bonafide clustered by utterance, since genuine speech has no
    synthesis system, and clusters the spoof side by **attack**. The eval
    partition's 1,495 spoof utterances come from 13 systems at 115 utterances
    each, so under this option the effective sample size on the spoof side is
    13, not 1,495. The interval is then a statement about speech synthesised by
    systems we have not seen, which is what a claim about articulation needs.

    `speaker` clusters both sides by speaker. Eval has 67 speakers, all 67 on
    the bonafide side and 48 of them on the spoof side, so under utterance
    clustering the bonafide side is treated as about 1,500 independent units
    when it holds 67. `speaker-paired` returns the same groups; the difference
    is in the draw, which the caller makes with `stratify=False` so a speaker
    is picked once for both classes.

    Which is correct depends on the claim, not on taste, so all are reported.
    `docs/PREREGISTRATION.md` specifies clustering by source utterance
    throughout, so `utterance` stays the default and the registered result is
    the one it produces.
    """
    if cluster not in CLUSTERS:
        raise ValueError(f"cluster must be one of {CLUSTERS}, got {cluster!r}")
    utterances = np.asarray(data["u"])
    if cluster == "utterance":
        return utterances
    if cluster in ("speaker", "speaker-paired"):
        return np.asarray(data["speaker"])
    attack = np.asarray(data["attack"])
    is_bona = np.asarray(data["y"]) == data["bonafide_label"]
    return np.where(is_bona, utterances, attack)


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
