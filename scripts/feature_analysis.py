"""
Which of PanPhon's 24 articulatory features predict phase separation?

What this does
    The confirmed result in `docs/PREREGISTRATION.md` is stated over a seven
    category phoneme map, `mgd_hf_std` separates bonafide from spoofed more
    strongly in sonorants than in obstruents. That map is English shaped and
    covers 41 of the 389 phone symbols the aligner emits, so it cannot carry a
    cross linguistic claim. Articulatory features can, because every phone in
    every language decomposes into the same 24 dimensions.

    So this asks the same question one dimension at a time. For each feature,
    slices split into the [+F] group and the [-F] group, 0 meaning unspecified
    is discarded and counted, and separation of `mgd_hf_std` between bonafide
    and spoofed is measured within each group, folded with max(auc, 1-auc) so a
    statistic running spoof-high counts the same as one running bonafide-high.
    The reported quantity is the DIFFERENCE, [+F] separation minus [-F]
    separation, with a cluster bootstrap interval.

    This is a restatement of the pre-registered contrast in a universal
    vocabulary, not a new confirmatory test. It is EXPLORATORY. Nothing here
    licences a claim that `docs/PREREGISTRATION.md` does not already licence.

Why it is built the way it is
    Clustering. Slices are not independent. One source utterance yields many
    slices sharing a speaker, a channel and, when spoofed, one synthesis
    system. Every interval here resamples whole utterances via
    `stats._cluster_draw`. Utterances, not slices, are the effective sample
    size, so both counts are printed everywhere.

    Paired draws. The [+F] and [-F] groups are drawn from the same utterances,
    so their two AUCs are correlated. Each bootstrap draw resamples utterances
    ONCE and recomputes both groups inside that draw, which is the only way the
    interval on the difference reflects that correlation. Two independent
    intervals that happen not to overlap would not have answered the question.

    The same draws for every feature. One rng, one sequence of utterance
    resamples, reused across all features, so differences between features are
    not differences between resampling accidents.

    Exact AUC on weighted samples. A bootstrap draw is a multiplicity vector
    over the original rows, so the AUC is the Mann-Whitney statistic with those
    multiplicities as weights, midranks included. It agrees with
    `stats.fast_auc` to floating point, and it avoids re-sorting tens of
    thousands of values two hundred thousand times. Verified in
    `tests/test_feature_analysis.py`.

    Collinearity is described, not decomposed. [+son] and [+voi] pick out
    nearly the same slices here. The pairwise correlations are printed so a
    reader can see which results are restatements of each other. No
    unregularised regression is fitted and no per feature coefficient is
    reported, because with correlations of this size a coefficient does not
    isolate anything. The optional `--multivariate` pass is a regularised fit
    scored on held out utterances and reports predictive performance only.

What it deliberately does not do
    It does not drop a feature for being uninteresting. Every feature is listed,
    including the ones with no result. A feature with no [+F] segment in the
    corpus is reported as NOT COMPUTABLE, which is a statement about the
    inventory and not about the feature.

Usage
    python scripts/feature_analysis.py --table results/eval_slices.csv.gz
    python scripts/feature_analysis.py --table results/eval_slices.csv.gz --seg window
    python scripts/feature_analysis.py --table results/eval_slices.csv.gz --multivariate

Reads .csv or .csv.gz. Runs on CPU in well under a minute.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from analyse_slices import open_table  # noqa: E402

from phonophase.phonology import FEATURE_NAMES, features  # noqa: E402
from phonophase.stats import (  # noqa: E402
    _cluster_draw,
    _cluster_index,
    benjamini_hochberg,
    fast_auc,
)

PHASE = "mgd_hf_std"

# Same usability floor as scripts/analyse_slices.py, applied per class per
# group. Counted in utterances as well as slices, because ten utterances split
# into four hundred slices is still ten utterances.
MIN_SLICES = 30
MIN_UTTERANCES = 10


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path, seg, stat=PHASE):
    """Read one segmentation out of the slice table into aligned arrays."""
    y, utt, phone, value = [], [], [], []
    with open_table(path, ("seg", "label", "utterance_id", "phoneme", stat)) as fh:
        for r in csv.DictReader(fh):
            if r["seg"] != seg:
                continue
            y.append(int(r["label"]))
            utt.append(r["utterance_id"])
            phone.append(r["phoneme"])
            value.append(float(r[stat]))
    return {
        "y": np.asarray(y, dtype=np.int64),
        "u": np.asarray(utt),
        "phone": np.asarray(phone),
        "x": np.asarray(value, dtype=np.float64),
    }


def feature_columns(phones):
    """
    Feature value per row, aligned to the input.

    Returns (matrix of shape (n, 24), boolean mask of rows that parsed). Rows
    that do not parse are NaN in the matrix and False in the mask, so the
    caller can count them rather than discovering a silent drop later.
    `phonology.feature_matrix` drops unparseable entries instead, which loses
    the alignment to the slice table.
    """
    phones = np.asarray(phones)
    lookup = {p: features(p) for p in set(phones.tolist())}
    matrix = np.full((phones.size, len(FEATURE_NAMES)), np.nan)
    ok = np.zeros(phones.size, dtype=bool)
    for i, p in enumerate(phones):
        v = lookup[p]
        if v is not None:
            matrix[i] = v
            ok[i] = True
    return matrix, ok


# --------------------------------------------------------------------------
# AUC on a weighted sample
# --------------------------------------------------------------------------

def dense_bins(values):
    """Map values onto dense integer bins preserving order, ties shared."""
    uniq, inverse = np.unique(np.asarray(values, dtype=np.float64), return_inverse=True)
    return inverse.astype(np.int64), int(uniq.size)


def weighted_auc(bins, n_bins, pos_rows, neg_rows, weights=None):
    """
    Exact AUC for a sample in which each row carries an integer multiplicity.

    The Mann-Whitney U with midrank tie handling, which is what `fast_auc`
    computes via `rankdata`. Writing it over precomputed bins means a bootstrap
    draw costs two bincounts instead of a sort, and the whole run drops from
    minutes to seconds. `tests/test_feature_analysis.py` asserts agreement with
    `fast_auc` including on tied data.

    Returns nan when either class is empty, matching `fast_auc`.
    """
    wp = None if weights is None else weights[pos_rows]
    wn = None if weights is None else weights[neg_rows]
    count_pos = np.bincount(bins[pos_rows], weights=wp, minlength=n_bins)
    count_neg = np.bincount(bins[neg_rows], weights=wn, minlength=n_bins)

    n_pos = count_pos.sum()
    n_neg = count_neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Negatives strictly below each bin, plus half the negatives tied with it.
    below = np.cumsum(count_neg) - count_neg
    u = float(np.dot(count_pos, below + 0.5 * count_neg))
    return u / float(n_pos * n_neg)


def folded_from_auc(auc):
    """
    Direction agnostic separation, max(auc, 1-auc).

    Identical to `analyse_slices.folded`, which writes it as
    abs(auc - 0.5) + 0.5. A statistic running bonafide-high separates just as
    well as one running spoof-high, and the sign is not the claim.
    """
    return abs(auc - 0.5) + 0.5


def cluster_weights(row_index, n_rows):
    """Turn the row indices of one cluster draw into per row multiplicities."""
    return np.bincount(row_index, minlength=n_rows).astype(np.float64)


# --------------------------------------------------------------------------
# the per feature test
# --------------------------------------------------------------------------

def check_clusters_are_single_class(y, utterances):
    """
    Refuse to run if any utterance carries both labels.

    `stats._prepare_clusters` builds its cluster index within one class at a
    time, so an utterance appearing as both bonafide and spoofed would be split
    into two clusters and resampled twice independently. The clustering would
    silently stop doing its job, which is the one thing that must not happen
    quietly here. On ASVspoof 2019 LA eval all 2,995 utterances are single
    class, so this is a guard for tables that are not this one.
    """
    y = np.asarray(y)
    utterances = np.asarray(utterances)
    order = np.argsort(utterances, kind="stable")
    u_sorted, y_sorted = utterances[order], y[order]
    _, starts = np.unique(u_sorted, return_index=True)
    first = y_sorted[np.repeat(starts, np.diff(np.append(starts, y_sorted.size)))]
    bad = np.unique(u_sorted[y_sorted != first])
    if bad.size:
        raise ValueError(
            f"{bad.size} utterance(s) carry both labels, e.g. {bad[:3].tolist()}. "
            "The cluster bootstrap would resample them as two clusters and the "
            "intervals would be too narrow.")


def bootstrap_p(draws):
    """
    Two sided achieved significance level from the bootstrap distribution of
    the difference, against the null that the difference is zero.

    Why this rather than `stats.permutation_test_auc`. That function shuffles
    labels row by row with no `groups` argument, so it would treat slices from
    one utterance as independent, which is the error this file exists to avoid.
    It also tests a different null, that the statistic carries no class
    information at all, whereas the question here is whether the two groups
    separate by DIFFERENT amounts. Both groups separating strongly and equally
    is a null for this question and not for that one.

    Plus one on numerator and denominator, so p is never zero. A resampling
    test cannot licence a claim it does not have the resolution to make.
    """
    d = np.asarray(draws, dtype=np.float64)
    n = d.size
    if n == 0:
        return float("nan")
    below = (np.sum(d <= 0) + 1) / (n + 1)
    above = (np.sum(d >= 0) + 1) / (n + 1)
    return float(min(1.0, 2 * min(below, above)))


def group_counts(y, utterances, mask):
    """Slices and distinct utterances per class inside one feature group."""
    out = {}
    for label, key in ((0, "bonafide"), (1, "spoof")):
        rows = mask & (y == label)
        out[f"slices_{key}"] = int(rows.sum())
        out[f"utterances_{key}"] = len(set(utterances[rows].tolist()))
    return out


def usable(counts):
    """Enough of both classes, counted in utterances as well as slices."""
    return all(
        counts[f"slices_{k}"] >= MIN_SLICES
        and counts[f"utterances_{k}"] >= MIN_UTTERANCES
        for k in ("bonafide", "spoof")
    )


def analyse(data, matrix, parsed, n_boot=2000, seed=0, alpha=0.05, verbose=True):
    """
    Run every feature against the same set of utterance resamples.

    Draw outer, feature inner, so one sequence of utterance resamples is shared
    by all features and the comparison between features is not contaminated by
    resampling noise that differs between them.
    """
    y, utterances, x = data["y"], data["u"], data["x"]
    keep = parsed
    y, utterances, x, matrix = y[keep], utterances[keep], x[keep], matrix[keep]
    phones = np.asarray(data["phone"])[keep] if "phone" in data else None
    n_rows = y.size
    check_clusters_are_single_class(y, utterances)

    bins, n_bins = dense_bins(x)

    # Per feature row sets, fixed before any resampling.
    plans, results = [], []
    for j, name in enumerate(FEATURE_NAMES):
        column = matrix[:, j]
        plus, minus = column == 1, column == -1
        record = {
            "feature": name,
            "n_plus": int(plus.sum()),
            "n_minus": int(minus.sum()),
            "n_zero": int((column == 0).sum()),
            "utterances_plus": len(set(utterances[plus].tolist())),
            "utterances_minus": len(set(utterances[minus].tolist())),
        }
        counts_plus = group_counts(y, utterances, plus)
        counts_minus = group_counts(y, utterances, minus)
        record["counts_plus"] = counts_plus
        record["counts_minus"] = counts_minus
        # How many distinct phones sit behind each side. A feature whose [+F]
        # group is one phone is a phoneme result wearing feature clothing, and
        # the reader has to be able to see that. In a 40 phone English
        # inventory several features collapse that far.
        if phones is not None:
            record["phones_plus"] = sorted(set(phones[plus].tolist()))
            record["phones_minus"] = sorted(set(phones[minus].tolist()))

        if not usable(counts_plus) or not usable(counts_minus):
            record["computable"] = False
            record["reason"] = _why_not(counts_plus, counts_minus)
            results.append(record)
            continue

        record["computable"] = True
        auc_plus = weighted_auc(bins, n_bins,
                                np.flatnonzero(plus & (y == 1)),
                                np.flatnonzero(plus & (y == 0)))
        auc_minus = weighted_auc(bins, n_bins,
                                 np.flatnonzero(minus & (y == 1)),
                                 np.flatnonzero(minus & (y == 0)))
        record["auc_plus"] = auc_plus
        record["auc_minus"] = auc_minus
        record["sep_plus"] = folded_from_auc(auc_plus)
        record["sep_minus"] = folded_from_auc(auc_minus)
        record["difference"] = record["sep_plus"] - record["sep_minus"]
        results.append(record)

        plans.append({
            "record": record,
            "plus_pos": np.flatnonzero(plus & (y == 1)),
            "plus_neg": np.flatnonzero(plus & (y == 0)),
            "minus_pos": np.flatnonzero(minus & (y == 1)),
            "minus_neg": np.flatnonzero(minus & (y == 0)),
            "draws": [],
            "folds_flipped": 0,
            # Which side of chance the full sample sat on, per group. Every AUC
            # here lands below 0.5 because bonafide runs the higher mgd_hf_std,
            # so the fold is a uniform sign change and costs nothing. What would
            # cost something is a group whose draws straddle chance, since
            # max(auc, 1-auc) is biased upward there and two unequally sized
            # groups inherit unequal amounts of that bias. That is what is
            # counted, not the flip itself.
            "side_plus": 1.0 if auc_plus >= 0.5 else -1.0,
            "side_minus": 1.0 if auc_minus >= 0.5 else -1.0,
        })

    if plans:
        prepared = _cluster_index(y, utterances)
        rng = np.random.default_rng(seed)
        for b in range(n_boot):
            weights = cluster_weights(_cluster_draw(rng, prepared), n_rows)
            for plan in plans:
                a = weighted_auc(bins, n_bins, plan["plus_pos"], plan["plus_neg"], weights)
                c = weighted_auc(bins, n_bins, plan["minus_pos"], plan["minus_neg"], weights)
                if not (np.isfinite(a) and np.isfinite(c)):
                    continue
                if ((a - 0.5) * plan["side_plus"] < 0
                        or (c - 0.5) * plan["side_minus"] < 0):
                    plan["folds_flipped"] += 1
                plan["draws"].append(folded_from_auc(a) - folded_from_auc(c))
            if verbose and n_boot >= 500 and (b + 1) % 500 == 0:
                print(f"  ... {b + 1}/{n_boot} draws", file=sys.stderr)

        for plan in plans:
            draws = np.asarray(plan["draws"], dtype=np.float64)
            record = plan["record"]
            lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            record["ci_low"] = float(lo)
            record["ci_high"] = float(hi)
            record["excludes_zero"] = bool(lo > 0 or hi < 0)
            record["p_value"] = bootstrap_p(draws)
            record["n_draws_used"] = int(draws.size)
            record["folds_flipped"] = int(plan["folds_flipped"])
            record["_draws"] = draws

        _head_to_head(plans, reference="son", alpha=alpha)

    tested = [r for r in results if r["computable"]]
    if tested:
        bh = benjamini_hochberg([r["p_value"] for r in tested], alpha=alpha)
        for record, q, rejected in zip(tested, bh["q_values"], bh["rejected"]):
            record["q_value"] = float(q)
            record["survives_fdr"] = bool(rejected)
    return results


def _head_to_head(plans, reference="son", alpha=0.05):
    """
    Is a feature's difference actually LARGER than the reference feature's?

    Reading the ordering of point estimates as a ranking is the error this
    guards against. The features sit within a few thousandths of each other and
    their intervals overlap almost entirely, so "x beats son" needs testing, not
    eyeballing. Because every feature was measured on the SAME utterance
    resamples, the difference of differences can be taken inside each draw,
    which is the paired comparison and is far tighter than comparing two
    marginal intervals.

    Descriptive, and deliberately outside the FDR family. The 24 feature tests
    are the registered set. These 16 extra contrasts are a sanity check on how
    to read the ordering, not additional findings.
    """
    ref = next((p for p in plans if p["record"]["feature"] == reference), None)
    if ref is None:
        return
    ref_draws = np.asarray(ref["record"]["_draws"], dtype=np.float64)
    for plan in plans:
        record = plan["record"]
        if record["feature"] == reference:
            continue
        draws = np.asarray(record["_draws"], dtype=np.float64)
        if draws.size != ref_draws.size:
            continue
        gap = draws - ref_draws
        lo, hi = np.percentile(gap, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        record["vs_reference"] = {
            "reference": reference,
            "gap": float(record["difference"] - ref["record"]["difference"]),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "distinguishable": bool(lo > 0 or hi < 0),
        }


def _why_not(counts_plus, counts_minus):
    """Say which side of the split failed and how, rather than just 'skipped'."""
    reasons = []
    for side, counts in (("[+F]", counts_plus), ("[-F]", counts_minus)):
        if counts["slices_bonafide"] + counts["slices_spoof"] == 0:
            reasons.append(f"no {side} segment in the corpus inventory")
        elif not usable(counts):
            reasons.append(
                f"{side} has {counts['slices_bonafide']}/{counts['slices_spoof']} slices "
                f"and {counts['utterances_bonafide']}/{counts['utterances_spoof']} utterances "
                f"bonafide/spoof, below the {MIN_SLICES} slice "
                f"and {MIN_UTTERANCES} utterance floor")
    return "; ".join(reasons)


# --------------------------------------------------------------------------
# collinearity, described
# --------------------------------------------------------------------------

def collinearity(matrix, parsed, names=None):
    """
    Pairwise correlation between feature columns over slices.

    Description only. These numbers say which results are restatements of each
    other, which is exactly what a reader needs in order not to read [+son] and
    [+voi] as two findings. They do not support any decomposition into separate
    contributions, and none is attempted.

    Columns that do not vary, which here means features with no [+F] segment in
    the inventory, are excluded and returned as `constant`.
    """
    names = list(FEATURE_NAMES if names is None else names)
    m = matrix[parsed]
    varying = [j for j in range(m.shape[1]) if np.unique(m[:, j]).size > 1]
    constant = [names[j] for j in range(m.shape[1]) if j not in varying]
    if not varying:
        return {"names": [], "matrix": np.empty((0, 0)), "constant": constant}
    sub = m[:, varying]
    return {
        "names": [names[j] for j in varying],
        "matrix": np.corrcoef(sub, rowvar=False),
        "constant": constant,
    }


def top_pairs(collin, k=10):
    """The most collinear feature pairs, strongest absolute correlation first."""
    names, c = collin["names"], collin["matrix"]
    pairs = [
        (abs(c[i, j]), names[i], names[j], float(c[i, j]))
        for i in range(len(names)) for j in range(i + 1, len(names))
    ]
    pairs.sort(reverse=True)
    return [(a, b, r) for _, a, b, r in pairs[:k]]


# --------------------------------------------------------------------------
# exploratory multivariate view
# --------------------------------------------------------------------------

def multivariate(data, matrix, parsed, collin, seed=0, n_splits=5):
    """
    EXPLORATORY. A regularised fit scored on held out utterances.

    Deliberately reports held out AUC and nothing else. In sample coefficients
    over columns correlating above 0.9 are not interpretable as per feature
    contributions, so they are not printed, and no one should be tempted to
    read them as a decomposition.

    Three models, so the comparison means something.
      phase      label from mgd_hf_std alone
      features   label from the feature vector alone. This is the baseline that
                 matters. If bonafide and spoofed utterances differ in which
                 phones they contain, a model can score above chance without
                 using phase at all, and any uplift has to clear this.
      combined   both, plus every feature by phase interaction. Interactions
                 are the multivariate form of the question, since the claim is
                 that features modulate how well phase separates.

    Folds are split by utterance, never within one.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = data["y"][parsed]
    utterances = data["u"][parsed]
    x = data["x"][parsed]
    names = collin["names"]
    columns = matrix[parsed][:, [FEATURE_NAMES.index(n) for n in names]]

    z = ((x - x.mean()) / x.std()).reshape(-1, 1)
    designs = {
        "phase": z,
        "features": columns,
        "combined": np.hstack([z, columns, columns * z]),
    }

    out = {}
    for label, design in designs.items():
        scores = []
        for train, test in GroupKFold(n_splits=n_splits).split(design, y, groups=utterances):
            if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
                continue
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=seed),
            ).fit(design[train], y[train])
            predicted = model.predict_proba(design[test])[:, 1]
            scores.append(folded_from_auc(fast_auc(y[test], predicted)))
        out[label] = {
            "mean": float(np.mean(scores)) if scores else float("nan"),
            "folds": [float(s) for s in scores],
            "n_columns": int(design.shape[1]),
        }
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def report(results, collin, data, parsed, seg, table, n_boot):
    tested = sorted(
        (r for r in results if r["computable"]),
        key=lambda r: -r["difference"],
    )
    skipped = [r for r in results if not r["computable"]]
    total_utterances = len(set(data["u"].tolist()))

    print(f"table {table}, segmentation {seg}, statistic {PHASE}")
    print(f"{int(parsed.sum())} slices from {total_utterances} utterances, "
          f"{int((~parsed).sum())} slices dropped as unparseable phones")
    print(f"cluster bootstrap by utterance_id, {n_boot} draws, shared across features\n")

    print("SEPARATION DIFFERENCE, [+F] minus [-F]. Positive means the feature's")
    print("plus value carries the stronger bonafide/spoof separation.\n")
    header = (f"{'FEATURE':<9}{'DIFF':>8} {'95% CI':>20} {'p':>8}{'q':>8}  "
              f"{'SEP+':>6}{'SEP-':>6}  {'vs son':>8} {'95% CI':>20}")
    print(header)
    print("-" * len(header))
    for r in tested:
        mark = "*" if r.get("survives_fdr") else " "
        vs = r.get("vs_reference")
        if vs is None:
            gap, gap_ci = "reference", ""
        else:
            gap = f"{vs['gap']:+.4f}"
            gap_ci = "[%+.4f, %+.4f]%s" % (
                vs["ci_low"], vs["ci_high"], "!" if vs["distinguishable"] else " ")
        print(f"{r['feature']:<9}{r['difference']:>+8.4f} "
              f"{'[%+.4f, %+.4f]' % (r['ci_low'], r['ci_high']):>20} "
              f"{r['p_value']:>8.4f}{r['q_value']:>7.4f}{mark} "
              f"{r['sep_plus']:>6.3f}{r['sep_minus']:>6.3f}  "
              f"{gap:>8} {gap_ci:>20}")
    print(f"\n* survives Benjamini-Hochberg at q < 0.05 across the {len(tested)} "
          f"computable features.")
    print("! the paired interval on this feature's difference minus son's excludes")
    print("  zero, so the two are actually distinguishable rather than merely")
    print("  ordered. Without a !, the row above son is not beating it.")

    print("\nCOUNTS BEHIND EVERY ROW. Utterances are the effective sample size.")
    counts_header = (f"{'FEATURE':<9} {'[+F] SLICES b/s':>18} {'[+F] UTT b/s':>15} "
                     f"{'[-F] SLICES b/s':>18} {'[-F] UTT b/s':>15} {'0 DROP':>8} "
                     f"{'PHONES +/-':>11}")
    print(counts_header)
    print("-" * len(counts_header))
    def pair(counts, unit):
        return f"{counts[unit + '_bonafide']}/{counts[unit + '_spoof']}"

    for r in tested:
        p_, m_ = r["counts_plus"], r["counts_minus"]
        n_phones = (f"{len(r['phones_plus'])}/{len(r['phones_minus'])}"
                    if "phones_plus" in r else "-")
        print(f"{r['feature']:<9} "
              f"{pair(p_, 'slices'):>18} {pair(p_, 'utterances'):>15} "
              f"{pair(m_, 'slices'):>18} {pair(m_, 'utterances'):>15} "
              f"{r['n_zero']:>8} {n_phones:>11}")

    narrow = [r for r in tested if len(r.get("phones_plus", [0] * 99)) <= 3]
    if narrow:
        print("\nFeatures whose [+F] side is three phones or fewer in this inventory.")
        print("For these the feature and the phoneme are the same variable here, so")
        print("the result is not evidence that the FEATURE is what matters. Only a")
        print("second language, where the same feature is carried by different")
        print("phones, can separate the two.")
        for r in narrow:
            print(f"  {r['feature']:<9} [+F] = {' '.join(r['phones_plus'])}")
    print("0 DROPPED is slices whose feature value is unspecified. It also absorbs")
    print("multi segment phones whose segments disagree, because phonology.features")
    print("averages across segments and a disagreement averages to 0. The affricates")
    print("land there for cont and strid, and the diphthongs for hi, lo, back and")
    print("round. Where the 0 column is very large the feature is simply undefined")
    print("over a whole natural class, and that row rests on much less of the corpus")
    print("than the others do.")

    flipped = sum(r.get("folds_flipped", 0) for r in tested)
    print()
    if flipped == 0:
        print("Every group sat on the same side of chance in every bootstrap draw as")
        print("it did in the full sample, so max(auc, 1-auc) is a uniform sign change")
        print("here and its upward bias near chance is not in play.")
    else:
        print("WARNING. A group crossed chance relative to the full sample in")
        print(f"{flipped} feature-draws. max(auc, 1-auc) is biased upward near chance")
        print("and two unequally sized groups inherit unequal amounts of it, so the")
        print("difference for those features is not trustworthy. Check which.")

    print(f"\nNOT COMPUTABLE, {len(skipped)} of {len(results)} features. Listed rather")
    print("than dropped, because absence here is a fact about the corpus inventory.")
    for r in sorted(skipped, key=lambda r: FEATURE_NAMES.index(r["feature"])):
        print(f"  {r['feature']:<9} {r['reason']}")

    print("\nCOLLINEARITY, description only.")
    print("Strongest pairs among the computable features, over slices.")
    for a, b, r in top_pairs(collin, k=10):
        print(f"  {a:<7} {b:<7} r = {r:+.3f}")
    if "son" in collin["names"]:
        i = collin["names"].index("son")
        row = [(abs(collin["matrix"][i, j]), collin["names"][j], float(collin["matrix"][i, j]))
               for j in range(len(collin["names"])) if j != i]
        row.sort(reverse=True)
        print("\n  what 'son' is nearly the same partition as:")
        for _, name, r in row[:5]:
            print(f"    son vs {name:<7} r = {r:+.3f}")
    print("\n  These correlations are why no per feature coefficient is reported.")
    print("  Features high in the table may be one finding stated several ways.")


def report_multivariate(mv):
    print("\nEXPLORATORY MULTIVARIATE VIEW, held out utterances, L2 logistic.")
    print("Held out folded AUC per model. Not a decomposition. No coefficients.")
    for label in ("phase", "features", "combined"):
        m = mv[label]
        folds = ", ".join(f"{s:.3f}" for s in m["folds"])
        print(f"  {label:<9} {m['mean']:.4f}   ({m['n_columns']} columns)  folds {folds}")
    uplift = mv["combined"]["mean"] - max(mv["phase"]["mean"], mv["features"]["mean"])
    print(f"  uplift of combined over the better single model: {uplift:+.4f}")
    print("  A small uplift means the features add little beyond what phase alone")
    print("  and the phone inventory already give. It does not tell you which")
    print("  feature did the work, and nothing here should be read as if it did.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--table", required=True, help=".csv or .csv.gz slice table")
    ap.add_argument("--seg", default="run", choices=["run", "window"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--multivariate", action="store_true",
                    help="also run the exploratory regularised fit")
    args = ap.parse_args()

    data = load(args.table, args.seg)
    if data["y"].size == 0:
        raise SystemExit(f"no rows with seg={args.seg} in {args.table}")

    matrix, parsed = feature_columns(data["phone"])
    results = analyse(data, matrix, parsed,
                      n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
    collin = collinearity(matrix, parsed)
    report(results, collin, data, parsed, args.seg, args.table, args.n_boot)

    if args.multivariate:
        report_multivariate(multivariate(data, matrix, parsed, collin, seed=args.seed))

    print("\n" + "=" * 78)
    print("This is an exploratory restatement of the pre-registered sonorant")
    print("contrast in feature terms. It is not a second confirmation of it. The")
    print("features are heavily collinear and the tests are not independent, so")
    print("treat the ordering as description and not as a ranking of causes.")
    print("=" * 78)


if __name__ == "__main__":
    main()
