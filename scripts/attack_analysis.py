"""
Why does the sonorant effect vary across attacks, and does it really reverse?

The problem this addresses
    The research log carries a SERIOUS escalation. The sonorant advantage was
    recorded as holding in 8 of 13 ASVspoof eval attacks and reversing on A08,
    A13, A16 and A19, with the reasoning that a genuine articulatory mechanism
    should differ in degree across vocoders and not in sign. Four reversals
    would mean the mechanism is not articulatory, or is conditional on something
    about the synthesiser nobody has identified, and either way H2 could return
    a positive for reasons unrelated to articulation.

    That claim was never written down with the command that produced it, which
    is the one thing this project requires of a finding. So it is recomputed
    here, with intervals, from a script with tests.

The distinction the whole question turns on
    Two different quantities can both be called "the sonorant advantage" and
    they behave differently.

        level        does phase beat magnitude within sonorants at all
                     folded AUC of mgd_hf_std minus folded AUC of
                     mfcc_frame_spread, averaged over the sonorant categories

        asymmetry    does phase beat magnitude MORE in sonorants than in
                     obstruents, which is the level minus the same quantity
                     computed over the obstruent categories

    `docs/PREREGISTRATION.md` states H1 as the asymmetry. Its primary test is
    that the sonorant advantage exceeds the obstruent advantage, and its second
    falsification condition is that the ordering reverses. So an attack where
    phase loses to magnitude everywhere, but loses less in sonorants, is an
    attack where H1 holds. Reading the level alone would call it a reversal.

    Both are reported below, per attack, with intervals.

The grouping, and where it came from
    `configs/asvspoof2019_attacks.yaml` records each attack's waveform
    generator, transcribed from the ASVspoof 2019 database paper and verified
    against the arXiv full text. From it one binary is derived, whether the fine
    phase structure of the output originates in a human recording. Concatenating
    recorded units or filtering a real waveform inherits natural phase. A
    vocoder or a neural waveform model has to produce it.

    The prediction is that a system inheriting natural phase has no sonorant
    phase deficit to find, because it never attempted the reconstruction the
    mechanism is about.

    This is NOT a confirmatory test. The reversal set was known before the
    grouping was written, so a person who knew the answer chose the partition.
    What makes it more than a story is that the grouping comes from someone
    else's documentation of how the systems work, it was fixed before the per
    attack numbers were computed with this script, and it predicts an out of
    sample case that has not been looked at, which is registered separately in
    `docs/PREREGISTRATION-H3.md`.

Two limits on the group test, stated rather than buried
    The eval partition holds only three attacks that inherit natural phase, so
    the smallest attainable two sided p over 3 against 9 is about 0.009 and the
    test has little power. A negative result here is undecided, not evidence of
    no association.

    The two tests reported answer different questions and will disagree, so
    which one to read depends on the claim being made.

        permutation over attack labels
            treats the thirteen attacks as the sample, so its noise is the
            variation BETWEEN synthesis systems. This is the test for "would a
            new system of this kind behave this way", which is the claim the
            project cares about, and it is the conservative one.

        bootstrap over utterances
            holds the thirteen systems fixed and resamples the speech, so its
            noise is sampling of utterances and slices. It answers "is this
            difference between these particular systems stable", which is a
            narrower claim. It cannot see between-system variation at all, so
            it will report an interval clear of zero in cases where the
            permutation test is nowhere near significant. That is not a
            contradiction, and the bootstrap must not be quoted as if it
            licensed the general claim.

    Both are printed. The general claim needs the permutation test, and with
    three attacks on one side that test is underpowered, which is why H3 exists.

Usage
    python scripts/attack_analysis.py --table results/eval_slices.csv.gz
    python scripts/attack_analysis.py --table results/eval_slices.csv.gz --seg window

CPU, about a minute at the default 1000 draws.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import math
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.stats import _cluster_draw, _prepare_clusters, fast_auc  # noqa: E402

# The pre-registered grouping, copied from scripts/analyse_slices.py so the two
# analyses are demonstrably about the same categories.
SONORANTS = ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"]
OBSTRUENTS = ["FRICATIVES", "PLOSIVES", "AFFRICATES"]

PHASE = "mgd_hf_std"
MAGNITUDE = "mfcc_frame_spread"

MIN_SLICES = 30
MIN_UTTERANCES = 10

BONAFIDE_MARKERS = {"-", "", "bonafide", "none", "nan", "real"}

# Above this many assignments the permutation null is sampled rather than
# enumerated. Thirteen attacks never reach it. A larger corpus might.
ENUMERATION_LIMIT = 500_000


# --------------------------------------------------------------------------
# the taxonomy
# --------------------------------------------------------------------------

def load_taxonomy(path):
    """
    Read the attack taxonomy, with the tri-state derived field kept tri-state.

    A YAML `unknown` must not collapse to False. The difference between "the
    paper says this system filters a real recording" and "the paper does not
    say" is the difference between evidence and a guess, and it is the reason
    A13 is excluded from the grouping rather than assigned to a side.
    """
    import yaml

    with open(path) as fh:
        document = yaml.safe_load(fh)

    taxonomy = {}
    for attack, record in document["attacks"].items():
        value = record.get("natural_phase_origin")
        if isinstance(value, str) and value.lower() == "unknown":
            value = None
        if value not in (True, False, None):
            raise ValueError(
                f"{attack}: natural_phase_origin must be true, false or unknown, "
                f"got {value!r}"
            )
        taxonomy[attack] = {**record, "natural_phase_origin": value}
    return taxonomy


def check_coverage(attacks, taxonomy):
    """Every attack in the data must have a taxonomy row, or the run stops."""
    missing = sorted(set(attacks) - set(taxonomy))
    if missing:
        raise ValueError(
            f"no taxonomy row for {', '.join(missing)}. Add them to "
            f"configs/asvspoof2019_attacks.yaml from the source paper rather "
            f"than letting the analysis drop them"
        )
    return True


def group_means(statistics, taxonomy, key):
    """Split a per attack statistic on a taxonomy field, holding out unknowns."""
    groups = {True: [], False: [], None: []}
    for attack, value in statistics.items():
        groups[taxonomy[attack][key]].append(attack)

    def mean(names):
        return float(np.mean([statistics[n] for n in names])) if names else float("nan")

    return {
        "true": sorted(groups[True]),
        "false": sorted(groups[False]),
        "excluded": sorted(groups[None]),
        "mean_true": mean(groups[True]),
        "mean_false": mean(groups[False]),
        "difference": mean(groups[True]) - mean(groups[False]),
    }


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path, seg):
    """Read one segmentation of a slice table into aligned arrays."""
    opener = gzip.open if str(path).endswith(".gz") else open
    columns = {k: [] for k in ("y", "u", "attack", "category", "phase", "magnitude")}
    with opener(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["seg"] != seg:
                continue
            columns["y"].append(int(row["label"]))
            columns["u"].append(row["utterance_id"])
            columns["attack"].append(row["attack"])
            columns["category"].append(row["category"])
            columns["phase"].append(float(row[PHASE]))
            columns["magnitude"].append(float(row[MAGNITUDE]))

    data = {k: np.asarray(v) for k, v in columns.items()}
    data["y"] = data["y"].astype(np.int64)
    data["phase"] = data["phase"].astype(np.float64)
    data["magnitude"] = data["magnitude"].astype(np.float64)

    marked = np.isin(np.char.lower(data["attack"].astype(str)), sorted(BONAFIDE_MARKERS))
    labels = np.unique(data["y"][marked])
    if labels.size != 1:
        raise ValueError(
            f"rows with no attack carry labels {labels.tolist()}, so which label "
            f"means bonafide cannot be read from the table"
        )
    data["bonafide_label"] = int(labels[0])
    return data


# --------------------------------------------------------------------------
# the per attack statistics
# --------------------------------------------------------------------------

def folded(y, values):
    """Separation without regard to direction, as the pre-registered analysis."""
    return abs(fast_auc(y, values) - 0.5) + 0.5


def _advantage(y, phase, magnitude):
    """Phase minus magnitude, both folded, on identical rows."""
    return folded(y, phase) - folded(y, magnitude)


def _usable(y, utterances):
    for label in (0, 1):
        mask = y == label
        if int(mask.sum()) < MIN_SLICES:
            return False
        if np.unique(utterances[mask]).size < MIN_UTTERANCES:
            return False
    return True


def per_attack(data, n_boot=1000, seed=0, alpha=0.05):
    """
    Level and asymmetry for every attack, against the shared bonafide pool.

    One draw resamples the bonafide utterances once and every attack's
    utterances within that attack, and all thirteen attacks are recomputed
    inside it. That reflects the real design, where the bonafide side is shared,
    and it means the intervals on two attacks are correlated in the way the
    underlying estimates are.
    """
    bona = data["bonafide_label"]
    is_bona = (data["y"] == bona).astype(np.int64)
    utterances, attacks, categories = data["u"], data["attack"], data["category"]
    phase, magnitude = data["phase"], data["magnitude"]

    bona_rows = np.flatnonzero(is_bona == 1)
    names = sorted(set(attacks[is_bona == 0].tolist()))

    # One stratum per attack plus one for the bonafide pool, each resampled to
    # its own size so an attack cannot shrink to nothing in a draw.
    stratum = np.where(is_bona == 1, "bonafide", attacks)
    prepared = {
        name: _prepare_clusters(stratum, utterances, name)
        for name in ["bonafide"] + names
    }

    observed, rows_by_attack = {}, {}
    for name in names:
        rows = np.concatenate([bona_rows, np.flatnonzero(attacks == name)])
        rows_by_attack[name] = rows
        observed[name] = _summarise(is_bona, utterances, categories, phase, magnitude, rows)

    rng = np.random.default_rng(seed)
    draws = {name: {"level": [], "obstruent": [], "asymmetry": []} for name in names}
    for _ in range(n_boot):
        drawn_bona = _cluster_draw(rng, [prepared["bonafide"]])
        for name in names:
            drawn = np.concatenate([drawn_bona, _cluster_draw(rng, [prepared[name]])])
            summary = _summarise(is_bona, utterances, categories, phase, magnitude,
                                 drawn, check_usable=False,
                                 allowed=observed[name]["categories"])
            for key, target in (("sonorant_advantage", "level"),
                                ("obstruent_advantage", "obstruent"),
                                ("asymmetry", "asymmetry")):
                draws[name][target].append(summary[key])

    for name in names:
        for key, target in (("sonorant", "level"), ("obstruent", "obstruent"),
                            ("asymmetry", "asymmetry")):
            values = np.asarray(draws[name][target], dtype=np.float64)
            finite = values[np.isfinite(values)]
            if finite.size < 2:
                low = high = float("nan")
            else:
                low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            observed[name][f"{key}_ci_low"] = float(low)
            observed[name][f"{key}_ci_high"] = float(high)
        observed[name]["draws"] = {k: np.asarray(v, dtype=np.float64)
                                   for k, v in draws[name].items()}
    return observed


def _summarise(is_bona, utterances, categories, phase, magnitude, rows,
               check_usable=True, allowed=None):
    """The two group advantages and their difference, on one set of rows."""
    y = is_bona[rows]
    cat = categories[rows]

    def group(names):
        values, used = [], []
        for name in names:
            if allowed is not None and name not in allowed:
                continue
            selected = rows[cat == name]
            if selected.size == 0:
                continue
            ys = is_bona[selected]
            if check_usable and not _usable(ys, utterances[selected]):
                continue
            if np.unique(ys).size < 2:
                continue
            values.append(_advantage(ys, phase[selected], magnitude[selected]))
            used.append(name)
        return (float(np.mean(values)) if values else float("nan")), used

    sonorant, son_used = group(SONORANTS)
    obstruent, obs_used = group(OBSTRUENTS)

    return {
        "sonorant_advantage": sonorant,
        "obstruent_advantage": obstruent,
        "asymmetry": sonorant - obstruent,
        "categories": son_used + obs_used,
        "n_slices": int(rows.size),
        "n_utterances_bonafide": int(np.unique(utterances[rows][y == 1]).size),
        "n_utterances_spoof": int(np.unique(utterances[rows][y == 0]).size),
    }


# --------------------------------------------------------------------------
# the group test
# --------------------------------------------------------------------------

def exact_permutation_p(values, in_group, seed=0):
    """
    Two sided permutation test on a difference of group means, enumerated.

    With thirteen attacks split three against ten there are 286 assignments, so
    the null can be enumerated and the p value is exact. A sampled
    approximation would return a different number for a different seed, which
    is needless noise at this size.

    `smallest_attainable_p` is reported beside the p value because a test over a
    handful of units can be incapable of returning anything small, and a p of
    0.4 from such a test means nothing at all.
    """
    values = np.asarray(values, dtype=np.float64)
    in_group = np.asarray(in_group, dtype=bool)
    n, k = values.size, int(in_group.sum())
    if k == 0 or k == n:
        raise ValueError("one group is empty, so there is nothing to compare it against")

    def difference(mask):
        return float(values[mask].mean() - values[~mask].mean())

    observed = difference(in_group)
    total = math.comb(n, k)

    if total <= ENUMERATION_LIMIT:
        extreme = 0
        for combination in itertools.combinations(range(n), k):
            mask = np.zeros(n, dtype=bool)
            mask[list(combination)] = True
            if abs(difference(mask)) >= abs(observed) - 1e-12:
                extreme += 1
        p_value = extreme / total
        exact = True
        n_assignments = total
    else:
        rng = np.random.default_rng(seed)
        n_assignments = 20000
        extreme = 0
        for _ in range(n_assignments):
            mask = np.zeros(n, dtype=bool)
            mask[rng.choice(n, size=k, replace=False)] = True
            if abs(difference(mask)) >= abs(observed) - 1e-12:
                extreme += 1
        p_value = (extreme + 1) / (n_assignments + 1)
        exact = False

    smallest = 2.0 / total if total > 1 else 1.0
    return {
        "difference": observed,
        "p_value": float(p_value),
        "n_assignments": int(n_assignments),
        "exact": exact,
        "smallest_attainable_p": float(smallest),
        # A test that could never have returned a small number has not tested
        # anything, whatever it returns.
        "underpowered": bool(smallest > 0.05),
    }


def bootstrap_group_difference(results, members_true, members_false, alpha=0.05):
    """
    Interval on the difference in mean asymmetry between the two groups.

    Built from the per attack draws, which share one bonafide resample per draw,
    so the shared bonafide pool is handled rather than ignored.

    This holds the thirteen systems FIXED. It answers whether the difference
    between these particular systems is stable under resampling the speech, and
    it cannot see variation between synthesis systems, which is the variation
    that matters for any claim about a class of system. Read it beside the
    permutation test, never instead of it.
    """
    if not members_true or not members_false:
        return None
    stack = lambda names: np.vstack([results[n]["draws"]["asymmetry"] for n in names])
    difference = stack(members_true).mean(axis=0) - stack(members_false).mean(axis=0)
    finite = difference[np.isfinite(difference)]
    if finite.size < 2:
        return None
    low, high = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "difference": float(np.mean(finite)),
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_report(results, taxonomy, seg, n_boot):
    print(f"\nPer attack, seg={seg}, {n_boot} draws clustered by utterance")
    print("=" * 78)
    print("\nlevel     = phase minus magnitude within sonorants")
    print("asymmetry = the same, minus the obstruent value. H1 is about this one.\n")

    header = (f"{'attack':<7} {'origin':<9} {'waveform generator':<26} "
              f"{'level':>8} {'asymmetry':>10} {'95% CI':>20}")
    print(header)
    print("-" * len(header))

    origin_name = {True: "natural", False: "generated", None: "unknown"}
    for attack in sorted(results):
        r = results[attack]
        t = taxonomy[attack]
        ci = f"[{r['asymmetry_ci_low']:+.3f}, {r['asymmetry_ci_high']:+.3f}]"
        print(f"{attack:<7} {origin_name[t['natural_phase_origin']]:<9} "
              f"{t['waveform_generator'][:26]:<26} "
              f"{r['sonorant_advantage']:>+8.3f} {r['asymmetry']:>+10.3f} {ci:>20}")

    levels = {a: r["sonorant_advantage"] for a, r in results.items()}
    asymmetries = {a: r["asymmetry"] for a, r in results.items()}

    holds = [a for a, v in asymmetries.items() if v > 0]
    clear = [a for a, r in results.items() if r["asymmetry_ci_low"] > 0]
    level_negative = [a for a, v in levels.items() if v < 0]

    print(f"\nAsymmetry positive in {len(holds)} of {len(asymmetries)} attacks, "
          f"clear of zero in {len(clear)}.")
    print(f"Asymmetry negative in: {', '.join(sorted(set(asymmetries) - set(holds))) or 'none'}")
    print(f"Level negative in: {', '.join(sorted(level_negative)) or 'none'}")
    print("\nAn attack in the second list but not the first is one where phase loses to")
    print("magnitude everywhere and loses LESS in sonorants. H1 holds there.")

    for label, statistics in (("asymmetry", asymmetries), ("level", levels)):
        split = group_means(statistics, taxonomy, "natural_phase_origin")
        tested = split["true"] + split["false"]
        in_group = np.array([a in split["true"] for a in tested])
        values = np.array([statistics[a] for a in tested])

        print(f"\nGrouped by phase origin, on {label}")
        print(f"  natural   {', '.join(split['true'])}: mean {split['mean_true']:+.4f}")
        print(f"  generated {len(split['false'])} attacks: mean {split['mean_false']:+.4f}")
        if split["excluded"]:
            print(f"  held out, origin not decidable from the paper: "
                  f"{', '.join(split['excluded'])}")

        test = exact_permutation_p(values, in_group)
        print(f"  difference {test['difference']:+.4f}, exact p {test['p_value']:.4f} "
              f"over {test['n_assignments']} assignments")
        if test["underpowered"]:
            print(f"  UNDERPOWERED. The smallest attainable p is "
                  f"{test['smallest_attainable_p']:.3f}")

        if label == "asymmetry":
            interval = bootstrap_group_difference(results, split["true"], split["false"])
            if interval:
                print(f"  bootstrap difference {interval['difference']:+.4f} "
                      f"[{interval['ci_low']:+.4f}, {interval['ci_high']:+.4f}], "
                      f"{'clear of zero' if interval['excludes_zero'] else 'spans zero'}")
                print("  that interval holds these systems fixed and resamples the "
                      "speech, so it\n  cannot speak about systems not in this corpus. "
                      "The permutation test can,\n  and it is the underpowered one.")

    print("\nExploratory. The grouping was written from the source paper before these")
    print("numbers were computed, but the reversal set was already known, so this is a")
    print("hypothesis. docs/PREREGISTRATION-H3.md states the confirmatory form.")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--table", required=True)
    ap.add_argument("--seg", default="run", choices=["run", "window"])
    ap.add_argument("--taxonomy", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "configs",
        "asvspoof2019_attacks.yaml"))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    taxonomy = load_taxonomy(args.taxonomy)
    data = load(args.table, args.seg)
    names = sorted(set(data["attack"][data["y"] != data["bonafide_label"]].tolist()))
    check_coverage(names, taxonomy)

    counts = Counter(data["attack"])
    print(f"{args.table}, seg={args.seg}: {len(names)} attacks, "
          f"{counts['-']} bonafide slices")

    results = per_attack(data, n_boot=args.n_boot, seed=args.seed)
    print_report(results, taxonomy, args.seg, args.n_boot)


if __name__ == "__main__":
    main()
