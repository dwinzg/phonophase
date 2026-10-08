"""
Run the pre-registered test on the table produced by slice_and_probe.py.

The decision procedure in docs/PREREGISTRATION.md is implemented literally,
including all four falsification conditions, and the verdict is printed. That
is the point. A procedure fixed in advance and executed mechanically is the
only defence against fitting a story to whatever the data happens to show,
which has already happened twice in this project.

Runs locally in seconds on the downloaded CSV. No GPU, no corpus.

Usage
    python scripts/analyse_slices.py --table results/eval_slices.csv.gz
    python scripts/analyse_slices.py --table results/eval_slices.csv.gz --seg window
    python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster system
    python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster speaker
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.corpus import read_speaker_map  # noqa: E402
from phonophase.stats import (  # noqa: E402
    BONAFIDE_MARKERS, CLUSTERS, benjamini_hochberg, bonafide_label,
    bootstrap_auc_ci, cluster_groups, fast_auc, percentile_interval,
    shared_cluster_draws,
)

# Re-exported so a reader of this script can see the whole contract, and so the
# tests can assert there is exactly one of each rather than a copy per script.
__all__ = ["BONAFIDE_MARKERS", "bonafide_label", "cluster_groups",
           "shared_cluster_draws"]

SONORANTS = ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"]
OBSTRUENTS = ["FRICATIVES", "PLOSIVES", "AFFRICATES"]

PHASE = "mgd_hf_std"
MAGNITUDE = "mfcc_frame_spread"


def open_table(path):
    """
    Open the slice table, decompressing gzip transparently.

    The only table artifact in the repository is results/eval_slices.csv.gz and
    the documented command pointed at a plain .csv that is not there, so the
    pre-registered analysis crashed on the gzip magic bytes with
    UnicodeDecodeError before reading a single row.

    The format is decided by those magic bytes rather than by the suffix. A
    table is often renamed on the way out of a notebook or a bundle, and the
    bytes are what the reader actually has to cope with.
    """
    with open(path, "rb") as fh:
        magic = fh.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    return opener(path, "rt", newline="")


UNKNOWN_SPEAKER = "UNKNOWN"
DEFAULT_PROTOCOL = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "configs", "protocols",
    "ASVspoof2019.LA.cm.eval.trl.txt")


def load(path, seg, language=None, speakers=None):
    """
    Read one segmentation of the slice table, optionally one language of it.

    The speaker of a row is its `speaker` column when the table has one. The
    shipped eval table predates that column, so `speakers`, a map from
    utterance id to speaker read off the ASVspoof protocol, fills it in. A row
    neither source can place is UNKNOWN, which require_cluster refuses.
    """
    speakers = speakers or {}
    rows = defaultdict(lambda: {"y": [], "u": [], "attack": [], "speaker": [],
                                PHASE: [], MAGNITUDE: []})
    with open_table(path) as fh:
        for r in csv.DictReader(fh):
            if r["seg"] != seg:
                continue
            if language is not None and r.get("language", "-") != language:
                continue
            d = rows[r["category"]]
            d["y"].append(int(r["label"]))
            d["u"].append(r["utterance_id"])
            d["attack"].append(r.get("attack", ""))
            d["speaker"].append(r.get("speaker") or speakers.get(
                r["utterance_id"], UNKNOWN_SPEAKER))
            d[PHASE].append(float(r[PHASE]))
            d[MAGNITUDE].append(float(r[MAGNITUDE]))
    data = {k: {n: np.asarray(v) for n, v in d.items()} for k, d in rows.items()}
    return with_bonafide_label(data)


def with_bonafide_label(data):
    """
    Attach the bonafide label, read once over the whole table.

    Read over every category rather than per category, because a category
    holding one class alone cannot answer the question and would otherwise get
    None while its neighbours got 0.
    """
    if not data:
        return data
    labels = np.concatenate([d["y"] for d in data.values()])
    attacks = np.concatenate([d["attack"] for d in data.values()])
    answer = bonafide_label(labels, attacks)
    return {k: {**d, "bonafide_label": answer} for k, d in data.items()}


def require_cluster(data, cluster):
    """
    Refuse system clustering on a table whose bonafide rows cannot be located.

    Falling back to utterance clustering here would report a narrow interval
    under a heading that says system, which is the kind of quiet substitution
    this project has already shipped twice.
    """
    if cluster in ("speaker", "speaker-paired"):
        placeholders = {UNKNOWN_SPEAKER, "-", ""}
        if any(np.isin(d["speaker"], sorted(placeholders)).any()
               for d in data.values()):
            raise SystemExit(
                f"--cluster {cluster} needs a speaker for every row. This table "
                "has rows with none, and --protocol does not list their "
                "utterance ids.")
        return
    if cluster != "system":
        return
    if any(d.get("bonafide_label") is None for d in data.values()):
        raise SystemExit(
            "--cluster system needs an attack column that agrees with the label "
            "column, so bonafide rows can be told from spoofed ones. This table "
            "does not have one. Use --cluster utterance, which is what "
            "docs/PREREGISTRATION.md specifies in any case."
        )


def folded(y, v):
    """Direction agnostic separation. A statistic running bonafide-high
    separates just as well as one running spoof-high."""
    return abs(fast_auc(y, v) - 0.5) + 0.5


# A third verdict beside True and False. A string so it is truthy by accident
# nowhere: every use compares with `is`.
UNDECIDED = "undecided"


def at_least(needed, clear, unsettled):
    """
    Whether at least `needed` cells are clear, when `unsettled` more of them
    could go either way. True, False, or UNDECIDED when the unsettled cells
    would decide it.
    """
    if clear >= needed:
        return True
    return UNDECIDED if clear + unsettled >= needed else False


def settle_note(result):
    """How many draws would settle an undecided bound, as a phrase."""
    needed = result["draws_to_settle"]
    if needed is None:
        return "and the bound sits on zero, so no draw count settles it"
    return f"about {needed:,} draws would settle it"


MIN_SLICES = 30
MIN_UTTERANCES = 10


def usable(d) -> bool:
    """Enough of both classes, counted in utterances as well as slices."""
    for label in (0, 1):
        mask = d["y"] == label
        if int(mask.sum()) < MIN_SLICES:
            return False
        if len(set(d["u"][mask].tolist())) < MIN_UTTERANCES:
            return False
    return True


def pooled(data, categories, cluster):
    """
    One table over `categories`, plus which category each row came from.

    The bootstrap has to draw over the pooled table, because the clusters are
    utterances and one utterance yields slices in several categories. Pooling is
    what lets a single draw carry that utterance's rows into every category at
    once.
    """
    y = np.concatenate([data[c]["y"] for c in categories])
    groups = np.concatenate([cluster_groups(data[c], cluster) for c in categories])
    phase = np.concatenate([data[c][PHASE] for c in categories])
    magnitude = np.concatenate([data[c][MAGNITUDE] for c in categories])
    owner = np.concatenate([
        np.full(data[c]["y"].size, i, dtype=np.int64)
        for i, c in enumerate(categories)
    ])
    return y, groups, phase, magnitude, owner


def draw_mean(idx, owner, y, phase, magnitude, which):
    """
    The mean per category advantage inside one draw, over the categories `which`.

    A category the draw missed entirely, or landed on with one class only, is
    SKIPPED rather than counted as zero. Counting it as zero would drag the mean
    toward zero in proportion to how often the draw misses, which is a bias that
    grows as a category gets rarer. Returns None when the draw reached none of
    them.
    """
    per_category = []
    for i in which:
        rows = idx[owner[idx] == i]
        if rows.size == 0:
            continue
        ys = y[rows]
        if np.unique(ys).size < 2:
            continue
        per_category.append(folded(ys, phase[rows]) - folded(ys, magnitude[rows]))
    return float(np.mean(per_category)) if per_category else None


def observed_mean(data, categories):
    """The mean per category advantage on the sample as it stands."""
    return float(np.mean([
        folded(data[c]["y"], data[c][PHASE]) - folded(data[c]["y"], data[c][MAGNITUDE])
        for c in categories
    ]))


REQUIRED_BONAFIDE_UTTERANCES = 200


def adequately_powered(per_category, categories,
                       minimum=REQUIRED_BONAFIDE_UTTERANCES):
    """
    Which categories meet the Required sample size precondition.

    docs/PREREGISTRATION.md asks for "at least 200 bonafide source utterances
    per phoneme category" and says the binding constraint is bonafide
    utterances. So the bonafide count alone decides it.

    This read min(bonafide, spoof), which also demanded 200 spoofed utterances
    under a label that says bonafide. That is stricter than the registered
    requirement, and on a table with a small spoof side it would demote a result
    the registration calls adequately powered. It changes no published verdict:
    the eval run's spoof counts were all above 200 as well.

    Note that this is a power precondition and not one of the four falsification
    conditions, whatever the numbering in the printed output suggests.
    """
    return [c for c in categories
            if per_category.get(c, {}).get("utterances", (0, 0))[0] >= minimum]


def _interval_fields(interval):
    """
    The interval as this script reports it. `undecided` means the draw count
    does not settle whether the interval excludes zero, see percentile_interval
    in src/phonophase/stats.py.
    """
    return {"ci_low": interval["ci_low"], "ci_high": interval["ci_high"],
            "excludes_zero": interval["excludes_null"],
            "undecided": interval["undecided"],
            "draws_to_settle": interval["draws_to_settle"]}


def advantage_ci(data, categories, n_boot, seed, alpha=0.05, cluster="utterance"):
    """
    Bootstrap the mean phase-minus-magnitude advantage over a set of categories,
    resampling SOURCE UTTERANCES rather than slices.

    Slices from one utterance share a speaker, a channel and, for spoof, the
    same synthesis system. Resampling slices would understate the uncertainty by
    roughly the square root of slices per utterance.

    ONE draw serves every category, which is a correction. This function used to
    resample each category independently and average the results. One utterance
    feeds all three categories, so independent draws discarded the dependence
    they share and shrank the variance of their mean: every published LEVEL
    interval was about 40 percent too narrow. The conclusions survive the
    correction, the widths do not. See shared_cluster_draws in
    src/phonophase/stats.py.

    `cluster` selects what gets resampled and therefore what the interval is a
    statement about. See cluster_groups in src/phonophase/stats.py.
    `utterance` is the registered setting and the default.
    """
    present = [c for c in categories if c in data and usable(data[c])]
    if not present:
        return None

    y, groups, phase, magnitude, owner = pooled(data, present, cluster)

    draws = []
    for idx in shared_cluster_draws(y, groups, n_boot, seed,
                                    stratify=cluster != "speaker-paired"):
        if idx.size == 0:
            continue
        mean = draw_mean(idx, owner, y, phase, magnitude, range(len(present)))
        if mean is not None:
            draws.append(mean)

    if not draws:
        return None

    interval = percentile_interval(draws, alpha)
    return {"advantage": observed_mean(data, present), "categories": present,
            **_interval_fields(interval), "n_draws": len(draws)}


def asymmetry_ci(data, n_boot, seed, alpha=0.05, cluster="utterance"):
    """
    Bootstrap the ASYMMETRY: sonorant advantage minus obstruent advantage.

    Distinct from the LEVEL, which is the sonorant advantage on its own. The two
    are easy to confuse and this project has already shipped two claims that
    confused them, so both are reported and neither is called "the effect".

    Both group means come from ONE draw over all six categories, so the
    difference is a difference rather than two separately reported numbers. That
    matters in the opposite direction from the LEVEL: sonorant and obstruent
    slices largely come from the same utterances, so whatever those utterances
    contribute is common to both means and cancels here. Pairing therefore
    TIGHTENS this interval where it WIDENS the LEVEL. An earlier version drew
    each category independently and said in this docstring that the real
    asymmetry interval was probably narrower than the one it reported; it was,
    and this is the estimator that measures it.

    No verdict is computed from this. The falsification conditions are left
    exactly as docs/PREREGISTRATION.md states them.
    """
    sides = []
    for categories in (SONORANTS, OBSTRUENTS):
        present = [c for c in categories if c in data and usable(data[c])]
        if not present:
            return None
        sides.append(present)

    every = sides[0] + sides[1]
    y, groups, phase, magnitude, owner = pooled(data, every, cluster)
    sonorant_owners = range(len(sides[0]))
    obstruent_owners = range(len(sides[0]), len(every))

    draws = []
    for idx in shared_cluster_draws(y, groups, n_boot, seed,
                                    stratify=cluster != "speaker-paired"):
        if idx.size == 0:
            continue
        son = draw_mean(idx, owner, y, phase, magnitude, sonorant_owners)
        obs = draw_mean(idx, owner, y, phase, magnitude, obstruent_owners)
        if son is not None and obs is not None:
            draws.append(son - obs)

    if not draws:
        return None

    observed = observed_mean(data, sides[0]) - observed_mean(data, sides[1])
    return {"asymmetry": observed,
            **_interval_fields(percentile_interval(draws, alpha)),
            "n_draws": len(draws)}


def run_by_language(args, speakers):
    """
    H2. The same pre-registered test, once per language.

    If the artifact is articulatory rather than linguistic it should appear in
    every language. If it appears only in English it was a property of the
    training corpus and the cross lingual story is over, which is worth knowing
    cheaply.
    """
    languages = set()
    with open_table(args.table) as fh:
        for r in csv.DictReader(fh):
            if r["seg"] == args.seg:
                languages.add(r.get("language", "-"))

    print(f"H2, per language. table {args.table}, segmentation {args.seg}, "
          f"clustered by {args.cluster}")
    print(f"languages: {', '.join(sorted(languages))}\n")
    print(f"{'LANGUAGE':<12} {'SONORANT ADV':>14} {'95% CI':>20} {'OBSTRUENT ADV':>15}  VERDICT")
    print("-" * 78)

    for lang in sorted(languages):
        data = load(args.table, args.seg, language=lang, speakers=speakers)
        require_cluster(data, args.cluster)

        son = advantage_ci(data, SONORANTS, args.n_boot, args.seed, cluster=args.cluster)
        obs = advantage_ci(data, OBSTRUENTS, args.n_boot, args.seed + 1, cluster=args.cluster)
        if son is None:
            print(f"{lang:<12} {'too few':>14}")
            continue
        holds = bool(son["excludes_zero"] and son["advantage"] > 0 and (
            obs is None or (np.isfinite(obs["advantage"])
                            and son["advantage"] > obs["advantage"])))
        verdict = ("undecided at this draw count" if son["undecided"]
                   else "holds" if holds else "does not hold")
        print(f"{lang:<12} {son['advantage']:>+14.3f} "
              f"{'[%+.3f, %+.3f]' % (son['ci_low'], son['ci_high']):>20} "
              f"{(obs['advantage'] if obs else float('nan')):>+15.3f}  "
              f"{verdict}")

    print("\nH2 is supported only if the effect holds in a language absent from")
    print("training. Holding in English alone is the null result for H2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--seg", default="run", choices=["run", "window"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", default="utterance", choices=CLUSTERS,
                    help="what the bootstrap resamples. 'utterance' is the "
                         "registered setting and makes the interval a statement "
                         "about more utterances from the same 13 synthesis "
                         "systems. 'system' clusters the spoof side by attack, "
                         "making it a statement about systems we have not seen. "
                         "'speaker' clusters both sides by speaker, and "
                         "'speaker-paired' picks each speaker once for both "
                         "classes")
    ap.add_argument("--protocol", default=DEFAULT_PROTOCOL,
                    help="ASVspoof protocol file, used to find the speaker of a "
                         "row when the table has no speaker column")
    ap.add_argument("--by-language", action="store_true",
                    help="run the same test separately per language. This is "
                         "the H2 cross lingual test. If the effect is "
                         "articulatory it should hold in every language")
    args = ap.parse_args()

    speakers = (read_speaker_map(args.protocol)
                if os.path.exists(args.protocol) else {})

    if args.by_language:
        run_by_language(args, speakers)
        return

    data = load(args.table, args.seg, speakers=speakers)
    if not data:
        raise SystemExit(f"no rows with seg={args.seg} in {args.table}")
    require_cluster(data, args.cluster)

    print(f"table {args.table}, segmentation {args.seg}\n")
    print(f"{'CATEGORY':<16} {'SLICES':>12} {'UTTERANCES':>12} {'PHASE SEP':>10} {'95% CI':>16}")
    print("-" * 70)

    per_category, p_like = {}, []
    for cat in sorted(data):
        d = data[cat]
        n_b, n_s = int((d["y"] == 0).sum()), int((d["y"] == 1).sum())
        u_b = len(set(d["u"][d["y"] == 0].tolist()))
        u_s = len(set(d["u"][d["y"] == 1].tolist()))
        if not usable(d):
            print(f"{cat:<16} {f'{n_b}/{n_s}':>12} {f'{u_b}/{u_s}':>12}   too few")
            continue
        r = bootstrap_auc_ci(d["y"], d[PHASE], n_boot=args.n_boot,
                             seed=args.seed, groups=cluster_groups(d, args.cluster))
        sep = max(r["auc"], 1 - r["auc"])
        lo, hi = ((r["ci_low"], r["ci_high"]) if r["auc"] >= 0.5
                  else (1 - r["ci_high"], 1 - r["ci_low"]))
        per_category[cat] = {"sep": sep, "lo": lo, "hi": hi,
                             "clears_chance": not r["crosses_chance"],
                             "chance_undecided": r["chance_undecided"],
                             "utterances": (u_b, u_s)}
        print(f"{cat:<16} {f'{n_b}/{n_s}':>12} {f'{u_b}/{u_s}':>12} "
              f"{sep:>10.3f} {'[%.3f, %.3f]' % (lo, hi):>16}")

    son = advantage_ci(data, SONORANTS, args.n_boot, args.seed, cluster=args.cluster)
    obs = advantage_ci(data, OBSTRUENTS, args.n_boot, args.seed + 1, cluster=args.cluster)
    asym = asymmetry_ci(data, args.n_boot, args.seed, cluster=args.cluster)

    print("\n" + "=" * 70)
    print("PRE-REGISTERED TEST, docs/PREREGISTRATION.md")
    print("=" * 70)
    for name, r in (("sonorants", son), ("obstruents", obs)):
        if r is None:
            print(f"  {name:<12} unavailable")
            continue
        print(f"  {name:<12} advantage {r['advantage']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]  "
              f"over {', '.join(r['categories'])}")

    # The LEVEL is the sonorant advantage above. The ASYMMETRY is the difference
    # between the two, which is the quantity H1 is about. Reported side by side
    # because confusing them is the failure this project has already shipped
    # twice. No falsification condition is computed from the line below.
    if asym is not None:
        print(f"  {'asymmetry':<12} sonorant minus obstruent {asym['asymmetry']:+.3f} "
              f"[{asym['ci_low']:+.3f}, {asym['ci_high']:+.3f}]")
    print(f"  {'clustered by':<12} {args.cluster}")

    print("\nFalsification conditions, stated in advance:")
    verdicts = []

    if son is None:
        verdicts.append(("1. sonorant advantage interval excludes zero", None, "unavailable"))
    else:
        ok = (UNDECIDED if son["undecided"]
              else son["excludes_zero"] and son["advantage"] > 0)
        detail = f"{son['advantage']:+.3f} [{son['ci_low']:+.3f}, {son['ci_high']:+.3f}]"
        if son["undecided"]:
            detail += ("\n         the bound nearest zero is within Monte Carlo "
                       f"error of it at {son['n_draws']} draws, "
                       + settle_note(son))
        verdicts.append(("1. sonorant advantage interval excludes zero and is positive",
                         ok, detail))

    if son and obs:
        ok = bool(np.isfinite(son["advantage"]) and np.isfinite(obs["advantage"])
                  and son["advantage"] > obs["advantage"])
        verdicts.append(("2. sonorant advantage exceeds obstruent advantage", ok,
                         f"{son['advantage']:+.3f} against {obs['advantage']:+.3f}"))
    else:
        verdicts.append(("2. sonorant advantage exceeds obstruent advantage", False,
                         "one group had too few usable categories to compare"))

    unsettled = [c for c in SONORANTS
                 if per_category.get(c, {}).get("chance_undecided")]
    clears = [c for c in SONORANTS
              if per_category.get(c, {}).get("clears_chance") and c not in unsettled]
    ok = at_least(2, len(clears), len(unsettled))
    detail = f"{len(clears)} of 3, {', '.join(clears) or 'none'}"
    if unsettled:
        detail += f". Not settled at this draw count: {', '.join(unsettled)}"
    verdicts.append(("3. phase clears chance in at least 2 of 3 sonorant categories",
                     ok, detail))

    powered = adequately_powered(per_category, SONORANTS)
    verdicts.append(("Required sample size, at least 200 bonafide utterances "
                     "per sonorant category",
                     len(powered) == 3, f"{len(powered)} of 3 adequately powered"))

    for text, ok, detail in verdicts:
        mark = ("n/a " if ok is None else "UNDECIDED" if ok is UNDECIDED
                else "PASS" if ok else "FAIL")
        print(f"  [{mark}] {text}\n         {detail}")

    # Condition 4 compares two partitions, so one table cannot decide it and
    # this script must not appear to. It was reported here for months as the
    # sample size row above, which is a different requirement from a different
    # section of the registration.
    print("  [ -- ] 4. present on train attacks A01 to A06 but absent on eval")
    print("         needs both partitions, run scripts/falsification_4.py")

    print("\n" + "=" * 70)

    # Undecided is a distinct outcome from rejected, and conflating them is the
    # single most consequential error available here. A run with too little data
    # to evaluate the primary test has not refuted anything, it has failed to
    # test anything. The pilot's split-half was exactly this case.
    if son is None or obs is None or not np.isfinite(son["advantage"]):
        print("UNDECIDED. Not enough usable data to run the primary test.")
        print("This is NOT a rejection. Nothing has been tested.")
        print("Raise the utterance counts and rerun. Needed, 200 bonafide")
        print("utterances per sonorant category.")
        print("=" * 70)
        return

    decided = [v[1] for v in verdicts[:3] if v[1] is not None]
    if UNDECIDED in decided and False not in decided:
        # A bound within Monte Carlo error of its threshold prints PASS on one
        # seed and FAIL on another. Neither is a result.
        print("UNDECIDED at this draw count. A falsification condition turns on")
        print("a bootstrap bound that another seed would put on the other side")
        print("of its threshold. This is NOT a rejection and NOT a confirmation.")
        print("Rerun with a larger --n-boot.")
    elif len(decided) == 3 and all(decided):
        print("H1 SUPPORTED on this partition.")
        if len(powered) < 3:
            print("Power condition not met, so treat as provisional and say so.")
    elif len(powered) < 3:
        print("H1 NOT SUPPORTED, but the run is underpowered.")
        print(f"Only {len(powered)} of 3 sonorant categories reach 200 bonafide")
        print("utterances, so this is weak evidence against rather than a")
        print("rejection. Raise the counts before concluding anything.")
    else:
        print("H1 REJECTED on this partition, per the conditions fixed in advance,")
        print("at adequate power. Write it up as a negative result rather than")
        print("reframing it.")
    print("=" * 70)


if __name__ == "__main__":
    main()
