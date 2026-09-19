"""
Run the pre-registered test on the table produced by slice_and_probe.py.

The decision procedure in docs/PREREGISTRATION.md is implemented literally,
including all four falsification conditions, and the verdict is printed. That
is the point. A procedure fixed in advance and executed mechanically is the
only defence against fitting a story to whatever the data happens to show,
which has already happened twice in this project.

Runs locally in seconds on the downloaded CSV. No GPU, no corpus.

Usage
    python scripts/analyse_slices.py --table results/eval_slices.csv
    python scripts/analyse_slices.py --table results/eval_slices.csv --seg window
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score  # noqa: F401

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.stats import (  # noqa: E402
    _cluster_draw, _cluster_index, benjamini_hochberg, bootstrap_auc_ci, fast_auc,
)

SONORANTS = ["NASALS", "APPROXIMANTS", "VOWELS_STEADY"]
OBSTRUENTS = ["FRICATIVES", "PLOSIVES", "AFFRICATES"]

PHASE = "mgd_hf_std"
MAGNITUDE = "mfcc_frame_spread"


def load(path, seg):
    rows = defaultdict(lambda: {"y": [], "u": [], PHASE: [], MAGNITUDE: []})
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["seg"] != seg:
                continue
            d = rows[r["category"]]
            d["y"].append(int(r["label"]))
            d["u"].append(r["utterance_id"])
            d[PHASE].append(float(r[PHASE]))
            d[MAGNITUDE].append(float(r[MAGNITUDE]))
    return {k: {n: np.asarray(v) for n, v in d.items()} for k, d in rows.items()}


def folded(y, v):
    """Direction agnostic separation. A statistic running bonafide-high
    separates just as well as one running spoof-high."""
    return abs(fast_auc(y, v) - 0.5) + 0.5


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


def advantage_ci(data, categories, n_boot, seed, alpha=0.05):
    """
    Bootstrap the mean phase-minus-magnitude advantage over a set of categories,
    resampling SOURCE UTTERANCES rather than slices.

    Slices from one utterance share a speaker, a channel and, for spoof, the
    same synthesis system. Resampling slices would understate the uncertainty by
    roughly the square root of slices per utterance.

    The cluster index is prepared once per category and each draw is a
    vectorised gather. An earlier version rebuilt the utterance list and
    boolean-scanned every row once per picked utterance inside the draw loop,
    which on the eval run was about 29 million comparisons per category per
    draw and turned a sub-minute analysis into one that ran past ten minutes.
    """
    rng = np.random.default_rng(seed)
    present = [c for c in categories if c in data and usable(data[c])]
    if not present:
        return None

    prepared = {c: _cluster_index(data[c]["y"], data[c]["u"]) for c in present}

    draws = []
    for _ in range(n_boot):
        per_category = []
        for cat in present:
            d = data[cat]
            idx = _cluster_draw(rng, prepared[cat])
            if idx.size == 0 or np.unique(d["y"][idx]).size < 2:
                continue
            y = d["y"][idx]
            per_category.append(
                folded(y, d[PHASE][idx]) - folded(y, d[MAGNITUDE][idx])
            )
        if per_category:
            draws.append(float(np.mean(per_category)))

    if not draws:
        return None

    observed = float(np.mean([
        folded(data[c]["y"], data[c][PHASE]) - folded(data[c]["y"], data[c][MAGNITUDE])
        for c in present
    ]))
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"advantage": observed, "ci_low": float(lo), "ci_high": float(hi),
            "categories": present, "excludes_zero": bool(lo > 0 or hi < 0)}


def run_by_language(args):
    """
    H2. The same pre-registered test, once per language.

    If the artifact is articulatory rather than linguistic it should appear in
    every language. If it appears only in English it was a property of the
    training corpus and the cross lingual story is over, which is worth knowing
    cheaply.
    """
    languages = set()
    with open(args.table, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["seg"] == args.seg:
                languages.add(r.get("language", "-"))

    print(f"H2, per language. table {args.table}, segmentation {args.seg}")
    print(f"languages: {', '.join(sorted(languages))}\n")
    print(f"{'LANGUAGE':<12} {'SONORANT ADV':>14} {'95% CI':>20} {'OBSTRUENT ADV':>15}  VERDICT")
    print("-" * 78)

    for lang in sorted(languages):
        rows = defaultdict(lambda: {"y": [], "u": [], PHASE: [], MAGNITUDE: []})
        with open(args.table, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["seg"] != args.seg or r.get("language", "-") != lang:
                    continue
                d = rows[r["category"]]
                d["y"].append(int(r["label"]))
                d["u"].append(r["utterance_id"])
                d[PHASE].append(float(r[PHASE]))
                d[MAGNITUDE].append(float(r[MAGNITUDE]))
        data = {k: {n: np.asarray(v) for n, v in d.items()} for k, d in rows.items()}

        son = advantage_ci(data, SONORANTS, args.n_boot, args.seed)
        obs = advantage_ci(data, OBSTRUENTS, args.n_boot, args.seed + 1)
        if son is None:
            print(f"{lang:<12} {'too few':>14}")
            continue
        holds = bool(son["excludes_zero"] and son["advantage"] > 0 and (
            obs is None or (np.isfinite(obs["advantage"])
                            and son["advantage"] > obs["advantage"])))
        print(f"{lang:<12} {son['advantage']:>+14.3f} "
              f"{'[%+.3f, %+.3f]' % (son['ci_low'], son['ci_high']):>20} "
              f"{(obs['advantage'] if obs else float('nan')):>+15.3f}  "
              f"{'holds' if holds else 'does not hold'}")

    print("\nH2 is supported only if the effect holds in a language absent from")
    print("training. Holding in English alone is the null result for H2.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--seg", default="run", choices=["run", "window"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--by-language", action="store_true",
                    help="run the same test separately per language. This is "
                         "the H2 cross lingual test. If the effect is "
                         "articulatory it should hold in every language")
    args = ap.parse_args()

    if args.by_language:
        run_by_language(args)
        return

    data = load(args.table, args.seg)
    if not data:
        raise SystemExit(f"no rows with seg={args.seg} in {args.table}")

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
                             seed=args.seed, groups=d["u"])
        sep = max(r["auc"], 1 - r["auc"])
        lo, hi = ((r["ci_low"], r["ci_high"]) if r["auc"] >= 0.5
                  else (1 - r["ci_high"], 1 - r["ci_low"]))
        per_category[cat] = {"sep": sep, "lo": lo, "hi": hi,
                             "clears_chance": not r["crosses_chance"],
                             "utterances": (u_b, u_s)}
        print(f"{cat:<16} {f'{n_b}/{n_s}':>12} {f'{u_b}/{u_s}':>12} "
              f"{sep:>10.3f} {'[%.3f, %.3f]' % (lo, hi):>16}")

    son = advantage_ci(data, SONORANTS, args.n_boot, args.seed)
    obs = advantage_ci(data, OBSTRUENTS, args.n_boot, args.seed + 1)

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

    print("\nFalsification conditions, stated in advance:")
    verdicts = []

    if son is None:
        verdicts.append(("1. sonorant advantage interval excludes zero", None, "unavailable"))
    else:
        ok = son["excludes_zero"] and son["advantage"] > 0
        verdicts.append(("1. sonorant advantage interval excludes zero and is positive",
                         ok, f"{son['advantage']:+.3f} [{son['ci_low']:+.3f}, {son['ci_high']:+.3f}]"))

    if son and obs:
        ok = bool(np.isfinite(son["advantage"]) and np.isfinite(obs["advantage"])
                  and son["advantage"] > obs["advantage"])
        verdicts.append(("2. sonorant advantage exceeds obstruent advantage", ok,
                         f"{son['advantage']:+.3f} against {obs['advantage']:+.3f}"))
    else:
        verdicts.append(("2. sonorant advantage exceeds obstruent advantage", False,
                         "one group had too few usable categories to compare"))

    clears = [c for c in SONORANTS if per_category.get(c, {}).get("clears_chance")]
    verdicts.append((f"3. phase clears chance in at least 2 of 3 sonorant categories",
                     len(clears) >= 2, f"{len(clears)} of 3, {', '.join(clears) or 'none'}"))

    powered = [c for c in SONORANTS
               if min(per_category.get(c, {}).get("utterances", (0, 0))) >= 200]
    verdicts.append(("4. at least 200 bonafide utterances per sonorant category",
                     len(powered) == 3, f"{len(powered)} of 3 adequately powered"))

    for text, ok, detail in verdicts:
        mark = "PASS" if ok else ("n/a " if ok is None else "FAIL")
        print(f"  [{mark}] {text}\n         {detail}")

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
    if len(decided) == 3 and all(decided):
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
