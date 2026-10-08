"""
Evaluate the FOURTH falsification condition of docs/PREREGISTRATION.md.

The condition, quoted from the registration:

    The effect is present on train attacks A01 to A06 but absent on eval, which
    would make it a property of those six synthesis systems rather than of
    speech.

Why this script exists
    No script in the repository evaluated it. `scripts/analyse_slices.py` prints
    four numbered conditions, and its fourth is the *Required sample size*
    requirement from a different section of the registration, "at least 200
    bonafide utterances per sonorant category". The outcome table in
    docs/PREREGISTRATION.md reports that same sample-size row as condition 4.
    The cross partition comparison the registration actually fixed in advance
    was never run, so "all four falsification conditions passed" rested on three
    conditions and a power check.

    This script leaves analyse_slices.py alone and runs the condition as worded.

What it measures, and why two numbers and not one
    "The sonorant advantage" names two different quantities, and conflating them
    has already produced wrong claims in this project. Both are carried
    separately end to end and neither is called "the effect".

        LEVEL        the sonorant mean phase-minus-magnitude advantage on its
                     own. Folded AUC of mgd_hf_std minus folded AUC of
                     mfcc_frame_spread, averaged over NASALS, APPROXIMANTS and
                     VOWELS_STEADY.

        ASYMMETRY    the LEVEL minus the same quantity averaged over
                     FRICATIVES, PLOSIVES and AFFRICATES. This is the quantity
                     the primary test of H1 is stated in.

    Condition 4 is a conjunction, so it is evaluated once per quantity. Both
    verdicts are printed.

The data
    A01 to A06 are the six systems the countermeasure training partition uses.
    The ASVspoof 2019 LA **development** partition uses those same six, so dev
    stands in for "train attacks A01 to A06" here. Eval is A07 to A19.

    Dev and eval differ in speakers and in size as well as in attacks, so a
    difference between the two partitions is not by itself an attack effect.
    That confound is printed with the result rather than left implicit.

Estimator
    Bootstrap over source utterances, the setting docs/PREREGISTRATION.md
    specifies throughout. The estimator is the registered one, imported from
    scripts/analyse_slices.py rather than reimplemented, so this script cannot
    disagree with the published eval numbers by construction.

Usage
    results/dev_slices.csv.gz is a data artifact this branch does not track.
    Retrieve it first:

        git show task/T6-attack-reversals:results/dev_slices.csv.gz \\
            > results/dev_slices.csv.gz

    then

        python scripts/falsification_4.py \\
            --train-table results/dev_slices.csv.gz \\
            --eval-table results/eval_slices.csv.gz
        python scripts/falsification_4.py \\
            --train-table results/dev_slices.csv.gz \\
            --eval-table results/eval_slices.csv.gz --seg window
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
sys.path.insert(0, _HERE)

from analyse_slices import (  # noqa: E402
    BONAFIDE_MARKERS,
    MAGNITUDE,
    OBSTRUENTS,
    PHASE,
    SONORANTS,
    advantage_ci,
    asymmetry_ci,
    open_table,
    usable,
    with_bonafide_label,
)

# The six systems the registration names. The countermeasure training partition
# and the development partition of ASVspoof 2019 LA both use exactly these.
TRAIN_ATTACKS = ("A01", "A02", "A03", "A04", "A05", "A06")

FIRES = "FIRES"
DOES_NOT_FIRE = "DOES NOT FIRE"
UNRESOLVED = "UNRESOLVED"

CONDITION = ("The effect is present on train attacks A01 to A06 but absent on "
             "eval, which would make it a property of those six synthesis "
             "systems rather than of speech.")


def is_bonafide_attack(attack) -> bool:
    """
    Whether an attack field marks genuine speech rather than a synthesis system.

    Uses the same markers analyse_slices.bonafide_label reads the label
    convention from, so the attack filter and the label convention cannot drift
    apart.
    """
    return str(attack).lower() in BONAFIDE_MARKERS


def load_attacks(path, seg, attacks=None):
    """
    Load a slice table restricted to a set of attacks, keeping ALL bonafide rows.

    The bonafide side is what both partitions are scored against, so it is never
    filtered. Bonafide rows carry attack '-', so the obvious
    `row["attack"] in attacks` would delete every genuine row and leave a single
    class table. Nothing downstream raises on that: every AUC simply collapses
    towards chance and the condition would read as absent everywhere for a
    reason that has nothing to do with speech.

    `attacks=None` keeps every row, which is how the eval side is loaded.
    """
    keep = None if attacks is None else set(attacks)
    rows = defaultdict(lambda: {"y": [], "u": [], "attack": [],
                                PHASE: [], MAGNITUDE: []})
    with open_table(path) as fh:
        for r in csv.DictReader(fh):
            if r["seg"] != seg:
                continue
            attack = r.get("attack", "")
            if keep is not None and attack not in keep and not is_bonafide_attack(attack):
                continue
            d = rows[r["category"]]
            d["y"].append(int(r["label"]))
            d["u"].append(r["utterance_id"])
            d["attack"].append(attack)
            d[PHASE].append(float(r[PHASE]))
            d[MAGNITUDE].append(float(r[MAGNITUDE]))
    data = {k: {n: np.asarray(v) for n, v in d.items()} for k, d in rows.items()}
    return with_bonafide_label(data)


def _normalise(result, key):
    """
    One shape for both quantities, with the estimate under a neutral name.

    The name of the quantity lives in the key of the dict `measure` returns, so
    a LEVEL can never be printed under an ASYMMETRY heading by accident.
    """
    if result is None:
        return None
    return {"estimate": float(result[key]),
            "ci_low": float(result["ci_low"]),
            "ci_high": float(result["ci_high"]),
            "excludes_zero": bool(result["excludes_zero"]),
            "positive": result.get("positive", bool(
                result["excludes_zero"] and result[key] > 0)),
            "categories": tuple(result.get("categories", ()))}


def measure(data, n_boot=2000, seed=0, cluster="utterance"):
    """
    The LEVEL, the obstruent advantage, and the ASYMMETRY, each with a 95%
    bootstrap interval clustered by source utterance.

    The obstruent advantage is returned because the ASYMMETRY is built from it
    and an unlabelled difference is not checkable. It is not itself a claim.

    Seeds follow analyse_slices.main so the numbers this prints for a whole
    table match what the registered script prints for it.
    """
    son = advantage_ci(data, SONORANTS, n_boot, seed, cluster=cluster)
    obs = advantage_ci(data, OBSTRUENTS, n_boot, seed + 1, cluster=cluster)
    asym = asymmetry_ci(data, n_boot, seed, cluster=cluster)
    return {"level": _normalise(son, "advantage"),
            "obstruent_level": _normalise(obs, "advantage"),
            "asymmetry": _normalise(asym, "asymmetry")}


def presence(result):
    """
    Is the effect present: True, False, or None for not measurable.

    Presence is a property of the interval, not of the size of the estimate. A
    small advantage whose interval clears zero is present; a large one whose
    interval straddles zero is not. The registration's first falsification
    condition is worded the same way, "the sonorant advantage interval includes
    zero".

    Direction counts. H1 states which way the effect goes, so an interval that
    excludes zero from below is a reversal and not a presence.

    None is returned when there was nothing to measure, and when the interval's
    lower bound is within Monte Carlo error of zero, so that another seed could
    answer the other way. That is a third outcome and collapsing it into False would
    let missing data, or the seed, fire a condition.
    """
    if result is None:
        return None
    estimate = result.get("estimate")
    if estimate is None or not np.isfinite(estimate):
        return None
    if "positive" in result:
        return result["positive"]
    return bool(result["excludes_zero"] and estimate > 0)


def condition_4(train_present, eval_present):
    """
    The condition as worded: present on A01 to A06 AND absent on eval.

    Either side unmeasured gives UNRESOLVED. Undecided is not rejected, and it
    is not a pass either.
    """
    if train_present is None or eval_present is None:
        return UNRESOLVED
    return FIRES if (train_present and not eval_present) else DOES_NOT_FIRE


def describe(data):
    """Counts per category, so the size difference between partitions is visible."""
    out = {}
    for cat, d in data.items():
        bona, spoof = d["y"] == d["bonafide_label"], d["y"] != d["bonafide_label"]
        out[cat] = {
            "slices": (int(bona.sum()), int(spoof.sum())),
            "utterances": (len(set(d["u"][bona].tolist())),
                           len(set(d["u"][spoof].tolist()))),
            "usable": usable(d),
        }
    return out


def attacks_present(data):
    seen = set()
    for d in data.values():
        seen |= {a for a in np.unique(d["attack"]).tolist()
                 if not is_bonafide_attack(a)}
    return sorted(seen)


def _print_partition(name, path, data, result):
    counts = describe(data)
    print(f"\n{name}")
    print(f"  table      {path}")
    print(f"  attacks    {', '.join(attacks_present(data)) or 'none'}")
    print(f"  {'CATEGORY':<16} {'SLICES bona/spoof':>20} {'UTTERANCES bona/spoof':>24}")
    for cat in sorted(counts):
        c = counts[cat]
        mark = "" if c["usable"] else "   too few"
        print(f"  {cat:<16} {'%d/%d' % c['slices']:>20} "
              f"{'%d/%d' % c['utterances']:>24}{mark}")

    print(f"\n  {'QUANTITY':<18} {'ESTIMATE':>10} {'95% CI':>22}  PRESENT")
    for key, label in (("level", "LEVEL"),
                       ("obstruent_level", "obstruent adv"),
                       ("asymmetry", "ASYMMETRY")):
        r = result[key]
        if r is None:
            print(f"  {label:<18} {'unmeasurable':>10}")
            continue
        verdict = {True: "yes", False: "no", None: "not measurable"}[presence(r)]
        shown = verdict if key != "obstruent_level" else "not a claim"
        print(f"  {label:<18} {r['estimate']:>+10.4f} "
              f"{'[%+.4f, %+.4f]' % (r['ci_low'], r['ci_high']):>22}  {shown}")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate falsification condition 4 of docs/PREREGISTRATION.md")
    ap.add_argument("--train-table", required=True,
                    help="slice table for the A01 to A06 systems, i.e. the "
                         "ASVspoof 2019 LA development partition")
    ap.add_argument("--eval-table", required=True,
                    help="slice table for eval, attacks A07 to A19")
    ap.add_argument("--seg", default="run", choices=["run", "window"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cluster", default="utterance",
                    choices=["utterance", "system"],
                    help="what the bootstrap resamples. 'utterance' is the "
                         "registered setting and the default")
    ap.add_argument("--train-attacks", default=",".join(TRAIN_ATTACKS),
                    help="comma separated. Bonafide rows are kept regardless")
    args = ap.parse_args()

    wanted = [a.strip() for a in args.train_attacks.split(",") if a.strip()]

    print("=" * 74)
    print("FALSIFICATION CONDITION 4, docs/PREREGISTRATION.md")
    print("=" * 74)
    print(f"\n  {CONDITION}\n")
    print(f"segmentation {args.seg}, {args.n_boot} bootstrap draws clustered by "
          f"{args.cluster}, seed {args.seed}")
    print("\nTwo quantities, reported separately and never merged:")
    print("  LEVEL      sonorant mean phase-minus-magnitude advantage, alone")
    print("  ASYMMETRY  that LEVEL minus the obstruent advantage")

    train = load_attacks(args.train_table, args.seg, wanted)
    if not train:
        raise SystemExit(f"no rows with seg={args.seg} in {args.train_table}")
    held_out = load_attacks(args.eval_table, args.seg, None)
    if not held_out:
        raise SystemExit(f"no rows with seg={args.seg} in {args.eval_table}")

    train_r = measure(train, args.n_boot, args.seed, args.cluster)
    eval_r = measure(held_out, args.n_boot, args.seed, args.cluster)

    _print_partition(f"TRAIN SYSTEMS, {', '.join(wanted)}",
                     args.train_table, train, train_r)
    _print_partition("HELD OUT EVAL, A07 to A19", args.eval_table,
                     held_out, eval_r)

    print("\n" + "=" * 74)
    print("VERDICT ON CONDITION 4")
    print("=" * 74)
    print("The condition is a conjunction, so it is evaluated once per quantity.")

    verdicts = {}
    for key, label in (("level", "LEVEL"), ("asymmetry", "ASYMMETRY")):
        t, e = presence(train_r[key]), presence(eval_r[key])
        verdicts[key] = condition_4(t, e)
        shown = {True: "present", False: "absent", None: "not measurable"}
        print(f"\n  {label}")
        print(f"    on {', '.join(wanted)}   {shown[t]}")
        print(f"    on eval A07 to A19        {shown[e]}")
        print(f"    condition 4               {verdicts[key]}")

    print("\nWhat would have made it fire, for either quantity: the interval on")
    print("A01 to A06 excluding zero from above, together with the interval on")
    print("eval including zero, or excluding it from below. Both clauses are")
    print("required. Presence on the train systems alone does not falsify H1,")
    print("because H1 predicts the effect there too.")

    print("\nConfound, stated because it bears on any cross partition reading:")
    print("dev and eval differ in speakers and in sample size as well as in")
    print("attacks, so a difference between the partitions is not by itself an")
    print("attack effect. It limits what a FIRES verdict could be attributed")
    print("to; it does not weaken a DOES NOT FIRE verdict, which needs only")
    print("that the effect be present on eval.")

    print("\n" + "=" * 74)
    if all(v == DOES_NOT_FIRE for v in verdicts.values()):
        print("CONDITION 4 DOES NOT FIRE on either quantity. H1 is not")
        print("falsified by this condition.")
    elif any(v == FIRES for v in verdicts.values()):
        fired = [k for k, v in verdicts.items() if v == FIRES]
        print(f"CONDITION 4 FIRES on {', '.join(fired)}. Per the registration "
              f"H1 is\nrejected. Write it up as a negative result.")
    else:
        print("CONDITION 4 UNRESOLVED. Not enough data on at least one side to")
        print("evaluate it. This is NOT a pass and NOT a rejection.")
    print("=" * 74)


if __name__ == "__main__":
    main()
