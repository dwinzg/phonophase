"""
Is the phase artifact indexed to sonority, or to voicing?

The question
    The confirmed result is stated over sonorants against obstruents. In English
    those two classes are nearly the same set as voiced against voiceless,
    r = +0.602, and the feature level re-analysis found `son` and `voi`
    statistically indistinguishable, +0.0476 against +0.0505. So the finding as
    written may be about periodicity of the excitation rather than about
    sonority, and the vocoder literature's mechanism is about periodicity. That
    makes the alternative live rather than pedantic.

    The two readings differ in what they predict, which is the only reason this
    is worth an analysis rather than a footnote. Under sonority the artifact
    follows manner of articulation. Under voicing it follows whether the vocal
    folds were vibrating, so it should appear in voiced obstruents too, and
    should be absent from voiceless sonorants.

The design
    English populates all four cells of son crossed with voi, which is what
    separates the two factors.

        A  [+son, +voi]   vowels, m n l r w j
        B  [-son, +voi]   b d g v z ð dʒ
        C  [-son, -voi]   p t k f s θ ʃ tʃ
        D  [+son, -voi]   h ʔ

    The ladder A, B, C is the contrast the task asked for, and it is primary
    because it does not depend on any contested classification. Cell D exists
    because PanPhon, following Hayes, treats glottals as [+son] on the grounds
    that they involve no oral constriction. Many phonologists would call /h/ a
    voiceless fricative instead. That disagreement is a convention rather than a
    fact about the audio, so cell D is reported separately, the crossed test
    that uses it is reported as secondary, and `--glottals obstruent` reruns the
    whole thing with D folded into C so a reader can see what the convention is
    worth.

    A slice in cell D is also the place where the silence concern bites hardest.
    A glottal stop is close to silence, and the proportion of silence is known
    to differ between genuine and synthetic speech, so a per phone breakdown is
    printed for every cell and `--drop-phone` removes one by name.

What is measured, and why the folding convention changes here
    AUC of `mgd_hf_std`, bonafide against spoof, within each cell.

    The pre-registered analysis folds, reporting max(auc, 1-auc), because it
    compares a phase statistic against a magnitude statistic and the two can run
    in opposite directions. Every cell here uses the one statistic, and the
    mechanism predicts one direction for all of them, bonafide high. So
    direction is information rather than nuisance, and every difference below is
    computed on the **directional** AUC.

    That is not a presentational choice. Folding reflects a value about 0.5, so
    for a cell sitting near chance it maps draws on both sides onto one side.
    That inflates a null cell by about one standard error, and it destroys the
    correlation between that cell's draws and another cell's, which is the
    correlation the paired bootstrap exists to exploit. Measured here, folding
    made the interval on a difference wider than treating the two cells as
    independent, which is the wrong direction and a sign the quantity was wrong.

    The folded value is still printed per cell, so the numbers can be read
    against the pre-registered ones, and so is the direction.

Why the bootstrap is built this way
    Slices are not independent. One utterance yields many, sharing a speaker, a
    channel and, when spoofed, one synthesis system. Every interval resamples
    whole utterances.

    The cells are drawn from the same utterances, so their AUCs are correlated.
    Each draw resamples utterances ONCE and recomputes all four cells inside
    that draw, so an interval on a difference carries that correlation. Two
    separate intervals that happen not to overlap would not have answered this.

What this is not
    Not confirmatory. `docs/PREREGISTRATION.md` registered the sonorant against
    obstruent contrast and nothing else, and this runs on the same eval data
    that confirmed it. It cannot license a new claim. What it can do is say
    which of two readings of the confirmed result the existing data supports,
    and what an H2 language would have to look like to separate them properly.

Usage
    python scripts/sonority_voicing.py --table results/eval_slices.csv.gz
    python scripts/sonority_voicing.py --table results/eval_slices.csv.gz --seg window
    python scripts/sonority_voicing.py --table results/eval_slices.csv.gz --glottals obstruent

Reads .csv or .csv.gz. CPU, under a minute.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.phonology import FEATURE_NAMES, features  # noqa: E402
from phonophase.stats import (  # noqa: E402
    _cluster_draw, _cluster_index, benjamini_hochberg, fast_auc,
)

PHASE = "mgd_hf_std"

SON_VOI = "[+son,+voi]"
OBS_VOI = "[-son,+voi]"
OBS_UNVOI = "[-son,-voi]"
SON_UNVOI = "[+son,-voi]"

# A, B, C. The contrast the task asked for, and the one that does not rest on
# how glottals are classified.
LADDER = [SON_VOI, OBS_VOI, OBS_UNVOI]
CELLS = LADDER + [SON_UNVOI]

CONTRASTS = {"A-B": (SON_VOI, OBS_VOI), "B-C": (OBS_VOI, OBS_UNVOI), "A-C": (SON_VOI, OBS_UNVOI)}

# Same usability floor as the pre-registered analysis, applied per cell.
# Counted in utterances as well as slices, because four hundred slices from ten
# utterances is still ten utterances.
MIN_SLICES = 30
MIN_UTTERANCES = 10

_SON = FEATURE_NAMES.index("son")
_VOI = FEATURE_NAMES.index("voi")


# --------------------------------------------------------------------------
# the partition
# --------------------------------------------------------------------------

def cell_of(phone: str):
    """
    Which of the four cells a phone belongs to, or None.

    Derived from the feature vector rather than from a list of English phones,
    so the same code partitions a language whose inventory we have never seen.
    A feature that PanPhon leaves unspecified, value 0, puts the phone in no
    cell rather than in a guessed one.
    """
    vector = features(phone)
    if vector is None:
        return None
    son, voi = vector[_SON], vector[_VOI]
    if son == 0 or voi == 0:
        return None
    if son > 0:
        return SON_VOI if voi > 0 else SON_UNVOI
    return OBS_VOI if voi > 0 else OBS_UNVOI


def assign_cells(phones):
    """Cells for a sequence of phones, plus a count of what was dropped by symbol."""
    cells, dropped = [], Counter()
    for phone in phones:
        cell = cell_of(phone)
        if cell is None:
            dropped[phone] += 1
        cells.append(cell)
    return np.asarray(cells, dtype=object), dict(dropped)


# A bonafide row carries no attack. Which symbol marks that varies by corpus,
# so every spelling seen in the registered corpora is accepted and anything else
# is treated as an attack.
BONAFIDE_MARKERS = {"-", "", "bonafide", "none", "nan", "real"}


def bonafide_label(data) -> int:
    """
    Which value of the label column means bonafide, read from the attack column.

    In the ASVspoof tables label 1 is SPOOF and bonafide rows carry attack '-'.
    Assuming the other convention inverts every direction in the output while
    leaving every magnitude correct, which is an error that survives review.
    This project has already shipped one inverted hypothesis, so the convention
    is read from the data and cross checked rather than assumed.
    """
    y = np.asarray(data["y"])
    attack = np.asarray(data.get("attack") if data.get("attack") is not None else [])
    if attack.size != y.size:
        raise ValueError(
            "cannot tell which label is bonafide without an attack column of the "
            "same length as the labels"
        )

    marked = np.isin(np.char.lower(attack.astype(str)), sorted(BONAFIDE_MARKERS))
    if not marked.any() or marked.all():
        raise ValueError(
            "the attack column marks either no rows or every row as bonafide, so "
            "the bonafide label cannot be read from it"
        )

    labels = np.unique(y[marked])
    if labels.size != 1:
        raise ValueError(
            f"attack column and label column disagree: rows with no attack carry "
            f"labels {labels.tolist()}"
        )
    label = int(labels[0])
    if bool((y[~marked] == label).any()):
        raise ValueError(
            f"attack column and label column disagree: attacked rows also carry "
            f"label {label}, which the bonafide rows carry"
        )
    return label


def drop_quietest(data, fraction: float):
    """
    Remove the quietest `fraction` of slices by high frequency energy.

    Why this exists. Zhang et al. 2023 report that the proportion of silence is
    lower in TTS speech than in genuine speech, and that removing silence at
    test time degrades detection sharply. So a near silent slice can separate
    the classes for a reason that has nothing to do with phase. This project
    inherits that exposure directly: cell D is carried entirely by /h/, the
    quietest segment in the English inventory.

    The threshold is a single global percentile rather than one per class,
    because a per class threshold would force the two sides to be balanced and
    hide exactly the asymmetry being checked for. The cost is that the drop can
    fall mostly on one class, which changes the comparison rather than cleaning
    it, so the per class counts are returned and flagged when they are lopsided.
    """
    if fraction <= 0:
        return data, {"n_dropped": 0, "n_dropped_bonafide": 0, "n_dropped_spoof": 0,
                      "threshold": float("-inf"), "lopsided": False}

    energy = np.asarray(data["energy"], dtype=np.float64)
    threshold = float(np.nanpercentile(energy, 100.0 * fraction))
    keep = ~(energy < threshold)

    is_bona = np.asarray(data["y"]) == data["bonafide_label"]
    dropped_bona = int((~keep & is_bona).sum())
    dropped_spoof = int((~keep & ~is_bona).sum())

    out = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
           for k, v in data.items()}

    total = dropped_bona + dropped_spoof
    # One class losing more than four fifths of the removed slices means the
    # threshold is separating the classes rather than removing silence.
    lopsided = bool(total > 0 and max(dropped_bona, dropped_spoof) / total > 0.8)
    return out, {
        "n_dropped": total,
        "n_dropped_bonafide": dropped_bona,
        "n_dropped_spoof": dropped_spoof,
        "threshold": threshold,
        "lopsided": lopsided,
    }


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

    Which is correct depends on the claim, not on taste, so both are reported.
    `docs/PREREGISTRATION.md` specifies clustering by source utterance
    throughout, so `utterance` stays the default and the registered result is
    the one it produces.
    """
    if cluster not in ("utterance", "system"):
        raise ValueError(f"cluster must be 'utterance' or 'system', got {cluster!r}")
    utterances = np.asarray(data["u"])
    if cluster == "utterance":
        return utterances
    attack = np.asarray(data["attack"])
    is_bona = np.asarray(data["y"]) == data["bonafide_label"]
    return np.where(is_bona, utterances, attack)


def subset_rows(data, keep):
    """
    Keep a boolean selection of rows, across every per row array in `data`.

    The list of arrays is derived rather than written out. An earlier version
    named seven of them by hand, `energy` was added to the loader afterwards and
    not to that list, and combining --drop-phone with --drop-quietest then
    compared a full length mask against short labels and crashed. A hand written
    list drifts from the loader. This cannot.
    """
    keep = np.asarray(keep, dtype=bool)
    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == keep.shape:
            out[key] = value[keep]
        else:
            out[key] = value
    return out


def contrast_label(contrast) -> str:
    """
    The three distinct states a contrast can be in, named so they cannot be
    confused on the page.

    `clear` requires the interval to exclude zero AND the corrected q to survive.
    The report used to print the verdict beside the interval alone, so a
    contrast with an interval of [+0.0011, +0.0400] and q = 0.06 printed as
    "spans zero", which contradicts the number next to it.
    """
    if contrast.get("clear"):
        return "clear of zero"
    low, high = contrast.get("ci_low"), contrast.get("ci_high")
    if low is not None and high is not None and not (low <= 0 <= high):
        return "not clear after correction"
    return "spans zero"


def folded_from_auc(auc: float) -> float:
    """
    Separation without regard to direction.

    A statistic running spoof-high separates the classes exactly as well as one
    running bonafide-high. Direction is reported separately rather than folded
    away, because direction is part of the mechanism.
    """
    return max(auc, 1.0 - auc)


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load(path, seg, stat=PHASE):
    """Read one segmentation out of a slice table into aligned arrays."""
    opener = gzip.open if str(path).endswith(".gz") else open
    y, utt, phone, value, attack, duration, energy = [], [], [], [], [], [], []
    with opener(path, "rt", newline="") as fh:
        for row in csv.DictReader(fh):
            if row["seg"] != seg:
                continue
            y.append(int(row["label"]))
            utt.append(row["utterance_id"])
            phone.append(row["phoneme"])
            value.append(float(row[stat]))
            attack.append(row.get("attack", ""))
            duration.append(float(row.get("duration_s") or "nan"))
            energy.append(float(row.get("hf_energy") or "nan"))
    cells, dropped = assign_cells(phone)
    data = {
        "y": np.asarray(y, dtype=np.int64),
        "u": np.asarray(utt),
        "phone": np.asarray(phone),
        "x": np.asarray(value, dtype=np.float64),
        "attack": np.asarray(attack),
        "duration": np.asarray(duration, dtype=np.float64),
        "energy": np.asarray(energy, dtype=np.float64),
        "cell": cells,
        "dropped": dropped,
    }
    data["bonafide_label"] = bonafide_label(data)
    return data


# --------------------------------------------------------------------------
# one cell
# --------------------------------------------------------------------------

def describe_cell(data, mask, min_slices=MIN_SLICES, min_utterances=MIN_UTTERANCES):
    """
    Counts, separation and direction for one cell, with a usability verdict.

    `data["is_bonafide"]` is 1 for bonafide and 0 for spoof, whatever the source
    table's own convention was, so everything below reads in one direction.
    """
    is_bona, u, x, phone = data["is_bonafide"], data["u"], data["x"], data["phone"]
    ys, us, xs = is_bona[mask], u[mask], x[mask]

    n_bona_utt = int(np.unique(us[ys == 1]).size)
    n_spoof_utt = int(np.unique(us[ys == 0]).size)
    n_bona = int((ys == 1).sum())
    n_spoof = int((ys == 0).sum())

    # A longer slice estimates its own statistic more precisely, so a cell of
    # longer slices could separate better for that reason alone. Reported rather
    # than assumed away.
    duration = np.asarray(data.get("duration", []), dtype=np.float64)
    in_cell = duration[mask] if duration.size == mask.size else np.array([np.nan])

    out = {
        "n_slices": int(ys.size),
        "n_slices_bonafide": n_bona,
        "n_slices_spoof": n_spoof,
        "duration_median": float(np.nanmedian(in_cell)) if in_cell.size else float("nan"),
        "duration_mean": float(np.nanmean(in_cell)) if in_cell.size else float("nan"),
        "n_utterances_bonafide": n_bona_utt,
        "n_utterances_spoof": n_spoof_utt,
        "phones": Counter(phone[mask]).most_common(),
        "auc": float("nan"),
        "auc_directional": float("nan"),
        "direction": None,
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "crosses_chance": True,
        "usable": False,
        "why_unusable": None,
    }

    reasons = []
    if n_bona < min_slices or n_spoof < min_slices:
        reasons.append(f"{n_bona} bonafide and {n_spoof} spoof slices, floor is {min_slices}")
    if n_bona_utt < min_utterances or n_spoof_utt < min_utterances:
        reasons.append(
            f"{n_bona_utt} bonafide and {n_spoof_utt} spoof utterances, "
            f"floor is {min_utterances}"
        )
    if reasons:
        out["why_unusable"] = "; ".join(reasons)
        return out

    directional = fast_auc(ys, xs)
    out["usable"] = True
    out["auc_directional"] = float(directional)
    out["auc"] = folded_from_auc(float(directional))
    out["direction"] = "bonafide-high" if directional >= 0.5 else "spoof-high"
    return out


# --------------------------------------------------------------------------
# the shared bootstrap
# --------------------------------------------------------------------------

def _interval(draws, alpha):
    finite = draws[np.isfinite(draws)]
    if finite.size < 2:
        return float("nan"), float("nan")
    lo, hi = np.percentile(finite, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def _bootstrap_p(draws):
    """
    Two sided bootstrap p for a difference, against the null of no difference.

    Add one to numerator and denominator, since a resampling test cannot licence
    a claim of p equals zero.
    """
    finite = draws[np.isfinite(draws)]
    if finite.size < 2:
        return 1.0
    below = int((finite <= 0).sum())
    above = int((finite >= 0).sum())
    n = finite.size
    return float(min(1.0, 2.0 * (min(below, above) + 1) / (n + 1)))


def analyse(data, n_boot=2000, seed=0, alpha=0.05,
            min_slices=MIN_SLICES, min_utterances=MIN_UTTERANCES,
            cluster="utterance"):
    """
    Separation per cell, the three ladder contrasts, and the crossed test.

    One sequence of utterance resamples serves every cell and every contrast, so
    differences between cells are differences between cells rather than between
    resampling accidents.
    """
    bona = data.get("bonafide_label")
    if bona is None:
        raise ValueError(
            "data has no bonafide_label. Call bonafide_label() on the table and "
            "carry the answer, rather than letting the direction be assumed"
        )
    y = (np.asarray(data["y"]) == bona).astype(np.int64)
    u, x = data["u"], data["x"]
    cell = data["cell"]
    data = {**data, "is_bonafide": y}

    masks = {c: (cell == c) for c in CELLS}
    cells = {c: describe_cell(data, masks[c], min_slices, min_utterances) for c in CELLS}

    usable = [c for c in CELLS if cells[c]["usable"]]
    rng = np.random.default_rng(seed)
    clusters = _cluster_index(y, cluster_groups(data, cluster))

    folded_draws = {c: np.full(n_boot, np.nan) for c in CELLS}
    direct_draws = {c: np.full(n_boot, np.nan) for c in CELLS}

    for b in range(n_boot):
        idx = _cluster_draw(rng, clusters)
        if idx.size == 0:
            continue
        for c in usable:
            rows = idx[masks[c][idx]]
            if rows.size == 0:
                continue
            ys = y[rows]
            if np.unique(ys).size < 2:
                continue
            auc = fast_auc(ys, x[rows])
            direct_draws[c][b] = auc
            folded_draws[c][b] = folded_from_auc(auc)

    for c in usable:
        # The interval on separation is taken from the directional draws, which
        # are unbiased. Folding would push the lower end up and make a cell that
        # separates nothing look as though it separated something.
        lo, hi = _interval(direct_draws[c], alpha)
        cells[c]["ci_low"], cells[c]["ci_high"] = lo, hi
        cells[c]["crosses_chance"] = bool(np.isnan(lo) or lo <= 0.5 <= hi)

    contrasts, p_values, keys = {}, [], []
    for name, (left, right) in CONTRASTS.items():
        if not (cells[left]["usable"] and cells[right]["usable"]):
            contrasts[name] = {
                "delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "p": 1.0, "q": 1.0, "clear": False,
                "why": f"{left} or {right} unusable",
            }
            continue
        draws = direct_draws[left] - direct_draws[right]
        lo, hi = _interval(draws, alpha)
        contrasts[name] = {
            "delta": cells[left]["auc_directional"] - cells[right]["auc_directional"],
            # Carried so the ladder can be read against the pre-registered
            # numbers, which are folded. No inference is run on it.
            "delta_folded": cells[left]["auc"] - cells[right]["auc"],
            "ci_low": lo, "ci_high": hi,
            "p": _bootstrap_p(draws), "q": 1.0, "clear": False, "why": None,
        }
        # A-C is A-B plus B-C exactly, so it carries no degree of freedom of its
        # own and must not enter the correction family as a third test.
        if name != "A-C":
            p_values.append(contrasts[name]["p"])
            keys.append(name)

    if p_values:
        corrected = benjamini_hochberg(np.asarray(p_values), alpha=alpha)
        for name, q in zip(keys, corrected["q_values"]):
            c = contrasts[name]
            c["q"] = float(q)
            c["clear"] = bool((c["ci_low"] > 0 or c["ci_high"] < 0) and q <= alpha)

    derived = contrasts.get("A-C")
    if derived is not None and not derived.get("why"):
        # Reported with its interval, but no q, because it is a sum of the two
        # tested steps rather than an independent test.
        derived["q"] = None
        derived["clear"] = bool(derived["ci_low"] > 0 or derived["ci_high"] < 0)

    return {
        "cells": cells,
        "contrasts": contrasts,
        "step_difference": _step_difference(cells, direct_draws, alpha),
        "main_effects": _main_effects(cells, direct_draws, alpha),
        "n_boot": n_boot,
        "alpha": alpha,
        "cluster": cluster,
    }


def _step_difference(cells, draws, alpha):
    """
    (A-B) minus (B-C). Whether the two ladder steps differ FROM EACH OTHER.

    Why the ladder needs this to say anything about a factor
        The ladder names sonority when the A-B step carries the effect and
        voicing when the B-C step does. But "this step cleared zero and that one
        did not" is a statement about power, not about effect. On the eval
        partition the two steps were +0.0282 and +0.0319, a ratio of 0.88, while
        their standard errors differed by a factor of 1.85. Under utterance
        clustering both cleared and the reading was graded; under system
        clustering only one cleared and the reading became voicing, on identical
        point estimates. Their difference is -0.0037 with p about 0.9.

        So a label taken from which step cleared is decided by how much data each
        cell happened to have. Naming a factor requires this quantity to be
        clear, which is a claim the data can refuse.

    Computed inside the shared draw, so its interval carries the correlation
    between the two steps. Not in the FDR family with them: it is one follow-up
    question asked only when the ladder is interpretable, and adding it to the
    family would move the q values of the very steps it is asked about.
    """
    if not all(cells[c]["usable"] for c in LADDER):
        return None

    a, b, c = draws[SON_VOI], draws[OBS_VOI], draws[OBS_UNVOI]
    draw = (a - b) - (b - c)
    seen = {k: cells[k]["auc_directional"] for k in LADDER}
    observed = ((seen[SON_VOI] - seen[OBS_VOI]) - (seen[OBS_VOI] - seen[OBS_UNVOI]))

    lo, hi = _interval(draw, alpha)
    return {
        "delta": float(observed), "ci_low": lo, "ci_high": hi,
        "p": _bootstrap_p(draw),
        "clear": bool(lo > 0 or hi < 0),
    }


def _main_effects(cells, draws, alpha):
    """
    The crossed test, available only when all four cells are usable.

    son          the sonority main effect, (A+D)/2 minus (B+C)/2
    voi          the voicing main effect,  (A+B)/2 minus (C+D)/2
    interaction  the sonority effect among voiced segments minus the same among
                 voiceless ones, (A-B) minus (D-C)

    A main effect is the honest way to ask which factor the artifact follows,
    because each one averages over the other. The ladder cannot do that.
    """
    if not all(cells[c]["usable"] for c in CELLS):
        return None

    a, b, c, d = (draws[SON_VOI], draws[OBS_VOI], draws[OBS_UNVOI], draws[SON_UNVOI])
    observed = {k: cells[k]["auc_directional"] for k in CELLS}

    definitions = {
        "son": ((a + d) / 2 - (b + c) / 2,
                (observed[SON_VOI] + observed[SON_UNVOI]) / 2
                - (observed[OBS_VOI] + observed[OBS_UNVOI]) / 2),
        "voi": ((a + b) / 2 - (c + d) / 2,
                (observed[SON_VOI] + observed[OBS_VOI]) / 2
                - (observed[OBS_UNVOI] + observed[SON_UNVOI]) / 2),
        "interaction": ((a - b) - (d - c),
                        (observed[SON_VOI] - observed[OBS_VOI])
                        - (observed[SON_UNVOI] - observed[OBS_UNVOI])),
    }

    out, p_values, keys = {}, [], []
    for name, (draw, point) in definitions.items():
        lo, hi = _interval(draw, alpha)
        out[name] = {"delta": float(point), "ci_low": lo, "ci_high": hi,
                     "p": _bootstrap_p(draw), "q": 1.0, "clear": False}
        p_values.append(out[name]["p"])
        keys.append(name)

    corrected = benjamini_hochberg(np.asarray(p_values), alpha=alpha)
    for name, q in zip(keys, corrected["q_values"]):
        out[name]["q"] = float(q)
        out[name]["clear"] = bool(
            (out[name]["ci_low"] > 0 or out[name]["ci_high"] < 0) and q <= alpha
        )
    return out


# --------------------------------------------------------------------------
# the verdict
# --------------------------------------------------------------------------

def verdict(result):
    """
    Apply the decision rule fixed before the run.

    Undecided and rejected are different outcomes, and this returns undecided
    whenever a cell is absent or underpowered rather than reading a missing cell
    as evidence of no effect.
    """
    cells = result["cells"]

    unusable = [c for c in LADDER if not cells.get(c, {}).get("usable", False)]
    if unusable:
        why = "; ".join(
            f"{c} {cells.get(c, {}).get('why_unusable') or 'absent from the table'}"
            for c in unusable
        )
        return {
            "reading": "undecided",
            "surprise": False,
            "why": f"underpowered or absent cells, so no contrast is interpretable. {why}",
        }

    ab, bc, ac = (result["contrasts"][k] for k in ("A-B", "B-C", "A-C"))

    def positive(c):
        return bool(c["clear"] and c["ci_low"] > 0)

    def negative(c):
        return bool(c["clear"] and c["ci_high"] < 0)

    if negative(ab):
        return {
            "reading": "voicing-not-sonority",
            "surprise": True,
            "why": ("A-B is clear and negative. Voiced obstruents separate MORE than "
                    "sonorants, which the sonorant framing does not survive."),
        }
    if negative(bc):
        return {
            "reading": "reversed-voicing",
            "surprise": True,
            "why": ("B-C is clear and negative. Voiceless obstruents separate more than "
                    "voiced ones, which neither reading predicts."),
        }

    # Naming a factor requires the two steps to DIFFER, not merely one of them
    # to fail to clear. See _step_difference: one step clearing while the other
    # does not is a statement about power, and this rule used to read it as a
    # statement about mechanism.
    steps = result.get("step_difference")

    def steps_differ(sign):
        if not steps or not steps.get("clear"):
            return False
        return steps["ci_low"] > 0 if sign > 0 else steps["ci_high"] < 0

    def undecidable_label(clear_step, other_step):
        p = steps.get("p") if steps else None
        measured = "" if p is None else f" Their difference has p={p:.4f}."
        return {
            "reading": "steps-indistinguishable",
            "surprise": False,
            "why": (f"{clear_step} clear, {other_step} not, but the two steps are "
                    f"not distinguishable from each other.{measured} Which step "
                    f"clears is then decided by power rather than by effect size, "
                    f"so neither sonority nor voicing is named."),
        }

    if positive(ab) and not positive(bc):
        if steps_differ(+1):
            return {
                "reading": "sonority",
                "surprise": False,
                "why": ("A-B clear, B-C not, and the A-B step is clearly the larger "
                        "of the two. Voiced obstruents pattern with voiceless ones, "
                        "so the split follows sonority rather than voicing."),
            }
        return undecidable_label("A-B", "B-C")
    if positive(bc) and not positive(ab):
        if steps_differ(-1):
            return {
                "reading": "voicing",
                "surprise": False,
                "why": ("B-C clear, A-B not, and the B-C step is clearly the larger "
                        "of the two. Voiced obstruents pattern with sonorants, so "
                        "the split follows voicing rather than sonority."),
            }
        return undecidable_label("B-C", "A-B")
    if positive(ab) and positive(bc):
        return {
            "reading": "graded",
            "surprise": False,
            "why": ("A-B and B-C both clear. Neither label is right. Separation tracks "
                    "how much of the slice carries periodic excitation."),
        }
    if positive(ac):
        return {
            "reading": "graded-weak",
            "surprise": False,
            "why": ("A-C is clear but neither single step is. There is an ordering and "
                    "the data cannot say where the step falls."),
        }
    return {
        "reading": "indistinguishable",
        "surprise": False,
        "why": ("No contrast clears zero. On this corpus the three cells cannot be "
                "told apart, which refutes nothing."),
    }


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------

def print_report(result, data, title):
    cells = result["cells"]
    print(f"\n{title}")
    print("=" * len(title))

    print("\nSeparation of mgd_hf_std within each cell, bonafide against spoof.")
    print(f"{'cell':<14} {'AUC':>6} {'95% CI':>18} {'dir':>14} {'slices':>8} "
          f"{'utt b/s':>12} {'median s':>9}")
    for c in CELLS:
        d = cells[c]
        if not d["usable"]:
            print(f"{c:<14} {'unusable':>6}   {d['why_unusable']}")
            continue
        ci = f"[{d['ci_low']:.3f}, {d['ci_high']:.3f}]"
        chance = " (crosses chance)" if d["crosses_chance"] else ""
        print(f"{c:<14} {d['auc_directional']:>6.3f} {ci:>18} "
              f"{d['direction'] or '-':>14} {d['n_slices']:>8} "
              f"{d['n_utterances_bonafide']:>5}/{d['n_utterances_spoof']:<6} "
              f"{d['duration_median']:>9.4f}{chance}")

    print("\nPhones carrying each cell, most frequent first, top six.")
    for c in CELLS:
        top = cells[c]["phones"][:6]
        total = len(cells[c]["phones"])
        shown = "  ".join(f"{p} {n}" for p, n in top)
        print(f"  {c:<14} {total:>3} types   {shown}")

    print("\nLadder contrasts on directional AUC, paired cluster bootstrap. FDR across "
          "the two\nindependent steps; A-C is their sum and is reported without a q "
          "of its own.")
    print(f"{'contrast':<10} {'difference':>11} {'95% CI':>20} {'q':>8} {'folded':>8}   "
          f"verdict")
    for name in ("A-B", "B-C", "A-C"):
        c = result["contrasts"][name]
        if c.get("why"):
            print(f"{name:<10} {'-':>11}   {c['why']}")
            continue
        ci = f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]"
        q = "  derived" if c["q"] is None else f"{c['q']:>8.4f}"
        print(f"{name:<10} {c['delta']:>+11.4f} {ci:>20} {q} "
              f"{c['delta_folded']:>+8.4f}   {contrast_label(c)}")

    # Printed next to the steps, because the reading depends on it and a reader
    # who sees only that one step cleared will otherwise draw the conclusion
    # this script used to draw.
    step = result.get("step_difference")
    if step is None:
        print("\n(A-B)-(B-C)  unavailable, the ladder has an unusable cell.")
    else:
        ci = f"[{step['ci_low']:+.4f}, {step['ci_high']:+.4f}]"
        settled = "the steps differ" if step["clear"] else "NOT distinguishable"
        print(f"\n{'(A-B)-(B-C)':<10} {step['delta']:>+11.4f} {ci:>20} "
              f"p={step['p']:.4f}   {settled}")
        print("  Whether the two steps differ from each other. A factor is named "
              "only if they do,\n  since otherwise which step clears is decided by "
              "power rather than by effect size.")

    main = result["main_effects"]
    if main is None:
        print("\nCrossed test unavailable, one of the four cells is unusable.")
    else:
        print("\nCrossed test, secondary. Depends on glottals being [+son], which is a "
              "convention.")
        print(f"{'effect':<12} {'size':>9} {'95% CI':>20} {'q':>8}   verdict")
        for name in ("son", "voi", "interaction"):
            m = main[name]
            ci = f"[{m['ci_low']:+.4f}, {m['ci_high']:+.4f}]"
            print(f"{name:<12} {m['delta']:>+9.4f} {ci:>20} {m['q']:>8.4f}   "
                  f"{contrast_label(m)}")

    v = verdict(result)
    if result.get("cluster") == "system":
        print("\nIntervals below resample SYNTHESIS SYSTEMS on the spoof side, 13 of "
              "them, not\nutterances. They generalise to unseen systems and are much "
              "wider for that reason.")
    print(f"\nREADING: {v['reading'].upper()}")
    print(f"  {v['why']}")
    if v["surprise"]:
        print("  This is the outcome recorded in advance as a surprise. Treat it as a "
              "finding about the framing, not as noise.")

    if data.get("dropped"):
        total = sum(data["dropped"].values())
        top = ", ".join(f"{p} {n}" for p, n in
                        Counter(data["dropped"]).most_common(5))
        print(f"\nDropped {total} slices whose phone has no son or voi value: {top}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--table", required=True, help="slice table, .csv or .csv.gz")
    ap.add_argument("--seg", default="run", choices=["run", "window"])
    ap.add_argument("--stat", default=PHASE)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--glottals", default="separate", choices=["separate", "obstruent"],
                    help="fold [+son,-voi] into [-son,-voi], the sensitivity run")
    ap.add_argument("--cluster", default="utterance", choices=["utterance", "system"],
                    help="what the bootstrap resamples on the spoof side. "
                         "'system' clusters by attack, making the interval a "
                         "statement about unseen synthesis systems rather than "
                         "about more utterances from these ones")
    ap.add_argument("--drop-quietest", type=float, default=0.0,
                    help="remove this fraction of the quietest slices by hf_energy, "
                         "the silence sensitivity run. 0.2 is a reasonable probe")
    ap.add_argument("--drop-phone", action="append", default=[],
                    help="remove one phone by symbol, repeatable. For the silence check")
    ap.add_argument("--both-segmentations", action="store_true",
                    help="run seg=run and seg=window, since a finding on one only is a "
                         "segmentation artifact")
    args = ap.parse_args()

    segs = ["run", "window"] if args.both_segmentations else [args.seg]
    for seg in segs:
        data = load(args.table, seg, args.stat)

        if args.drop_phone:
            keep = ~np.isin(data["phone"], args.drop_phone)
            removed = int((~keep).sum())
            data = subset_rows(data, keep)
            print(f"Dropped {removed} slices for phones {args.drop_phone}")

        if args.drop_quietest:
            data, report = drop_quietest(data, args.drop_quietest)
            print(f"Silence sensitivity: dropped {report['n_dropped']} slices below "
                  f"hf_energy {report['threshold']:.3f}, "
                  f"{report['n_dropped_bonafide']} bonafide and "
                  f"{report['n_dropped_spoof']} spoof")
            if report["lopsided"]:
                print("  WARNING: the drop fell almost entirely on one class, so this "
                      "run has changed\n  the comparison rather than cleaned it. Read "
                      "it as a confound check, not a robustness check.")

        if args.glottals == "obstruent":
            data["cell"] = np.where(data["cell"] == SON_UNVOI, OBS_UNVOI, data["cell"])

        result = analyse(data, n_boot=args.n_boot, seed=args.seed,
                         cluster=args.cluster)
        title = (f"Sonority against voicing, seg={seg}, {args.stat}, "
                 f"glottals={args.glottals}, clustered by {args.cluster}")
        print_report(result, data, title)

    print("\nExploratory. This runs on the same eval data that confirmed H1 and "
          "cannot license a new claim.\nIt says which reading the existing data "
          "supports, and what an H2 language would have to break.")


if __name__ == "__main__":
    main()
