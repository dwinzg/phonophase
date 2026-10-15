"""
Tests for results/report_numbers.csv, the figures' data.

The figures in report/ are drawn from this file, so a row that points at a
document which is not there, or an estimate outside its own interval, would be
printed in a figure without anything complaining.
"""

import csv
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _rows():
    with open(os.path.join(ROOT, "results", "report_numbers.csv"), newline="") as fh:
        return list(csv.DictReader(fh))


def test_every_row_names_a_document_that_exists():
    for r in _rows():
        assert os.path.exists(os.path.join(ROOT, r["source"])), r


def test_every_estimate_lies_inside_its_interval():
    for r in _rows():
        low, value, high = float(r["low"]), float(r["value"]), float(r["high"])
        assert low <= value <= high, r


def test_the_headline_row_is_the_registered_result():
    registered = [r for r in _rows() if r["figure"] == "clustering"
                  and r["row"].startswith("utterance") and r["series"] == "asymmetry"]
    assert [(r["value"], r["low"], r["high"]) for r in registered] == [
        ("0.063", "0.045", "0.080")]


# Columns behind each row of the pair figure, as docs/ROBUSTNESS.md names them.
PAIRS = {
    "MFCC spread (registered)": ("mgd_hf_std", "mfcc_frame_spread"),
    "MFCC std": ("mgd_hf_std", "mfcc_std"),
    "MFCC IQR": ("mgd_hf_std", "mfcc_iqr"),
    "HF energy": ("mgd_hf_std", "hf_energy"),
    "spectral tilt": ("mgd_hf_std", "hf_lf_tilt"),
    "phase: MGD IQR": ("mgd_iqr", "mfcc_frame_spread"),
    "phase: MGD std": ("mgd_std", "mfcc_frame_spread"),
    "phase: MGD spread": ("mgd_frame_spread", "mfcc_frame_spread"),
}
SONORANTS = ("NASALS", "APPROXIMANTS", "VOWELS_STEADY")
OBSTRUENTS = ("FRICATIVES", "PLOSIVES", "AFFRICATES")


def _eval_columns():
    """seg=run rows of the eval table as arrays, read with nothing from the package."""
    import gzip

    import numpy as np

    wanted = {c for pair in PAIRS.values() for c in pair}
    cols = {k: [] for k in ("utterance_id", "attack", "category", *wanted)}
    with gzip.open(os.path.join(ROOT, "results", "eval_slices.csv.gz"), "rt") as fh:
        for r in csv.DictReader(fh):
            if r["seg"] == "run":
                for k in cols:
                    cols[k].append(r[k])
    out = {k: np.asarray(v) for k, v in cols.items()}
    for k in wanted:
        out[k] = out[k].astype(float)
    out["spoof"] = out["attack"] != "-"
    return out


def _folded_auc(spoof, x):
    from scipy.stats import rankdata

    ranks = rankdata(x)
    n1 = spoof.sum()
    auc = (ranks[spoof].sum() - n1 * (n1 + 1) / 2) / (n1 * (~spoof).sum())
    return max(auc, 1 - auc)


def _usable(t, m):
    """scripts/attack_analysis.py's rule: 30 slices and 10 utterances per class."""
    for cls in (t["spoof"], ~t["spoof"]):
        rows = m & cls
        if rows.sum() < 30 or len(set(t["utterance_id"][rows])) < 10:
            return False
    return True


def _point(t, keep, phase, magnitude, check_usable=False):
    def mean_advantage(categories):
        adv = []
        for c in categories:
            m = keep & (t["category"] == c)
            if check_usable and not _usable(t, m):
                continue
            adv.append(_folded_auc(t["spoof"][m], t[phase][m])
                       - _folded_auc(t["spoof"][m], t[magnitude][m]))
        return sum(adv) / len(adv)
    level = mean_advantage(SONORANTS)
    return level, level - mean_advantage(OBSTRUENTS)


def test_the_figure_point_estimates_regenerate_from_the_shipped_table():
    """
    The intervals need the bootstrap, but every point estimate in the
    clustering and pair figures is a plain function of the eval table, so a
    typed in value that drifted from it is caught here, to the printed digit.
    """
    import numpy as np

    t = _eval_columns()
    every = np.ones(t["spoof"].size, dtype=bool)
    computed = {}
    for row, (phase, magnitude) in PAIRS.items():
        computed[("pair", row, "asymmetry")] = _point(t, every, phase, magnitude)[1]
    level, asym = _point(t, every, "mgd_hf_std", "mfcc_frame_spread")
    # Per system, a category too thin in one attack is skipped, as the script does.
    per_system = [_point(t, ~t["spoof"] | (t["attack"] == a), "mgd_hf_std",
                         "mfcc_frame_spread", check_usable=True)
                  for a in sorted(set(t["attack"][t["spoof"]]))]
    for r in _rows():
        if r["figure"] != "clustering":
            continue
        if r["row"].startswith("mean of 13"):
            idx = 0 if r["series"] == "level" else 1
            computed[("clustering", r["row"], r["series"])] = float(
                np.mean([p[idx] for p in per_system]))
        else:
            computed[("clustering", r["row"], r["series"])] = (
                level if r["series"] == "level" else asym)

    checked = 0
    for r in _rows():
        key = (r["figure"], r["row"], r["series"])
        if key in computed:
            assert f"{computed[key]:.3f}" == f"{float(r['value']):.3f}", (key, computed[key])
            checked += 1
    assert checked == 18, checked


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_report_numbers")
