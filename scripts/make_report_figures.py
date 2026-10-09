"""
Draw the figures of report/phonophase.tex.

Figure 1 is computed here from results/eval_slices.csv.gz, point estimates
only. The others are intervals that take minutes to bootstrap, so they are
drawn from results/report_numbers.csv, whose `source` column names the document
that gives the command behind each row.

Usage
    pip install -e ".[report]"
    python scripts/make_report_figures.py
"""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import analyse_slices as A  # noqa: E402

OUT = os.path.join(ROOT, "report", "figures")
NUMBERS = os.path.join(ROOT, "results", "report_numbers.csv")

# Two categorical hues, checked for colour vision deficiency separation, and
# neutral inks. Every series is also named in a legend and drawn with its own
# marker, so nothing is carried by colour alone.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e4e3df"
MARKERS = ("o", "s")
WIDTH = 3.3  # inches, one column

plt.rcParams.update({
    "font.size": 7.5, "font.family": "serif", "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK, "ytick.labelcolor": INK, "legend.frameon": False,
    "pdf.fonttype": 42,
})


def read_numbers(figure):
    """Rows of one figure, in file order, as {row: {series: (value, low, high)}}."""
    rows = defaultdict(dict)
    with open(NUMBERS, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["figure"] == figure:
                rows[r["row"]][r["series"]] = tuple(
                    float(r[k]) for k in ("value", "low", "high"))
    return rows


def _axes(n_rows, per_row=0.24, extra=0.75):
    fig, ax = plt.subplots(figsize=(WIDTH, n_rows * per_row + extra))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    return fig, ax


def forest(rows, series, colours, xlabel, path, legend=True):
    """Intervals by row, one marker per series, with a line at zero."""
    labels = list(rows)
    fig, ax = _axes(len(labels))
    named = set()
    for i, label in enumerate(labels):
        present = [k for k, name in enumerate(series) if name in rows[label]]
        for j, k in enumerate(present):
            # Rows holding both series are drawn side by side; a row holding
            # one sits on its line.
            y = len(labels) - 1 - i - (j - (len(present) - 1) / 2) * 0.32
            value, low, high = rows[label][series[k]]
            ax.plot([low, high], [y, y], color=colours[k], linewidth=1.3,
                    solid_capstyle="round")
            ax.plot(value, y, MARKERS[k], color=colours[k], markersize=4.2,
                    markeredgecolor="white", markeredgewidth=0.6,
                    label=None if series[k] in named else series[k])
            named.add(series[k])
    ax.axvline(0, color=INK, linewidth=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels[::-1])
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.set_xlabel(xlabel)
    if legend and len(series) > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(series),
                  handletextpad=0.3, columnspacing=1.2)
    _save(fig, path)


def _save(fig, path):
    """PDF for the report, and a PNG beside it for the README."""
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path[:-4] + ".png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def separation_by_category(path):
    """Folded separation of each statistic in each category, eval, seg=run."""
    data = A.load(os.path.join(ROOT, "results", "eval_slices.csv.gz"), "run")
    order = A.SONORANTS + A.OBSTRUENTS
    names = {"NASALS": "nasals", "APPROXIMANTS": "approximants",
             "VOWELS_STEADY": "steady vowels", "FRICATIVES": "fricatives",
             "PLOSIVES": "plosives", "AFFRICATES": "affricates"}
    fig, ax = _axes(len(order))
    for i, cat in enumerate(order):
        y = len(order) - 1 - i
        phase = A.folded(data[cat]["y"], data[cat][A.PHASE])
        magnitude = A.folded(data[cat]["y"], data[cat][A.MAGNITUDE])
        ax.plot([magnitude, phase], [y, y], color=MUTED, linewidth=0.9, zorder=1)
        ax.plot(phase, y, MARKERS[0], color=BLUE, markersize=4.6,
                markeredgecolor="white", markeredgewidth=0.6,
                label="group delay (MGD-HF)" if i == 0 else None)
        ax.plot(magnitude, y, MARKERS[1], color=ORANGE, markersize=4.4,
                markeredgecolor="white", markeredgewidth=0.6,
                label="cepstral (MFCC spread)" if i == 0 else None)
    ax.axhline(2.5, color=GRID, linewidth=0.8)
    ax.text(0.602, 5.3, "sonorants", ha="right", va="center", color=MUTED)
    ax.text(0.602, 2.2, "obstruents", ha="right", va="center", color=MUTED)
    ax.axvline(0.5, color=INK, linewidth=0.7)
    ax.set_xlim(0.497, 0.605)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[c] for c in order][::-1])
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.set_xlabel("folded separation of bonafide from spoof (0.5 is chance)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              handletextpad=0.3, columnspacing=1.2)
    _save(fig, path)


def main():
    os.makedirs(OUT, exist_ok=True)
    separation_by_category(os.path.join(OUT, "separation.pdf"))
    forest(read_numbers("clustering"), ["asymmetry", "level"], [BLUE, ORANGE],
           "estimate and 95% interval", os.path.join(OUT, "clustering.pdf"))
    forest(read_numbers("pair"), ["asymmetry"], [BLUE],
           "asymmetry and 95% interval", os.path.join(OUT, "pair.pdf"))
    forest(read_numbers("resampler"), ["synthesis", "no synthesis"], [BLUE, ORANGE],
           "asymmetry and 95% interval", os.path.join(OUT, "resampler.pdf"))
    forest(read_numbers("split"), ["unmodified", "low-passed"], [BLUE, ORANGE],
           "asymmetry and 95% interval", os.path.join(OUT, "split.pdf"))
    print(f"wrote 5 figures to {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
