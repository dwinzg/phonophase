"""
Can a band edge or a resampling difference between the classes account for H1?

The threat
    On real speech with no synthesis anywhere, a 16 to 22.05 to 16 kHz round
    trip applied to one arm separates the arms on mgd_hf_std and leaves
    mfcc_frame_spread at chance. An anti-alias filter is a group delay
    phenomenon at the band edge, and mgd_hf_std is Modified Group Delay
    dispersion over the upper half of the spectrum, so a difference in how
    bonafide and spoofed audio were band limited would look like H1's phase
    advantage. Whether it would also look like H1's ASYMMETRY across manner
    classes is what this script measures. See docs/BAND-EDGE.md.

One alignment, many versions of the audio
    `align` runs the alignment model once and caches every phone run. `slice`
    then cuts any treated version of the audio at exactly those boundaries, so
    an intervention changes the audio and nothing else. The slicing arithmetic
    and the statistics are the ones scripts/slice_and_probe.py uses, imported
    rather than copied, and `slice --treatment none` reproduces its table.

Treatments, applied to the whole utterance before slicing
    none                    the file as it is
    rt:<resampler>:<rate>   16 kHz to <rate> and back, e.g. rt:soxr_hq:22050
    lp:<cutoff_hz>          one linear phase FIR low pass, zero net delay
    pcm16                   round to 16 bit, as a file written to disk is
    a > b                   a, then b, e.g. lp:6900>pcm16

Usage
    python scripts/band_edge.py align --manifest M.csv --out align.csv
    python scripts/band_edge.py slice --manifest M.csv --alignment align.csv \\
        --treatment rt:soxr_hq:22050 --out rt.csv.gz
    python scripts/band_edge.py describe --manifest M.csv --out ltas.csv.gz
    python scripts/band_edge.py m1 --descriptors ltas.csv.gz
    python scripts/band_edge.py m2 --table base.csv.gz --descriptors ltas.csv.gz
    python scripts/band_edge.py r1 --base base.csv.gz --treated rt.csv.gz
    python scripts/band_edge.py r2 --table rt_all.csv.gz
    python scripts/band_edge.py change --before base.csv.gz --after rt_all.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from math import gcd

import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

import analyse_slices as A  # noqa: E402
import slice_and_probe as S  # noqa: E402

from phonophase.features import modified_group_delay  # noqa: E402
from phonophase.probe import HOP, N_FFT, WIN, slice_statistics  # noqa: E402
from phonophase.stats import (  # noqa: E402
    bootstrap_auc_ci,
    percentile_interval,
    shared_cluster_draws,
)

SR = S.SR
ALIGN_FIELDS = ["utterance_id", "n_samples", "phoneme", "first", "last"]
BAND_COLUMN = "mgd_band_std"
DESCRIPTORS = ["bw30_hz", "bw40_hz", "edge_ratio_db", "edge_slope_db_per_khz",
               "edge_slope_hz"]


def open_out(path, mode="wt"):
    return (gzip.open if path.endswith(".gz") else open)(path, mode, newline="")


def read_csv(path):
    with A.open_table(path) as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------
# treatments
# --------------------------------------------------------------------------

def _soxr(quality):
    def resample(y, sr_in, sr_out):
        import soxr
        return soxr.resample(y, sr_in, sr_out, quality=quality)
    return resample


def _poly(y, sr_in, sr_out):
    from scipy.signal import resample_poly
    g = gcd(sr_in, sr_out)
    return resample_poly(y, sr_out // g, sr_in // g)


# librosa.resample's default, soxr_hq, is the first entry: the two return
# identical samples, so it has no entry of its own.
RESAMPLERS = {
    "soxr_hq": _soxr("HQ"), "soxr_vhq": _soxr("VHQ"), "soxr_mq": _soxr("MQ"),
    "soxr_lq": _soxr("LQ"), "poly": _poly,
}


def round_trip(y, rate, resampler):
    """
    16 kHz to `rate` and back through one resampler, at the original length.

    The length is forced back because the cached alignment indexes samples. A
    resampler that returned one sample more or fewer would otherwise move the
    last slice boundary.
    """
    f = RESAMPLERS[resampler]
    back = f(f(y, SR, rate), rate, SR)
    if back.size >= y.size:
        return back[:y.size]
    return np.pad(back, (0, y.size - back.size))


LOW_PASS_TAPS = 511


def low_pass(y, cutoff_hz, taps=LOW_PASS_TAPS):
    """
    One linear phase FIR low pass with its delay removed.

    Symmetric with an odd tap count, so the group delay is a whole number of
    samples, and "same" convolution centres it out. The output is therefore
    time aligned with the input sample for sample.
    """
    from scipy.signal import firwin
    h = firwin(taps, cutoff_hz, fs=SR, window=("kaiser", 8.0))
    return np.convolve(y, h, mode="same")


def pcm16(y):
    """Round to the 16 bit grid the corpus is stored on."""
    return np.clip(np.round(y * 32768.0), -32768, 32767) / 32768.0


def treat(y, spec):
    """Apply a treatment named as in the module docstring."""
    for step in spec.split(">"):
        kind, _, rest = step.partition(":")
        if step == "none":
            pass
        elif step == "pcm16":
            y = pcm16(y)
        elif kind == "rt":
            resampler, _, rate = rest.partition(":")
            y = round_trip(y, int(rate), resampler)
        elif kind == "lp":
            y = low_pass(y, float(rest))
        else:
            raise ValueError(f"unknown treatment {step!r} in {spec!r}")
    return y


# --------------------------------------------------------------------------
# alignment cache and slicing
# --------------------------------------------------------------------------

def read_audio(path):
    """The read slice_and_probe.run performs, for 16 kHz input."""
    speech, sr = sf.read(path)
    if speech.ndim > 1:
        speech = speech[:, 0]
    if sr != SR:
        raise ValueError(f"{path} is {sr} Hz, this script expects {SR}")
    return speech.astype(np.float64)


def cmd_align(args):
    import torch
    items = S.read_manifest(args.manifest)
    done = S.already_done(args.out)
    items = [r for r in items if r[1] not in done]
    print(f"{len(done)} utterances cached, {len(items)} to align")
    if not items:
        return
    chars_for = S.load_aligner("cuda" if torch.cuda.is_available() else "cpu")

    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ALIGN_FIELDS)
        if new_file:
            writer.writeheader()
        for n, item in enumerate(items, 1):
            speech = read_audio(item[0])
            runs = [r for r in S.group_runs(chars_for(speech))
                    if r[0] in S.CHAR_TO_CATEGORY]
            # A row with no phone marks an utterance as done even when it
            # yielded nothing, so a resume does not align it again.
            for char, first, last in runs or [("", "", "")]:
                writer.writerow({"utterance_id": item[1], "n_samples": speech.size,
                                 "phoneme": char, "first": first, "last": last})
            if n % 50 == 0:
                fh.flush()
                os.fsync(fh.fileno())
                print(f"  {n}/{len(items)}", flush=True)


def read_alignment(path):
    """{utterance_id: [(phoneme, first, last), ...]} from the cache."""
    runs = {}
    for r in read_csv(path):
        entry = runs.setdefault(r["utterance_id"], [])
        if r["phoneme"]:
            entry.append((r["phoneme"], int(r["first"]), int(r["last"])))
    return runs


def band_rows(band_hz):
    """Rows of the MGD plane whose centre frequency lies in [lo, hi)."""
    freqs = np.arange(N_FFT // 2 + 1) * SR / N_FFT
    return np.flatnonzero((freqs >= band_hz[0]) & (freqs < band_hz[1]))


def slice_utterance(item, runs, treatment, segs, band_hz=None):
    """
    Rows for one utterance, shaped like slice_and_probe.FIELDS.

    The treatment sees the whole utterance and the cached runs decide the cuts,
    so two treatments of one utterance yield slices over identical samples.
    `band_hz` adds MGD dispersion over a sub band as an extra column. That is a
    different statistic from mgd_hf_std and is never substituted for it
    silently.
    """
    path, utt, attack, language, corpus, label_str, speaker = item
    speech = treat(read_audio(path), treatment)
    rows = []
    for category, char, seg, first, last, a, b in S.iter_slices(speech.size, runs, segs):
        stats = slice_statistics(speech[a:b])
        if stats is None:
            continue
        row = {"utterance_id": utt, "speaker": speaker,
               "label": 0 if label_str == "bonafide" else 1, "attack": attack,
               "language": language, "corpus": corpus, "category": category,
               "phoneme": char, "seg": seg, "start_frame": first,
               "n_frames": last - first, "duration_s": round((b - a) / SR, 4),
               **stats}
        if band_hz:
            mgd = modified_group_delay(speech[a:b], n_fft=N_FFT, hop_length=HOP,
                                       win_length=WIN)
            row[BAND_COLUMN] = float(np.std(mgd[band_rows(band_hz)]))
        rows.append(row)
    return rows


def _slice_job(job):
    return slice_utterance(*job)


def cmd_slice(args):
    alignment = read_alignment(args.alignment)
    items = [r for r in S.read_manifest(args.manifest)
             if args.only in ("all", r[5])]
    missing = [r[1] for r in items if r[1] not in alignment]
    if missing:
        raise SystemExit(f"{len(missing)} utterances have no cached alignment, "
                         f"first {missing[0]}. Run align to completion first.")
    segs = args.segmentations.split(",")
    jobs = [(r, alignment[r[1]], args.treatment, segs, args.band) for r in items]
    fields = S.FIELDS + ([BAND_COLUMN] if args.band else [])
    written = 0
    try:
        with open_out(args.out) as fh, ProcessPoolExecutor(args.jobs) as pool:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for rows in pool.map(_slice_job, jobs, chunksize=16):
                writer.writerows(rows)
                written += len(rows)
    except BaseException:
        # A table cut short by a missing file reads as a complete one later.
        os.remove(args.out)
        raise
    print(f"{args.treatment}: {written} rows from {len(items)} utterances to {args.out}")


# --------------------------------------------------------------------------
# M1, where the band edges sit
# --------------------------------------------------------------------------

def ltas_db(y, nperseg=1024):
    """Long term average spectrum, (frequencies in Hz, power in dB)."""
    from scipy.signal import welch
    freqs, power = welch(y, fs=SR, nperseg=nperseg)
    return freqs, 10.0 * np.log10(power + 1e-20)


def describe(y):
    """
    Magnitude domain descriptors of one file's band edge.

    bw30_hz, bw40_hz
        Highest frequency whose level is within 30 or 40 dB of the mean level
        over 1 to 4 kHz. Where the spectrum stops.
    edge_ratio_db
        Mean power over 7 to 8 kHz against 4 to 7 kHz. How empty the top of
        the band is.
    edge_slope_db_per_khz, edge_slope_hz
        The steepest fall of the spectrum between 4 and 8 kHz, measured across
        500 Hz, and where it is. Speech rolls off gradually and an anti-alias
        filter falls off a cliff, so this responds to the filter itself and not
        to how much high frequency energy the talker happened to produce, and
        it does so wherever the cutoff sits. The two descriptors above only
        respond to a cutoff inside the band they were told to look at.
    """
    freqs, db = ltas_db(y)
    power = 10.0 ** (db / 10.0)

    def level(lo, hi):
        return 10.0 * np.log10(power[(freqs >= lo) & (freqs < hi)].mean() + 1e-20)

    ref = level(1000, 4000)
    out = {}
    for drop in (30, 40):
        above = np.flatnonzero(db >= ref - drop)
        out[f"bw{drop}_hz"] = float(freqs[above[-1]]) if above.size else 0.0
    out["edge_ratio_db"] = float(level(7000, 8000) - level(4000, 7000))

    # Edge padded so the smoothing does not invent a fall at Nyquist, which is
    # where the edge being looked for sits.
    step = int(round(500 / (freqs[1] - freqs[0])))
    smooth = np.convolve(np.pad(db, 4, mode="edge"), np.ones(8) / 8, mode="same")[4:-4]
    lo = np.searchsorted(freqs, 4000)
    falls = smooth[lo + step:] - smooth[lo:-step]
    k = int(np.argmin(falls))
    out["edge_slope_db_per_khz"] = float(falls[k] * 2.0)
    out["edge_slope_hz"] = float(freqs[lo + k] + 250.0)
    return out


def _describe_job(job):
    item, treatment = job
    return {"utterance_id": item[1], "label": 0 if item[5] == "bonafide" else 1,
            "attack": item[2], **describe(treat(read_audio(item[0]), treatment))}


def cmd_describe(args):
    jobs = [(r, args.treatment) for r in S.read_manifest(args.manifest)]
    with open_out(args.out) as fh, ProcessPoolExecutor(args.jobs) as pool:
        writer = csv.DictWriter(
            fh, fieldnames=["utterance_id", "label", "attack"] + DESCRIPTORS)
        writer.writeheader()
        writer.writerows(pool.map(_describe_job, jobs, chunksize=32))
    print(f"{len(jobs)} files described to {args.out}")


def load_descriptors(path):
    rows = read_csv(path)
    out = {"attack": np.asarray([r["attack"] for r in rows]),
           "y": np.asarray([int(r["label"]) for r in rows])}
    for name in DESCRIPTORS:
        out[name] = np.asarray([float(r[name]) for r in rows])
    return out


def descriptor_aucs(desc, n_boot, seed):
    """
    AUC of every descriptor for bonafide against each attack, spoof positive.

    Signed and not folded: these are compared across attacks sharing one
    descriptor, where folding would reflect a near chance attack upward.
    One row per file, so the bootstrap resamples utterances.
    """
    bona = A.bonafide_label(desc["y"], desc["attack"])
    if bona is None:
        raise SystemExit("cannot read which label is bonafide from the descriptors")
    is_bona = desc["y"] == bona
    results = {}
    for attack in sorted(set(desc["attack"][~is_bona].tolist())):
        rows = is_bona | (desc["attack"] == attack)
        y = (~is_bona[rows]).astype(int)
        results[attack] = {
            name: bootstrap_auc_ci(y, desc[name][rows], n_boot=n_boot, seed=seed)
            for name in DESCRIPTORS}
    return results


def cmd_m1(args):
    desc = load_descriptors(args.descriptors)
    bona = A.bonafide_label(desc["y"], desc["attack"])
    groups = ["-"] + sorted(set(desc["attack"][desc["y"] != bona].tolist()))
    print("Per group quantiles, 5 / 50 / 95 percent\n")
    print(f"{'GROUP':<6} {'N':>5} " + " ".join(f"{n:>26}" for n in DESCRIPTORS))
    for g in groups:
        rows = (desc["y"] == bona) if g == "-" else (desc["attack"] == g)
        cells = []
        for name in DESCRIPTORS:
            q = np.percentile(desc[name][rows], [5, 50, 95])
            cells.append(f"{q[0]:>8.1f}/{q[1]:>8.1f}/{q[2]:>8.1f}")
        print(f"{g:<6} {int(rows.sum()):>5} " + " ".join(cells))

    aucs = descriptor_aucs(desc, args.n_boot, args.seed)
    print("\nAUC, bonafide against each attack, spoof positive, signed, "
          "95% utterance bootstrap\n")
    print(f"{'ATTACK':<6} " + " ".join(f"{n:>26}" for n in DESCRIPTORS))
    for attack, per in aucs.items():
        print(f"{attack:<6} " + " ".join(
            f"{per[n]['auc']:>8.3f} [{per[n]['ci_low']:.3f}, {per[n]['ci_high']:.3f}]"
            for n in DESCRIPTORS))
    lows = {n: min(np.median(desc[n][(desc["y"] == bona) if g == "-"
                                     else (desc["attack"] == g)]) for g in groups)
            for n in ("bw30_hz", "bw40_hz")}
    print(f"\nlowest group median bandwidth: 30 dB {lows['bw30_hz']:.0f} Hz, "
          f"40 dB {lows['bw40_hz']:.0f} Hz")
    print(f"lowest single file: 30 dB {desc['bw30_hz'].min():.0f} Hz, "
          f"40 dB {desc['bw40_hz'].min():.0f} Hz")


# --------------------------------------------------------------------------
# tables, LEVEL and ASYMMETRY
# --------------------------------------------------------------------------

def load_rows(path, seg="run"):
    return [r for r in read_csv(path) if r["seg"] == seg]


def to_data(rows, phase=A.PHASE):
    """
    The per category arrays analyse_slices works on.

    `phase` names the column filed under the registered phase statistic's key.
    Anything other than the default is a redefinition and the caller says so.
    """
    data = {}
    for r in rows:
        d = data.setdefault(r["category"], {"y": [], "u": [], "attack": [],
                                            A.PHASE: [], A.MAGNITUDE: []})
        d["y"].append(int(r["label"]))
        d["u"].append(r["utterance_id"])
        d["attack"].append(r["attack"])
        d[A.PHASE].append(float(r[phase]))
        d[A.MAGNITUDE].append(float(r[A.MAGNITUDE]))
    return A.with_bonafide_label(
        {k: {n: np.asarray(v) for n, v in d.items()} for k, d in data.items()})


def two_arm_rows(base_rows, treated_rows, name):
    """
    Bonafide against the same bonafide treated, as one two class table.

    Arm 0 keeps its rows as they are. Arm 1 takes label 1 and the treatment as
    its attack, so the direction convention is still read off the attack column
    and not assumed. Utterance ids are left identical across arms, which is
    what lets a draw take both versions of an utterance together.
    """
    labels = [int(r["label"]) for r in base_rows]
    bona = A.bonafide_label(labels, [r["attack"] for r in base_rows])
    if bona is None:
        bona = A.bonafide_label(labels + [1 - labels[0]],
                                [r["attack"] for r in base_rows] + ["x"])
    arm0 = [r for r in base_rows if int(r["label"]) == bona]
    arm1 = [{**r, "label": 1 - bona, "attack": name} for r in treated_rows
            if int(r["label"]) == bona]
    key = ("utterance_id", "category", "seg")
    if [[r[k] for k in key] for r in arm0] != [[r[k] for k in key] for r in arm1]:
        raise ValueError("the two arms do not hold the same slices in the same order")
    return arm0 + arm1


def _interval(value, draws, alpha):
    """
    A percentile interval that knows whether the draw count settles which side
    of zero it lies on, see percentile_interval in src/phonophase/stats.py.
    """
    r = percentile_interval(draws, alpha)
    return {"value": value, "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "undecided": r["undecided"]}


def level_asymmetry(data, n_boot, seed, joint=False, alpha=0.05):
    """
    LEVEL, the obstruent advantage and the ASYMMETRY from one set of draws.

    joint=False draws each class's utterances separately, through
    shared_cluster_draws, and its ASYMMETRY interval is the one
    analyse_slices.asymmetry_ci returns for the same seed.

    joint=True is for two arms holding the SAME utterances. A draw picks
    utterance ids and takes both versions of each, so whatever an utterance
    contributes to both arms cancels in the separation. Drawing the arms
    separately treats them as unrelated recordings, which they are not.
    """
    sides = [[c for c in cats if c in data and A.usable(data[c])]
             for cats in (A.SONORANTS, A.OBSTRUENTS)]
    if not sides[0] or not sides[1]:
        return None
    every = sides[0] + sides[1]
    y, groups, phase, magnitude, owner = A.pooled(data, every, "utterance")
    son_i, obs_i = range(len(sides[0])), range(len(sides[0]), len(every))

    indices = shared_cluster_draws(y, groups, n_boot, seed, stratify=not joint)

    draws = []
    for idx in indices:
        son = A.draw_mean(idx, owner, y, phase, magnitude, son_i)
        obs = A.draw_mean(idx, owner, y, phase, magnitude, obs_i)
        if son is not None and obs is not None:
            draws.append((son, obs, son - obs))
    draws = np.asarray(draws)

    level, obstruent = A.observed_mean(data, sides[0]), A.observed_mean(data, sides[1])
    out = {"categories": every, "n_draws": len(draws), "draws": draws}
    for k, (name, value) in enumerate((("level", level), ("obstruent", obstruent),
                                       ("asymmetry", level - obstruent))):
        out[name] = _interval(value, draws[:, k], alpha)
    return out


def per_category(data):
    """Signed AUC and folded separation of both statistics, per category."""
    out = {}
    for c in sorted(data):
        d = data[c]
        if not A.usable(d):
            continue
        out[c] = {"n": (int((d["y"] == 0).sum()), int((d["y"] == 1).sum())),
                  "phase_auc": A.fast_auc(d["y"], d[A.PHASE]),
                  "magnitude_auc": A.fast_auc(d["y"], d[A.MAGNITUDE]),
                  "phase_sep": A.folded(d["y"], d[A.PHASE]),
                  "magnitude_sep": A.folded(d["y"], d[A.MAGNITUDE])}
    return out


def print_per_category(data):
    print(f"  {'CATEGORY':<14} {'SLICES 0/1':>14} {'PHASE AUC':>10} {'MAG AUC':>9} "
          f"{'PHASE SEP':>10} {'MAG SEP':>9} {'ADVANTAGE':>10}")
    for c, r in per_category(data).items():
        print(f"  {c:<14} {'%d/%d' % r['n']:>14} {r['phase_auc']:>10.3f} "
              f"{r['magnitude_auc']:>9.3f} {r['phase_sep']:>10.3f} "
              f"{r['magnitude_sep']:>9.3f} {r['phase_sep'] - r['magnitude_sep']:>+10.3f}")


def fmt(r):
    """An interval, marked when the draw count does not settle its sign."""
    return (f"{r['value']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
            + (" (not settled)" if r["undecided"] else ""))


def print_level_asymmetry(data, n_boot, seeds, joint):
    for seed in seeds:
        r = level_asymmetry(data, n_boot, seed, joint=joint)
        if r is None:
            raise SystemExit("too few sonorant or obstruent slices on one side to compare")
        print(f"  seed {seed}  LEVEL {fmt(r['level'])}  OBSTRUENT {fmt(r['obstruent'])}  "
              f"ASYMMETRY {fmt(r['asymmetry'])}")


def cmd_r1(args):
    """Bonafide against treated bonafide. No synthesis on either side."""
    arm0 = load_rows(args.treated_base or args.base, args.seg)
    rows = two_arm_rows(arm0, load_rows(args.treated, args.seg), "treated")
    data = to_data(rows)
    print(f"arm 0 {args.treated_base or args.base} (bonafide rows)\n"
          f"arm 1 {args.treated} (bonafide rows)\nseg={args.seg}, "
          f"bonafide label read as {next(iter(data.values()))['bonafide_label']}\n")
    print_per_category(data)
    for joint in (True, False):
        print(f"\n{'JOINT draw, both versions of each utterance' if joint else 'UNPAIRED draw, each arm resampled separately'}, "
              f"{args.n_boot} draws, utterance clustered")
        print_level_asymmetry(data, args.n_boot, args.seeds, joint)


def against(rows, attacks):
    """Bonafide rows plus the named attacks, or every row when none are named."""
    if not attacks:
        return rows
    print(f"bonafide against {' '.join(sorted(attacks))} only")
    keep = set(attacks) | A.BONAFIDE_MARKERS
    return [r for r in rows if r["attack"] in keep]


def change(before_rows, after_rows, n_boot, seed, alpha=0.05):
    """
    How far a treatment moved LEVEL and ASYMMETRY, as after minus before.

    Both tables hold the same slices in the same order, so one seed gives both
    the same utterances in every draw and the difference is taken inside the
    draw. Two separate intervals that overlap say nothing about whether the
    treatment moved the quantity; this does.
    """
    key = ("utterance_id", "category", "phoneme", "start_frame", "n_frames", "label")
    if [[r[k] for k in key] for r in before_rows] != [[r[k] for k in key] for r in after_rows]:
        raise ValueError("the two tables do not hold the same slices in the same order")
    before = level_asymmetry(to_data(before_rows), n_boot, seed)
    after = level_asymmetry(to_data(after_rows), n_boot, seed)
    if before["categories"] != after["categories"] or before["n_draws"] != after["n_draws"]:
        raise ValueError("the two tables did not yield the same draws")
    out = {}
    for k, name in enumerate(("level", "obstruent", "asymmetry")):
        out[name] = _interval(after[name]["value"] - before[name]["value"],
                              after["draws"][:, k] - before["draws"][:, k], alpha)
    return out


def cmd_change(args):
    before = against(load_rows(args.before, args.seg), args.attacks)
    after = [r for r in against(load_rows(args.after, args.seg), args.attacks)]
    print(f"after {args.after} minus before {args.before}, seg={args.seg}, "
          f"{args.n_boot} draws, utterance clustered, same draw for both")
    for seed in args.seeds:
        r = change(before, after, args.n_boot, seed)
        print(f"  seed {seed}  LEVEL {fmt(r['level'])}  ASYMMETRY {fmt(r['asymmetry'])}")


def cmd_r2(args):
    """The registered comparison on one table, with the registered estimator."""
    data = to_data(against(load_rows(args.table, args.seg), args.attacks),
                   phase=args.phase)
    if args.phase != A.PHASE:
        print(f"REDEFINITION: the phase statistic is {args.phase}, not {A.PHASE}\n")
    print(f"table {args.table}, seg={args.seg}\n")
    print_per_category(data)
    print(f"\n{args.n_boot} draws, utterance clustered, one draw for all categories")
    for seed in args.seeds:
        son = A.advantage_ci(data, A.SONORANTS, args.n_boot, seed)
        asym = A.asymmetry_ci(data, args.n_boot, seed)
        if son is None or asym is None:
            raise SystemExit("too few sonorant or obstruent slices on one side to compare")
        print(f"  seed {seed}  LEVEL {son['advantage']:+.4f} "
              f"[{son['ci_low']:+.4f}, {son['ci_high']:+.4f}]  "
              f"ASYMMETRY {asym['asymmetry']:+.4f} "
              f"[{asym['ci_low']:+.4f}, {asym['ci_high']:+.4f}]")


# --------------------------------------------------------------------------
# M2, does the asymmetry track the band edge mismatch across attacks
# --------------------------------------------------------------------------

def spearman(x, y, n_perm=100000, seed=0):
    """Spearman rho with a two sided Monte Carlo permutation p, +1 corrected."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    shuffled = rng.permuted(np.tile(ry, (n_perm, 1)), axis=1)
    null = ((shuffled - ry.mean()) @ (rx - rx.mean())) / (rx.size * rx.std() * ry.std())
    p = (int((np.abs(null) >= abs(rho) - 1e-12).sum()) + 1) / (n_perm + 1)
    return rho, float(p), float(np.quantile(np.abs(null), 0.95))


def cmd_m2(args):
    import attack_analysis as AA
    desc = load_descriptors(args.descriptors)
    aucs = descriptor_aucs(desc, 200, args.seed)
    per = AA.per_attack(AA.load(args.table, args.seg), n_boot=args.n_boot,
                        seed=args.seed)
    attacks = sorted(per)
    print(f"{'ATTACK':<6} {'LEVEL':>28} {'ASYMMETRY':>28} " +
          " ".join(f"{n[:14]:>14}" for n in DESCRIPTORS))
    for a in attacks:
        r = per[a]
        print(f"{a:<6} "
              f"{'%+.3f [%+.3f, %+.3f]' % (r['sonorant_advantage'], r['sonorant_ci_low'], r['sonorant_ci_high']):>28} "
              f"{'%+.3f [%+.3f, %+.3f]' % (r['asymmetry'], r['asymmetry_ci_low'], r['asymmetry_ci_high']):>28} " +
              " ".join(f"{aucs[a][n]['auc']:>14.3f}" for n in DESCRIPTORS))
    print("\ndescriptor columns are signed AUC, bonafide against the attack. "
          "Mismatch below is |AUC - 0.5|.\n")
    print(f"{'DESCRIPTOR':<24} {'QUANTITY':<10} {'RHO':>7} {'PERM P':>8}")
    for name in DESCRIPTORS:
        mismatch = [abs(aucs[a][name]["auc"] - 0.5) for a in attacks]
        for label, key in (("ASYMMETRY", "asymmetry"), ("LEVEL", "sonorant_advantage")):
            rho, p, critical = spearman(mismatch, [per[a][key] for a in attacks])
            print(f"{name:<24} {label:<10} {rho:>+7.3f} {p:>8.4f}")
    print(f"\nn={len(attacks)}. Smallest |rho| reaching two sided p<0.05 under "
          f"permutation: {critical:.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)

    def add(name, fn, **arguments):
        p = sub.add_parser(name)
        for flag, kw in arguments.items():
            p.add_argument("--" + flag.replace("_", "-"), **kw)
        p.set_defaults(fn=fn)

    manifest = dict(required=True)
    out = dict(required=True)
    jobs = dict(type=int, default=max(1, (os.cpu_count() or 2) - 2))
    seg = dict(default="run", choices=["run", "window"])
    n_boot = dict(type=int, default=2000)
    seeds = dict(type=int, nargs="+", default=[0, 1, 2])
    seed = dict(type=int, default=0)

    add("align", cmd_align, manifest=manifest, out=out)
    add("slice", cmd_slice, manifest=manifest, out=out, jobs=jobs,
        alignment=dict(required=True), treatment=dict(default="none"),
        only=dict(default="all", choices=["all", "bonafide", "spoof"]),
        segmentations=dict(default="run"),
        band=dict(type=float, nargs=2, default=None, metavar=("LO_HZ", "HI_HZ")))
    add("describe", cmd_describe, manifest=manifest, out=out, jobs=jobs,
        treatment=dict(default="none"))
    add("m1", cmd_m1, descriptors=dict(required=True), n_boot=n_boot, seed=seed)
    add("m2", cmd_m2, table=dict(required=True), descriptors=dict(required=True),
        seg=seg, n_boot=n_boot, seed=seed)
    add("r1", cmd_r1, base=dict(required=True), treated=dict(required=True),
        treated_base=dict(default=None, help="arm 0 when both arms are treated"),
        seg=seg, n_boot=n_boot, seeds=seeds)
    add("r2", cmd_r2, table=dict(required=True), phase=dict(default=A.PHASE),
        attacks=dict(nargs="+", default=None, help="restrict the spoof side"),
        seg=seg, n_boot=n_boot, seeds=seeds)

    add("change", cmd_change, before=dict(required=True), after=dict(required=True),
        attacks=dict(nargs="+", default=None, help="restrict the spoof side"),
        seg=seg, n_boot=n_boot, seeds=seeds)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
