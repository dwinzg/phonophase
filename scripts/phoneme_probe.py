"""
Model free probe of the slicing hypothesis.

The question
    Does isolating particular phoneme classes actually help, and if so does
    PHASE behave differently across classes than magnitude does?

    The first half of that is already answered in the literature. Fricatives,
    plosives and diphthongs carry the strongest cues, see
    configs/literature.yaml. Reporting it again is not a finding.

    The second half is the part that might be open. Every phoneme level paper
    found so far uses self supervised embeddings or magnitude domain features.
    None was found using group delay. There is a physical reason to expect a
    difference. Fricatives are aperiodic so phase is close to random, vowels
    are periodic so phase carries harmonic structure, and plosives are
    transient so phase is discontinuous. A vocoder has no reason to fail
    uniformly across those regimes, and magnitude domain work cannot see it.

Why this is model free
    Training a detector per phoneme class to answer this would be slow and
    would confound the question with optimisation. Instead each slice is
    reduced to a few scalars and the separability of those scalars is measured
    directly with AUC. That runs in minutes on CPU and gives a clean answer.

    A high AUC for an MGD statistic where the matching MFCC statistic is near
    chance is the result worth having. If MGD never beats MFCC, the phase
    thesis is not supported and no amount of framing rescues it.

Normalisation
    Statistics are computed with normalisation off. Per file z scoring forces
    every slice to unit variance, which divides out the exact dispersion this
    probe measures. That was the defect in the original analysis path.

Input layout, as produced by src/01_phoneme_slicer.py
    <input_dir>/<CATEGORY>/<bonafide|spoof>/*.wav

Usage
    python scripts/phoneme_probe.py --input_dir data/processed_atomic/train \\
        --out results/phoneme_probe
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import soundfile as sf
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.features import modified_group_delay  # noqa: E402
from phonophase.metrics import compute_eer  # noqa: E402
from phonophase.stats import (  # noqa: E402
    benjamini_hochberg,
    bootstrap_auc_ci,
    paired_bootstrap_delta,
    permutation_test_auc,
)

import librosa  # noqa: E402

SR = 16000
N_FFT = 512
HOP = 128
WIN = 400
MIN_SAMPLES = 512

# Slice filenames look like LA_T_1000406_70_s.wav, so the source utterance is
# recoverable. This matters more than it sounds. Slices from one utterance share
# a speaker, a channel and, for spoofed audio, the same synthesis system, so
# they are not independent observations. 129 fricative slices came from 23
# utterances. Bootstrapping slices would understate the uncertainty badly, so
# the utterance id is carried through and used as the bootstrap cluster.
UTTERANCE_RE = re.compile(r"^(LA_[TED]_\d+)_")

# Pre-registered primary comparison. Fixed before running at scale, so the
# headline claim is one test rather than the best of many.
#
# frame_spread is the mean over time of the per frame spread across frequency,
# which is the closest scalar to the Spectral Collapse idea that synthetic
# speech is unnaturally smooth within a frame. It is computed identically on
# both planes, so the comparison isolates the domain.
#
# Everything else is exploratory and is reported under FDR correction. Picking
# the best MGD statistic and comparing it to the best MFCC statistic, which is
# what the first pass did, inflates the difference by selection and cannot
# support a claim.
PRIMARY_PHASE = "mgd_frame_spread"
PRIMARY_MAGNITUDE = "mfcc_frame_spread"


def fix_duration(audio: np.ndarray, seconds: float | None) -> np.ndarray:
    """
    Crop or pad every slice to the same length.

    Sonorants run longer than plosives, and a longer slice yields more frames
    and therefore a differently behaved variance estimate. Without this the
    sonorant advantage reported on the first pass could be duration rather than
    phase. Centre cropping keeps the steady portion of the segment.
    """
    if seconds is None:
        return audio
    target = int(seconds * SR)
    if audio.size == target:
        return audio
    if audio.size > target:
        start = (audio.size - target) // 2
        return audio[start:start + target]
    pad = target - audio.size
    return np.pad(audio, (pad // 2, pad - pad // 2))


def summarise(audio: np.ndarray) -> dict | None:
    """
    Reduce one slice to a handful of scalars.

    Paired by design. Every phase statistic has a magnitude counterpart
    computed over the same framing, so the comparison isolates the domain
    rather than the analysis settings.
    """
    if audio.size < MIN_SAMPLES:
        return None

    mgd = modified_group_delay(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=40, n_fft=N_FFT, hop_length=HOP)

    if not (np.all(np.isfinite(mgd)) and np.all(np.isfinite(mfcc))):
        return None

    def spread(plane):
        q75, q25 = np.percentile(plane, [75, 25])
        return {
            "std": float(np.std(plane)),
            "iqr": float(q75 - q25),
            # Mean over time of the per frame spread across frequency. This is
            # the closest scalar to the Spectral Collapse idea, that synthetic
            # speech is unnaturally smooth within a frame.
            "frame_spread": float(np.mean(np.std(plane, axis=0))),
        }

    out = {}
    for name, plane in (("mgd", mgd), ("mfcc", mfcc)):
        for key, value in spread(plane).items():
            out[f"{name}_{key}"] = value

    # High frequency half only. Fricative energy lives up there, so if the
    # artifact is fricative driven this should separate better than the full band.
    half = mgd.shape[0] // 2
    out["mgd_hf_std"] = float(np.std(mgd[half:]))
    return out


def utterance_id(filename: str) -> str:
    """Recover the source utterance, falling back to the filename itself."""
    match = UTTERANCE_RE.match(filename)
    return match.group(1) if match else filename


def collect(category_dir: str, limit: int | None, fixed_duration: float | None):
    """Return (features, labels, durations, utterance ids)."""
    features, labels, durations, utterances = [], [], [], []
    for label_name, label in (("bonafide", 0), ("spoof", 1)):
        folder = os.path.join(category_dir, label_name)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
        if limit:
            files = files[:limit]
        for name in files:
            try:
                audio, sr = sf.read(os.path.join(folder, name))
                if audio.ndim > 1:
                    audio = audio[:, 0]
                if sr != SR:
                    continue
                raw_duration = audio.size / SR
                row = summarise(fix_duration(audio.astype(np.float64), fixed_duration))
                if row is not None:
                    features.append(row)
                    labels.append(label)
                    durations.append(raw_duration)
                    utterances.append(utterance_id(name))
            except Exception:
                continue
    return features, labels, durations, utterances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True,
                    help="directory of <CATEGORY>/<bonafide|spoof>/*.wav")
    ap.add_argument("--out", default="results/phoneme_probe")
    ap.add_argument("--limit", type=int, default=400,
                    help="slices per class per category, caps runtime")
    ap.add_argument("--fixed-duration", type=float, default=None,
                    help="crop or pad every slice to this many seconds, to remove "
                         "duration as an explanation for any category difference")
    ap.add_argument("--min-per-class", type=int, default=30)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cluster", action="store_true",
                    help="bootstrap slices instead of source utterances. Wrong, and "
                         "kept only to show how much it understates uncertainty")
    ap.add_argument("--primary-phase", default=PRIMARY_PHASE,
                    help="pre-registered phase statistic for the primary test")
    ap.add_argument("--primary-magnitude", default=PRIMARY_MAGNITUDE,
                    help="pre-registered magnitude statistic for the primary test")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"not a directory: {args.input_dir}")

    categories = sorted(
        d for d in os.listdir(args.input_dir)
        if os.path.isdir(os.path.join(args.input_dir, d))
    )
    if not categories:
        raise SystemExit(f"no phoneme category directories under {args.input_dir}")

    print(f"categories      {', '.join(categories)}")
    print(f"primary test    {args.primary_phase} against {args.primary_magnitude}, paired bootstrap")
    print(f"duration        {'fixed at %.3fs' % args.fixed_duration if args.fixed_duration else 'as sliced, NOT controlled'}")
    print(f"bootstrap       {args.n_boot} draws, "
          f"{'SLICE level, understates uncertainty' if args.no_cluster else 'clustered by source utterance'}")
    print(f"permutation     {args.n_perm} draws\n")
    os.makedirs(args.out, exist_ok=True)

    results = {}
    exploratory = []  # (category, statistic, p_value) for FDR across everything

    for category in categories:
        features, labels, durations, utterances = collect(
            os.path.join(args.input_dir, category), args.limit, args.fixed_duration
        )
        y = np.asarray(labels)
        clusters = np.asarray(utterances) if not args.no_cluster else None
        n_bona = int(np.sum(y == 0))
        n_spoof = int(np.sum(y == 1))

        if n_bona < args.min_per_class or n_spoof < args.min_per_class:
            print(f"{category:<16} SKIPPED, {n_bona} bonafide and {n_spoof} spoof, "
                  f"below the minimum of {args.min_per_class}")
            results[category] = {"skipped": True, "n_bonafide": n_bona, "n_spoof": n_spoof}
            continue

        dur = np.asarray(durations)
        u = np.asarray(utterances)
        entry = {
            "n_bonafide": n_bona,
            "n_spoof": n_spoof,
            "n_utterances_bonafide": int(len(set(u[y == 0].tolist()))),
            "n_utterances_spoof": int(len(set(u[y == 1].tolist()))),
            "duration_mean_s": float(dur.mean()),
            "duration_std_s": float(dur.std()),
            "statistics": {},
        }

        for stat in sorted(features[0]):
            values = np.array([f[stat] for f in features], dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.std(values) == 0:
                continue
            ci = bootstrap_auc_ci(y, values, n_boot=args.n_boot,
                                  alpha=args.alpha, seed=args.seed, groups=clusters)
            perm = permutation_test_auc(y, values, n_perm=args.n_perm, seed=args.seed)
            entry["statistics"][stat] = {**ci, "p_value": perm["p_value"]}
            exploratory.append((category, stat, perm["p_value"]))

        # Pre-registered primary comparison, paired on the same slices.
        if args.primary_phase in entry["statistics"] and args.primary_magnitude in entry["statistics"]:
            phase = np.array([f[args.primary_phase] for f in features])
            magnitude = np.array([f[args.primary_magnitude] for f in features])
            # Fold both to the same orientation so the difference compares
            # separation strength rather than direction.
            if entry["statistics"][args.primary_phase]["auc"] < 0.5:
                phase = -phase
            if entry["statistics"][args.primary_magnitude]["auc"] < 0.5:
                magnitude = -magnitude
            entry["primary"] = paired_bootstrap_delta(
                y, phase, magnitude, n_boot=args.n_boot, alpha=args.alpha,
                seed=args.seed, groups=clusters
            )

        results[category] = entry
        prim = entry.get("primary")
        verdict = "no difference shown"
        if prim and prim["significant"]:
            verdict = "phase ahead" if prim["delta"] > 0 else "magnitude ahead"
        print(f"{category:<16} slices {n_bona}/{n_spoof}  "
              f"utterances {entry['n_utterances_bonafide']}/{entry['n_utterances_spoof']}  "
              f"primary delta {prim['delta']:+.3f} "
              f"[{prim['ci_low']:+.3f}, {prim['ci_high']:+.3f}]  {verdict}"
              if prim else f"{category:<16} primary unavailable")

    # FDR across every exploratory test at once. The first pass ran roughly 49
    # of these uncorrected, where about two false positives are expected.
    if exploratory:
        p_values = np.array([e[2] for e in exploratory])
        bh = benjamini_hochberg(p_values, alpha=args.alpha)
        for (category, stat, _), q, rejected in zip(exploratory, bh["q_values"], bh["rejected"]):
            results[category]["statistics"][stat]["q_value"] = float(q)
            results[category]["statistics"][stat]["survives_fdr"] = bool(rejected)
        n_raw = int((p_values < args.alpha).sum())
        print(f"\n{len(exploratory)} exploratory tests, {n_raw} below raw alpha, "
              f"{int(bh['rejected'].sum())} survive FDR")

    with open(os.path.join(args.out, "phoneme_probe.json"), "w") as fh:
        json.dump(results, fh, indent=2)

    scored = {k: v for k, v in results.items() if not v.get("skipped") and "primary" in v}
    if scored:
        print()
        print("=" * 78)
        print("PRIMARY, pre-registered. Phase minus magnitude on frame spread, paired.")
        print("=" * 78)
        print(f"{'CATEGORY':<16} {'N':>9} {'DUR ms':>8} {'DELTA':>8} {'95% CI':>18}  VERDICT")
        print("-" * 78)
        for cat, e in sorted(scored.items(), key=lambda kv: -kv[1]["primary"]["delta"]):
            pr = e["primary"]
            verdict = ("phase" if pr["delta"] > 0 else "magnitude") if pr["significant"] else "inconclusive"
            print(f"{cat:<16} {e['n_bonafide']:>4}/{e['n_spoof']:<4} "
                  f"{e['duration_mean_s']*1000:>8.1f} {pr['delta']:>+8.3f} "
                  f"{'[%+.3f, %+.3f]' % (pr['ci_low'], pr['ci_high']):>18}  {verdict}")
        print("=" * 78)

        decided = [e for e in scored.values() if e["primary"]["significant"]]
        phase_wins = sum(1 for e in decided if e["primary"]["delta"] > 0)
        print(f"\n{len(decided)} of {len(scored)} categories decided, "
              f"phase ahead in {phase_wins}")
        if not decided:
            print("Nothing is decided at this sample size. That is the honest reading,")
            print("and it means the earlier bare AUC table should not be relied on.")

        if args.fixed_duration is None:
            print("\nDuration was not controlled. Sonorants run longer than obstruents,")
            print("so rerun with --fixed-duration before reading any category pattern")
            print("as being about phase rather than about segment length.")

    # Exploratory table, ranked by folded separation. Folding is necessary
    # because a statistic running bonafide-higher separates just as well as one
    # running spoof-higher, and bonafide-higher is the Spectral Collapse
    # direction. Ranking on raw AUC would bury exactly the statistics of
    # interest, which is what the first pass of this script did.
    if scored:
        print()
        print("=" * 78)
        print("EXPLORATORY, FDR corrected. Strongest statistic per category.")
        print("Not a claim. Any pattern here needs confirming on held out data.")
        print("=" * 78)
        print(f"{'CATEGORY':<16} {'STATISTIC':<18} {'SEP':>6} {'95% CI':>16} {'q':>8}  DIRECTION")
        print("-" * 78)
        for cat, e in scored.items():
            best = None
            for name, v in e["statistics"].items():
                sep = max(v["auc"], 1 - v["auc"])
                if best is None or sep > best[0]:
                    lo, hi = ((v["ci_low"], v["ci_high"]) if v["auc"] >= 0.5
                              else (1 - v["ci_high"], 1 - v["ci_low"]))
                    best = (sep, name, lo, hi, v.get("q_value", 1.0),
                            "bonafide_higher" if v["auc"] < 0.5 else "spoof_higher",
                            v["crosses_chance"])
            if best:
                sep, name, lo, hi, q, direction, crosses = best
                flag = "  crosses chance" if crosses else ""
                print(f"{cat:<16} {name:<18} {sep:>6.3f} "
                      f"{'[%.3f, %.3f]' % (lo, hi):>16} {q:>8.4f}  {direction}{flag}")
        print("=" * 78)

    print(f"\nwrote {os.path.join(args.out, 'phoneme_probe.json')}")


if __name__ == "__main__":
    main()
