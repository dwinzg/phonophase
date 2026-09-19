"""
Build a manifest for slice_and_probe.py --manifest from a directory of audio.

Why this exists
    The measurement should not be tied to ASVspoof. Cross lingual replication
    needs MLAAD, In-the-Wild, IndicSynth or a Bengali corpus, and every one of
    those ships a different layout. A manifest is the one thing they can all be
    reduced to, and it keeps corpus specific mess out of the analysis code.

Layouts handled
    --layout label_dirs      <root>/<bonafide|spoof>/*.wav
    --layout lang_label      <root>/<language>/<bonafide|spoof>/*.wav
    --layout label_lang      <root>/<bonafide|spoof>/<language>/*.wav

    Label directory names are matched loosely, so real, genuine and bonafide
    all count as bonafide, and fake, synthetic and spoof all count as spoof.
    Anything else is reported and skipped rather than guessed at.

Usage
    python scripts/make_manifest.py --root data/mlaad --layout lang_label \\
        --corpus mlaad --out manifests/mlaad.csv
"""

from __future__ import annotations

import argparse
import csv
import os

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}

BONAFIDE = {"bonafide", "real", "genuine", "human", "original"}
SPOOF = {"spoof", "fake", "synthetic", "generated", "tts", "vc"}


def classify(name: str) -> str | None:
    n = name.strip().lower()
    if n in BONAFIDE:
        return "bonafide"
    if n in SPOOF:
        return "spoof"
    return None


def walk(root: str, layout: str):
    """Yield (path, language, label) for every audio file that can be labelled."""
    unknown = set()
    for dirpath, _, filenames in os.walk(root):
        rel = os.path.relpath(dirpath, root)
        if rel == ".":
            continue
        parts = rel.split(os.sep)

        language, label_dir = "-", None
        if layout == "label_dirs" and len(parts) >= 1:
            label_dir = parts[0]
        elif layout == "lang_label" and len(parts) >= 2:
            language, label_dir = parts[0], parts[1]
        elif layout == "label_lang" and len(parts) >= 2:
            label_dir, language = parts[0], parts[1]

        if label_dir is None:
            continue
        label = classify(label_dir)
        if label is None:
            unknown.add(label_dir)
            continue

        for name in sorted(filenames):
            if os.path.splitext(name)[1].lower() in AUDIO_EXT:
                yield os.path.join(dirpath, name), language, label

    if unknown:
        print(f"  skipped unrecognised label directories: {', '.join(sorted(unknown))}")
        print("  extend BONAFIDE or SPOOF in this script if any of those are labels")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--layout", default="lang_label",
                    choices=["label_dirs", "lang_label", "label_lang"])
    ap.add_argument("--corpus", default="-", help="recorded in every row")
    ap.add_argument("--language", default=None,
                    help="override, for a corpus that is a single language")
    ap.add_argument("--limit-per-class", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        raise SystemExit(f"not a directory: {args.root}")

    rows = []
    for path, language, label in walk(args.root, args.layout):
        rows.append({
            "path": path,
            "label": label,
            "utterance_id": os.path.splitext(os.path.basename(path))[0],
            "attack": "-",
            "language": args.language or language,
            "corpus": args.corpus,
        })

    if not rows:
        raise SystemExit("nothing found. Check --root and --layout")

    if args.limit_per_class:
        import random
        random.Random(args.seed).shuffle(rows)
        kept, counts = [], {}
        for r in rows:
            key = (r["language"], r["label"])
            if counts.get(key, 0) < args.limit_per_class:
                kept.append(r)
                counts[key] = counts.get(key, 0) + 1
        rows = kept

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "utterance_id",
                                           "attack", "language", "corpus"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {len(rows)} rows to {args.out}")
    by = {}
    for r in rows:
        by[(r["language"], r["label"])] = by.get((r["language"], r["label"]), 0) + 1
    print(f"\n{'LANGUAGE':<12} {'BONAFIDE':>10} {'SPOOF':>10}")
    print("-" * 34)
    for lang in sorted({k[0] for k in by}):
        print(f"{lang:<12} {by.get((lang,'bonafide'),0):>10} {by.get((lang,'spoof'),0):>10}")

    thin = [l for l in {k[0] for k in by} if by.get((l, "bonafide"), 0) < 200]
    if thin:
        print(f"\nUnder 200 bonafide utterances: {', '.join(sorted(thin))}")
        print("The pilot showed per phoneme estimates are unstable below roughly")
        print("200, so treat those languages as underpowered.")


if __name__ == "__main__":
    main()
