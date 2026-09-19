"""
Export exactly the audio the confirmatory run will read, as a small archive.

Why this exists
    Fetching the 7.12 GB corpus inside Colab fails. Edinburgh DataShare rate
    limits by IP and returns 403, and Colab egresses from shared Google
    datacenter addresses whose quota is routinely already spent. That is not
    fixable by changing the request.

    The run only touches a few thousand specific utterances, and an ASVspoof
    flac averages about 61 KB, so the audio it actually needs is roughly 190 MB
    rather than 7.12 GB. Export that once from a machine that already holds the
    corpus, upload it, and the download problem disappears.

Why the selection is guaranteed to match
    Both this and slice_and_probe.py call phonophase.corpus.select_balanced with the
    same seed and limits, so the exported files are precisely the files the run
    will open. Reimplementing the selection here would be the obvious way to
    get a subtly different subset and never notice.

Usage
    python scripts/export_audio_subset.py --subset eval \\
        --limit-bonafide 1500 --limit-spoof 1500 --out eval_subset.zip

Then in Colab, upload it and run

    python scripts/slice_and_probe.py --manifest <unzipped>/manifest.csv \\
        --out results/eval_slices.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from phonophase.config import load_config  # noqa: E402
from phonophase.corpus import partition_paths, read_protocol, select_balanced  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--subset", default="eval", choices=["train", "dev", "eval"])
    ap.add_argument("--limit-bonafide", type=int, default=1500)
    ap.add_argument("--limit-spoof", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=1234,
                    help="must match the seed passed to slice_and_probe.py")
    ap.add_argument("--out", default="eval_subset.zip")
    args = ap.parse_args()

    config = load_config(args.config)
    audio_dir, protocol = partition_paths(config, args.subset)
    for path in (audio_dir, protocol):
        if not os.path.exists(path):
            raise SystemExit(f"missing: {path}")

    selected = select_balanced(read_protocol(protocol), args.limit_bonafide,
                               args.limit_spoof, args.seed)
    n_bona = sum(1 for r in selected if r[2] == "bonafide")
    print(f"{args.subset}: {len(selected)} utterances, {n_bona} bonafide, "
          f"{len(selected) - n_bona} spoof, seed {args.seed}")

    with tempfile.TemporaryDirectory() as staging:
        audio_out = os.path.join(staging, "audio")
        os.makedirs(audio_out)

        rows, missing, total = [], 0, 0
        for utt, attack, label in selected:
            src = os.path.join(audio_dir, f"{utt}.flac")
            if not os.path.exists(src):
                missing += 1
                continue
            shutil.copy2(src, os.path.join(audio_out, f"{utt}.flac"))
            total += os.path.getsize(src)
            rows.append({
                # Relative to the manifest, so it resolves wherever it is
                # unzipped. slice_and_probe resolves manifest relative paths
                # against the manifest's own directory.
                "path": f"audio/{utt}.flac",
                "label": label,
                "utterance_id": utt,
                "attack": attack,
                "language": "en",
                "corpus": f"asvspoof2019_la_{args.subset}",
            })

        if missing:
            print(f"  {missing} files listed in the protocol were not on disk")
        if not rows:
            raise SystemExit("nothing exported")

        manifest = os.path.join(staging, "manifest.csv")
        with open(manifest, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "label", "utterance_id",
                                               "attack", "language", "corpus"])
            w.writeheader()
            w.writerows(rows)

        with open(os.path.join(staging, "SUBSET_INFO"), "w") as fh:
            fh.write(f"subset:   {args.subset}\n")
            fh.write(f"seed:     {args.seed}\n")
            fh.write(f"bonafide: {n_bona}\n")
            fh.write(f"spoof:    {len(rows) - n_bona}\n")
            fh.write(f"note:     selection must match slice_and_probe.py "
                     f"--seed {args.seed}\n")

        print(f"  staged {len(rows)} files, {total/1e6:.0f} MB of audio")
        print("  compressing, flac is already compressed so expect little gain")

        out = os.path.abspath(args.out)
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _, names in os.walk(staging):
                for name in names:
                    full = os.path.join(root, name)
                    z.write(full, os.path.relpath(full, staging))

    print(f"\nwrote {out}, {os.path.getsize(out)/1e6:.0f} MB")
    print("\nUpload this to Drive once, then in Colab:")
    print("  !unzip -q /content/drive/MyDrive/phonophase/eval_subset.zip -d /content/eval_subset")
    print("  !python scripts/slice_and_probe.py \\")
    print("      --manifest /content/eval_subset/manifest.csv \\")
    print("      --out results/eval_slices.csv")


if __name__ == "__main__":
    main()
