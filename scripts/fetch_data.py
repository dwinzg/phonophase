"""
Fetch corpora from their original sources.

Why this exists
    Nothing recorded where the data came from, which made the published results
    unverifiable. This reads configs/datasets.yaml and pulls from the source,
    so a fresh machine, a Colab runtime or a reviewer can reproduce the setup
    without anyone shipping 7 GB around by hand.

Licence guard
    Anything in the registry with licence_verified false refuses to download
    unless --licence-checked is passed. That is deliberate. This repository is
    MIT, several of these corpora are not, and at least one is non commercial
    only. A licence you have not read is not a licence you can rely on.

Usage
    python scripts/fetch_data.py --list
    python scripts/fetch_data.py --dataset asvspoof2019_la --out data
    python scripts/fetch_data.py --dataset asvspoof2019_la --out data --no-extract
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import zipfile

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REGISTRY = os.path.join(HERE, "..", "configs", "datasets.yaml")


def load_registry() -> dict:
    with open(REGISTRY) as fh:
        return yaml.safe_load(fh)


def show(registry: dict) -> None:
    print(f"{'KEY':<22} {'STATUS':<11} {'SIZE':>8}  LICENCE")
    print("-" * 78)
    for key, entry in registry.items():
        size = f"{entry['size_gb']:.1f} GB" if entry.get("size_gb") else "-"
        lic = entry.get("licence", "see landing page")
        mark = "" if entry.get("licence_verified") else "  [CHECK LICENCE]"
        print(f"{key:<22} {entry.get('status','-'):<11} {size:>8}  {lic}{mark}")
    print()
    print("Entries without a direct_url are registered for reference only and")
    print("are fetched from their landing page, usually HuggingFace.")


def md5sum(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: str) -> None:
    """Resumable download. curl -C - picks up a partial file rather than restarting."""
    print(f"downloading to {dest}")
    print("large corpora take a while, and the transfer resumes if interrupted")
    print("note: hosts rate limit by IP. From Colab or another shared cloud")
    print("address this often returns 403 regardless of the request. If it does,")
    print("use scripts/export_audio_subset.py from a machine that already holds")
    print("the corpus and take the --manifest route instead.")
    result = subprocess.run(
        ["curl", "-L", "-C", "-", "--fail", "--progress-bar", "-o", dest, url]
    )
    if result.returncode != 0:
        if result.returncode == 22:
            raise SystemExit(
                "download refused by the host, usually HTTP 403 from IP rate\n"
                "limiting. Colab and other shared cloud addresses hit this\n"
                "routinely. Export the subset you need locally instead:\n\n"
                "  python scripts/export_audio_subset.py --subset eval \\\n"
                "      --limit-bonafide 1500 --limit-spoof 1500 --out eval_subset.zip\n\n"
                "That is about 190 MB rather than 7.12 GB."
            )
        raise SystemExit(f"download failed with code {result.returncode}")


def extract(path: str, out_dir: str) -> None:
    print(f"extracting {os.path.basename(path)}")
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            z.extractall(out_dir)
    elif path.endswith((".tar.gz", ".tgz")):
        subprocess.run(["tar", "-xzf", path, "-C", out_dir], check=True)
    else:
        print("  unrecognised archive type, leaving it alone")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show the registry and exit")
    ap.add_argument("--dataset", help="registry key to fetch")
    ap.add_argument("--out", default="data", help="destination directory")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument("--licence-checked", action="store_true",
                    help="confirm you have read the licence for an unverified entry")
    args = ap.parse_args()

    registry = load_registry()

    if args.list or not args.dataset:
        show(registry)
        return

    if args.dataset not in registry:
        raise SystemExit(f"unknown dataset {args.dataset!r}. Try --list.")

    entry = registry[args.dataset]
    print(f"{entry['name']}")
    print(f"  licence     {entry.get('licence', 'see landing page')}")
    print(f"  source      {entry.get('landing_page')}")
    if entry.get("citation"):
        print(f"  cite        {entry['citation']}")
    if entry.get("requires"):
        print(f"  also needs  {entry['requires']}")
    print()

    if not entry.get("licence_verified") and not args.licence_checked:
        raise SystemExit(
            f"The licence for {args.dataset} has not been verified in the registry.\n"
            f"Read it at {entry.get('landing_page')}, then rerun with "
            "--licence-checked.\nThis guard exists because this repository is MIT "
            "and several of these corpora are not."
        )

    url = entry.get("direct_url")
    if not url:
        raise SystemExit(
            f"{args.dataset} has no scriptable direct link. Fetch it from\n"
            f"  {entry.get('landing_page')}\n"
            "HuggingFace datasets are usually one line:\n"
            "  from datasets import load_dataset; load_dataset(\"<repo id>\")"
        )

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, entry.get("filename", f"{args.dataset}.download"))

    if os.path.exists(dest) and entry.get("size_gb"):
        have = os.path.getsize(dest) / 1e9
        print(f"  found an existing file at {have:.2f} GB of {entry['size_gb']:.2f} GB")

    download(url, dest)

    if entry.get("md5"):
        print("verifying checksum")
        got = md5sum(dest)
        if got != entry["md5"]:
            raise SystemExit(f"checksum mismatch\n  expected {entry['md5']}\n  got      {got}")
        print("  checksum ok")

    if not args.no_extract:
        extract(dest, args.out)

    print(f"\ndone. {entry['name']} is in {args.out}")
    if entry.get("note"):
        print(f"\nnote: {' '.join(entry['note'].split())}")


if __name__ == "__main__":
    main()
