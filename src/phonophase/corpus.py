"""
ASVspoof partition handling, in one place.

Why this file exists
    The selection logic lived inside slice_and_probe.py. Exporting an audio
    subset for Colab has to pick EXACTLY the same utterances the run would pick,
    and the only way to guarantee that is to share the code rather than
    reimplement it. Three divergent feature implementations is a mistake this
    repository already made once.
"""

from __future__ import annotations

import os

import numpy as np

from phonophase.config import load_config, resolve_path

# train ships as .trn, dev and eval as .trl. Hardcoding .trn is why the original
# slicer could never read a held out partition.
PROTOCOL_SUFFIX = {"train": "trn", "dev": "trl", "eval": "trl"}


def partition_paths(config: dict, subset: str) -> tuple[str, str]:
    """Return (audio_dir, protocol_path) for a partition."""
    # audio_train is data/LA/ASVspoof2019_LA_train/flac, so the corpus root is
    # two levels up. One dirname leaves the partition folder and every path
    # built from it doubles, which is a bug this code shipped once already.
    base = resolve_path(
        os.path.dirname(os.path.dirname(config["paths"]["audio_train"].rstrip("/")))
    )
    audio_dir = os.path.join(base, f"ASVspoof2019_LA_{subset}", "flac")
    protocol = os.path.join(
        base, "ASVspoof2019_LA_cm_protocols",
        f"ASVspoof2019.LA.cm.{subset}.{PROTOCOL_SUFFIX[subset]}.txt",
    )
    return audio_dir, protocol


def read_protocol(path: str) -> list[tuple[str, str, str]]:
    """Return (utterance_id, attack_id, label) per line."""
    rows = []
    with open(path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 5:
                rows.append((parts[1], parts[3], parts[4]))
    return rows


def select_balanced(rows, n_bonafide: int, n_spoof: int, seed: int):
    """
    Balanced selection, spoof stratified evenly across attack types so one
    synthesis system cannot dominate.

    Deterministic given the seed. Both the exporter and the run call this, so
    the exported audio is exactly the audio the run will read.
    """
    rng = np.random.default_rng(seed)

    bona = [r for r in rows if r[2] == "bonafide"]
    spoof = [r for r in rows if r[2] == "spoof"]

    chosen = [bona[i] for i in rng.permutation(len(bona))[:n_bonafide]]

    by_attack: dict[str, list] = {}
    for r in spoof:
        by_attack.setdefault(r[1], []).append(r)

    quota = max(1, n_spoof // max(1, len(by_attack)))
    for attack in sorted(by_attack):
        pool = by_attack[attack]
        chosen += [pool[i] for i in rng.permutation(len(pool))[:quota]]

    return chosen
