"""
Configuration loading.

Kept in the package so nothing here depends on src/utils.py, which is legacy
and due to be replaced by whatever comes back from the cluster.

Paths in config.yaml are relative to the repository root, so resolve_path
anchors them rather than relying on the caller's working directory. That was a
real failure mode. 05_train.py calls load_config("config.yaml") with a bare
relative path, which breaks whenever a job does not start in the root.
"""

from __future__ import annotations

import os

import yaml

# This file sits at <root>/src/phonophase/config.py, so the root is three levels up.
REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)

def load_config(path: str | None = None) -> dict:
    """Load config.yaml from the working directory, else from the repository root."""
    # A bare name is looked up in the working directory first, then at the
    # repository root, which only exists when installed from a clone.
    path = path or "config.yaml"
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(REPO_ROOT, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"configuration file not found at {path}")
    with open(path) as fh:
        return yaml.safe_load(fh)


def resolve_path(relative: str) -> str:
    """Anchor a config path to the repository root."""
    return relative if os.path.isabs(relative) else os.path.join(REPO_ROOT, relative)
