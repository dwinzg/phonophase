#!/bin/bash
# Build a small zip of the code for Colab, so nothing has to be pushed first.
#
# The confirmatory notebook can either clone from GitHub or unzip this. Cloning
# needs the tooling commits pushed. This does not, which matters while the repo
# is deliberately held local until a result verifies.
#
# Excludes the corpus, the virtualenv, checkpoints and results. Code only.
set -eu

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/phonophase-colab.zip}"

cd "$ROOT"
rm -f "$OUT"

# Stamp the bundle. Without this there is no way to tell inside Colab whether
# the code in the runtime is the code you just built, and re-running a stale
# bundle is a very easy mistake to make.
#
# Read the git state BEFORE writing the stamp. BUNDLE_INFO is itself an
# untracked file in the repository root, so writing it first makes the tree
# look dirty and the stamp reports on its own existence.
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
DIRTY="$(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"

{
    echo "built:  $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "commit: $COMMIT"
    echo "dirty:  $DIRTY"
} > BUNDLE_INFO

zip -q -r "$OUT" \
    src scripts tests configs docs notebooks \
    pyproject.toml config.yaml README.md BUNDLE_INFO \
    -x '*__pycache__*' '*.pyc' '*.DS_Store' '*.pth' '*.pt' \
       'scripts/*.zip' 'docs/assets/*'

rm -f BUNDLE_INFO
echo "wrote $OUT"
echo "size $(du -h "$OUT" | cut -f1)"
echo
echo "Upload this in the notebook's setup cell, or run:"
echo "  unzip -q phonophase-colab.zip -d phonophase"
