"""Generate notebooks/confirmatory_run.ipynb. Kept as source, not JSON."""
import json, pathlib

def _lines(t):
    """
    nbformat stores source as a list of strings, and every element except the
    last must END WITH a newline. Jupyter joins them with "".join(), so
    stripping the newlines concatenates the lines into one unparseable line.
    """
    parts = t.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": _lines(t.strip())}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": _lines(t.strip("\n"))}

cells = [
md("""
# PADS confirmatory run

One job. Align and slice the ASVspoof 2019 LA **eval** partition, reduce every
phoneme slice to nine statistics, and bring back one CSV.

The hypothesis and all four falsification conditions were fixed in advance in
`docs/PREREGISTRATION.md`. Read it before running this. The analysis is
mechanical afterwards, which is the point.

**No wav files are written.** Slicing eval to disk would be roughly 145,000 tiny
files, slow locally and unusable on Drive. The probe only needs statistics, so
this aligns, slices in memory and appends rows to a table. Output is a few MB.

**Both segmentations in one pass.** `seg=run` is the true phoneme extent,
`seg=window` reproduces the old fixed 60ms window. Same audio, same alignment,
so the pilot result stays comparable and a null cannot be confounded between
sample size and the segmentation fix.

Set the runtime to GPU. Runtime, Change runtime type, T4 or better.
"""),
md("## 1. Runtime"),
code("""
import torch, platform
print("python", platform.python_version(), "| torch", torch.__version__)
print("cuda  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
else:
    print()
    print("STOP. This is a GPU job. On CPU it takes many hours.")
    print("Runtime, Change runtime type, Hardware accelerator, GPU.")
"""),
md("""
## 2. Code

Two routes. **Upload is the default**, because it needs nothing pushed.

**Route U, upload a bundle.** Run `./scripts/bundle_for_colab.sh` locally, which
writes a 160 KB `phonophase-colab.zip` of code only, no corpus and no virtualenv. Run
the cell below and pick that file. Use this while the repository is being held
local until a result verifies.

**Route G, clone from GitHub.** Only works once the tooling commits are pushed,
and needs a token with `repo` scope from https://github.com/settings/tokens
because the repository is private. Read with `getpass` so it never lands in
saved notebook output.
"""),
code("""
ROUTE = "U"     # U uploads a bundle, G clones from GitHub
REFRESH = False # True replaces code already in the runtime with a fresh upload

import os, shutil, subprocess

TARGET = "/content/Deepfake-PADS-Net"

# Reusing a session after a failed run is the easy way to keep running stale
# code. Rebuild the bundle locally after ANY change to scripts or src, then set
# REFRESH = True here.
if os.path.exists(TARGET) and REFRESH:
    shutil.rmtree(TARGET)
    print("removed the previous copy")

if os.path.exists(TARGET):
    print("code already present, NOT re-uploaded.")
    print("If you changed anything locally, set REFRESH = True and rerun.")
elif ROUTE == "U":
    from google.colab import files
    print("Pick phonophase-colab.zip, produced by ./scripts/bundle_for_colab.sh")
    uploaded = files.upload()
    name = next(iter(uploaded))
    os.makedirs(TARGET, exist_ok=True)
    !unzip -q -o "{name}" -d "{TARGET}"
    print(f"unpacked {name}")
else:
    import getpass
    token = getpass.getpass("GitHub token (hidden): ").strip()
    r = subprocess.run(["git", "clone",
                        f"https://{token}@github.com/dwinzg/Deepfake-PADS-Net.git",
                        TARGET], capture_output=True, text=True)
    # Never echo the output, it carries the token in the remote URL.
    print("clone ok" if r.returncode == 0 else "clone failed, check the token")
    del token

os.chdir(TARGET)
assert os.path.exists("scripts/slice_and_probe.py"), \\
    "slice_and_probe.py is missing. If you cloned, the tooling commits are not pushed yet. Use ROUTE = 'U'."

# Show exactly which build is in the runtime, so a stale bundle is obvious
# rather than silent.
if os.path.exists("BUNDLE_INFO"):
    print("--- bundle in this runtime ---")
    print(open("BUNDLE_INFO").read().strip())
    print("Compare against your local ./scripts/bundle_for_colab.sh output.")
else:
    print("no BUNDLE_INFO, this came from a clone rather than a bundle")
print("\\ncwd", os.getcwd())
"""),
code("""
# transformers is needed for wav2vec2 and is not a package dependency, since
# only the slicer uses it. torch is left alone so Colab's CUDA build survives.
!pip install -q -e .
!pip install -q transformers
!PYTHON=python bash scripts/run_tests.sh 2>&1 | grep -E "passed|failed"
"""),
md("""
## 3. Audio

**Route H, stream from a HuggingFace mirror. Default, nothing to upload.**

The original host, Edinburgh DataShare, rate limits by IP and returns **403** to
Colab, whose egress addresses are shared and usually already over quota. No
change to the request fixes that.

`SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA` mirrors the same 71,237 eval
utterances, redistributed under ODC-BY which permits it. HuggingFace is not rate
limited from Colab, so the whole run becomes self contained.

The mirror carries path, audio and label but not the attack id, so the 2.2 MB
protocol ships inside `phonophase-colab.zip` and supplies it. That also keeps the
utterance selection identical to every other route, since it feeds the same
`select_balanced`.

**Route S, an exported subset.** Use this if route H misbehaves. Run
`scripts/export_audio_subset.py` locally, which writes a 170 MB
`eval_subset.zip`, and drop it in Drive at `MyDrive/phonophase/`.

**Route D, the original download.** Kept for completeness. Expect a 403.
"""),
code("""
ROUTE_AUDIO = "H"   # H streams from HuggingFace, S uses a Drive subset, D downloads

import os

HF_DATASET = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
MANIFEST = None

if ROUTE_AUDIO == "H":
    !pip install -q datasets
    assert os.path.exists("configs/protocols/ASVspoof2019.LA.cm.eval.trl.txt"), \\
        "protocol missing from the bundle, rebuild phonophase-colab.zip"
    print("streaming from", HF_DATASET)

elif ROUTE_AUDIO == "S":
    from google.colab import drive
    drive.mount("/content/drive")

    ARCHIVE = "/content/drive/MyDrive/phonophase/eval_subset.zip"
    assert os.path.exists(ARCHIVE), (
        f"{ARCHIVE} not found. Run scripts/export_audio_subset.py locally and "
        "upload the result to Drive at MyDrive/phonophase/. Or switch to ROUTE_AUDIO = 'H'."
    )
    if not os.path.exists("/content/eval_subset/manifest.csv"):
        !unzip -q "{ARCHIVE}" -d /content/eval_subset
    MANIFEST = "/content/eval_subset/manifest.csv"
    print(open("/content/eval_subset/SUBSET_INFO").read().strip())

else:
    if not os.path.exists("data/LA"):
        !python scripts/fetch_data.py --dataset asvspoof2019_la --out data
"""),
md("""
## 4. The run

Roughly 3,000 utterances, spoof stratified evenly across attacks A07 to A19.
Every utterance yields many phoneme slices, so this clears the 200 bonafide
utterances per category the pre-registration requires.

Both segmentations come from the same alignment. `seg=run` is the true phoneme
extent, `seg=window` reproduces the legacy fixed 60ms window, so if the result
is null you can tell whether that is about sample size or about the
segmentation change.

Resumable. Rows flush periodically and processed utterances are skipped on
restart, so a reclaimed session costs a few utterances rather than the run. If
it stops, rerun the cell.
"""),
code("""
if ROUTE_AUDIO == "H":
    !python scripts/slice_and_probe.py \\
        --hf-dataset "{HF_DATASET}" \\
        --limit-bonafide 1500 --limit-spoof 1500 \\
        --segmentations run,window \\
        --out results/eval_slices.csv
elif MANIFEST:
    !python scripts/slice_and_probe.py \\
        --manifest "{MANIFEST}" \\
        --segmentations run,window \\
        --out results/eval_slices.csv
else:
    !python scripts/slice_and_probe.py \\
        --subset eval --limit-bonafide 1500 --limit-spoof 1500 \\
        --segmentations run,window \\
        --out results/eval_slices.csv
"""),
code("""
# Sanity check before downloading. Utterance counts are what matter, not slices.
import pandas as pd

df = pd.read_csv("results/eval_slices.csv")
print(f"{len(df):,} rows, {df.utterance_id.nunique():,} utterances, "
      f"{os.path.getsize('results/eval_slices.csv')/1e6:.1f} MB\\n")
for seg in df.seg.unique():
    d = df[df.seg == seg]
    print(f"--- seg={seg} ---")
    t = d.groupby("category").apply(
        lambda g: pd.Series({
            "slices": len(g),
            "utt_bonafide": g[g.label == 0].utterance_id.nunique(),
            "utt_spoof": g[g.label == 1].utterance_id.nunique(),
        }), include_groups=False)
    print(t.to_string(), "\\n")
print("Need at least 200 bonafide utterances per sonorant category, which is")
print("NASALS, APPROXIMANTS and VOWELS_STEADY. Short? Export a larger subset.")
"""),
md("""
## 5. Bring it home

One CSV. Nothing else. This is not a training run, so there are no checkpoints
and no model weights to retrieve.

Download the one CSV, then run the analysis locally.

```bash
python scripts/analyse_slices.py --table results/eval_slices.csv --seg run
python scripts/analyse_slices.py --table results/eval_slices.csv --seg window
```

It prints each falsification condition and a verdict. Whatever it says is the
answer. Both segmentations are reported so a null can be attributed correctly.
"""),
code("""
from google.colab import files
files.download("results/eval_slices.csv")
"""),
code("""
# Optional, also keep a copy in Drive in case the download is interrupted.
from google.colab import drive
import shutil, os

drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/pads", exist_ok=True)
shutil.copy("results/eval_slices.csv", "/content/drive/MyDrive/phonophase/eval_slices.csv")
print("saved to Drive")
"""),
md("""
## If it fails

That is a result. `docs/PREREGISTRATION.md` commits in advance to writing it up
as a negative rather than reframing it, and the analysis script prints that
reminder. Three findings in this project have already evaporated under
checking, so a fourth would say something real about how fragile phase based
phoneme level effects are at the sample sizes this literature typically uses.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU",
                   "colab": {"provenance": [], "gpuType": "T4", "toc_visible": True},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

pathlib.Path("notebooks/confirmatory_run.ipynb").write_text(json.dumps(nb, indent=1))
print(f"wrote notebooks/confirmatory_run.ipynb, {len(cells)} cells")
