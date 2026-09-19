# PhonoPhase

**Where does the phase artifact of speech synthesis live, and is that location a
property of articulation rather than of language?**

Speech synthesis reproduces what a voice sounds like far more readily than it
reproduces the phase structure a real vocal tract produces. The residue is
detectable, but it is not spread evenly through speech. This project measures
where it concentrates, in articulatory terms, and tests whether that map holds
in languages absent from training.

This is a measurement study. It is not a detector, and it does not compete with
one. See `docs/SCOPE.md`.

## The finding so far

Pre-registered in `docs/PREREGISTRATION.md` before the data was touched, then
run on the ASVspoof 2019 LA **eval** partition, attacks A07 to A19, none seen in
training. 2,995 utterances, 137,992 slices, bootstraps clustered by source
utterance.

**Dispersion of Modified Group Delay over the upper spectrum separates bonafide
from spoofed speech more strongly in sonorants than in obstruents.**

| Category | separation | 95% CI |
|---|---|---|
| NASALS | 0.599 | [0.581, 0.616] |
| VOWELS_MOVING | 0.581 | [0.561, 0.603] |
| APPROXIMANTS | 0.580 | [0.560, 0.599] |
| VOWELS_STEADY | 0.572 | [0.557, 0.587] |
| FRICATIVES | 0.548 | [0.537, 0.559] |
| AFFRICATES | 0.536 | [0.496, 0.574] |
| PLOSIVES | 0.511 | [0.499, 0.522] |

Sonorant advantage **+0.048 [+0.039, +0.057]** against obstruent −0.015. All
four falsification conditions passed at 6 to 7 times the required power.

The result agrees to within 0.002 on both segmentations, which is worth less
than it sounds and is stated here rather than left to look impressive. The two
segmentations are largely the same data. 73 percent of slices are numerically
identical between them, the statistics correlate at 0.988, and 89 percent of
`seg=run` slices are a single frame, because CTC argmax is peaky and the slicer
has a 60 ms floor. Their agreement rules out a coding difference between the two
paths. It does not show the finding survives a different way of cutting the
audio, and forced alignment would be the test that does.

There is a physical account. Sonorants are voiced and periodic and carry
coherent harmonic phase that a vocoder must reconstruct. Obstruents are
aperiodic, so phase is close to random in genuine speech too and there is little
structure for an artifact to disturb.

### Three things it does not say

**It is small.** Separation runs 0.51 to 0.60. This locates phase information,
it does not detect anything.

**It is heterogeneous across attacks.** Scored per attack with
`scripts/attack_analysis.py`, the asymmetry that H1 states, the sonorant
advantage minus the obstruent one, is positive on **12 of 13** attacks and clear
of zero on 8. It runs from +0.218 on A09, which is Vocaine, to −0.079 on A19,
which is the one genuine reversal. A tendency across synthesis systems rather
than a property of all of them, and the exceptions are more interesting than the
average.

An earlier version of this section reported the effect as reversing on four
attacks. That was a different quantity, whether phase beats magnitude in
sonorants at all, which H1 does not claim. A08 and A13 are negative on that
quantity and clearly positive on the asymmetry: phase loses to magnitude
everywhere on those attacks and loses less in sonorants. The correction is
recorded rather than quietly applied, because the four reversals figure was
treated as a serious threat to the mechanism for a week and is the kind of
number that outlives its source.

**One language.** Whether it holds elsewhere is the open question.

## What comes next

The seven category grouping above is too coarse and too English. It covers 41 of
the 387 phone symbols the aligner emits, discarding aspiration, breathy voice,
nasalisation, length and retroflexion, which are exactly the contrasts that
distinguish languages.

`phonophase.phonology` replaces it with PanPhon's 24 articulatory features,
raising coverage to 296 phones. Features are universal by construction, so the
comparison stays well defined between languages whose inventories do not
overlap.

That turns the open question into something sharper and more useful. Regress
separation on articulatory features, find which features carry the signal, and
use that to predict from a language's phone inventory how much signal to expect
**before any labelled data exists in it**. Then test at both extremes of that
prediction.

Full plan in `docs/PLAN.md`. The design threat to watch is in `docs/SCOPE.md`,
and it is real. Different languages are synthesised by different systems, so a
language effect and a vocoder effect are confounded unless the comparison is
made within a single multilingual synthesiser.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # instrument and tests
pip install -e ".[align]"      # adds transformers and datasets for the GPU pass
```

Install torch first for your platform. It is deliberately not a dependency,
because the right build is platform specific and Colab already has one.

## Use

Measure on ASVspoof, streaming the corpus so nothing needs downloading.

```bash
python scripts/slice_and_probe.py \
    --hf-dataset SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA \
    --limit-bonafide 1500 --limit-spoof 1500 \
    --segmentations run,window --out results/eval_slices.csv

python scripts/analyse_slices.py --table results/eval_slices.csv --seg run
```

Measure on any other corpus, which is how the cross lingual test runs.

```bash
python scripts/make_manifest.py --root data/mlaad --layout lang_label \
    --corpus mlaad --out manifests/mlaad.csv
python scripts/slice_and_probe.py --manifest manifests/mlaad.csv --out results/mlaad.csv
python scripts/analyse_slices.py --table results/mlaad.csv --by-language
```

`notebooks/confirmatory_run.ipynb` runs the whole thing on a Colab GPU with two
uploads and no corpus download.

## Reproducing the finding

`results/eval_slices.csv.gz` is the data behind the table above.

```bash
gunzip -k results/eval_slices.csv.gz
python scripts/analyse_slices.py --table results/eval_slices.csv --seg run
python scripts/analyse_slices.py --table results/eval_slices.csv --seg window
```

Roughly thirty seconds, no GPU. It prints each falsification condition and a
verdict.

## Tests

```bash
./scripts/run_tests.sh     # no pytest required
pytest tests/              # after pip install -e ".[dev]"
```

## Layout

```
src/phonophase/    the instrument
scripts/           command line entry points
tests/             run standalone or under pytest
configs/           dataset and literature registries, ASVspoof protocols
docs/              scope, research design, pre-registration, plan
notebooks/         Colab confirmatory run
results/           the confirmed result and its data
```

## Method notes

Six failure modes were found and corrected while building this, every one of
them general rather than local to one codebase. They are documented in
`docs/SCOPE.md` and the instrument detects the last two directly.

Group delay computed with a signal wide ramp instead of a per frame one,
correlating 0.999999 with frame index. Raw group delay presented as Modified
Group Delay with no cepstral smoothing or compression. Per file normalisation
dividing out the dispersion being measured. Fixed width windows around CTC peaks
described as phoneme segments. Best of many statistics compared against best of
many others, turning zero of six into four of six. Slices from one utterance
bootstrapped as independent observations, when 129 came from 23.

## Citing

`CITATION.cff` is in the repository root.

## License

MIT, see `LICENSE`. Covers the source code only. The speech corpora retain their
own terms, recorded per entry in `configs/datasets.yaml`, and at least one is
non commercial.
