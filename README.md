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
four falsification conditions passed at 6 to 7 times the required power, and the
result is identical to within 0.002 on both segmentations.

There is a physical account. Sonorants are voiced and periodic and carry
coherent harmonic phase that a vocoder must reconstruct. Obstruents are
aperiodic, so phase is close to random in genuine speech too and there is little
structure for an artifact to disturb.

### Three things it does not say

**It is small.** Separation runs 0.51 to 0.60. This locates phase information,
it does not detect anything.

**It is heterogeneous across attacks.** Scored per attack, the sonorant
advantage holds in **8 of 13**, from +0.183 on A14 to −0.109 on A08, reversing
on A08, A13, A16 and A19. A tendency across synthesis systems, not a property of
all of them, and the exceptions are more interesting than the average.

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

## Who this is for

Four uses, ordered by how ready each one is. Nothing here is a detector, and
none of it competes with AASIST-L at 0.99% EER on 85,000 parameters.

**You need cluster robust statistics for a per segment speech measurement.**
`phonophase.stats` has no dependency on the rest of the package and is the piece
most worth lifting. If you compute a statistic per phoneme, per frame or per
window, your observations are not independent: many come from one utterance and
share a speaker, a channel and, for synthetic audio, one synthesis system.
Bootstrapping them as independent understates the interval by roughly the square
root of slices per utterance. In our pilot, 129 slices came from 23 utterances.
`bootstrap_auc_ci(..., groups=utterance_id)` is the fix, and
`paired_bootstrap_delta` is the right way to compare two statistics measured on
the same samples.

**You want to know where a signal concentrates across phoneme classes, in your
own corpus.** `scripts/make_manifest.py` builds a manifest from common directory
layouts, `slice_and_probe.py --manifest` aligns and reduces in one GPU pass
emitting both segmentations, and `analyse_slices.py` reports per class with
clustered intervals. Language is carried through, so `--by-language` works. None
of this is specific to deepfakes: it needs a per segment statistic and a class
label, whatever they mean.

**You are checking anti-spoofing work, yours or someone else's, for known
failure modes.** `docs/RESOURCES.md` lists nine ways this measurement goes
wrong, every one of them made and caught here, with what it looked like. Two are
detected by the instrument directly. It is worth reading even if you never touch
phase.

**You want to check or extend the finding itself.** The eval table ships in the
repository, the analysis runs in about thirty seconds on CPU, and the
falsification conditions were written down before the data was touched. Trying
to break it is a welcome use and is easier than reproducing it.

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

## Contributing

Issues and pull requests are welcome. The project has an unusual bar, described
below, and it is worth reading first because it is the reason anything here can
be trusted.

### What would help most

**Run the instrument on a corpus we do not have.** The open question is whether
the effect is a property of articulation or of English, and it is falsifiable in
one additional language. Two properties make a corpus especially useful: a
phone inventory that breaks the sonority and voicing collinearity, for example a
sizeable voiceless sonorant inventory, and more than five genuine speakers,
which every corpus available to us currently fails.

**Try to break a claim.** The falsifiers are written down in
`docs/PREREGISTRATION.md` and `docs/PREREGISTRATION-H3.md`. A clean negative is
a contribution and gets written up as one, not reframed.

**Tone marked phones.** `phonophase.phonology` rejects any symbol containing a
digit, which drops 91 symbols and excludes Mandarin, Vietnamese and Thai
entirely. Stripping the tone digit, parsing the base vowel and carrying tone as
a separate covariate would open every tonal language.

**Forced alignment.** Both current segmentations are approximately a 60 ms
window around a CTC argmax peak, and 73% of slices are numerically identical
between them. A real test of segmentation sensitivity needs proper alignment.

### The rules a contribution has to meet

They exist because this project's predecessor produced headline numbers nobody,
including their author, could reproduce.

- **Every claim ships with the command that produced it and its output.** A
  result that cannot be reproduced from what is written is not recorded.
- **Pre-registration is binding.** Do not change a primary test after seeing
  results. Register a new hypothesis and test it on data that had no part in
  forming it.
- **Undecided is not rejected.** A test with too little data has refuted
  nothing, and the analysis scripts distinguish the two.
- **Cluster every bootstrap by source utterance.** Always pass `groups=`.
- **No best of many comparisons.** Selecting the best phase statistic and
  comparing it against the best magnitude statistic once turned zero of six
  into four of six.
- **Report the magnitude, not only the interval.** Effects here run 0.51 to 0.60
  separation. Solid and meaningful are different things.

### How to send it

Branch from `main` as `task/T<id>-<slug>`, commit, push, and open a pull
request. Tests run with `./scripts/run_tests.sh`, no pytest required. New
analysis code is expected to carry tests built on data whose answer is known by
construction, which is how everything in `tests/` is written.

### What will be turned down

A claim with no command behind it. A result from a primary test that changed
after the data was seen. A p value from a test incapable of returning a small
one, without that stated. A count of how many cells came out positive, where the
cells have intervals that were never corrected or reported.

## Citing

`CITATION.cff` is in the repository root.

## License

MIT, see `LICENSE`. Covers the source code only. The speech corpora retain their
own terms, recorded per entry in `configs/datasets.yaml`, and at least one is
non commercial.
