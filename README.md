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
training. 2,995 utterances, 68,996 slices in each of two segmentations, bootstraps clustered by source
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

Sonorant advantage **+0.048 [+0.035, +0.061]** against obstruent −0.015, and
the difference between them, which is the quantity H1 asserts, **+0.063
[+0.045, +0.080]**. The first three falsification conditions passed with 6 to 7
times the required number of bonafide utterances, and the fourth, which compares partitions, does not
fire (`scripts/falsification_4.py`).

An earlier version of this page gave the first interval as [+0.039, +0.057].
That came from resampling each phoneme category independently, which ignores
that one utterance feeds all three, and was too narrow.

**How far that reaches depends on what is resampled**, and
`docs/ROBUSTNESS.md` has the table. The difference between sonorants and
obstruents excludes zero whether the bootstrap resamples utterances, speakers
or synthesis systems. The sonorant advantage on its own does not: resampling
the 13 synthesis systems puts its lower bound on zero, and the analysis script
reports that as undecided rather than printing whichever verdict the random
seed produced.

**It is also specific to this pair of statistics.** Against the three cepstral
statistics and high frequency energy the difference holds. Against spectral
tilt, a magnitude statistic with no phase content, it is +0.009 [-0.003,
+0.033], not distinguishable from zero: tilt shows nearly the same sonorant
over obstruent pattern. So this is not evidence that the pattern is unique to
phase.

The result agrees to within 0.002 on both segmentations, which is worth less
than it sounds and is stated here rather than left to look impressive. The two
segmentations are largely the same data. 73 percent of slices are numerically
identical between them, the statistics correlate at 0.988, and 89 percent of
`seg=run` slices are a single frame, because CTC argmax is peaky and the slicer
has a 60 ms floor. Their agreement rules out a coding difference between the two
paths. It does not show the finding survives a different way of cutting the
audio, and forced alignment would be the test that does.

A physical account has been proposed and is not established. Sonorants are
voiced and periodic and carry coherent harmonic phase that a vocoder must
reconstruct. Obstruents are aperiodic, so phase is close to random in genuine
speech too and there is little structure for an artifact to disturb. Two
measurements stand in the way of asserting it. A magnitude statistic, spectral
tilt, shows nearly the same pattern (`docs/ROBUSTNESS.md`). And a sample rate
conversion applied to genuine speech, with no synthesis involved, reproduces
the sonorant advantage in full, though not the difference between sonorants and
obstruents, which survives when both classes are low passed identically
(`docs/BAND-EDGE.md`). On four of the thirteen attacks, the ones whose audio
stops near 7.7 kHz, that difference does not survive, a split noticed after the
fact and not yet tested on other data.

### Three things it does not say

**It is small.** Separation runs 0.51 to 0.60. This locates phase information,
it does not detect anything.

**Per attack, this data mostly cannot decide.** Scored per attack with
`scripts/attack_analysis.py`, with intervals and FDR correction across the
thirteen, the asymmetry that H1 states is

| | count | attacks |
|---|---|---|
| clear and positive | 4 | A08, A09, A14, A17 |
| clear and negative, a reversal | 1 | A19, and only just, q = 0.047 |
| undecided | 8 | A07, A10, A11, A12, A13, A15, A16, A18 |

Undecided is not a reversal. Each attack contributes about 230 utterances, so
most per attack intervals are simply too wide to rank. The pooled result above
is strong because it pools.

The decided set is not stable either. On `seg=window` the same procedure gives
seven clear positive rather than four, because A07, A10 and A13 sit close to the
decision boundary and cross it. What holds on both segmentations is the core:
A08, A09, A14 and A17 clear positive, A19 the only clear negative, A16 and A18
undecided. Anything finer than that is being read off noise.

**An earlier version of this section reported the effect as reversing on four
attacks, and a later one as holding on twelve of thirteen. Both were counting
the sign of a point estimate, and neither is a result.** The four came from a
different quantity, whether phase beats magnitude in sonorants at all, which H1
does not claim: A08 and A13 are negative on that and clearly positive on the
asymmetry, meaning phase loses to magnitude everywhere on those attacks and
loses less in sonorants. The twelve came from counting uncorrected signs, which
is the same error in the opposite direction.

The count of reversals is not even stable under the definition. Averaging the
per category advantages, as the pre-registration specifies, one attack is
negative. Pooling the slices instead, three are, because A16 and A18 sit within
0.0005 of zero and change side. `scripts/attack_analysis.py` prints both, for
exactly this reason.

**One language.** Whether it holds elsewhere is the open question.

## What comes next

The seven category grouping above is too coarse and too English. It covers 41 of
the 387 phone symbols the aligner emits, discarding aspiration, breathy voice,
nasalisation, length and retroflexion, which are exactly the contrasts that
distinguish languages.

`phonophase.phonology` replaces it with PanPhon's 24 articulatory features,
raising coverage to 380 of the 388 symbols, tone marked vowels included. Features are universal by construction, so the
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
repository, the analysis runs in about a minute on CPU, and the
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
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --seg run
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --seg window
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster system
```

About a minute each on a laptop CPU, no GPU. It prints each falsification
condition and a verdict, and says UNDECIDED when a bootstrap bound is too close
to its threshold for the draw count to settle.

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
