# PhonoPhase

**Where in speech does a deepfake differ from a real voice, and what does the
difference depend on?**

This is the first experiment in a line of work on speech deepfakes. Detectors
already tell real from synthetic speech very well on the data they were trained
on, and much less well on a new synthesiser, a new channel or a new language.
One reason is that nobody can say what they have learned. This project works on
that gap from the measurement side: instead of training a model, it takes one
simple, interpretable statistic and asks where in the speech signal it carries
information, sound class by sound class.

## What we are trying to do

1. **Locate the cue.** Cut every utterance into phones, group them by how they
   are produced (vowels, nasals, approximants, fricatives, stops, affricates),
   and measure how well a statistic separates real from synthetic speech inside
   each group.
2. **Test one specific idea.** The starting idea was that vocoders handle
   voiced, periodic sounds differently from noisy ones, so a phase statistic
   might help most in sonorants. That was written down as a hypothesis, with the conditions that
   would refute it, before the evaluation data was touched.
3. **Find out what causes it.** A pattern is only useful if it comes from the
   synthesis and not from the corpus: its sample rate history, its band edges,
   its speakers. Most of the work here is controls built to tell those apart.
4. **Then ask whether it transfers.** If the cue is tied to how sounds are
   produced and not to which language produces them, it should appear in a
   language the synthesiser was never trained on. That experiment is designed
   and not yet run.

Where it stands: step 1 and step 2 are done and the pattern is confirmed. Step
3 is where the first explanation lost its support, which is the main result so far. Step
4 waits on step 3.

## How this helps, and how to use it

This is not a detector and does not compete with one. The statistics here
separate the classes at 0.51 to 0.60 AUC, where a trained model (AASIST-L)
reaches an equal error rate under 1%. What it offers is different:

- **A way to ask what a cue depends on.** Point the pipeline at any corpus of
  real and synthetic speech and it reports, per sound class and per synthesis
  system, where a statistic separates the two, with intervals that respect the
  fact that thousands of phones come from a few hundred utterances.
- **A check for dataset shortcuts.** The controls here showed that a low
  quality sample rate conversion on real speech, with no synthesis at all,
  produces the same signature as the deepfakes do. Anyone training or evaluating a detector on
  ASVspoof-style data can run the same treatments (`scripts/band_edge.py`) to
  see whether a feature is reading the synthesis or the file's processing
  history.
- **Statistics you can lift out.** See [What is worth reusing](#what-is-worth-reusing).
- **A worked example of a result that was tested until part of it broke.** The
  registrations, the falsifiers and the errors caught are all in `docs/`, for
  anyone setting up a similar study.

The short report, with figures, is [`report/phonophase.pdf`](report/phonophase.pdf).
Scope and what is deliberately left out: `docs/SCOPE.md`.

## Terms

| term | meaning |
|---|---|
| sonorants, obstruents | nasals, approximants and steady vowels; fricatives, plosives and affricates |
| phase statistic, `mgd_hf_std` | spread of the Modified Group Delay over the upper half of the spectrum (4 to 8 kHz) in a slice |
| magnitude statistic, `mfcc_frame_spread` | spread of the cepstral (MFCC) frames in a slice; the registered comparison |
| separation | how well a statistic tells bonafide from spoof in one category, as AUC folded so 0.5 is chance and 1 is perfect |
| LEVEL | phase separation minus magnitude separation, averaged over the sonorant categories |
| ASYMMETRY | LEVEL minus the same quantity over the obstruent categories. The registered quantity of H1 |
| A01 to A19 | the ASVspoof 2019 LA synthesis systems; A01 to A06 in the train and development partitions, A07 to A19 in eval |
| hard edged | a system whose output stops abruptly near 7.7 kHz rather than rolling off |
| H1 | the ASYMMETRY is positive on eval. Registered before the data, confirmed |
| H2 | the same pattern in a language the synthesiser was not trained on. Designed, not run |
| H3 | the ASYMMETRY tracks whether a system generates or inherits its waveform phase. Registered, run, not supported |

## Findings

**1. The pattern is confirmed (H1).** Dispersion of Modified Group Delay over the
upper spectrum separates bonafide from spoofed speech more strongly in
sonorants than in obstruents, relative to a cepstral statistic. The registered
quantity, the ASYMMETRY, is **+0.063 [+0.045, +0.080]** on the eval
partition, attacks A07 to A19, with the interval clustered by utterance.

![Separation by category](report/figures/separation.png)

**2. Pooled, it is robust.** The difference excludes zero whether the bootstrap
resamples utterances, speakers or synthesis systems, and it survives an
identical low pass on both classes on the eval partition. System by system it
varies: one eval system and two development systems show the reverse, and on
the development partition the same low pass lowers the pooled difference by a
quarter.

**3. Its cause is not established.** The idea behind the test was a phase
artifact of vocoding. Two results count against asserting that:

- A low quality sample rate conversion applied to genuine speech, with no
  synthesis involved, reproduces the pattern at full size: +0.059 [+0.040,
  +0.077], same direction, same category order.
- A magnitude statistic with no phase content, spectral tilt, shows nearly the
  same pattern. Against it the difference is +0.009 [-0.003, +0.033].

![What reproduces it](report/figures/resampler.png)

That particular resampling artifact is ruled out as the cause here: low passing
both classes removes it and leaves the real difference at +0.061 [+0.042,
+0.078]. A difference in the upper band spectral envelope of synthetic speech,
with nothing particular to phase, is not ruled out. It is also not shown:
against upper band energy, another magnitude statistic, the difference stays
at +0.054 [+0.040, +0.069]. The control that could tell the two accounts apart
is registered and not yet run.

**4. What is not established.**

- The LEVEL on its own, +0.048, is not distinguishable from zero when the 13
  synthesis systems are resampled.
- The pattern belongs to upper band group delay dispersion. Whole band group
  delay statistics do not show it.
- A split by band edge that appeared on eval after the fact was registered and
  tested on the development partition, and did not replicate.
- H3, that the pattern follows whether a system generates or inherits its
  waveform phase, was registered and run on the development partition. Its
  direction held on all six development systems, and the test pooled over both
  partitions gave p = 0.021. The registration counts an algorithm that appears
  in both partitions only once, and with those removed p = 0.16. Not supported.
- Per system, the ASYMMETRY is clearly positive on 4 of the 13 eval systems,
  clearly negative on one (A19) and undecided on the other 8.

## Where to read next

**Current** documents state results as they stand. **Records** are kept for
the history of how the result was reached, and say so at the top.

| document | what it is | |
|---|---|---|
| [`report/phonophase.pdf`](report/phonophase.pdf) | the four page report with figures | current |
| [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md) | H1 as registered, its falsifiers and its outcome | current |
| [`docs/ROBUSTNESS.md`](docs/ROBUSTNESS.md) | how far H1 reaches, by clustering and by statistic | current |
| [`docs/BAND-EDGE.md`](docs/BAND-EDGE.md) | the resampling and band edge test that left the cause undecided | current |
| [`docs/PREREGISTRATION-BANDEDGE.md`](docs/PREREGISTRATION-BANDEDGE.md) | the band edge split, registered and run on dev | current |
| [`docs/PREREGISTRATION-H3.md`](docs/PREREGISTRATION-H3.md) | H3, registered, run, not supported | current |
| [`docs/PREREGISTRATION-GRIFFINLIM.md`](docs/PREREGISTRATION-GRIFFINLIM.md) | the positive control that can tell phase from envelope; not yet run | current |
| [`docs/PREREGISTRATION-ADDENDUM.md`](docs/PREREGISTRATION-ADDENDUM.md) | corrections to the H1 registration's wording and estimator | current |
| [`docs/SCOPE.md`](docs/SCOPE.md) | what is in and out of scope, and why | current |
| [`docs/RESOURCES.md`](docs/RESOURCES.md) | what each idea rests on, its status, and nine ways this measurement goes wrong | current |
| [`docs/WORKING-PROTOCOL.md`](docs/WORKING-PROTOCOL.md) | the method standard a claim here has to meet | current |
| [`results/PROVENANCE.md`](results/PROVENANCE.md) | where each shipped table came from | current |
| [`docs/H2-DESIGN.md`](docs/H2-DESIGN.md) | the cross lingual design draft | record |
| [`docs/RESEARCH.md`](docs/RESEARCH.md) | the opening design memo | record |
| [`docs/PLAN.md`](docs/PLAN.md) | the original phased plan | record |

## What comes next

The cause. A positive control is registered in
`docs/PREREGISTRATION-GRIFFINLIM.md`, on a system whose magnitude is a real
recording's and whose phase is wholly synthetic. It can separate a phase
account from a spectral envelope account, and it has not been run. Cross
lingual work (`docs/H2-DESIGN.md`) waits on it.

## Reproduce the result

The eval table ships in the repository. No GPU, no corpus, about a minute each.
Where every table came from, and its checksum, is in `results/PROVENANCE.md`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" -c constraints.txt   # Python 3.12; the versions that produced the numbers

python scripts/analyse_slices.py --table results/eval_slices.csv.gz
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster system
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --magnitude hf_lf_tilt
```

The script prints the primary test, each falsification condition and a verdict.
It says UNDECIDED when a bootstrap bound is too close to its threshold for the
draw count to settle, and gives no verdict on H1 for any setting that is not
the registered one.

## Use it on your own audio

```bash
pip install torch                 # your platform's build, see pyproject.toml
pip install -e ".[align]"

python scripts/make_manifest.py --root data/mycorpus --layout lang_label \
    --corpus mycorpus --out manifests/mycorpus.csv
python scripts/slice_and_probe.py --manifest manifests/mycorpus.csv --out results/mycorpus.csv
python scripts/analyse_slices.py --table results/mycorpus.csv --by-language
```

`slice_and_probe.py` aligns, slices and reduces each slice to nine statistics
in one pass. `scripts/band_edge.py` applies a treatment to the audio (a
resampling round trip, a low pass) with the alignment held fixed, which is how
the band edge tests were run. `notebooks/confirmatory_run.ipynb` runs the
confirmatory pass on a Colab GPU.

## What is worth reusing

- **`phonophase.stats`**, which has no dependency on the rest of the package.
  A cluster bootstrap in which one draw serves every category, a percentile
  interval that knows whether the draw count settles its sign, and paired
  comparisons of two statistics on the same samples. If you compute a statistic
  per phoneme, per frame or per window, your observations are not independent,
  and this is the fix.
- **The failure modes**, in `docs/RESOURCES.md`: nine ways this kind of
  measurement goes wrong, each one made and caught here.

## Tests

Each test builds data whose answer is known by construction.

```bash
pytest tests/              # after pip install -e ".[dev]"
./scripts/run_tests.sh     # no pytest required
```

## Layout

```
src/phonophase/    the instrument: features, statistics, phonology
scripts/           command line entry points
tests/             run standalone or under pytest
configs/           dataset and literature registries, ASVspoof protocol
docs/              scope, registrations, robustness and band edge records
report/            the short report and its figures
results/           slice tables and the numbers behind the figures
notebooks/         Colab confirmatory run
```

## Contributing

Issues and pull requests are welcome, and trying to break a claim is the most
useful kind: there are issue templates for a number that does not reproduce
and for a challenged claim. How to set up, the standard a claim has to meet
and how to send a change are in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Citing

See `CITATION.cff`.

## License

MIT for the source code, see `LICENSE`. The result tables and the protocol file
derive from ASVspoof 2019 and carry its attribution terms, see `NOTICE.md`. The
terms of every corpus this project reads are in `configs/datasets.yaml`.
