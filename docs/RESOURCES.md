# Resources, and what each idea rests on

Written 2026-09-19. A review rather than a catalogue. `configs/datasets.yaml`
and `configs/literature.yaml` are the registries and stay the source of truth
for fields like licence and URL. This page is for the judgements the registries
cannot hold: which resources actually unblock something, which claims rest on
what, and which ideas are currently floating free.

Every number here has a command behind it in the project log.

## How to read the status labels

| label | means |
|---|---|
| **confirmed** | pre-registered, run on held out data, falsification conditions passed |
| **supported** | measured with intervals, but exploratory or on data that formed it |
| **underpowered** | direction as predicted, test incapable of resolving it |
| **undecided** | measured, interval spans zero. Refutes nothing |
| **contradicted** | our own data points the other way |
| **speculative** | an idea with no measurement behind it yet |

Undecided and contradicted are different, and neither is the same as refuted.
The project has lost four claims to conflating them.

---

## 1. The review, briefly

**The instrument is in better shape than the evidence.** The measurement code is
corpus agnostic, tested, and detects the specific failure modes that destroyed
the predecessor project. The evidence is one confirmed asymmetry in one language
on one corpus, plus three exploratory results that qualify it.

**The binding constraint is not language coverage, it is genuine speech.** Every
synthetic corpus in the registry is spoof only. MLAAD, LRLspoof, SpeechFake, all
of them. The real side for H2 comes from M-AILABS, which has **2 to 5 speakers
per language**. So the cross linguistic test is limited by how few real voices
are available, not by how many languages have synthetic audio. Anyone planning
work here should start from that fact.

**Four registry entries still have unverified licences**, and the one verified
non commercial entry, MLAAD at CC-BY-NC 4.0, is stricter than this repository's
MIT. Anything trained or published from MLAAD inherits that restriction.

**Per attack and per feature, this data mostly cannot decide.** The pooled
results are solid because they pool. Broken down, most cells are undecided, and
two separate claims have already been built by counting signs across those
undecided cells.

---

## 2. Ideas, and what each one rests on

The point of this table is that an idea with nothing in its evidence column is
not thereby wrong. It is unfunded, and it should not appear in a write up as
though it were funded.

### The phase advantage over magnitude is not spread evenly through speech

**Status: confirmed.** The project's one confirmed result.

Pre-registered in `docs/PREREGISTRATION.md` before the data was touched, run on
ASVspoof 2019 LA eval, attacks A07 to A19, none seen in training. 2,995
utterances, 137,992 slices, bootstraps clustered by source utterance. Sonorant
advantage **+0.048 [+0.039, +0.057]** against obstruent −0.015, all four
falsification conditions passed at 6 to 7 times the required power.

**Falsifier:** the sonorant advantage interval includes zero, or the ordering
reverses. Neither occurred.

**What limits it:** one language, one corpus, separations of 0.51 to 0.60.

### It concentrates in sonorants specifically

**Status: superseded.** Do not repeat this wording.

T7 crossed sonority against voicing on the same data. The **ladder** is the
robust part: separation runs 0.576, 0.547, 0.515 across [+son,+voi],
[-son,+voi] and [-son,-voi], both steps clear of zero, on both segmentations and
under a silence control. Voiced obstruents sit between sonorants and voiceless
obstruents rather than with either.

The defensible wording is that separation tracks **how much of the segment
carries periodic, resonant structure**, which is a mechanism rather than a label.

**The decomposition into the two factors is NOT robust, and an earlier version
of this page said it was.** The baseline crossed test gives son +0.0323 and voi
+0.0278, which reads as two factors of similar size. Dropping the quietest
quarter of slices, the silence control that Zhang et al. 2023 motivates, gives
son +0.0523 and voi +0.0120 with an interval covering zero, on both
segmentations. And the honest reading of that is not "sonority after all": what
moves is cell [+son,-voi], which is 1,471 slices of the single phone /h/. Both
the original claim and its replacement rest on that one segment. Report the
ladder, not the decomposition.

**Why the wording matters beyond pedantry:** two 2026 papers rank obstruents
*above* sonorants. Stated as "the artifact concentrates in sonorants", the claim
contradicts published work. Stated as the phase advantage over magnitude, it
does not, because they measure magnitude domain discriminativeness and we
measure the difference between the domains. See section 4.

### Phase and magnitude carry complementary information

**Status: supported, and independently corroborated.** The best grounded idea
here.

Ours: the asymmetry above, which is by construction a phase minus magnitude
quantity. Theirs: MOSAIC (arXiv:2607.04314, July 2026) reports that phase and
biophonetic tokens activate on bonafide while spectral tokens activate on spoof,
at utterance level. And the two phoneme level papers find obstruents leading in
the magnitude domain, which is the other half of a complementarity.

**What is ours and what is not:** the direction is published, so it is
corroboration and not priority. What is not published is the phoneme
conditioning.

### The effect is governed by whether a system generates or inherits its phase

**Status: underpowered, direction as predicted.**

T6 grouped the 13 eval attacks by whether the output's fine phase originates in
a human recording, taken from the ASVspoof 2019 paper and fixed in
`configs/asvspoof2019_attacks.yaml` before the numbers were computed. Systems
inheriting natural phase average +0.006 asymmetry, systems generating it +0.069.

The exact permutation test over the 13 attack labels gives **p = 0.24**, with a
smallest attainable p of 0.009. Three attacks on one side is not a test. The
bootstrap interval excludes zero but holds the systems fixed and so cannot speak
about systems outside the corpus.

**Falsifier, registered:** `docs/PREREGISTRATION-H3.md`, on the dev partition
A01 to A06, which has never been sliced.

**Known counterexample inside the group:** A17 inherits natural phase and shows a
clear positive asymmetry, +0.070.

### The effect is consistent across synthesis systems

**Status: undecided, and previously overstated twice.**

Corrected per attack with FDR across the thirteen: 4 clear positive, 1 clear
negative (A19, q = 0.047, marginal), 8 undecided on `seg=run`. On `seg=window`
it is 7, 1 and 5, because three attacks sit near the boundary and cross it.
Stable on both: A08, A09, A14, A17 positive, A19 the only reversal, A16 and A18
undecided.

Two earlier claims were built from this cell by counting signs. "Reverses on 4 of
13" counted the sign of a different quantity. "Positive on 12 of 13" counted
uncorrected signs of point estimates. Both are recorded in the project log as
errors rather than quietly restated.

### Griffin-Lim should show the largest phase deficit

**Status: contradicted by our own data.**

The reasoning is that Griffin-Lim reconstructs phase from magnitude with no
learned phase model, so it should be the extreme case. ASVspoof contains exactly
that system: A11 is Tacotron 2 with Griffin-Lim. Its asymmetry is +0.034 with an
interval spanning zero, and it is beaten by A09 Vocaine at +0.218 and A14
STRAIGHT at +0.122, both classical parametric source filter vocoders.

`docs/H2-DESIGN.md` names MLAAD's `griffin_lim` a positive control on this
reasoning. That should not be relied on until T8 settles what the MLAAD system
actually is.

**Alternative worth registering rather than asserting:** the operative variable
may be parametric excitation modelling rather than phase reconstruction as such.

### The artifact is articulatory rather than linguistic

**Status: speculative.** This is the project's thesis and it has not been tested.

H1 establishes the asymmetry in English. Whether the same articulatory classes
carry it in a language absent from training is H2, and H2 has not been run.
Everything above is about English.

### The instrument is useful to people outside this project

**Status: speculative, plausible.**

It is corpus agnostic through the manifest route, has 73 tests on `main`, and
measures things that are generally hard to get right. Nobody outside this
project has used it, so the claim is untested.

---

## 3. Data

Registry: `configs/datasets.yaml`. Licence fields marked unverified there have
not been checked and must not be relied on.

| corpus | status | serves | licence | verified |
|---|---|---|---|---|
| ASVspoof 2019 LA | in use | H1, confirmed. H3, dev partition unsliced | Open Data Commons Attribution | yes |
| M-AILABS | in use | the genuine side for H2 | modified BSD 3 clause | yes |
| ASVspoof 2021 LA / DF | planned | replication under codec and channel effects | ODC-By v1.0 | yes |
| MLAAD | candidate | H2, 18 systems span 4 languages at 1000 utterances each | CC-BY-NC 4.0 | yes |
| LRLspoof | candidate | H2 at 66 languages, ungated | MIT on card, CC-BY-SA in paper | **no, they conflict** |
| IndicSynth | candidate | H2 for Indic languages including Bengali | unknown | no |
| In-the-Wild | candidate | robustness outside studio conditions | unknown | no |
| SpeechFake | planned | H2 at scale, 46 languages | unknown | no |
| XMAD-Bench | planned | an established protocol, so H2 is comparable | unknown | no |

### What actually constrains H2

**Every synthetic corpus above is spoof only.** The real side is the problem.

M-AILABS supplies it for four languages under a permissive licence, already
16 kHz, from a live host, with **2 to 5 speakers per language**. Measured from
the data rather than from documentation: fr 5, en 4, es 3, pl 2.

With two real voices in Polish, a measured real against synthetic separation
could be separating those two particular voices from TTS voices. The mitigation
is not speaker balancing, which is impossible at these counts. It is MLAAD's
`original_file` field, which names the specific M-AILABS recording each sample
was derived from, giving **utterance level pairing**: same speaker, same text,
same session. That pairing is the load bearing element of the H2 design, and any
corpus proposed as a replacement has to be checked for an equivalent before it
is adopted.

LRLspoof is larger and freer but has not been checked for such a pairing. Until
it has, adopting it would trade the strongest available control for a larger
language count.

### Attack metadata

`configs/asvspoof2019_attacks.yaml` records the waveform generator for all 19
ASVspoof 2019 LA attacks, transcribed from the database paper (arXiv:1911.01601)
and verified against the full text on 2026-09-19, plus one derived field for
whether the output's phase originates in a human recording. A13 is recorded as
`unknown` because the paper says it filters "the TTS voice" without naming which
TTS produced it, and that decides the answer.

---

## 4. Where this sits in the literature

Registry: `configs/literature.yaml`, two search passes, 2026-09-18 and
2026-09-19. Entries are marked `fetched` or `search_summary` so nobody mistakes
one for the other.

### Foreclosed

| claim | by |
|---|---|
| phoneme level features for detection, as a method | Zhang et al. 2025, arXiv:2412.12619, code public |
| phoneme level explainability of detectors | Taylor et al. 2026, arXiv:2607.08586 |
| phonetic explainability framing | arXiv:2606.15454, Computer Speech and Language |
| phoneme level analysis tied to speaker identity | Salvi et al. 2025, arXiv:2507.08626 |
| zero shot cross lingual detection | Kim et al. 2026, arXiv:2609.16458 |
| phase runs bonafide high, magnitude runs spoof high | MOSAIC, arXiv:2607.04314 |
| detector performance measured across many languages | LRLspoof, arXiv:2603.02364, 66 languages |
| lightweight and accurate as a contribution | AASIST-L, 0.99% EER at 85k parameters |

### Open, as far as four searches can tell

Conditioning a **phase domain** measure on phoneme class. Every phoneme level
paper found works on magnitude, SSL embeddings or handcrafted spectral features.
The one paper using phase does no phoneme conditioning.

Separating **sonority from voicing** as independent variables in a synthesis
artifact.

Explaining cross system heterogeneity of a phonetic effect by **how the
synthesiser produced its waveform**.

### The wording constraint this creates

*Phonetically Explainable* (arXiv:2606.15454) ranks stops, fricatives,
affricates, nasals and closures most discriminative and puts vowels and
semivowels lower. The emotional conditions paper reports fricatives and
diphthongs. **Both rank obstruents above sonorants.**

That is compatible with our result only because the quantities differ. Theirs is
magnitude domain discriminativeness; ours is the phase advantage over magnitude,
and one corpus can give both. The claim must therefore be stated as the
asymmetry between domains. Stated as "sonorants are where the artifact is", it
contradicts two published results and would be shot down on first reading.

### What this does not establish

Absence. Four searches cannot prove a negative, two entries are search summaries
rather than readings, and two of the four findings above post dated the previous
pass by weeks. The next pass should read MOSAIC and *Phonetically Explainable*
in full, and search specifically for phase or group delay conditioned on voicing.

---

## 5. The instrument, and what is reusable today

The part with value independent of whether H1 and H2 hold. Corpus agnostic, no
dependency on the rest of the package, MIT licensed.

### `phonophase.stats`

| function | answers |
|---|---|
| `bootstrap_auc_ci` | how uncertain is this AUC, with `groups=` for clustered data |
| `permutation_test_auc` | could this separation have arisen by chance |
| `paired_bootstrap_delta` | is A genuinely better than B **on the same samples** |
| `benjamini_hochberg` | how many of these many tests survive correction |
| `fast_auc` | Mann-Whitney AUC, exact to 1e-16 against sklearn, without the per call overhead |

The clustered bootstrap is the piece most worth lifting. Anyone measuring a
per segment statistic over speech has the pseudoreplication problem, usually
without knowing it.

### `phonophase.phonology`

IPA to PanPhon's 24 articulatory features, 296 of 387 phone symbols. Replaces a
seven category English shaped map that covered 41. **Known limitation:** symbols
containing digits are rejected, which drops 91 tone marked phones and excludes
Mandarin, Vietnamese and Thai entirely. Fix before any tonal language work, task
T4.

### `phonophase.features`

Modified Group Delay following Murthy and Gadde, with cepstral smoothing of the
denominator and sign preserving power compression. Written after finding that
the predecessor computed the group delay numerator with a ramp across the whole
signal rather than within each frame, producing a channel that correlated
0.999999 with frame index.

### `phonophase.metrics`

EER, t-DCF validated against the reference implementation and the ASV score
files shipped with ASVspoof 2019, per attack and per group breakdowns, and a
warning whenever AUC falls below 0.5, which is more often an inverted score
convention than genuine anti learning.

### Scripts

`slice_and_probe.py` aligns, slices and reduces in one GPU pass, emitting both
segmentations so a null can be attributed. `--manifest` takes any corpus.
`analyse_slices.py` runs the pre-registered procedure and distinguishes
undecided, not supported but underpowered, and rejected at adequate power.
`attack_analysis.py`, `sonority_voicing.py` and `feature_analysis.py` are the
exploratory analyses, each with tests built on data whose answer is known by
construction.

---

## 6. Nine ways this measurement goes wrong

Every one was made here, found here, and is a general failure mode rather than a
local mistake. This is the most portable thing the project has produced.

1. **Group delay with a signal wide ramp** instead of a per frame one. The
   channel correlated 0.999999 with frame index and was two orders of magnitude
   more clock than physics.
2. **Raw group delay presented as Modified Group Delay.** Without cepstral
   smoothing and compression the feature is dominated by spikes at magnitude
   zeros, which is exactly what the modification exists to suppress.
3. **Per file normalisation** dividing out the dispersion being measured. Per
   file z scoring forces every utterance to unit variance.
4. **Fixed width windows around CTC peaks described as phoneme segments.** They
   are not segments, and 89 percent of ours are a single 60 ms frame.
5. **Best of many compared against best of many.** Selecting the best phase
   statistic and comparing it to the best magnitude statistic turned zero of six
   into four of six.
6. **Slices from one utterance treated as independent.** 129 fricative slices
   came from 23 utterances. Effective n is utterances.
7. **The label convention assumed rather than read.** Here label 1 is SPOOF.
   Assuming otherwise inverts every direction in the output while leaving every
   magnitude correct, which is an error that survives review.
8. **Folding applied across groups.** `max(auc, 1-auc)` is right when comparing
   a phase statistic against a magnitude one. Across groups sharing a statistic
   it reflects draws about 0.5, inflating a near chance group and destroying the
   correlation a paired bootstrap exists to exploit.
9. **Counting signs instead of reading intervals.** Reducing a per cell estimate
   to its sign and counting how many are positive. This produced two separate
   wrong claims here, in opposite directions, and the count was not even stable
   under a defensible change of definition.

The instrument detects 6 and 9 directly. 7 is now read from the data and cross
checked. 5 is why the pre-registered pair exists.

---

## 7. What is missing

**A second language.** H2 is the project's thesis and has not been run.

**A real side with more than five speakers.** The binding constraint. No
registered corpus solves it. The paired design mitigates rather than removes it.

**Forced alignment.** Both current segmentations are approximately a 60 ms
window around a CTC peak, and they agree with each other because 73 percent of
slices are numerically identical. That agreement rules out a coding difference
between the two paths and nothing more. A real segmentation test needs forced
alignment.

**Tone.** 91 tone marked symbols are dropped, excluding every tonal language.

**Full readings of two papers.** MOSAIC and *Phonetically Explainable* are the
two closest to this work and both are currently fetched summaries.

**A use of the instrument by someone who did not write it.** Until then, "useful
to others" is a hope.
