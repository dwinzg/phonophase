# Pre-registration

> **OUTCOME, 2026-09-18. H1 SUPPORTED.** All four falsification conditions
> passed on the held out eval partition, at full power, on both segmentations.
> Details at the end of this document, including a heterogeneity caveat that
> qualifies the result.


Written 2026-09-18, **before** any additional data was sliced or examined.

Registered because the exploratory statistic below was chosen after seeing the
data. Writing the test down first is the only thing that separates a confirmed
finding from a story fitted to noise, and the phase 0 correction in
`docs/PLAN.md` is a live demonstration of how easily that happens here.

## Hypothesis

**H1.** `mgd_hf_std`, the standard deviation of Modified Group Delay over the
upper half of the spectrum, separates bonafide from spoofed speech more
strongly in **sonorants** (nasals, approximants, steady vowels) than in
**obstruents** (fricatives, plosives), with bonafide showing the higher value.

**Mechanism.** Sonorants are voiced and periodic and carry coherent harmonic
phase structure that a vocoder has to reconstruct. Obstruents are aperiodic, so
phase is close to random in genuine speech too and there is little structure for
a synthesis artifact to disturb.

**H2, conditional on H1.** If the artifact is articulatory rather than
linguistic, H1 holds in a language absent from training.

## Primary test

Paired bootstrap on folded AUC, `mgd_hf_std` against `mfcc_frame_spread`,
clustered by source utterance, on the ASVspoof 2019 LA **eval** partition
(attacks A07 to A19, none seen in training).

H1 is supported if the sonorant mean advantage exceeds the obstruent mean
advantage with a 95% interval excluding zero.

## Falsification, stated in advance

H1 is rejected if any of the following holds.

The sonorant advantage interval includes zero on eval.

The ordering reverses, so obstruents show the larger advantage.

`mgd_hf_std` fails to clear chance in at least two of the three sonorant
categories under a cluster bootstrap.

The effect is present on train attacks A01 to A06 but absent on eval, which
would make it a property of those six synthesis systems rather than of speech.

## Required sample size

At least **200 bonafide source utterances per phoneme category**, not slices.
The pilot had roughly 20, and at that size the effect does not replicate across
a split by utterance. See below.

## Evidence so far, and why it is not enough

On the pilot, 300 source utterances, after duration control, FDR across 42
tests and a bootstrap clustered by utterance, `mgd_hf_std` was the strongest
statistic in all three sonorant categories with intervals clear of chance.

It then failed a split-half replication, split by utterance.

| Category | half A | half B | replicates |
|---|---|---|---|
| NASALS | 0.670 [0.462, 0.818] | 0.738 [0.628, 0.828] | no |
| APPROXIMANTS | 0.747 [0.633, 0.848] | 0.624 [0.334, 0.844] | no |
| VOWELS_STEADY | 0.657 [0.544, 0.771] | 0.748 [0.631, 0.839] | yes |
| FRICATIVES | 0.650 [0.549, 0.742] | 0.539 [0.466, 0.605] | no |
| PLOSIVES | 0.559 [0.459, 0.700] | 0.537 [0.437, 0.679] | no |

One of five replicates. The point estimates stay directionally consistent and
similar in size, so this reads as insufficient power rather than a reversed
effect, but it is not evidence for H1 and must not be reported as such.

Each half holds roughly ten bonafide utterances per category. Nothing can be
concluded at that size, and the binding constraint is bonafide **utterances**,
not slices.

## Confound control, run 2026-09-18 before the confirmatory test

`mgd_hf_std` could plausibly have been a proxy for high-frequency energy
differences arising from how bonafide and spoofed audio were produced and
stored, which would have nothing to do with articulation. It is not.

| Category | mgd_hf_std | HF energy alone | mgd_hf_std after removing HF energy and tilt |
|---|---|---|---|
| NASALS | 0.702 | 0.533, crosses chance | 0.699 [0.594, 0.792] |
| APPROXIMANTS | 0.713 | 0.613 | 0.651 [0.574, 0.727] |
| VOWELS_STEADY | 0.700 | 0.580, crosses chance | 0.675 [0.592, 0.749] |

Correlation with HF energy is 0.50 to 0.66, and the separation barely moves
once energy and spectral tilt are partialled out. All bootstraps clustered by
source utterance.

The hypothesis is therefore confound controlled, cluster robust and FDR
surviving, and fails only on power. It is not refuted.

## Additional risk introduced by the segmentation fix

The effect above was measured on fixed 60ms windows. The slicer now emits real
phoneme extents, so the confirmatory run changes the data as well as the sample
size. If the effect vanishes, that is confounded between the two changes. Run
the fixed-window variant once at scale as well, so the two are separable.

## Known limitation of the current segmentation

Slices are fixed 60ms windows centred on a CTC peak frame, not phoneme
boundaries. Long phonemes are truncated and short ones carry neighbouring
context. Any phoneme conditioned claim is weakened by this and it is fixed
before the confirmatory run.

---

# Outcome

Run 2026-09-18 on ASVspoof 2019 LA **eval**, attacks A07 to A19, none seen in
training. 2,995 utterances, 137,992 slices, streamed from the ODC-BY HuggingFace
mirror. Bootstraps clustered by source utterance throughout.

## Result

| Condition, fixed in advance | seg=run | seg=window |
|---|---|---|
| 1. sonorant advantage excludes zero and is positive | PASS, +0.048 [+0.039, +0.057] | PASS, +0.046 [+0.037, +0.056] |
| 2. sonorant advantage exceeds obstruent | PASS, +0.048 vs −0.015 | PASS, +0.046 vs −0.017 |
| 3. phase clears chance in 2 of 3 sonorants | PASS, 3 of 3 | PASS, 3 of 3 |
| 4. 200+ bonafide utterances per sonorant category | PASS, 1344 / 1211 / 1493 | PASS |

Per category separation of `mgd_hf_std`, seg=run.

| Category | separation | 95% CI | group |
|---|---|---|---|
| NASALS | 0.599 | [0.581, 0.616] | sonorant |
| VOWELS_MOVING | 0.581 | [0.561, 0.603] | not pre-registered |
| APPROXIMANTS | 0.580 | [0.560, 0.599] | sonorant |
| VOWELS_STEADY | 0.572 | [0.557, 0.587] | sonorant |
| FRICATIVES | 0.548 | [0.537, 0.559] | obstruent |
| AFFRICATES | 0.536 | [0.496, 0.574] | obstruent |
| PLOSIVES | 0.511 | [0.499, 0.522] | obstruent |

The two segmentations agree to within 0.002, so the effect is not an artifact
of how the audio was cut. That control was built in precisely because the pilot
effect was measured on fixed windows.

## What this does not say

**The effect is small.** Separation runs 0.51 to 0.60 and the sonorant advantage
is 0.048. This is a statement about where phase information concentrates, not a
detector. A single statistic at 0.60 is nowhere near useful on its own.

**It is heterogeneous across attacks.** Scoring each attack against the full
bonafide pool, the sonorant advantage holds in **8 of 13** attacks, mean +0.031,
ranging from +0.183 on A14 to −0.109 on A08. It reverses on A08, A13, A16 and
A19. So this is a tendency across synthesis systems, not a property of all of
them, and pooling hides that. Which systems break the pattern, and why, is the
obvious next question and a mechanistic lead.

**VOWELS_MOVING was not in the pre-registered sonorant group** but lands at
0.581, squarely with the sonorants. Diphthongs are phonetically sonorants, so
this is consistent supporting evidence, but it was not part of the test and
must not be folded in after the fact.

**One language, one corpus.** H2 is untested. If the mechanism is articulatory
rather than linguistic it should replicate in a language absent from training.
That is the test that decides whether this is interesting or a curiosity.
