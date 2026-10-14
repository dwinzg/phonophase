# Pre-registration

> **OUTCOME, 2026-09-18. H1 SUPPORTED.** All four falsification conditions
> passed (condition 4 decided 2026-09-19) on the held out eval partition, at full power, on both segmentations.
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

Note, 2026-10-14: rule R1 of `docs/BAND-EDGE.md` fired, and its registered
consequence is that no mechanism is stated for H1. The paragraph above is kept
as registered and is not a current claim.

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

Note, 2026-10-14: "confound controlled" refers to the HF energy and tilt
partialling above, on the pilot. Against `hf_lf_tilt` as the comparator in place
of `mfcc_frame_spread`, the eval ASYMMETRY is +0.009 [-0.003, +0.033], not
distinguishable from zero (`docs/ROBUSTNESS.md`). The confirmed statement is
about `mgd_hf_std` against cepstral dispersion, not against every magnitude
statistic.

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
| 1. sonorant advantage excludes zero and is positive | PASS, +0.048 [+0.035, +0.061] | PASS, +0.046 [+0.033, +0.060] |
| 2. sonorant advantage exceeds obstruent | PASS, +0.048 vs −0.015 | PASS, +0.046 vs −0.017 |
| 3. phase clears chance in 2 of 3 sonorants | PASS, 3 of 3 | PASS, 3 of 3 |
| 4. present on A01 to A06 but absent on eval | does not fire, see below | does not fire, see below |
| Required sample size, 200+ bonafide utterances per sonorant category | met, 1344 / 1211 / 1493 | met |

The intervals in row 1 were first recorded here as [+0.039, +0.057] and
[+0.037, +0.056]. Those came from resampling each category independently and
were too narrow; the ones shown resample utterances once per draw for all three
categories. Point estimates and verdicts are unchanged.

The analysis script has since gained two things. It evaluates the primary test
as worded above, the interval on the sonorant minus obstruent difference, which
on this table is +0.063 [+0.045, +0.080] on seg=run and +0.063 [+0.045, +0.080]
on seg=window, and passes. And it prints a third state, UNDECIDED, for a test
that turns on a bootstrap bound within Monte Carlo error of its threshold. No
bound is near its threshold under the registered settings, so every line the
script printed for this table before, it still prints.

The last row is the *Required sample size* requirement, which is a power
precondition and not one of the four falsification conditions. Until
2026-09-19 it was tabulated here as condition 4 and the registered condition 4
was not evaluated anywhere, so the run of 2026-09-18 decided three of the four
conditions rather than all four. The fourth is decided below.

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

Note, 2026-10-14: the sentence above overstates the control. On
`results/eval_slices.csv.gz`, 72.9 per cent of slices have numerically identical
`mgd_hf_std` under seg=run and seg=window, the two correlate at 0.988, and 89.4
per cent of seg=run slices are a single frame. The two segmentations are largely
the same data, so their agreement rules out a coding difference between the two
paths and little more. It is not evidence that the effect is independent of how
the audio was cut; that needs forced alignment.

## Condition 4, run 2026-09-19

> The effect is present on train attacks A01 to A06 but absent on eval, which
> would make it a property of those six synthesis systems rather than of speech.

A01 to A06 are the six systems the countermeasure training partition uses, and
the ASVspoof 2019 LA **development** partition uses the same six. Dev therefore
stands in for the train systems here. 2,548 bonafide and 2,544 spoof
utterances, 114,181 slices per segmentation, against eval's A07 to A19.

Two quantities, kept separate because "the sonorant advantage" names both and
conflating them has produced wrong claims in this project. The **LEVEL** is the
sonorant mean phase-minus-magnitude advantage on its own. The **ASYMMETRY** is
that minus the obstruent advantage, which is the quantity the primary test is
stated in. Bootstraps clustered by source utterance, 2,000 draws.

**These intervals come from the shared draw estimator**, `shared_cluster_draws`
in `src/phonophase/stats.py`, in which one resample of source utterances serves
every phoneme category in a draw. Row 1 of the outcome table above was first
recorded from the earlier design, which resampled each category independently
and averaged the results, discarding the dependence between categories fed by
the same utterance. It has since been re-issued from the shared draw, as the
note under that table says, so the eval figures below and the ones above now
agree. Point estimates are identical under both designs.

| seg | quantity | A01 to A06 | eval A07 to A19 |
|---|---|---|---|
| run | LEVEL | +0.0736 [+0.0644, +0.0828] | +0.0479 [+0.0347, +0.0609] |
| run | ASYMMETRY | +0.1322 [+0.1179, +0.1472] | +0.0630 [+0.0451, +0.0795] |
| window | LEVEL | +0.0700 [+0.0613, +0.0792] | +0.0464 [+0.0333, +0.0596] |
| window | ASYMMETRY | +0.1316 [+0.1172, +0.1465] | +0.0629 [+0.0450, +0.0801] |

The correction moves the two quantities in opposite directions, which is what
the estimator predicts. Against the independent draw design the four LEVEL
intervals above are 37 to 42 per cent wider, because the categories averaged
into the LEVEL share utterances and that dependence was being discarded. The
four ASYMMETRY intervals move by under 10 per cent and on eval they **narrow**,
by 4 per cent on seg=run, because sonorant and obstruent slices largely come
from the same utterances, so a shared draw lets their contribution cancel in
the difference. The registered statistic also folds, and the obstruent side
sits near zero, which spends some of the correlation a paired draw would
otherwise exploit.

Both quantities are present, in the predicted direction, with intervals clear
of zero, on **both** partitions and on both segmentations. The condition
requires presence on A01 to A06 *and* absence on eval. The second clause is
false, so the condition **does not fire** on either quantity. It would have
fired had the eval interval included zero, or excluded it from below, while the
A01 to A06 interval excluded it from above. Presence on the train systems alone
does not falsify H1, because H1 predicts the effect there too.

Reproduce with

    python scripts/falsification_4.py \
        --train-table results/dev_slices.csv.gz \
        --eval-table results/eval_slices.csv.gz

The estimator is imported from `scripts/analyse_slices.py` rather than
reimplemented, so the two cannot disagree by construction. Run on the whole dev
table it reproduces that script's own output for dev to every printed digit.
Its eval figures reproduce row 1 of the table further up, +0.048 and +0.046 with
the re-issued intervals.

**The effect is roughly twice as large on the six train systems**, 0.074 against
0.048 on the LEVEL and 0.132 against 0.063 on the ASYMMETRY. That is not what
condition 4 tests and it does not falsify anything, but it is evidence that the
size of the effect is partly a property of the synthesis systems.

How firm that is differs between the two quantities once the widths are right.
The ASYMMETRY intervals stay far apart, +0.1179 and above against +0.0795 and
below on seg=run. The LEVEL intervals now only just clear each other, by 0.0035
on seg=run and 0.0017 on seg=window, where under the pre-correction widths the
gap looked comfortable. No interval was computed on the dev-minus-eval
difference itself, and non-overlap of two separately estimated intervals is a
weaker statement than a test of that difference, so the LEVEL gap should be
treated as suggestive only. It should not be read as a measured attack
effect: the partitions differ in speakers and in sample size as well as in
attacks, dev pools six systems where eval pools thirteen, and the eval pooling
already hides a documented spread across attacks. Separating those is a
question for a design that varies attacks while holding speakers fixed.

## What this does not say

**Corrected 2026-10-09.** Two things in this section are superseded and are
left below as written. First, the per attack figures: they came from an
earlier estimator and mix the LEVEL with the ASYMMETRY. With the current code
the LEVEL is positive on 10 of 13 attacks, mean +0.029, from +0.141 on A14 to
-0.092 on A08, and negative on A08, A13 and A19, not on A16. The ASYMMETRY,
which is what H1 states, has a positive point estimate on 12 of 13, is clear
after correction on four of them, and is clearly negative only on A19
(`scripts/attack_analysis.py`). Signs counted across attacks are not a test. Second, the wording "where phase information
concentrates", "a mechanistic lead" and "if the mechanism is articulatory":
rule R1 of `docs/BAND-EDGE.md` fired, and its registered consequence is that
no mechanism is stated for H1. Against a spectral tilt comparator the
ASYMMETRY is +0.009 [-0.003, +0.033], so "phase" here means one group delay
statistic against one cepstral statistic.

> **Superseded, kept for the record** (marked 2026-10-14; see the correction
> of 2026-10-09 above).
>
> **The effect is small.** Separation runs 0.51 to 0.60 and the sonorant advantage
> is 0.048. This is a statement about where phase information concentrates, not a
> detector. A single statistic at 0.60 is nowhere near useful on its own.
>
> **It is heterogeneous across attacks.** Scoring each attack against the full
> bonafide pool, the sonorant advantage holds in **8 of 13** attacks, mean +0.031,
> ranging from +0.183 on A14 to −0.109 on A08. It reverses on A08, A13, A16 and
> A19. So this is a tendency across synthesis systems, not a property of all of
> them, and pooling hides that. Which systems break the pattern, and why, is the
> obvious next question and a mechanistic lead.

**VOWELS_MOVING was not in the pre-registered sonorant group** but lands at
0.581, squarely with the sonorants. Diphthongs are phonetically sonorants, so
this is consistent supporting evidence, but it was not part of the test and
must not be folded in after the fact.

**One language, one corpus.** H2 is untested. If the mechanism is articulatory
rather than linguistic it should replicate in a language absent from training.
That is the test that decides whether this is interesting or a curiosity.
