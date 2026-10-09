# Pre-registration: the band edge split, on the development partition

Written 2026-10-08, **before** any development partition audio was read for
this purpose. No band edge descriptor has been computed on that partition, and
no treated version of its audio exists.

Registered because the split it tests was chosen after seeing the evaluation
partition. `docs/BAND-EDGE.md` found that four of the thirteen eval attacks
carry a hard band edge near 7.7 kHz, and that pooled over those four the
ASYMMETRY does not survive an identical low pass on both classes, while pooled
over the other nine it does. That is an observation made on the data that
suggested it. This document fixes a test of it on data that had no part in
forming it.

## Quantities

As in `docs/PREREGISTRATION.md` and its addendum. Phase statistic `mgd_hf_std`,
magnitude statistic `mfcc_frame_spread`, folded separation, seg=run.

- **LEVEL**: mean over NASALS, APPROXIMANTS, VOWELS_STEADY of phase separation
  minus magnitude separation.
- **ASYMMETRY**: LEVEL minus the same mean over FRICATIVES, PLOSIVES,
  AFFRICATES.

## Data

The ASVspoof 2019 LA development partition, attacks A01 to A06, restricted to
the utterances in `results/dev_slices.csv.gz`: 2,548 bonafide and 424 per
attack. Audio is read from a local copy of the corpus. Nothing is downloaded.

## What is already known about this partition, disclosed

- The unmodified LEVEL and ASYMMETRY on it, pooled and per attack, from
  `results/dev_slices.csv.gz`. Pooled ASYMMETRY is +0.132 [+0.118, +0.147].
- From the corpus paper, as recorded in `configs/asvspoof2019_attacks.yaml`: A04
  is the same algorithm as A16, and A06 the same as A19. A16 and A19 are two of
  the four hard edged eval attacks.
- Nothing about band edges, bandwidth or the effect of any treatment.

## Procedure, fixed

1. **Calibration.** Align and slice the audio with the existing pipeline. The
   recomputed table must reproduce `results/dev_slices.csv.gz` on the same
   utterances, with `mgd_hf_std` and `mfcc_frame_spread` agreeing to 1e-6 on
   every slice. If it does not, stop and report that. Nothing below is run.
2. **Classification**, from `scripts/band_edge.py describe` and `m1`, on the
   untreated audio, each attack against development bonafide:
   - **hard edged** if the AUC of the 40 dB bandwidth and the AUC of the edge
     slope are both within 0.05 of 0 or of 1;
   - **full band** if the AUC of the 40 dB bandwidth is between 0.30 and 0.70;
   - **intermediate** otherwise. Intermediate attacks enter neither group and
     are reported separately.
   These are the eval values: the four hard edged attacks had both AUCs below
   0.05, and the nine others had a bandwidth AUC of 0.44 to 0.52.
3. **Treatment.** `lp:6900>pcm16`, the identical linear phase low pass at
   6900 Hz then rounding to 16 bit, applied to every file of both classes. The
   cutoff is the eval one and is not re-derived from this partition.
4. **Estimator.** Bootstrap clustered by source utterance, one draw for all
   categories, 2,000 draws, seeds 0, 1 and 2, 95% percentile intervals, with
   the Monte Carlo band of `percentile_interval`. A bound inside that band of
   its threshold on any seed makes the prediction it decides undecided. The
   change under the treatment is taken inside the draw
   (`band_edge.py change`).

## Predictions

**B1, full band attacks, pooled against bonafide.** After the treatment the
ASYMMETRY is at least half of its unmodified value, with an interval excluding
zero.

- Supported if so on all three seeds.
- Refuted if it is below half of its unmodified value with an interval that
  includes zero.
- Undecided otherwise.

**B2, hard edged attacks, pooled against bonafide.** The treatment lowers the
ASYMMETRY: the same draw change is negative with an interval excluding zero,
its point estimate is at least 0.020 in size, and the ASYMMETRY after the
treatment is below half of its unmodified value. The eval change was -0.041.

- Supported if all three hold on all three seeds.
- Refuted if the ASYMMETRY after the treatment is at least half of its
  unmodified value with an interval excluding zero.
- Undecided otherwise.

**Preconditions.** A prediction is untestable, and reported as such, if its
group is empty, or if the group's unmodified ASYMMETRY is not positive with an
interval excluding zero, since then there is nothing to survive or to lose.

**Expectation, stated so it can be wrong.** A04 and A06 classify as hard edged
because they share algorithms with A16 and A19. The other four classify as full
band.

## What each outcome would mean

- **B1 and B2 both supported.** The split replicates. H1's ASYMMETRY is then a
  statement about attacks whose audio reaches Nyquist, and on hard edged
  attacks it is a top of band effect.
- **B1 supported, B2 refuted or untestable.** The eval split does not replicate
  and should not be carried into any statement of H1.
- **B1 refuted.** The ASYMMETRY on this partition does not survive
  equalisation even where there is no hard edge. That would count against the
  eval finding that it does, and both would be reported.
- Any undecided is reported as undecided.

## What this does not test

Whether H1 has a mechanism. `docs/BAND-EDGE.md` shows a resampler alone
reproduces H1's signature, and nothing here changes that. Per attack results
are descriptive: 424 utterances per attack resolve a change of roughly 0.02 at
best, and no per attack verdict is registered.

## Deviations

Any departure from the procedure above is recorded here, dated, with its
reason, before the result it affects is reported.

---

# Outcome

Run 2026-10-08, after the registration above was committed and pushed. Nothing
above this line was edited. No deviations.

**B1 is supported. B2 is refuted. The split found on the evaluation partition
does not replicate.**

## Calibration

The recomputed table reproduces `results/dev_slices.csv.gz` exactly: 5,092
utterances, 228,362 of 228,362 slices matched over both segmentations, and
`mgd_hf_std` and `mfcc_frame_spread` identical on every one.

## Classification

AUC of each descriptor, attack against development bonafide, spoof positive.

| attack | 40 dB bandwidth | edge slope | class |
|---|---|---|---|
| A01 | 0.001 [0.000, 0.002] | 0.000 [0.000, 0.000] | hard edged |
| A02 | 0.494 [0.484, 0.503] | 0.612 [0.587, 0.638] | full band |
| A03 | 0.484 [0.473, 0.495] | 0.519 [0.489, 0.550] | full band |
| A04 | 0.003 [0.000, 0.007] | 0.000 [0.000, 0.000] | hard edged |
| A05 | 0.500 [0.492, 0.507] | 0.281 [0.255, 0.307] | full band |
| A06 | 0.006 [0.002, 0.011] | 0.001 [0.000, 0.002] | hard edged |

No attack is intermediate. The stated expectation was right about A04 and A06
and wrong about A01, which is hard edged as well: median 40 dB bandwidth 7781
Hz against 8000 Hz for bonafide.

## The predictions

ASYMMETRY, pooled against bonafide, 2,000 draws clustered by utterance. Seed 0
shown. Seeds 1 and 2 move no bound by more than 0.003, and no bound is within
its Monte Carlo band of a threshold.

| group | unmodified | after the low pass | change, same draw |
|---|---|---|---|
| full band, A02 A03 A05 | +0.091 [+0.074, +0.107] | +0.089 [+0.073, +0.105] | -0.002 [-0.009, +0.006] |
| hard edged, A01 A04 A06 | +0.120 [+0.085, +0.151] | +0.105 [+0.083, +0.125] | -0.015 [-0.043, +0.014] |

**B1.** After the treatment the ASYMMETRY is +0.089, above half of +0.091, with
an interval excluding zero, on all three seeds. Supported.

**B2.** The precondition holds: the unmodified ASYMMETRY is positive with an
interval excluding zero. After the treatment it is +0.105, above half of
+0.120, with an interval excluding zero, on all three seeds. That is the
refutation condition as registered. The change is -0.015 with an interval that
includes zero, short of the 0.020 the prediction required.

## What follows

As fixed in advance for this outcome: the evaluation split does not replicate
and is not carried into any statement of H1.

**Corrected 2026-10-08, after an independent re-derivation.** This section
first said that the ASYMMETRY survives an identical low pass on both classes
whether or not the attacks carry a hard band edge. That reads more into the
verdict than it holds, in two ways.

The refutation of B2 rests on one attack. The hard edged pool is positive only
because A01 is in it. A04 and A06 pooled without it give -0.064 [-0.083, -0.043] unmodified,
which fails B2's precondition, so without A01 the prediction would have been
untestable, not refuted. The registered consequence is the same for both, so
the conclusion above does not depend on A01, but the word refuted does. After the low pass that pair gives +0.007 [-0.014, +0.026], a change of
+0.070 [+0.060, +0.080] on the same draw.

The treatment is not inert on this partition. Pooled over all six attacks the
ASYMMETRY is +0.132 [+0.118, +0.147] unmodified and +0.099 [+0.085, +0.113]
after, a change of -0.033 [-0.039, -0.027] on the same draw, which excludes
zero and is larger than either subgroup's change, because a folded statistic
of a pool is not the average of its parts. Each hard edged attack moves under
the low pass with an interval excluding zero, in opposite directions, and no
full band attack moves by more than 0.013 (table below). So band edge class
tracks whether the low pass moves the ASYMMETRY and not which way, and the
opposite signs cancel in the registered pool.

What the registered test does establish: the pooled prediction made from the
evaluation partition failed, and on the full band attacks the ASYMMETRY keeps
its size under equalisation.

## Sensitivity, from the re-derivation, not registered

Computed once, by independent code, and not reproduced with this repository's
scripts. Neither verdict changes under any of these. The size of the change
does.

| variation | full band change | hard edged change |
|---|---|---|
| as registered | -0.002 [-0.009, +0.006] | -0.015 [-0.043, +0.014] |
| affricates removed | -0.002 [-0.006, +0.002] | -0.030 [-0.041, -0.018] |
| low pass at 6500 Hz | +0.012 [+0.004, +0.021] | -0.003 [-0.030, +0.025] |
| low pass at 7300 Hz | +0.013 [+0.007, +0.019] | +0.008 [-0.018, +0.024] |

Affricates are 1,459 of 114,181 slices and carry a third of the obstruent mean.
The phase statistic after the low pass also depends on the filter design: a
different 6900 Hz filter of similar quality moves it by about a third of the
treatment's own effect. Read the changes above as signs and orders of
magnitude. An independently built spectral descriptor gives the same three classes at
40 dB and under any reasonable AUC threshold. At 30 dB it would class all three
hard edged attacks as intermediate, and at 60 dB A04 and A06, while this
repository's own 30 dB column still classes all three as hard edged. So the
grouping depends on how bandwidth is defined, within the registered rule's
limits.

## Per attack, descriptive only

No per attack verdict was registered, and these are reported because the pooled
hard edged group turns out to average over opposite signs. Seed 0.

| attack | class | unmodified | after the low pass | change, same draw |
|---|---|---|---|---|
| A01 | hard edged | +0.355 [+0.333, +0.377] | +0.295 [+0.274, +0.317] | -0.060 [-0.070, -0.050] |
| A04 | hard edged | -0.060 [-0.080, -0.028] | -0.012 [-0.034, +0.017] | +0.047 [+0.007, +0.073] |
| A06 | hard edged | -0.084 [-0.110, -0.057] | -0.014 [-0.039, +0.013] | +0.070 [+0.058, +0.083] |
| A02 | full band | +0.092 [+0.069, +0.113] | +0.096 [+0.073, +0.117] | +0.004 [-0.007, +0.015] |
| A03 | full band | +0.075 [+0.054, +0.097] | +0.079 [+0.058, +0.101] | +0.005 [-0.004, +0.013] |
| A05 | full band | +0.094 [+0.062, +0.122] | +0.080 [+0.046, +0.112] | -0.013 [-0.026, -0.000], not settled |

The pooled hard edged figure is carried by A01. A04 and A06 have a negative
ASYMMETRY, the reverse of H1, and the low pass moves both toward zero, to
intervals that include it. A04 is the
same algorithm as the evaluation partition's A16 and A06 the same as A19, and
A19 behaved the same way there: -0.079 unmodified, -0.024 after the low pass.
That is two partitions agreeing on one algorithm, noticed after the fact, and
it is a hypothesis.

## Commands

`$AUDIO` is a manifest of the 5,092 utterances, `$A` the tracked alignment
`results/band_edge_dev_alignment.csv.gz`.

```bash
B="python scripts/band_edge.py"
$B slice --manifest $AUDIO --alignment $A --segmentations run,window --out none.csv.gz
$B slice --manifest $AUDIO --alignment $A --segmentations run --treatment "lp:6900>pcm16" --out lp.csv.gz
$B describe --manifest $AUDIO --out ltas.csv.gz        # tracked as results/band_edge_dev_descriptors.csv.gz
$B m1 --descriptors results/band_edge_dev_descriptors.csv.gz
for G in "A02 A03 A05" "A01 A04 A06"; do
  $B r2 --table none.csv.gz --attacks $G --seeds 0 1 2
  $B r2 --table lp.csv.gz --attacks $G --seeds 0 1 2
  $B change --before none.csv.gz --after lp.csv.gz --attacks $G --seeds 0 1 2
done
```
