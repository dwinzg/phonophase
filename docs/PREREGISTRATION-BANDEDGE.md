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
