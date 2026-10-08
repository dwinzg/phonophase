# Band edges, resampling and H1

Run 2026-10-08 on the ASVspoof 2019 LA eval subset the confirmatory run used:
2,995 utterances, 1,500 bonafide and 115 from each of A07 to A19. All numbers
are seg=run. Tooling is `scripts/band_edge.py`, tests in
`tests/test_band_edge.py`.

## The question

`docs/PREREGISTRATION-GRIFFINLIM.md` records that on real speech, with no
synthesis anywhere, a 16 to 22.05 to 16 kHz round trip applied to one arm
separates the arms on `mgd_hf_std` at 0.576 folded and leaves
`mfcc_frame_spread` at 0.508, and that resampler choice alone gives 0.592
against 0.504. That is phase selective and inside H1's own range. The registered
confound control partials out high frequency energy and tilt, which is about
level, and an anti-alias filter is a group delay phenomenon at the band edge.

So: can a band edge or resampling difference between bonafide and spoofed audio
account for H1? H1 is the ASYMMETRY, the phase minus magnitude advantage in
sonorants minus the same in obstruents, +0.063 [+0.045, +0.080]. The LEVEL, the
sonorant advantage alone, is +0.048. Both are reported throughout and neither
is called "the effect".

## Decision rules, fixed before any intervention was run

**R1** fires if, for bonafide against the same bonafide after a 16 to 22.05 to
16 kHz round trip, the ASYMMETRY is positive with a 95% interval excluding
zero. Then a resampler reproduces H1's full signature.

**R2** fires if, after the identical treatment is applied to both classes of
the real comparison, the eval ASYMMETRY falls below half of the unmodified
value with a 95% interval that includes zero. R2 is quiet if it stays above
half with an interval excluding zero. Anything else is undecided.

**R3**, whether the per attack ASYMMETRY tracks the per attack band edge
mismatch, is descriptive only.

A mixed outcome is undecided and both numbers are given. A bound that changes
sign across seeds is undecided.

The rules were written down before any audio was read. They enter the
repository in the same commit as the results, so the repository itself cannot
show that.

Two things were fixed after M1 and before M3, because they depend on M1. The
low pass cutoff is 0.9 times the lowest group median 30 dB bandwidth, rounded
down to 100 Hz, which is 6900 Hz. The sub band for the redefined statistic is
4000 to 6500 Hz.

## Outcome against each rule

**R1: fires.** The rule names a 16 to 22.05 to 16 kHz round trip and no
resampler. Of the resamplers tried, the outcome depends on which, and on
whether the output is rounded back to 16 bit as every file in the corpus is.

| round trip, then 16 bit | LEVEL | ASYMMETRY, unpaired draw |
|---|---|---|
| soxr LQ | +0.055 [+0.043, +0.067] | **+0.059 [+0.040, +0.077]** |
| soxr HQ, librosa's default | -0.003 [-0.013, +0.012] | -0.028 [-0.045, +0.006] |
| `scipy.signal.resample_poly` | +0.009 | +0.000 [-0.016, +0.019] |
| for comparison, eval bonafide against spoof | +0.048 [+0.035, +0.061] | +0.063 [+0.045, +0.080] |

The unpaired draw is the estimator the registered analysis uses, so the rows
are comparable. With soxr LQ, whose passband starts to roll off near 6 kHz,
genuine speech against the same genuine speech reproduces the eval result in
size, in direction (the treated arm is lower on `mgd_hf_std` and higher on
`mfcc_frame_spread`, as the spoofed side is), and in category order (nasals
highest at 0.626, obstruents near zero advantage). Three seeds, no bound moves
by more than 0.001. With the two sharper resamplers there is no positive
ASYMMETRY once the output is 16 bit. Without the rounding soxr HQ gives +0.008
[+0.003, +0.013] under the joint draw; that compares float output with 16 bit
originals and is not a contrast the corpus contains.

So a band limiting step with no synthesis in it can produce H1's signature.
Whether one did, in the corpus, is what R2 asks. R1 fires and R2 is quiet,
which is the mixed outcome the rules call undecided.

**R2: quiet.** Half of the unmodified ASYMMETRY is +0.0315. After the identical
round trip on everything it is +0.047 [+0.028, +0.065] with soxr HQ and +0.062
[+0.044, +0.079] with resample_poly. After one identical low pass at 6900 Hz it
is +0.058 [+0.038, +0.076], and +0.061 [+0.042, +0.078] when the result is
rounded to 16 bit. Every variant stays above half with an interval excluding
zero, on three seeds, and no bound is near zero.

**R3: no tracking detected, at low power.** Across 13 attacks the largest
|rho| between ASYMMETRY and any band edge descriptor is 0.48, permutation
p = 0.10. With n = 13 only |rho| of 0.555 or more reaches p < 0.05.

**A finding the rules did not anticipate, selected after M1.** Four attacks,
A07, A08, A16 and A19, carry a hard band edge near 7.7 kHz that bonafide does
not have. Pooled, the ASYMMETRY against those four is +0.054 [+0.028, +0.081]
unmodified and +0.014 [-0.011, +0.041] after the low pass, a change of -0.041
[-0.052, -0.022]. Against the other nine it is +0.067 [+0.052, +0.086]
unmodified and +0.078 [+0.058, +0.096] after. That is one pooled contrast and
not a statement about each of the four. Taken singly only A07 loses its
ASYMMETRY with an interval excluding zero; A08 keeps it; A19's was negative and
moves toward zero. The movement is on the obstruent side and leans on the
affricate cell, 32 to 60 spoofed slices per attack. The split was chosen after
seeing M1 and M2, so this is exploratory and is not a test.

## What this changes about H1, and what it does not

**H1's signature is not specific to synthesis.** A low quality sample rate
round trip applied to genuine speech reproduces the LEVEL and the ASYMMETRY at
the size the eval comparison shows. The first rule was written to detect exactly
this and it fires. No articulatory or phase reconstruction mechanism can be
asserted for H1 on the strength of the eval result.

**The eval ASYMMETRY is nonetheless not that artifact, as far as these
equalisers can tell.** One identical low pass at 6900 Hz on both arms takes the
planted soxr LQ ASYMMETRY from +0.059 to +0.001 [-0.017, +0.018]. The same low
pass leaves the eval ASYMMETRY at +0.061 [+0.042, +0.078]. About a quarter of
the band the statistic reads is then stopband, so the sturdier figure is the
one restricted to 4000 to 6500 Hz, +0.046 [+0.028, +0.064], which still clears
half of the unmodified value. The two also behave
differently in LEVEL: under the low pass the planted difference grows to +0.130
in every category alike, and the eval LEVEL stays at +0.048. So whatever
separates bonafide from spoofed speech in the corpus, it is not a soxr LQ style
roll off, and it is not something above 6.9 kHz.

**What the low pass cannot rule out** is a difference in spectral shape below
its cutoff that is not a simple roll off. The nine attacks without a hard edge
are still separable from bonafide on edge slope, AUC 0.31 to 0.71. And spectral
tilt, a magnitude statistic, shows the same sonorant over obstruent pattern as
`mgd_hf_std` both before and after the low pass: with `hf_lf_tilt` as the
comparator the ASYMMETRY is +0.006 [-0.005, +0.032] on the low passed table
(`docs/ROBUSTNESS.md` has the unmodified figure). An account in which H1
reflects how the spectral envelope of synthetic speech differs in the upper
band, with nothing particular to phase, is consistent with everything here.

**Pooled over four attacks, the ASYMMETRY does not survive equalisation.** The
split was chosen after seeing the data, is not uniform within the four, and
depends on which attacks are pooled. And on these four the unmodified ASYMMETRY
was never a phase effect in sonorants: `mgd_hf_std` separates no better than
`mfcc_frame_spread` in any sonorant or obstruent category, and the ASYMMETRY is
positive only because the magnitude statistic separates obstruents better
still. It was a hypothesis for a new registration on a partition that had no
part in forming it.

**That registration has been run, and the split does not replicate.**
`docs/PREREGISTRATION-BANDEDGE.md` fixed the test on the development partition
before its audio was read. Three of its six attacks are hard edged. Pooled over
those three the ASYMMETRY is +0.120 unmodified and +0.105 [+0.083, +0.125]
after the same low pass, which is the registered refutation. Pooled over the
three full band attacks it is +0.091 and +0.089. So the split is not carried
into any statement of H1, and the ASYMMETRY survives equalisation on a second
partition.

Taken together: H1 stands as an observation made under pre-registration. Its
cause is undecided. A band limiting artifact of one kind reproduces it and is
ruled out for the eval comparison; a spectral envelope account is not ruled
out; an account specific to phase has no support that the envelope account does
not share.

## Calibration

The alignment was recomputed on CPU from the FLAC files and compared with
`results/eval_slices.csv.gz`, which was produced on a GPU from the HuggingFace
mirror.

| check | result |
|---|---|
| utterances | 2,995 in both, the same set |
| phone runs | 68,996 in both, 0 in one and not the other |
| slices, run and window | 137,992 in both, every one matched on utterance, phone, start frame and length |
| `mgd_hf_std`, largest absolute difference | 1.4e-09 |
| `mfcc_frame_spread`, largest absolute difference | 5.7e-14 |
| `analyse_slices.py` on the recomputed table | LEVEL +0.048 [+0.035, +0.061], ASYMMETRY +0.063 [+0.045, +0.080], identical to the published table to the digits printed |

`slice_and_probe.py` applies no normalisation to these statistics, per file or
per corpus: `slice_statistics` returns raw dispersions. There was nothing to
reproduce and nothing was added.

The archive holds 2,995 files, not 2,997.

## M1, where the band edges sit

Per file, from a Welch long term average spectrum, 1024 point segments.

- `bw30`, `bw40`: highest frequency within 30 or 40 dB of the mean 1 to 4 kHz
  level.
- `edge ratio`: mean power over 7 to 8 kHz against 4 to 7 kHz, in dB.
- `edge slope`: the steepest fall of the spectrum across any 500 Hz between
  4 and 8 kHz, in dB per kHz. Speech rolls off gradually and an anti-alias
  filter falls off a cliff, so this responds to a filter wherever its cutoff
  sits and does not depend on how much high frequency energy the talker
  produced. The first three only respond to a cutoff inside the band they
  were told to look at.

Group medians, and the AUC of each descriptor for bonafide against the attack
with a 95% bootstrap over utterances. AUC is signed, spoof positive, and not
folded, because it is compared across attacks.

| group | n | bw30 Hz | bw40 Hz | edge ratio dB | edge slope dB/kHz | AUC bw40 | AUC edge ratio | AUC edge slope |
|---|---|---|---|---|---|---|---|---|
| bonafide | 1500 | 8000 | 8000 | -5.6 | -20.3 | | | |
| A07 | 115 | 7719 | 7766 | -7.1 | -103.0 | 0.004 [0.001, 0.007] | 0.392 [0.344, 0.439] | 0.001 [0.000, 0.004] |
| A08 | 115 | 7672 | 7766 | -9.2 | -63.8 | 0.005 [0.002, 0.008] | 0.288 [0.236, 0.344] | 0.005 [0.001, 0.011] |
| A09 | 115 | 8000 | 8000 | -8.8 | -18.8 | 0.474 [0.447, 0.499] | 0.342 [0.287, 0.400] | 0.604 [0.547, 0.663] |
| A10 | 115 | 8000 | 8000 | -6.2 | -18.1 | 0.518 [0.513, 0.522] | 0.434 [0.380, 0.488] | 0.604 [0.554, 0.652] |
| A11 | 115 | 7984 | 8000 | -7.1 | -19.8 | 0.443 [0.410, 0.476] | 0.382 [0.333, 0.433] | 0.534 [0.482, 0.591] |
| A12 | 115 | 8000 | 8000 | -5.8 | -16.6 | 0.505 [0.487, 0.517] | 0.502 [0.444, 0.562] | 0.686 [0.635, 0.735] |
| A13 | 115 | 8000 | 8000 | -6.1 | -19.2 | 0.509 [0.494, 0.520] | 0.479 [0.416, 0.539] | 0.558 [0.508, 0.609] |
| A14 | 115 | 8000 | 8000 | -4.8 | -17.6 | 0.509 [0.494, 0.520] | 0.539 [0.491, 0.591] | 0.612 [0.556, 0.668] |
| A15 | 115 | 8000 | 8000 | -4.7 | -15.8 | 0.500 [0.481, 0.515] | 0.562 [0.513, 0.610] | 0.712 [0.663, 0.761] |
| A16 | 115 | 7719 | 7797 | -5.2 | -82.2 | 0.004 [0.001, 0.008] | 0.501 [0.448, 0.553] | 0.000 [0.000, 0.000] |
| A17 | 115 | 8000 | 8000 | -6.1 | -25.1 | 0.509 [0.495, 0.520] | 0.446 [0.406, 0.488] | 0.310 [0.261, 0.361] |
| A18 | 115 | 8000 | 8000 | -4.5 | -19.8 | 0.496 [0.476, 0.513] | 0.547 [0.489, 0.606] | 0.521 [0.476, 0.563] |
| A19 | 115 | 7766 | 7828 | -5.3 | -85.6 | 0.045 [0.022, 0.071] | 0.515 [0.469, 0.563] | 0.011 [0.001, 0.029] |

Bonafide reaches Nyquist with no measurable edge. A07, A08, A16 and A19 stop
between 7.6 and 7.8 kHz with a fall of 64 to 103 dB per kHz, and bandwidth
alone separates each of them from bonafide almost perfectly. The other nine
reach Nyquist, though the edge slope still separates several of them from
bonafide at 0.31 to 0.71, so no attack is matched to bonafide at the band edge.
The lowest group median is 7672 Hz at 30 dB and 7766 Hz at 40 dB. The lowest
single file is 4438 Hz at 30 dB, which is a quiet utterance and not a filter.

The full per file table is `results/band_edge_descriptors.csv.gz`. The 5 and
95 percent quantiles and the remaining descriptor columns are printed by the
`m1` command.

## M2, does the ASYMMETRY track the mismatch across attacks

Per attack against the shared bonafide pool, from
`attack_analysis.per_attack`, 2,000 draws. The last column is the same
ASYMMETRY after the 6900 Hz low pass and 16 bit rounding.

| attack | LEVEL | ASYMMETRY | ASYMMETRY after low pass |
|---|---|---|---|
| A07 | +0.022 [-0.019, +0.062] | +0.054 [+0.005, +0.093] | -0.001 [-0.044, +0.045] |
| A08 | -0.092 [-0.117, -0.032] | +0.060 [+0.018, +0.133] | +0.049 [+0.008, +0.103] |
| A09 | +0.125 [+0.091, +0.161] | +0.218 [+0.178, +0.259] | +0.241 [+0.202, +0.278] |
| A10 | +0.087 [+0.059, +0.108] | +0.031 [+0.002, +0.085] | +0.055 [+0.022, +0.095] |
| A11 | +0.084 [+0.051, +0.100] | +0.034 [-0.017, +0.082] | +0.062 [+0.004, +0.105] |
| A12 | +0.025 [-0.016, +0.054] | +0.045 [-0.007, +0.080] | +0.032 [-0.021, +0.061] |
| A13 | -0.047 [-0.070, -0.028] | +0.040 [+0.004, +0.071] | +0.049 [+0.008, +0.078] |
| A14 | +0.141 [+0.111, +0.174] | +0.122 [+0.086, +0.175] | +0.158 [+0.117, +0.194] |
| A15 | +0.020 [-0.005, +0.045] | +0.052 [+0.001, +0.099] | +0.043 [-0.013, +0.087] |
| A16 | +0.003 [-0.018, +0.027] | +0.027 [-0.039, +0.065] | -0.012 [-0.047, +0.034] |
| A17 | +0.051 [+0.022, +0.075] | +0.070 [+0.025, +0.111] | +0.118 [+0.071, +0.152] |
| A18 | +0.020 [-0.027, +0.053] | +0.007 [-0.044, +0.045] | -0.019 [-0.054, +0.008] |
| A19 | -0.056 [-0.086, -0.021] | -0.079 [-0.130, -0.022] | -0.024 [-0.076, +0.034] |

Spearman rho of each quantity against the mismatch |AUC - 0.5| of each
descriptor, 13 points, two sided Monte Carlo permutation p over 100,000
shuffles.

| descriptor | rho, ASYMMETRY | p | rho, LEVEL | p |
|---|---|---|---|---|
| bw30 | -0.077 | 0.81 | -0.396 | 0.18 |
| bw40 | -0.055 | 0.86 | -0.231 | 0.45 |
| edge ratio | +0.478 | 0.10 | +0.225 | 0.46 |
| edge slope | +0.071 | 0.82 | -0.445 | 0.13 |
| edge slope position | -0.264 | 0.38 | -0.308 | 0.31 |

**Power.** With n = 13 the smallest |rho| reaching two sided p < 0.05 is 0.555.
In a simulation of bivariate normal pairs the test detects a true correlation
of 0.5 in 36 percent of samples, 0.6 in 53 percent, 0.7 in 72 percent and 0.8
in 90 percent. Ten correlations are reported and none is corrected. A null here
says only that no very strong monotone relation exists.

## M3a, bonafide against treated bonafide

Arm 0 is each bonafide utterance as it is. Arm 1 is the same utterance after
the treatment, cut at the identical cached alignment. 1,500 utterances and
29,190 slices per arm. No synthesis on either side.

Two intervals are given. **Joint** draws utterance ids and takes both versions
of each, which is correct here because the arms are the same recordings.
**Unpaired** resamples each arm's utterances separately, as the registered
analysis does for two different sets of recordings; it is the wider one and is
the estimator `analyse_slices.asymmetry_ci` returns. 2,000 draws.

Folded separation by category, round trip through 22.05 kHz:

| category | soxr HQ, phase | soxr HQ, magnitude | poly, phase | poly, magnitude |
|---|---|---|---|---|
| NASALS | 0.609 | 0.506 | 0.516 | 0.504 |
| APPROXIMANTS | 0.582 | 0.508 | 0.513 | 0.505 |
| VOWELS_STEADY | 0.581 | 0.512 | 0.514 | 0.507 |
| FRICATIVES | 0.562 | 0.511 | 0.514 | 0.505 |
| PLOSIVES | 0.591 | 0.508 | 0.513 | 0.504 |
| AFFRICATES | 0.604 | 0.514 | 0.516 | 0.507 |

The soxr round trip raises `mgd_hf_std` (signed AUC above 0.5 in every
category); the poly round trip lowers it slightly (0.484 to 0.487). In the real
comparison the spoofed side is the lower one.

| contrast | LEVEL, joint | ASYMMETRY, joint | ASYMMETRY, unpaired |
|---|---|---|---|
| soxr HQ, 22.05 kHz | +0.0825 [+0.0801, +0.0848] | +0.0079 [+0.0033, +0.0127] | +0.0079 [-0.0167, +0.0395] |
| resample_poly, 22.05 kHz | +0.0089 [+0.0080, +0.0096] | -0.0001 [-0.0016, +0.0015] | -0.0001 [-0.0157, +0.0180] |
| resampler choice: soxr HQ arm against poly arm, both round tripped | +0.1018 [+0.0993, +0.1043] | +0.0087 [+0.0040, +0.0138] | +0.0087 [-0.0130, +0.0355] |
| soxr HQ, then rounded to 16 bit | -0.0029 [-0.0038, -0.0021] | -0.0277 [-0.0309, -0.0243] | -0.0277 [-0.0452, +0.0055] |
| resample_poly, then rounded to 16 bit | +0.0089 [+0.0079, +0.0096] | +0.0001 [-0.0015, +0.0018] | +0.0001 [-0.0155, +0.0186] |
| soxr LQ, then rounded to 16 bit | +0.0551 [+0.0517, +0.0583] | +0.0594 [+0.0536, +0.0654] | +0.0594 [+0.0404, +0.0773] |
| soxr LQ then 16 bit, both arms then low passed at 6900 Hz and 16 bit | +0.1302 [+0.1262, +0.1341] | +0.0005 [-0.0059, +0.0070] | +0.0005 [-0.0169, +0.0184] |

Seed 0 shown. Seeds 1 and 2 move no bound by more than 0.004 and change no
sign; all three are in `results/band_edge_summary.csv`.
`librosa.resample` at its default returns samples identical to soxr HQ, so it
has no row of its own.

The soxr LEVEL of +0.083 exceeds H1's +0.048, and it is in every category, so
the soxr round trip reproduces the phase selective separation and almost none
of the ASYMMETRY. Rounding the round tripped audio back to 16 bit, which is
what a file on disk carries, removes the LEVEL and turns the ASYMMETRY negative.
The effect of a soxr HQ round trip therefore lives largely below the 16 bit
noise floor of the original files.

### M3c, dose

Seed 0, joint draw.

| treatment | LEVEL | ASYMMETRY | ASYMMETRY, unpaired |
|---|---|---|---|
| soxr LQ, 22.05 kHz | +0.0569 [+0.0522, +0.0615] | +0.0418 [+0.0341, +0.0489] | +0.0418 [+0.0112, +0.0726] |
| soxr MQ, 22.05 kHz | +0.0624 [+0.0604, +0.0643] | -0.0121 [-0.0164, -0.0075] | -0.0121 [-0.0367, +0.0194] |
| soxr HQ, 22.05 kHz | +0.0825 [+0.0801, +0.0848] | +0.0079 [+0.0033, +0.0127] | +0.0079 [-0.0167, +0.0395] |
| soxr VHQ, 22.05 kHz | +0.0903 [+0.0877, +0.0927] | +0.0150 [+0.0103, +0.0198] | +0.0150 [-0.0100, +0.0461] |
| soxr HQ, 18 kHz | +0.0835 [+0.0811, +0.0859] | +0.0089 [+0.0043, +0.0136] | +0.0089 [-0.0156, +0.0405] |
| soxr HQ, 32 kHz | +0.0828 [+0.0804, +0.0851] | +0.0088 [+0.0042, +0.0135] | +0.0088 [-0.0158, +0.0403] |
| soxr HQ, 48 kHz | +0.0825 [+0.0801, +0.0848] | +0.0087 [+0.0041, +0.0134] | +0.0087 [-0.0159, +0.0402] |
| resample_poly, 32 kHz | +0.0105 [+0.0097, +0.0111] | +0.0015 [+0.0002, +0.0031] | +0.0015 [-0.0141, +0.0196] |

The intermediate rate does not matter. The filter does, and not monotonically
in steepness. soxr LQ is the one treatment whose ASYMMETRY excludes zero under
the unpaired draw as well. It also moves `mfcc_frame_spread`, to 0.53 to 0.62
folded, where the others leave it at 0.50 to 0.52. That does not set it apart
from the eval comparison, where the magnitude statistic separates at 0.51 to
0.56 too. Its 16 bit form is in the table above and is the contrast that
decides R1.

## M3b, the identical treatment on both classes

One code path, every file, bonafide against A07 to A19, cached alignment. The
registered estimator, `advantage_ci` and `asymmetry_ci`, utterance clustered,
one draw for all categories, 2,000 draws, seed 0. Seeds 1 and 2 move no bound
by more than 0.002. The last column is after minus before with both computed
inside the same draw.

| treatment | LEVEL | ASYMMETRY | change in ASYMMETRY |
|---|---|---|---|
| none | +0.0479 [+0.0347, +0.0609] | +0.0630 [+0.0451, +0.0795] | |
| b1, round trip, soxr HQ | +0.0488 [+0.0355, +0.0621] | +0.0471 [+0.0286, +0.0645] | -0.0159 [-0.0205, -0.0109] |
| b1, round trip, resample_poly | +0.0478 [+0.0345, +0.0610] | +0.0624 [+0.0444, +0.0794] | -0.0005 [-0.0023, +0.0014] |
| b1, soxr HQ, then 16 bit | +0.0466 [+0.0336, +0.0596] | +0.0496 [+0.0315, +0.0671] | -0.0134 [-0.0174, -0.0088] |
| b1, resample_poly, then 16 bit | +0.0482 [+0.0349, +0.0614] | +0.0616 [+0.0435, +0.0784] | |
| b2, low pass 6900 Hz | +0.0456 [+0.0325, +0.0578] | +0.0578 [+0.0382, +0.0759] | -0.0051 [-0.0141, +0.0030] |
| b2, low pass 6900 Hz, then 16 bit | +0.0484 [+0.0360, +0.0600] | +0.0608 [+0.0417, +0.0782] | -0.0021 [-0.0112, +0.0064] |
| 16 bit rounding alone | +0.0479 [+0.0347, +0.0609] | +0.0630 [+0.0451, +0.0795] | exactly zero |

The low pass is a 511 tap Kaiser FIR, linear phase, with its delay removed.

**b3, a redefinition and not the registered statistic.** MGD dispersion over a
sub band in place of the upper half.

| phase statistic | LEVEL | ASYMMETRY |
|---|---|---|
| `mgd_hf_std`, 4000 to 8000 Hz, registered | +0.0479 [+0.0347, +0.0609] | +0.0630 [+0.0451, +0.0795] |
| MGD dispersion, 4000 to 6500 Hz | +0.0445 [+0.0299, +0.0583] | +0.0461 [+0.0275, +0.0641] |
| MGD dispersion, 6500 to 8000 Hz | +0.0271 [+0.0137, +0.0396] | +0.0610 [+0.0326, +0.0761] |

### Do the treatments equalise anything

An identical treatment equalises a band edge only if it overrides the edge
that was there. This was tested two ways.

*On the descriptors.* AUC of the edge slope for bonafide against each of the
four attacks with a hard edge, 0.000 to 0.011 unmodified. After the soxr round
trip 0.40 to 0.50. After the 6900 Hz low pass 0.50 to 0.54. After the
resample_poly round trip 0.000 to 0.010: its filter leaves the edges where they
were, so **b1 with resample_poly equalises nothing and its row above is not
evidence.**

*On a planted difference.* Bonafide against round tripped bonafide, then the
equalising treatment applied to both arms. Seed 0, joint draw.

| planted | equaliser | LEVEL | ASYMMETRY |
|---|---|---|---|
| soxr HQ round trip | none | +0.0825 | +0.0079 |
| soxr HQ round trip | soxr HQ round trip | +0.0174 [+0.0163, +0.0185] | +0.0181 [+0.0154, +0.0193] |
| resample_poly round trip | soxr HQ round trip | +0.0044 [+0.0036, +0.0049] | -0.0031 [-0.0043, -0.0017] |
| soxr HQ round trip | low pass 6900 Hz | +0.0001 [-0.0001, +0.0000] | +0.0001 [+0.0001, +0.0007] |
| resample_poly round trip | low pass 6900 Hz | +0.0007 [+0.0002, +0.0011] | -0.0005 [-0.0010, +0.0005] |
| soxr HQ round trip | low pass 6900 Hz, 16 bit | +0.0001 [-0.0001, +0.0003] | -0.0004 [-0.0015, +0.0004] |
| resample_poly round trip | low pass 6900 Hz, 16 bit | +0.0009 [+0.0002, +0.0016] | -0.0014 [-0.0027, +0.0006] |

The low pass removes a planted resampling difference completely. A second soxr
round trip removes four fifths of the planted LEVEL and leaves an ASYMMETRY
larger than the one planted: passing through the same filter once more is not
the same as never having differed. **b2 is the valid equaliser here, b1 with
soxr a partial one.**

Two intervals in that table do not contain their own point estimate. That is
folding at chance: `max(auc, 1 - auc)` reflects every draw upward, and with
both statistics at 0.500 the reflection is the whole of what is left.

## The four attacks with a hard edge, and the nine without

Chosen after M1 and M2 had been seen, on the M1 descriptors alone: A07, A08,
A16 and A19 are the attacks whose bw40 and edge slope AUCs are below 0.05.
Exploratory.

| ASYMMETRY, bonafide against | A07 A08 A16 A19 | the other nine |
|---|---|---|
| unmodified | +0.0541 [+0.0280, +0.0806] | +0.0672 [+0.0521, +0.0864] |
| round trip, soxr HQ | +0.0004 [-0.0229, +0.0308] | +0.0682 [+0.0530, +0.0880] |
| round trip, soxr HQ, 16 bit | +0.0105 [-0.0109, +0.0438] | +0.0679 [+0.0532, +0.0876] |
| low pass 6900 Hz | +0.0125 [-0.0133, +0.0418] | +0.0738 [+0.0533, +0.0918] |
| low pass 6900 Hz, 16 bit | +0.0135 [-0.0113, +0.0411] | +0.0780 [+0.0580, +0.0961] |
| round trip, resample_poly (equalises nothing) | +0.0485 [+0.0251, +0.0766] | +0.0690 [+0.0539, +0.0884] |
| redefined, 4000 to 6500 Hz | +0.0032 [-0.0200, +0.0345] | +0.0656 [+0.0494, +0.0852] |
| redefined, 6500 to 8000 Hz | +0.0827 [+0.0299, +0.1271] | +0.0235 [+0.0056, +0.0466] |

| change in ASYMMETRY, same draw | A07 A08 A16 A19 | the other nine |
|---|---|---|
| low pass 6900 Hz, 16 bit | -0.0406 [-0.0522, -0.0215] | +0.0108 [-0.0110, +0.0202] |
| low pass 6900 Hz | -0.0417 [-0.0534, -0.0222] | +0.0066 [-0.0148, +0.0160] |
| round trip, soxr HQ | -0.0537 [-0.0596, -0.0346] | +0.0010 [-0.0036, +0.0055] |
| round trip, resample_poly | -0.0057 [-0.0075, +0.0036] | +0.0018 [-0.0002, +0.0039] |

LEVEL against the four is -0.018 [-0.037, +0.007] unmodified and stays within
0.01 of that under every treatment of the audio. LEVEL against the nine is +0.078
[+0.065, +0.090].

Against the nine attacks that reach Nyquist, the ASYMMETRY sits in 4000 to
6500 Hz, clear of every edge measured, and no treatment moves it. Against the
four with a hard edge, the ASYMMETRY sits above 6500 Hz and every treatment
that removes the edge removes it.

The four are not uniform. Per attack, low pass and 16 bit, same draw:

| attack | ASYMMETRY before | after | change |
|---|---|---|---|
| A07 | +0.054 | -0.001 | -0.0550 [-0.0705, -0.0083] |
| A08 | +0.060 | +0.049 | -0.0117 [-0.0467, +0.0057] |
| A16 | +0.027 | -0.012 | -0.0386 [-0.0665, +0.0376] |
| A19 | -0.079 | -0.024 | +0.0550 [+0.0295, +0.0808] |

The region above 6.9 kHz moves A07 and A19 by the same amount in opposite
directions. A19 is the one eval attack on which H1's ASYMMETRY is negative, and
most of that negative value comes from above 6.9 kHz. A08 keeps its ASYMMETRY.
The pooled figure for the four is not the mean of these, because a pooled AUC
is not a mean of per attack AUCs.

## The nine standing questions

1. **Is the quantity the one the hypothesis is about?** The rules are stated
   and evaluated on the ASYMMETRY. LEVEL is reported beside it everywhere. The
   0.576 and 0.592 that motivated this are bare separations, a third quantity,
   and are not what was tested.
2. **Is the bootstrap clustered by the unit that carries the information?**
   Utterances throughout, one draw for all categories. In M3a the arms are the
   same utterances, so the joint draw is the correct one and the unpaired
   interval is given beside it. System clustering was not run: with 4 and 9
   systems in the two subsets it would have almost no power.
3. **Is the direction convention read from the data?** Yes,
   `phonophase.stats.bonafide_label`, on every table. In M3a arm 1 carries the
   treatment as its attack so the same reader applies. Signed AUCs are printed
   beside folded ones.
4. **Multiple comparisons.** Nothing is corrected. R1 was evaluated on 3
   primary contrasts, 2 rounded variants and 7 dose contrasts. R2 on 6
   variants, of which the rule names 3; it is an "or" rule, so more variants
   make firing more likely, and none fired. M2 reports 10 correlations twice.
   The subset analysis reports 2 subsets by 7 treatments, plus 4 attacks.
5. **Is any count a count of signs?** No. The four attack figure is one pooled
   contrast. By paired intervals excluding zero, one of the four loses its
   ASYMMETRY and one gains.
6. **Does the conclusion survive a defensible change of definition?** R2 quiet
   survives float against 16 bit, two equalisers, and a sub band clear of the
   edges. R1 depends on the resampler: it fires at full size with soxr LQ and
   not with soxr HQ or resample_poly once the output is 16 bit. The subset finding survives three valid equalisers and the sub
   band redefinition, and is not uniform across the four attacks.
7. **Is undecided being reported as refuted?** R1 fires and R2 is quiet, and
   the cause of H1 is reported as undecided. R3 is reported as not detected at
   stated power.
8. **Does the power make the test capable of the answer it gives?** R2: the
   pooled interval has half width 0.018 against a threshold distance of 0.016
   to 0.031 from half, and the low pass was shown to remove a planted
   difference, so quiet is an answer the test could have failed to give. R3:
   no, below |rho| of about 0.7. Per attack: 115 utterances a side on the spoof
   side, intervals of half width 0.04, so only A07 and A19 are resolved.
9. **What was selected after seeing the data?** The 6900 Hz cutoff and the
   4000 to 6500 Hz band, by rules that needed M1. The 16 bit variants, added
   before any M3 number was seen, on the reasoning that MGD is close to scale
   invariant so a float low pass may not erase what it attenuates. The planted
   difference controls, added after the first R1 and R2 numbers. The split into
   four and nine attacks, the 6500 to 8000 Hz band and the per attack changes,
   all after M1, M2 and the pooled R2 numbers. The edge slope search was
   widened to reach Nyquist after the first M1 run showed it pinned at its
   upper limit; the other descriptors did not change.

## Limits

- The two arm contrasts fold each separation about 0.5, which discards its
  sign. They show what a treatment does to genuine speech against itself. They
  do not show how the same treatment on one class would add to or subtract from
  a real difference between classes, which depends on which class carries it.

- One corpus, one language, seg=run only. The window segmentation was
  calibrated and not carried through the interventions.
- An identical linear phase treatment equalises where the band ends. It does
  not undo the phase response of an unknown minimum phase or IIR anti-alias
  filter below its cutoff. If bonafide and spoofed audio differ in that, no
  treatment here removes it, and the nine attacks without a measurable edge are
  not thereby shown to be free of one.
- Bonafide is one population with one processing history. A band edge
  difference common to every attack, bonafide downsampled from a studio
  recording against speech generated at 16 kHz, cannot be separated across
  attacks. The low pass and the 4000 to 6500 Hz band address it only above
  6.9 and 6.5 kHz.
- The low pass removes everything above 6.9 kHz, edge or not. That the four
  attacks' ASYMMETRY disappears with it shows the ASYMMETRY lives up there, in
  a band where those attacks are empty above 7.7 kHz. It does not separate the
  edge itself from other content between 6.9 and 7.7 kHz. The soxr round trip,
  which acts above about 7.3 kHz, gives the same answer.
- The cepstral smoothing inside MGD spans the whole spectrum, so a sub band of
  the MGD plane is not strictly independent of an edge outside it.
- The bonafide arm of M3a is ASVspoof, English, 16 bit. The 0.576 and 0.592 it
  is compared with were measured on German M-AILABS at 60 ms windows, and
  whether those were rounded to 16 bit is not recorded.

## Not verified

- Why soxr HQ and resample_poly differ, and why soxr LQ produces an ASYMMETRY.
  The filters were not characterised beyond their effect on the descriptors.
- How ASVspoof's bonafide and each attack were actually band limited. M1
  describes the files; it does not recover the processing.
- Speaker clustered intervals. The recomputed tables carry no speaker column,
  because the archive's manifest has none.

## Commands

The two figures in the reading above that come from the registered script:

```bash
python scripts/analyse_slices.py --table <low passed table>
python scripts/analyse_slices.py --table <low passed table> --magnitude hf_lf_tilt
```

where the table is the output of `band_edge.py slice --treatment "lp:6900>pcm16"`
below.

The contrast that decides R1, and the equaliser applied to it. `$A` is the
tracked alignment, `results/band_edge_alignment.csv.gz`:

```bash
B="python scripts/band_edge.py"; S="--manifest $AUDIO/manifest.csv --alignment $A --only bonafide --segmentations run"
$B slice $S --treatment "rt:soxr_lq:22050>pcm16" --out $OUT/lq.csv.gz
$B slice $S --treatment "rt:soxr_lq:22050>pcm16>lp:6900>pcm16" --out $OUT/lq_lp.csv.gz
$B r1 --base $OUT/none.csv.gz --treated $OUT/lq.csv.gz --seeds 0 1 2
$B r1 --base $OUT/none.csv.gz --treated-base $OUT/lp_pcm16.csv.gz --treated $OUT/lq_lp.csv.gz
```

`$AUDIO` is the extracted archive, `$OUT` a scratch directory. Run with
`PYTHONPATH=src`.

```
B="python scripts/band_edge.py"; M=$AUDIO/manifest.csv

$B align --manifest $M --out $OUT/align.csv                      # about 50 min on CPU
$B slice --manifest $M --alignment $OUT/align.csv --segmentations run,window --out $OUT/none.csv.gz
python scripts/analyse_slices.py --table $OUT/none.csv.gz        # calibration

$B describe --manifest $M --out $OUT/ltas.csv.gz
$B m1 --descriptors $OUT/ltas.csv.gz
$B m2 --table $OUT/none.csv.gz --descriptors $OUT/ltas.csv.gz

# one table per treatment T, e.g. rt:soxr_hq:22050, rt:poly:22050, lp:6900, "lp:6900>pcm16"
$B slice --manifest $M --alignment $OUT/align.csv --treatment T --out $OUT/T.csv.gz
$B slice --manifest $M --alignment $OUT/align.csv --band 4000 6500 --out $OUT/band.csv.gz

$B r1 --base $OUT/none.csv.gz --treated $OUT/T.csv.gz                          # M3a
$B r1 --base $OUT/none.csv.gz --treated-base $OUT/T1.csv.gz --treated $OUT/T2.csv.gz
$B r2 --table $OUT/T.csv.gz                                                    # M3b
$B r2 --table $OUT/band.csv.gz --phase mgd_band_std                            # b3
$B r2 --table $OUT/T.csv.gz --attacks A07 A08 A16 A19
$B change --before $OUT/none.csv.gz --after $OUT/T.csv.gz [--attacks ...]
```

`results/band_edge_alignment.csv.gz` is the alignment cache, so `align` can be
skipped. `results/band_edge_descriptors.csv.gz` is the M1 table per file.
`results/band_edge_summary.csv` holds every LEVEL and ASYMMETRY line printed by
the runs above, all seeds. In it `pub_none` is the unmodified table, `all_` and
`bona_` prefix tables of every file and of bonafide only, `cliff` is A07 A08
A16 A19, `flat` is the other nine, and `PC` rows are the planted difference
controls with arm 0 and arm 1 named.
