# How far the H1 intervals reach, by what is resampled

Unregistered robustness analysis of the confirmed H1 result on
`results/eval_slices.csv.gz`. `docs/PREREGISTRATION.md` specifies clustering by
source utterance, and the registered verdict is the one that clustering
produces. Everything else on this page bounds the scope of the claim. It does
not adjudicate it, in either direction.

Two quantities, kept apart throughout:

- **LEVEL**: the mean, over the three sonorant categories, of the folded
  separation of `mgd_hf_std` minus that of `mfcc_frame_spread`. Falsification
  condition 1 is worded on this.
- **ASYMMETRY**: the LEVEL minus the same mean over the three obstruent
  categories. H1's primary test is worded on this.

## What each clustering claims

| `--cluster` | resampled unit | the interval is a statement about |
|---|---|---|
| `utterance` | source utterances, within each class | more utterances from the same speakers and the same 13 systems. Registered |
| `speaker-paired` | the 67 speakers, each picked once for both classes | new speakers, each recorded and spoofed as these were |
| `speaker` | speakers, picked separately within each class | new speakers on each side, with no speaker shared between the sides |
| `system` | bonafide by utterance, spoof by attack, 13 clusters | synthesis systems not seen |

Eval has 67 speakers. All 67 have bonafide utterances and 48 of them are also
spoofed, so for those 48 the same voice sits on both sides of the comparison.
The other 19 are bonafide only and supply 404 of the 1,500 bonafide utterances
in this table. `speaker-paired` keeps a speaker's two sides together in every
draw, which is how the data were collected. `speaker` picks the two sides
separately, so the speakers behind the bonafide class and the spoofed class
vary independently from draw to draw, far more than the 19 unshared speakers
make them differ in the data.

## Result

`seg=run`, 2,000 draws, 95% percentile intervals. Point estimates do not depend
on the clustering: LEVEL +0.048, ASYMMETRY +0.063. The bounds shown are seed 0,
and the bracketed figure is the largest movement of either bound across seeds 0,
1 and 2.

| cluster | LEVEL | width | ASYMMETRY | width | seed movement |
|---|---|---|---|---|---|
| utterance | [+0.035, +0.061] | 0.026 | [+0.045, +0.080] | 0.034 | 0.002 |
| speaker-paired | [+0.025, +0.071] | 0.046 | [+0.044, +0.082] | 0.038 | 0.001 |
| speaker | [+0.006, +0.087] | 0.081 | [+0.027, +0.094] | 0.067 | 0.004 |
| system | [+0.003, +0.082] | 0.080 | [+0.021, +0.100] | 0.079 | 0.004 |

`seg=window` gives the same picture: LEVEL +0.046, ASYMMETRY +0.063, with every
width within 0.004 of the row above it.

What this says:

1. **For the registered pair of statistics, the ASYMMETRY excludes zero under
   every clustering**, with a lower bound
   never below +0.018 on any seed or segmentation. It is the quantity H1
   asserts, and it is the robust one. The lower bound was computed on seeds 0
   to 2 of both segmentations.
2. **The registered LEVEL interval is too narrow as a statement about new
   speakers.** Respecting the crossed speaker design widens it by about 75
   percent, from 0.026 to 0.046. It still excludes zero.
3. **The LEVEL is sensitive to which speakers sit on each side.** When the two
   classes are picked separately, its interval is three times the registered
   width and its lower bound is within 0.01 of zero. The ASYMMETRY widens by
   less, a factor of two.
4. **The system clustered LEVEL sits on its threshold.** See below.

## The system clustered LEVEL, and why the script says UNDECIDED

At 2,000 draws the lower bound of the system clustered LEVEL landed at +0.003,
-0.001, +0.001 and +0.001 on seeds 0 to 3, so falsification condition 1 printed
PASS, FAIL, PASS and PASS on identical data. A percentile bound is an order
statistic of the draws and carries Monte Carlo error. `percentile_interval` in
`src/phonophase/stats.py` puts a distribution free 99 percent band on each
bound, and when the answer to "does the interval exclude zero" differs across
that band the script prints UNDECIDED instead of PASS or FAIL. All four seeds
now print UNDECIDED at 2,000 draws. The band rules out a pass on one seed and a
fail on another. A bound near the edge of its band can still read as settled on
one seed and UNDECIDED on the next.

At 40,000 draws the bound settles:

| seed | LEVEL, `seg=run`, `--cluster system`, 40,000 draws |
|---|---|
| 0 | +0.048 [+0.0016, +0.0832] |
| 1 | +0.048 [+0.0021, +0.0829] |
| 2 | +0.048 [+0.0017, +0.0831] |

On `seg=run` the interval excludes zero, by about 0.002. On `seg=window` the
same 40,000 draws give lower bounds of +0.0001, +0.0004 and +0.0001, which the
script still reports as UNDECIDED. The bound is on zero.

More draws answer the Monte Carlo question and nothing more. A percentile
interval over 13 clusters covers less often than its nominal rate, and a margin
of 0.002 is well inside what that shortfall could account for. The margin also
rests on treating the 1,500 bonafide utterances as independent, which the
speaker rows above show they are not. The reading this
supports is that under system clustering the LEVEL is not distinguishable from
zero, and the ASYMMETRY is.

## The result is specific to one pair of statistics

The registered pair is `mgd_hf_std` against `mfcc_frame_spread`. The table
carries other statistics of each kind, and swapping one side at a time is a fair
question to ask of a claim worded as "phase against magnitude". These are
sensitivity analyses on data already seen. Choosing the best of them would not
be a comparison, and none of them replaces the registered test.

`seg=run`, clustered by utterance, 2,000 draws, seed 0.

| phase statistic | magnitude statistic | LEVEL | ASYMMETRY |
|---|---|---|---|
| `mgd_hf_std` | `mfcc_frame_spread`, registered | +0.048 [+0.035, +0.061] | +0.063 [+0.045, +0.080] |
| `mgd_hf_std` | `mfcc_std` | +0.046 [+0.033, +0.059] | +0.055 [+0.037, +0.071] |
| `mgd_hf_std` | `mfcc_iqr` | +0.054 [+0.038, +0.070] | +0.052 [+0.030, +0.076] |
| `mgd_hf_std` | `hf_energy` | +0.053 [+0.043, +0.063] | +0.054 [+0.040, +0.069] |
| `mgd_hf_std` | `hf_lf_tilt` | +0.037 [+0.025, +0.048] | **+0.009 [-0.003, +0.033]** |
| `mgd_std` | `mfcc_frame_spread` | -0.023 [-0.037, -0.002] | -0.002 [-0.019, +0.018] |
| `mgd_iqr` | `mfcc_frame_spread` | +0.015 [-0.001, +0.031] | +0.030 [+0.000, +0.047], not settled at this draw count |
| `mgd_frame_spread` | `mfcc_frame_spread` | -0.023 [-0.037, -0.002] | +0.002 [-0.021, +0.023] |

Two things follow, and both narrow the claim.

1. **The asymmetry is established for upper band group delay dispersion and for
   no other phase statistic here.** `mgd_std` and `mgd_frame_spread`, which
   correlate at 0.998 and are in effect one statistic, give intervals centred on
   zero. `mgd_iqr` gives +0.030 with a lower bound on zero, which is undecided
   and not an absence.
2. **Against spectral tilt the asymmetry is not distinguishable from zero.**
   `hf_lf_tilt` has no phase content, and it separates bonafide from spoofed
   speech with nearly the same sonorant over obstruent pattern as `mgd_hf_std`
   does. The asymmetry holds against the three cepstral statistics and against
   high frequency energy. So the supported statement is that `mgd_hf_std` gains
   more in sonorants than cepstral dispersion does. That no magnitude statistic
   shows the same pattern is not supported: one comes close enough that this
   data cannot separate them.

## Limits

- One corpus, one language, 13 systems and 67 speakers.
- These are percentile intervals. No small sample correction is applied for the
  13 cluster case.
- The clusterings are alternatives, not a single nested model. No interval here
  resamples speakers and systems at once.
- The speaker of each row is read from the tracked protocol file,
  `configs/protocols/ASVspoof2019.LA.cm.eval.trl.txt`, because the shipped
  table predates the speaker column.

## Commands

```bash
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster utterance
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster speaker-paired
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster speaker
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster system
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --cluster system --n-boot 40000
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --magnitude hf_lf_tilt
python scripts/analyse_slices.py --table results/eval_slices.csv.gz --phase mgd_std
```

Add `--seed 1`, `--seed 2` and `--seg window` for the other columns, and the
other `--magnitude` and `--phase` values for the other rows of the last table. A run at
2,000 draws takes about a minute on a laptop CPU, and one at 40,000 about
twenty.
