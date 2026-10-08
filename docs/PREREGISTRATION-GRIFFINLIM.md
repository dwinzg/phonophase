# Pre-registration, the griffin_lim positive control

Written 2026-09-19, **before `mgd_hf_std` or `mfcc_frame_spread` has ever been
computed on any MLAAD audio**. That is the only thing that makes this test
confirmatory, and it is the only thing this test has going for it. Nothing in
this document may be changed once the first slice of MLAAD `griffin_lim` audio
has been measured.

Binding in the same way as `docs/PREREGISTRATION.md`, which registered H1 and
records its outcome, and `docs/PREREGISTRATION-H3.md`. This document registers
a separate test on separate data.

## Why this test exists

H1 is confirmed: the advantage of the phase statistic `mgd_hf_std` over the
magnitude statistic `mfcc_frame_spread` is larger in sonorants than in
obstruents, on ASVspoof 2019 LA eval, under pre-registration. The effect is
small. Separations run 0.51 to 0.60 and the asymmetry is 0.048.

Every result in this repository so far has the same shape: a phase statistic
beats a magnitude statistic by a little, on audio where the synthetic side
differs from the genuine side in **everything at once**, speaker, sentence,
recording chain, magnitude envelope and phase. Nothing has ever shown that
`mgd_hf_std` responds to phase rather than to one of the other things.

This project has never had a positive control: a system where the artifact is
known by construction to be in phase and nowhere else, so that the statistic
could be **shown capable of failing**. MLAAD's `griffin_lim` is that system.

## What has been verified about the construction

MLAAD `griffin_lim` takes the measured magnitude spectrogram of a real
M-AILABS recording and reconstructs a waveform from it by iterative
phase estimation. Its `original_file` field names the recording. The magnitude
is a real recording's. The phase is wholly synthetic.

Verified independently for this document, on files and metadata on disk:

| claim | measured | how |
|---|---|---|
| exact utterance pairing | 1000 rows, 1000 distinct `original_file`, in each of the 8 languages | `meta.csv` per language |
| the whole set is original-language | `is_original_language` True for 1000 of 1000 | `meta.csv`, de |
| no training data | `training_data: none`, `architecture: griffin_lim`, for 1000 of 1000 | `meta.csv`, de |
| duration matches the named recording | delta -0.45, -1.81, -6.68, -10.23 ms on 4 of 4 pairs, always slightly short | audio against the M-AILABS parquet |
| log-mel identity against its own source | r 0.9958 to 0.9976, against a null of 0.514 to 0.582 for unrelated real pairs | re-derived on 4 pairs |
| the waveform is **not** inherited | max abs cross-correlation 0.131 to 0.336 at best lag | re-derived on 4 pairs |
| `griffin_lim` exists only for M-AILABS languages | de, en, es, fr, it, pl, ru, uk, and no others of the 54 | repository tree |
| access | gated, and the project's credential reads `fake/` | `meta.csv` and audio fetched |

Two things came out of that re-derivation that the design has to carry.

**The magnitude is shared closely but not exactly.** Log-mel correlation is
0.996, but correlation of the **linear** log-|STFT|, which is the resolution
this project's front end actually works at, is 0.926 to 0.959. Griffin-Lim
converges to a signal whose magnitude approximates the target; it does not
reproduce it. So the magnitude side of the pair is near-identical, not
identical, and a magnitude statistic is predicted near chance rather than at
chance.

**Bandwidth is a property of the source speaker, not of the system.** Files are
stored at 22.05 kHz. For 21 of 23 German files sampled, the band above 8 kHz is
73 to 79 dB below the file's mean power per bin and holds 0.00000 of the energy:
an upsampled 16 kHz source. The two exceptions both come from one speaker,
`female/rebecca_braunert_plunkett`, whose spectral edge reaches 8786 and 9001 Hz.
That speaker is 116 of the 1000 German utterances. All four **English** source
speakers are clean: spectral edge 6869 to 7709 Hz and 0.00000 of energy above
8 kHz. The blanket statement "griffin_lim is empty above 8 kHz" is false at the
speaker level, and the design handles it by speaker rather than by assumption.

## The quantities, named

This project has shipped wrong claims by conflating quantities that are not the
same claim. Every clause below names the quantity it is about.

| name | definition |
|---|---|
| `SEP(f, cell)` | folded separation of feature `f`, real source against `griffin_lim`, within one phoneme category, `abs(AUC - 0.5) + 0.5` |
| `ADV(cell)` | `SEP(mgd_hf_std, cell) - SEP(mfcc_frame_spread, cell)`, on identical rows |
| `ADV_son` | `ADV` averaged over NASALS, APPROXIMANTS, VOWELS_STEADY |
| `ADV_obs` | `ADV` averaged over the obstruent categories that meet the count floor |
| `ASYM` | `ADV_son - ADV_obs` |
| `LADDER` | `SEP(mgd_hf_std, ·)` in the three cells `[+son,+voi]`, `[-son,+voi]`, `[-son,-voi]`, as `scripts/sonority_voicing.py` defines them |

`SEP(mgd_hf_std, ·)` is the bare separation of the phase statistic. `ADV_son`
is the sonorant advantage, which is a **level**. `ASYM` is the asymmetry. H1's
primary test is about `ASYM`; its first falsification condition is about
`ADV_son`; the per-category table it reports is about `SEP(mgd_hf_std, ·)`.
Those are three quantities and the three are not interchangeable. A statement
about one of them is not a statement about either other.

## Hypotheses

Three predictions follow from the construction, and a fourth from T7. They are
four different claims about four different quantities and they can come apart.

**G1, the magnitude prediction. About `SEP(mfcc_frame_spread, ·)`.** Because the
magnitude spectrogram is the source recording's own, a magnitude statistic
should sit near chance separating `griffin_lim` from its own M-AILABS source
side. Registered as: pooled `SEP(mfcc_frame_spread)` at or below **0.55**.
Not "at chance": the linear-magnitude correlation is 0.926 to 0.959, not 1.0,
so some magnitude separation is expected and is not a failure.

**G2, the phase prediction. About `SEP(mgd_hf_std, ·)`.** Because the phase is
wholly re-estimated, a phase statistic should separate, and by more than any
processing artifact of the comparison itself. Registered as: pooled
`SEP(mgd_hf_std)` exceeds the largest processing null defined below, with a 95
per cent cluster-bootstrap interval on the difference excluding zero.

**G3, the articulatory prediction. About `ASYM`.** If the mechanism behind H1 is
articulatory, coherent harmonic phase in voiced sonorants is what a system has
to reconstruct and gets wrong, then `ASYM` should be positive within this
system, where every phone's phase is reconstructed from scratch. Registered as:
`ASYM` positive with a 95 per cent interval excluding zero.

**G4, the ladder, registered as secondary. About `LADDER`.** `docs/RESOURCES.md`
records that "it concentrates in sonorants specifically" is **superseded
wording** and that the robust form of the T7 result is the ladder:
`SEP(mgd_hf_std, ·)` runs 0.576, 0.547, 0.515 across `[+son,+voi]`,
`[-son,+voi]` and `[-son,-voi]` on ASVspoof eval, both steps clear of zero.
Registered as: within `griffin_lim`, `LADDER` is monotone decreasing across
those three cells with both step intervals excluding zero. `[+son,-voi]` is
reported and is not part of the ladder, because on ASVspoof it is 1,471 slices
of the single phone /h/ and it is the cell that moved under the silence
control.

`LADDER` is a claim about the **separation** of the phase statistic within a
cell. It is not `ASYM` and it is not `ADV_son`. Its being registered as
secondary means it is reported whatever it does and it does not substitute for
G3.

G1 and G2 together are **not** diagnostic on this data, for a reason measured
below, and G3 is the load-bearing prediction. That is stated here so it cannot
be rearranged afterwards.

## Recorded prior expectation for G3, which is mixed

`docs/PREREGISTRATION-H3.md` already records the relevant prior, from the
ASVspoof eval run, before this document:

> A11 is Tacotron 2 with Griffin-Lim vocoding, the extreme case of phase
> generated with no learned phase model at all, and its asymmetry is +0.034
> with an interval spanning zero.

So the one Griffin-Lim system this project has already measured did **not**
show a clear asymmetry, and was not near the top of the table; the largest
asymmetries were classical source-filter vocoders, A09 at +0.218 and A14 at
+0.122. If `ASYM` comes back null here, that is the second Griffin-Lim system
to say so, and it is a result about the mechanism, not a missing data point.

H3's prediction runs the other way, and H3 has since been run. On the
development partition all six attacks were decided and the split was in the
predicted direction: A01, A02, A03 and A05, which generate their waveform,
clearly positive; A04 concatenation and A06 spectral filtering, which inherit a
real waveform, clearly negative. The registered pooled test over all nineteen
attacks then returned a difference of -0.1203 at exact p 0.0208, and -0.0890 at
p 0.1643 with the duplicated algorithms removed, which the registration demanded
in advance, so **H3's verdict is not supported** while its direction stands.

That makes the prior for G3 genuinely mixed rather than simply weak. Every
system in H3's dev generated group showed a clear positive asymmetry, and
`griffin_lim` generates its phase, which argues for G3. The one Griffin-Lim
system measured directly, A11, did not, which argues against it. `griffin_lim`
is the case that separates "generates its phase" from "is Griffin-Lim", and
those two readings make opposite predictions here.

## The rival account this test can discriminate

`docs/RESOURCES.md` records an alternative, as something to register rather
than assert: **the operative variable may be parametric excitation modelling
rather than phase reconstruction as such.** The evidence for it is the same A11
observation, the two largest asymmetries on ASVspoof eval are Vocaine and
STRAIGHT, both classical source-filter vocoders with explicit excitation
models, and the Griffin-Lim system is not near them.

`griffin_lim` separates the two accounts cleanly, because it reconstructs phase
with no excitation model of any kind.

| outcome | phase-reconstruction account, and H3's generate-versus-inherit reading | excitation-modelling account |
|---|---|---|
| `SEP(mgd_hf_std)` high and `ASYM` positive | supported | not supported |
| `SEP(mgd_hf_std)` high and `ASYM` null | not supported | supported |
| `SEP(mgd_hf_std)` at the nulls | refuted, F3 | untouched, since it predicts nothing here |

On ASVspoof the two readings are hard to separate, because the systems that
generate their phase vary in whether they model excitation at all, and the only
pure magnitude-inversion case in the corpus is a single attack, A11. Here the
same comparison runs on 1000 utterances of one algorithm rather than on one
attack, which is why this control is worth running even though its primary job
is G2.

This is written down now so that a null `ASYM` is read as evidence **for** a
rival account that already exists in this repository, rather than as a
disappointment to be explained.

## The data

Primary: MLAAD **`fake/en/griffin_lim`**, 1000 utterances, 2.07 hours, against
the 1000 M-AILABS recordings its `original_file` field names.

English, not German, and the reasons are fixed here in advance. H1 was
established in English, so the prediction transfers without invoking H2 and a
null cannot be explained away as a language effect. All four English source
speakers are band-clean below 8 kHz, where German has a speaker that is not.
The real side is available: `en_US.tgz` and `en_UK.tgz` both return HTTP 200 on
the route recorded in `configs/datasets.yaml`.

| property | value |
|---|---|
| pairs | 1000, matched 1000 of 1000 by `original_file` |
| source speakers | 4: elizabeth_klett, judy_bieber, mary_ann, elliot_miller |
| source archives | en_UK 355, en_US 645 |
| mean duration | 7.44 s, total 2.07 hours per side |
| stored rate, synthetic side | 22.05 kHz, verified on 4 of 4 speakers |
| spectral edge, synthetic side | 6869 to 7709 Hz, 0.00000 of energy above 8 kHz |

German `fake/de/griffin_lim` is registered as a **pre-specified replication**,
not as part of the primary test, and is run only after the English result is
recorded. Its `female/rebecca_braunert_plunkett` utterances are excluded by the
bandwidth rule below.

MLAAD is CC-BY-NC 4.0, non-commercial academic research only, and gated. That
is stricter than this repository's code licence and it constrains anything
trained on it. M-AILABS is modified BSD three-clause.

## The trap, and the mitigation written into the design

`mgd_hf_std` is dispersion of Modified Group Delay over the **upper half** of a
512-point spectrum. At 16 kHz that band is 4000 to 8000 Hz. At 22.05 kHz the
same rows are 5512 to 11025 Hz, and 55 per cent of that band is the empty
region above 8 kHz. So "the upper half" is not the same physical band in the
two classes unless something is done about it, and comparing MLAAD audio at
22.05 kHz against ASVspoof bonafide at 16 kHz would compare two different
frequency ranges across two different corpora.

Three things were measured on **real M-AILABS audio only**, with no
`griffin_lim` file on either side of any comparison, so that these numbers cost
nothing in prospectivity. The two tables below are 60 ms windows, the pilot
slice scale, through the repository's own `slice_statistics`, with bootstraps
clustered by source utterance; the dilution numbers in the first paragraph are
whole-utterance. In both tables the two arms are the same recordings processed
two ways, and each arm's clusters were resampled independently rather than in
pairs, which is the conservative choice here. Measured on the German source
speakers, because those are the M-AILABS recordings already on disk.

**The empty band is not noise, it is dilution.** In real audio upsampled to
22.05 kHz, MGD dispersion in the rows above 8 kHz is 0.347 against 4.028 in the
rows below it, a ratio of 0.086. Reading the band at 22.05 kHz rather than
16 kHz, on identical content, separates at only 0.529 folded. The band mismatch
attenuates real signal and shrinks between-utterance variance; it does not
manufacture a large separation.

**The resampling asymmetry does manufacture one, and it imitates the predicted
result exactly.** Real 16 kHz recordings against *the same recordings* passed
through 16 to 22.05 to 16 kHz:

| statistic | native | round-tripped | folded separation | 95% CI on AUC |
|---|---|---|---|---|
| `mgd_hf_std` | 3.3732 | 3.5048 | **0.576** | [0.387, 0.462], excludes chance |
| `mfcc_frame_spread` | 76.399 | 76.744 | 0.508 | [0.462, 0.524], crosses chance |
| `hf_energy` | -7.711 | -7.746 | 0.504 | crosses chance |
| `hf_lf_tilt` | -6.047 | -6.082 | 0.504 | crosses chance |

1003 slices per arm from 150 utterances. A resampler, with no synthesis
anywhere, moves the phase statistic to 0.576, inside the 0.51 to 0.60 range
that H1's whole result occupies, and leaves the magnitude statistic, the
energy control and the tilt control at chance. **That is precisely the
signature G1 and G2 predict.** If only the `griffin_lim` side were resampled,
this design would return a clean confirmation of G1 and G2 from nothing but
arithmetic.

**Equalising the rates is not sufficient, because the filter matters too.**
Both arms round-tripped through the same rates, one with librosa `soxr_hq` and
one with `scipy.signal.resample_poly`:

| statistic | folded separation | 95% CI on AUC |
|---|---|---|
| `mgd_hf_std` | **0.592** | [0.552, 0.634], excludes chance |
| `mfcc_frame_spread` | 0.504 | [0.469, 0.542], crosses chance |

798 slices per arm from 120 utterances. The choice of resampling filter alone
separates at 0.592, phase-selectively again. MLAAD's own 16 to 22.05 kHz
resampler is unknown and cannot be reproduced, so an artifact of this class
**cannot be eliminated from this comparison. It can only be bounded.**

### What the design therefore does

1. **Both sides resampled to 16 kHz through the identical code path.** The
   synthetic side 22.05 to 16 kHz; the real side 16 to 22.05 to 16 kHz with the
   same library and the same filter, so both carry one identical downsample and
   one identical anti-alias transition at 8 kHz. Never one side only.
2. **`griffin_lim` is compared only against its own M-AILABS source side.**
   Never against ASVspoof bonafide, and never against another MLAAD system.
3. **`mgd_hf_std` is not redefined.** Not re-banded, not re-normalised. The
   whole value of the control is that the registered statistic can fail it, and
   a statistic adjusted for this data would not be the statistic H1 is about.
4. **Bandwidth rule, fixed in advance.** Before any separation is computed, the
   fraction of energy above 8 kHz is measured per file on the synthetic side.
   Any source speaker whose median exceeds 0.5 per cent is dropped whole, both
   sides, and the drop is reported with its count. On English this is expected
   to drop nothing; on German it drops `rebecca_braunert_plunkett`.
5. **Both processing nulls are re-run on the English source side before the
   confirmatory pass**, phoneme-conditioned, and reported beside the result.
   Call the larger of the two `NULL_sep` for `SEP(mgd_hf_std)` and `NULL_asym`
   for `ASYM`. The numbers above stand as the German-side estimates, 0.576 and
   0.592; the English ones are not yet measured and the falsification conditions
   are written against whatever they turn out to be.

`NULL_asym` is the one that matters most and has not been measured at all yet.
A resampling artifact has no obvious reason to be indexed to articulation, but
it lives at the top of the band, where sonorants carry harmonics and obstruents
carry noise, so it cannot be assumed unindexed. If `NULL_asym` is itself
positive and comparable to the observed `ASYM`, nothing is attributable to
synthesis and F5 below fires.

## Primary test

**Registered as primary: the unpaired estimator with paired resampling.**
Folded separation as H1 defines it, `abs(AUC - 0.5) + 0.5`, computed over all
slices of each class within a phoneme category, with a cluster bootstrap whose
resampling unit is the **utterance pair**: a draw picks `original_file` pair ids
and takes both members of each drawn pair together, one shared draw serving
every category and both features, so `ADV`, `ADV_son`, `ADV_obs` and `ASYM` are
differenced inside each draw.

This is one test, decided in advance, and the paired alternative is not it.

Why unpaired-estimator-with-paired-resampling rather than a paired test on
slices. It is the same estimator as H1's, so the numbers are directly
comparable, which is the entire point of a control. A slice-level paired test
would require the CTC segmentation to correspond between the two files, and
the slices where the aligner disagrees are exactly the slices where Griffin-Lim
damaged the signal most, so dropping unmatched slices selects on the outcome.
Pairing still does the work it is good for: because each pair shares speaker,
sentence and magnitude by construction, speaker and content are controlled by
design rather than by resampling, and that is why speaker is not a bootstrap
level here despite there being only four of them.

**Secondary, always reported, never substituted for the primary.** A fully
paired analysis over the 1000 utterance pairs: per-utterance mean `mgd_hf_std`
within each phoneme category, differenced within pair, with a sign test and a
paired bootstrap over pairs. If the primary and the secondary disagree, both are
reported and the disagreement is the finding.

Clustering is by utterance pair and not by slice because slices are not
independent: many come from one utterance sharing a speaker, a sentence and a
system, and treating them as independent is pseudoreplication that makes
intervals far too narrow. On the pilot, 129 fricative slices came from 23
utterances.

## Required sample size

**1000 utterance pairs**, which is all of them: 1000 real plus 1000 synthetic,
2.07 hours per side.

**Floor, inherited from H1's registration: at least 200 source utterances
contributing slices to a phoneme category** before that category enters a group
mean. The floor is checked on counts, before any separation is computed, so
checking it is not looking at the answer. Categories below the floor are
dropped from their group mean by this rule and the drop is reported.
AFFRICATES is the category at risk: English has only two, and at 2,995
utterances on ASVspoof eval it already had the widest interval of any category,
0.536 [0.496, 0.574]. If AFFRICATES fails the floor, `ADV_obs` is the mean over
FRICATIVES and PLOSIVES, by the rule and not by choice.

**Precision, extrapolated from H1 and labelled as an extrapolation.** H1's eval
run at 2,995 utterances gave a half-width of about 0.009 on the sonorant
advantage and 0.010 to 0.017 per category. Scaling by the square root of
2995/1000 gives about 0.016 on the advantage and 0.017 to 0.029 per category at
1000 pairs. An `ASYM` of H1's size, +0.048, is comfortably detectable; one below
about +0.02 is not, and a null in that region will be reported as
underpowered rather than as a refutation. Utterances here are longer than
ASVspoof's, 7.44 s mean, so per-utterance information is greater and the true
precision may be better than this scaling; that is a reason not to claim the
number is exact, not a reason to assume it is better.

## Statistical discipline, fixed in advance

Answers to the standing questions in `docs/WORKING-PROTOCOL.md`, settled here
rather than at analysis time.

**Correction family.** G1 to G4 are separately registered predictions about
four different quantities and are not corrected against one another; that is
recorded as a choice, not an omission. Any
**per-category** claim carries Benjamini-Hochberg across the family of six
categories. Any **per-cell** claim about `LADDER` carries it across the four
cells. No claim is made from an uncorrected per-category or per-cell sign.

**Counting rule.** Any statement of the form "k of n categories show X" counts
**intervals**, never signs. Counting signs of point estimates is the operation
that produced two overstatements in this project's record, and it is banned
here in advance.

**Undecided is not refuted.** An interval that includes zero in a quantity
whose half-width exceeds the effect size being looked for is reported as
underpowered, with the half-width, and not as a negative result. The threshold
for this is the one given under sample size: an `ASYM` below about +0.02 is in
the underpowered region at 1000 pairs.

**Change of definition.** Both segmentations are run, `seg=run` and
`seg=window`, exactly as H1's confirmatory run did, and both are reported. On
ASVspoof they agreed to 0.002. A result that appears on one and not the other is
reported as segmentation-dependent and is not claimed.

**Direction is read, not assumed.** Folded separation hides direction, so the
sign of the unfolded AUC is reported for every cell alongside it, and F4 is
written against it.

## Falsification, stated in advance

Two tiers. The first says the control is not a control and the test should not
be interpreted. The second says the phase account is wrong.

### Tier 1, the control is invalid

**F1.** Pooled `SEP(mfcc_frame_spread)` exceeds **0.60**. The magnitude side is
then not shared as the construction requires, the pair set is not a positive
control, and G2 and G3 are uninterpretable on it. This falsifies the control,
not the phase account.

Between 0.55 and 0.60, G1 is not supported and the control survives: the
magnitude leakage is then reported as a standing caveat on G2 and G3, with
`ADV` carrying it directly since `ADV` is a difference of the two
separations.

**F2.** Fewer than 950 of the 1000 pairs resolve to a real M-AILABS file, or the
bandwidth rule drops more than 20 per cent of utterances. The design is then
running on something other than what was registered.

### Tier 2, the phase account is wrong

**F3.** Pooled `SEP(mgd_hf_std)` does not exceed `NULL_sep`, the larger of the
two processing nulls measured on the English source side, with a 95 per cent
interval on the difference excluding zero.

This is the condition that makes the statistic capable of failing, and it is
reachable: the German-side nulls are 0.576 and 0.592, and every per-category
separation H1 ever measured on real ASVspoof attacks was between 0.511 and
0.599. A statistic that cannot beat a resampler on audio whose phase has been
**entirely discarded and re-estimated**, with waveform cross-correlation of
0.13 to 0.34 against its own source, is not detecting phase reconstruction
artifacts. If F3 fires, the reading of `mgd_hf_std` as a phase-artifact
statistic is abandoned, and with it the mechanism H1 was given. H1's empirical
result stands, it was pre-registered and confirmed, but it stops being
evidence about phase and becomes an unexplained separation between two
statistics.

**F4.** `griffin_lim` runs the **higher** `mgd_hf_std` than its own source. On
ASVspoof, bonafide runs higher, which is why the analysis folds. A reversal here
means the statistic does not respond to phase reconstruction in the direction
H1 attributes to it. The folded statistic hides direction, so direction is
reported separately and must match.

**F5.** `NULL_asym`, the phoneme-conditioned processing null, is positive and
its interval overlaps the observed `ASYM`. Nothing is then attributable to
synthesis.

**F6.** `ASYM` is negative with an interval excluding zero. The articulatory
indexing then runs backwards in the one system where the artifact is purely
phase, and the articulatory reading of H1 is abandoned rather than qualified.

### What a null `ASYM` means, decided now

If `ASYM` comes back with an interval including zero while F3 and F4 do not
fire, the phase statistic separates strongly, the direction is right, and the
asymmetry is absent, then `mgd_hf_std` is confirmed as a phase-artifact
statistic and the **articulatory** part of the account is not supported by this
control. A11 makes that a live possibility and H3's dev result argues the other
way, which is why it is registered rather than assumed either way. It is
recorded here as a real result with a real consequence: H2 should not be run,
because a cross-lingual test of an articulatory profile is not interpretable
while the profile has failed in the one system where phase is the only
manipulation.

### Excuses ruled out in advance

None of the following may be offered after the fact as a reason a Tier 2
failure does not count. All were known and written down before the run.

- "Griffin-Lim has no learned phase model, so it is too extreme." That is the
  reason it was chosen. It was written down as the cleanest case, in
  `docs/H2-DESIGN.md`, before any of this was measured.
- "A11 already showed Griffin-Lim is weak." Recorded above, in advance, as the
  prior. It is a reason the test is informative, not a reason to discount it.
- "The band is wrong for this data." The band is registered, the bandwidth is
  verified per speaker, and the nulls bound the band artifact.
- "The magnitude is not perfectly shared." Quantified in advance at r 0.926 to
  0.959 on linear log-|STFT|, and G1's threshold was set at 0.55 rather than
  0.50 because of it.
- "One corpus, one language, small n." The sample size and its limits are
  registered above, including the region in which a null is underpowered rather
  than informative.
- "The asymmetry is there but smaller." A smaller-but-present `ASYM` supports a
  weaker claim and is recorded as supporting that weaker claim, exactly as
  H3's registration handles the same case. It is not support for G3 as stated.

## Pre-run checks, registered as assumptions

Each is verified before the confirmatory pass, and any that fails is reported
with what was done about it.

| assumption | status now |
|---|---|
| all 1000 English `original_file` paths resolve inside `en_US.tgz` / `en_UK.tgz` | **unverified**, route returns HTTP 200, contents not inspected |
| English real side is natively 16 kHz throughout | **unverified** for English; verified 16000 Hz for the German karlsson and ramona_deininger recordings |
| English synthetic side is 22.05 kHz and band-clean below 8 kHz | **verified**, 4 of 4 speakers, edge 6869 to 7709 Hz, 0.00000 of energy above 8 kHz |
| pairing is exact and one-to-one | **verified**, 1000 of 1000 distinct `original_file` |
| `NULL_sep` and `NULL_asym` on the English source side | **not yet measured**; German-side `NULL_sep` estimates are 0.576 and 0.592 |
| CTC segmentation succeeds on both sides at comparable rates | **unverified**; a large asymmetry in alignment failure rate between the two sides is itself reported |

## What this test cannot establish

**One corpus.** MLAAD, one gated release, CC-BY-NC.

**One algorithm.** Iterative magnitude-only phase reconstruction. Griffin-Lim
is not a vocoder anyone deploys, and it has no learned phase model, so it is a
statement about the extreme case and not about neural vocoders.

**Magnitude reconstruction, not natural phase inheritance.** The magnitude here
is the source recording's own measured magnitude. No deployed synthesiser has
that. This test asks whether the statistic can see a pure phase manipulation.
It does not ask whether the statistic sees the phase errors real systems make,
which is what H1 and H3 are about.

**One language for the primary test**, four source speakers, studio audiobook
speech.

**A positive result is not a detector.** If everything passes, what has been
shown is that `mgd_hf_std` responds to phase when phase is the only thing that
changed, and that the response is articulatorily indexed. Separations in this
project run 0.51 to 0.60 and a single statistic there is nowhere near useful
on its own.

---

# What this route closes

`docs/H2-DESIGN.md` named MLAAD `griffin_lim` as a candidate fourth
**waveform-inheriting** system, to strengthen H3's natural-phase group, which
stands at two attacks in dev (A04, A06) and three in eval (A16, A17, A19).

**It is not one, and this route is closed.** `griffin_lim` inherits the source
recording's magnitude and discards its phase entirely: max absolute waveform
cross-correlation against its own named source is 0.131 to 0.336, re-derived
here on four pairs, against log-mel correlation of 0.996 on the same pairs.
Nothing of the source waveform's fine phase structure survives.

Consequences, recorded so they are not rediscovered:

- H3's natural-phase group gains no members from MLAAD `griffin_lim`, and the
  constraint that made H3 not supported is untouched. Nineteen attacks contain
  only three distinct waveform-inheriting algorithms, two of them appearing
  twice, which is why the duplicate-removed pooled test sits at p 0.1643. What
  H3 needs is a **new inheriting algorithm from outside ASVspoof**, and
  `griffin_lim` is not one. The smallest attainable exact p over two attacks
  against four is 0.0667, not the 0.133 the H3 registration first stated;
  `scripts/attack_analysis.py` corrected that floor, because the complement of
  a size-k subset is enumerated only when k equals n-k.
- If `griffin_lim` were ever grouped for an H3-style test, it belongs in the
  **generated** group, at its extreme end.
- What `griffin_lim` is instead is the positive control this project never had,
  which is what this document registers. That is a better use of it than the
  one it was wanted for.
- The negative control for the same construction behaved as it should, and it
  is worth being exact about what it measured. `thorsten_vits`, over the same
  five sampled German source speakers, is a genuine 22.05 kHz generator: it
  carries 0.05 to 0.74 per cent of its spectral power above 8 kHz, where
  `griffin_lim` carries 0.00000 for four of those five and 0.08 to 0.18 per
  cent for `rebecca_braunert_plunkett`. And it leaks no source-speaker
  identity: WavLM x-vector cosine within source speaker 0.982 against 0.981
  between, gap +0.001 at p = 0.359, where `griffin_lim` gives 0.955 against
  0.757, gap +0.198 at p < 0.0001. Both are statements about speaker identity
  and bandwidth, not about `mgd_hf_std`, which has been computed on neither.

## Cost

One slicing pass over 2000 files, 4.1 hours of audio, the same pipeline as the
eval run. Analysis is CPU and takes minutes. The M-AILABS English archives are
7.47 and 3.47 GiB and both routes are live.

---

## Addendum, 2026-10-08, written before the run

Nothing above is edited. No `griffin_lim` file has been measured on either side
of any comparison. This addendum exists because a test run since on ASVspoof
2019 LA, recorded in `docs/BAND-EDGE.md`, bears on three statements in "The
trap, and the mitigation written into the design".

### 1. The two null figures do not say whether the audio was requantised

The 0.576 and 0.592 nulls above were computed on resampled arrays, and this
document does not record whether those arrays were rounded back to 16 bit
before the statistics were taken. It matters. On ASVspoof bonafide speech a
soxr HQ round trip gives a phase minus magnitude advantage of +0.083 in
sonorants when float output is compared with 16 bit originals, and -0.003 once
the output is rounded to 16 bit. The two figures stand as recorded, with that
caveat on what they measured.

**Added requirement.** Record the sample format of the MLAAD files and of the
M-AILABS sources before anything is computed. After the last resampling step,
round every array on both sides to 16 bit, and say so beside every result.

### 2. `NULL_asym` has now been measured elsewhere, and it is not small

The text above says a resampling artifact "has no obvious reason to be indexed
to articulation". On ASVspoof bonafide speech, a soxr LQ round trip through
22.05 kHz, rounded to 16 bit, against the same speech untreated, gives an
ASYMMETRY of +0.059 [+0.040, +0.077], the size of H1's own, in the same
direction and category order. soxr HQ and `resample_poly` give none. So the
artifact can be indexed to articulation, and whether it is depends on the
filter, which for MLAAD is unknown.

**Added null.** Step 5 re-runs two processing nulls on the English source
side. A third is added: the same round trip with soxr LQ. `NULL_sep` and
`NULL_asym` are the largest of the three.

### 3. Passing both sides through one resampler is only a partial equaliser

Mitigation 1 rests on both sides carrying "one identical downsample and one
identical anti-alias transition". Tested with a planted difference: a soxr HQ
round trip applied to one arm, then the same round trip applied to both, left
an ASYMMETRY of +0.018 where +0.008 had been planted. An identical linear phase
low pass on both arms, followed by rounding to 16 bit, removed planted
differences to within 0.001.

**Added secondary analysis.** After the registered analysis, and reported
beside it, the same comparison is repeated with one identical linear phase low
pass applied to both sides at 0.8625 of the common band limit, 6900 Hz for an
8 kHz limit, then rounding to 16 bit. `mgd_hf_std` is still not redefined.

### What is and is not changed

The hypotheses, the primary test, the sample size, the statistical discipline
and both tiers of falsification conditions are unchanged, and the registered
verdict is the one they produce. The additions are a recording requirement, one
more null, and one secondary analysis.

One reading rule is fixed now, because the secondary analysis could otherwise
be used selectively. If the registered analysis supports G3 and the low passed
one gives an `ASYM` below half of the registered value with an interval that
includes zero, the result is reported as not separable from a band edge
artifact, whatever the registered verdict says. If both agree, the secondary
analysis adds nothing and is reported in one line.
