# Pre-registration, H3

Written 2026-09-19, **before** the development partition was sliced or
examined. Nothing in this document may be changed after that data is touched.
If the test comes back negative, the negative gets written up.

Binding in the same way as `docs/PREREGISTRATION.md`. That document registered
H1 and its outcome is recorded there. This one registers a separate hypothesis
that came out of the per attack analysis (`scripts/attack_analysis.py`).

## Where H3 comes from

The per attack analysis asked why the sonorant effect varies so much across ASVspoof eval attacks.
Two things came out of it.

**The variation is smaller than the record claimed.** An earlier summary recorded
the effect as reversing on four of thirteen attacks, which was treated as a
serious threat to the articulatory reading. Recomputed with intervals, the
quantity H1 is about, the sonorant advantage minus the obstruent advantage,
is positive on twelve of thirteen and negative only on A19. The four reversals
were reversals of a different quantity, whether phase beats magnitude at all,
which H1 does not claim.

**What variation remains lines up with how the waveform was made.** Grouping the
thirteen eval attacks by whether the output's fine phase structure originates in
a human recording gives a mean asymmetry of +0.006 for the three that inherit it
against +0.069 for the nine that generate it.

That grouping was written from the ASVspoof 2019 database paper and recorded in
`configs/asvspoof2019_attacks.yaml` before the per attack numbers were computed
with `scripts/attack_analysis.py`. But the reversal set was already known from
that earlier summary, so the partition was chosen by people who knew part of the
answer. It is a hypothesis, and it is underpowered where it stands: three
attacks on one side, an exact permutation p of 0.24, and a smallest attainable p
of 0.009.

Note, 2026-10-14: the floor stated here is wrong by a factor of two. Three
against nine is 220 assignments, and with uneven groups the most extreme split
is enumerated once, so the smallest attainable two sided p is 1/220 = 0.0045,
not 0.009. `exact_permutation_p` in `scripts/attack_analysis.py` now computes
the floor this way. The observed p, 0.24 (0.2409 printed today), is unaffected.

## Hypothesis

**H3. The sonorant phase asymmetry is present when a synthesis system has to
generate the waveform's phase, and absent when the system inherits that phase
from a human recording.**

**Mechanism.** The asymmetry is claimed to arise because a model reconstructing
voiced, sonorant speech has to reproduce coherent harmonic phase and does it
imperfectly. A system that concatenates recorded units, or filters a real
recording, never attempts that reconstruction. It should therefore show no
sonorant deficit, and whatever artifact it leaves should not be indexed to
sonority.

Note, 2026-10-14: rule R1 of `docs/BAND-EDGE.md` has since fired, and its
registered consequence is that no mechanism is stated for H1. The paragraph
above is kept as registered and is not a current claim.

## The data, and why it is clean for this test

The ASVspoof 2019 LA **development** partition, attacks A01 to A06.

It has never been sliced. H1 was registered and confirmed on eval, A07 to A19,
and every analysis in this repository since has run on that eval table. The
development attacks played no part in forming H1, in forming the per attack grouping, or
in any number anyone here has seen.

The grouping is fixed in advance by `configs/asvspoof2019_attacks.yaml`, which
was written from the source paper before this document.

| attack | waveform generator | natural phase origin |
|---|---|---|
| A01 | WaveNet | no |
| A02 | WORLD | no |
| A03 | WORLD | no |
| A04 | waveform concatenation | **yes** |
| A05 | WORLD | no |
| A06 | spectral filtering + OLA | **yes** |

Two against four. Small, but the two are the same algorithms as A16 and A19 in
eval, so a disagreement between partitions would itself be informative.

## Primary test

For each of A01 to A06, the asymmetry as `scripts/attack_analysis.py` defines
it. Folded AUC of `mgd_hf_std` minus folded AUC of `mfcc_frame_spread`, averaged
over NASALS, APPROXIMANTS and VOWELS_STEADY, minus the same quantity averaged
over FRICATIVES, PLOSIVES and AFFRICATES. Bootstraps clustered by source
utterance, bonafide resampled once per draw and shared across attacks.

The test statistic is the difference in mean asymmetry between the
natural-origin group, A04 and A06, and the generated group, A01, A02, A03, A05.

Inference is the **exact permutation test over the six attack labels**, all
fifteen assignments enumerated. The bootstrap interval is reported beside it and
is not the test, because it holds the systems fixed and cannot speak about
systems outside the corpus.

**H3 is supported if the generated group's mean asymmetry exceeds the
natural-origin group's, and the pooled eval plus dev permutation test over
nineteen attacks reaches p < 0.05.**

Six attacks alone cannot reach p < 0.05, because the smallest attainable two
sided p over two against four is 2/15 = 0.133. This is stated here so nobody
reports the dev result alone as significant or as a failure. The dev partition
is registered as **confirmation of direction**, and the significance test is the
pooled one over all nineteen attacks, where the smallest attainable p over five
against thirteen is 0.0008.

Note, 2026-10-14, correction to the two floors above, checked by enumeration.
The test is two sided on the absolute difference in means, and when the groups
are of unequal size the complement of an assignment is never itself an
assignment, so the single most extreme split is counted once. Two against four
gives C(6,2) = 15 assignments and a floor of 1/15 = 0.0667, not 2/15 = 0.133;
`exact_permutation_p` returns 0.0667 for the dev asymmetries printed below. Six
attacks still cannot reach p < 0.05, so the registered role of dev as
confirmation of direction is unchanged. For the pooled test, A13 is held out
because its origin is not decidable from the source paper, so it runs over
eighteen attacks, five against thirteen, C(18,5) = 8568 assignments and a floor
of 1/8568 = 0.00012, not 0.0008. `scripts/attack_analysis.py` prints exactly
these counts.

The dev result enters the pooled test whatever it says. It is not looked at,
then included or excluded.

## Falsification, stated in advance

H3 is rejected if any of the following holds.

A04 or A06 shows an asymmetry clear of zero and positive, of a size comparable
to the generated group. One such case would mean a system that copies real
waveform still shows the sonorant deficit, which the mechanism forbids.

The pooled nineteen attack permutation test returns p > 0.05 with the observed
difference in the direction of H3 being smaller than +0.02 in mean asymmetry.

The generated dev attacks A01, A02, A03 and A05, which are three WORLD systems
and one WaveNet, fail to show a positive asymmetry. These are the same family as
eval attacks that did show it, so their failure would mean the eval result was
about something other than the synthesis method.

## Stated in advance so it cannot be claimed afterwards

A19 is the single genuine reversal in eval, at −0.079, and it is a
natural-origin system. A06 is the same algorithm. **If A06 also comes back
negative that is consistent with H3 but it is not additional evidence**, because
it is the same system measured on different utterances. The same applies to the
A04 and A16 pair. The pooled test therefore reports, beside the p value, the
result with one member of each duplicated pair removed.

The sonorant asymmetry being present but smaller in the natural-origin group,
rather than absent, supports a weaker version of H3 and is recorded as such
rather than as support for the stated one.

## What a positive H3 changes

It converts the attack heterogeneity from a threat into the mechanism's
strongest evidence. The effect would be shown to appear exactly where the
mechanism says it must, and to be absent exactly where it says it must be
absent, across systems that had no part in forming the claim.

It also makes H2 interpretable. Language and synthesis system are confounded in
any cross lingual corpus, and knowing which synthesis families carry the effect
is what lets a language result be read as a language result.

## What a negative H3 changes

The waveform provenance story is dropped. The attack heterogeneity goes back to
being unexplained, the articulatory reading stays underdetermined, and H2 should
not be run until something else explains the variation, because a positive H2
could not be attributed.

## Cost

One slicing pass over the dev partition, the same pipeline as the eval run.
2,548 bonafide and a balanced spoof sample, about twenty minutes on a Colab T4.
Analysis is CPU and takes under two minutes.

## Recorded before the run

Three things in the eval data that this test does not address, written here so
they are not quietly folded into whatever H3 returns.

**A17 does not fit.** It inherits natural phase by the same criterion and shows
a clear positive asymmetry of +0.070. The grouping predicts the group mean, not
every member, and A17 is the counterexample inside it.

**Griffin-Lim does not top the table.** A11 is Tacotron 2 with Griffin-Lim
vocoding, the extreme case of phase generated with no learned phase model at
all, and its asymmetry is +0.034 with an interval spanning zero. The largest are
A09 at +0.218, which is Vocaine, and A14 at +0.122, which is STRAIGHT, both
classical parametric source-filter vocoders. If the mechanism were simply
reconstruction of phase from magnitude, A11 should lead and it does not. This
matters directly for `docs/H2-DESIGN.md`, which names MLAAD's `griffin_lim` as a
positive control.

**The pooled effect is small.** Asymmetries run from −0.079 to +0.218 with most
between +0.03 and +0.07, on separations of 0.51 to 0.58. Everything here is a
small effect measured carefully, not a large one.

---

# Outcome

Run on the ASVspoof 2019 LA development partition, recorded 2026-10-14. Figures
re-derived on that date with

    python scripts/attack_analysis.py --table results/dev_slices.csv.gz
    python scripts/attack_analysis.py --table results/eval_slices.csv.gz \
        --pool-with results/dev_slices.csv.gz

seg=run, 1,000 draws clustered by utterance, seed 0. They agree with the figures
already recorded in `docs/PREREGISTRATION-GRIFFINLIM.md` to every digit that
document prints.

## Dev, confirmation of direction

| attack | origin | asymmetry | 95% CI | q |
|---|---|---|---|---|
| A01 | generated, WaveNet | +0.354 | [+0.334, +0.378] | 0.002 |
| A02 | generated, WORLD | +0.092 | [+0.069, +0.113] | 0.002 |
| A03 | generated, WORLD | +0.075 | [+0.053, +0.095] | 0.002 |
| A05 | generated, WORLD | +0.094 | [+0.062, +0.122] | 0.002 |
| A04 | natural, waveform concatenation | -0.059 | [-0.078, -0.031] | 0.002 |
| A06 | natural, spectral filtering + OLA | -0.084 | [-0.109, -0.058] | 0.002 |

All six intervals exclude zero after correction across the six attacks, and
the direction is as predicted on all six: the four generated attacks positive,
the two natural-origin attacks negative. The dev-only exact permutation over
the fifteen assignments gives a difference of -0.225 at p 0.0667, which is the
floor, and as registered that is a statement of direction and not of
significance.

## The registered pooled test

| | natural | generated | difference | exact p | assignments |
|---|---|---|---|---|---|
| all attacks | A04, A06, A16, A17, A19, mean -0.0250 | 13 attacks, mean +0.0952 | -0.1203 | 0.0208 | 8568 |
| duplicated algorithms removed | A16, A17, A19, mean +0.0062 | 13 attacks, mean +0.0952 | -0.0890 | 0.1643 | 560 |

A13 is held out of both, its origin not decidable from the source paper.

## Falsification conditions

1. A04 or A06 clear of zero and positive: does not fire. Both are clear and
   negative.
2. Pooled p > 0.05 with a difference in H3's direction smaller than +0.02: does
   not fire on either row. The difference is -0.1203 or -0.0890 in mean
   asymmetry, both well beyond 0.02 in H3's direction.
3. The generated dev attacks fail to show a positive asymmetry: does not fire.
   All four are clear and positive.

## Verdict

**H3 is not supported.** The pooled test reaches p < 0.05 only while the two
algorithms that appear in both partitions, A04/A16 and A06/A19, are each counted
twice. The registration states in advance that a duplicated algorithm is the
same system measured on different utterances and "is not additional evidence",
and requires the duplicate-removed result to be reported beside the p value.
With the duplicates removed the difference is -0.0890 at p 0.1643, and the
result is treated as not supported on that basis. No falsification condition
fires, so this is not a refutation either. The direction stands on dev, on all
six attacks.

Read literally, the registration's support criterion names only the pooled p,
and on all attacks that is 0.0208. Treating the duplicate-removed row as
binding is the stricter of the two readings, and it is the one taken here
because the registration's own reason for reporting that row is that the
doubled count is not evidence.

The constraint is structural, not a matter of sample size. Across all nineteen
attacks there are three distinct waveform-inheriting algorithms, two of them
appearing twice. More utterances from the same systems cannot change that;
only a new inheriting algorithm from outside ASVspoof can.

## Recorded beside the outcome

The negative dev asymmetries of A04 and A06 move under the identical 6900 Hz
low pass of the band edge test: A04 from -0.060 to -0.012, A06 from -0.084 to
-0.014, both intervals then spanning zero (`docs/PREREGISTRATION-BANDEDGE.md`,
per attack table; those figures are copied from it, not re-derived here, and
come from that test's recomputed table, so they differ from the ones above in
the third place). The eval counterpart A19 moved the same way there, A16 did
not, and no reading of either is registered. No mechanism is stated for the dev
negativity. Rule R1 of `docs/BAND-EDGE.md` fired, and no
mechanism is stated for H1 either.

