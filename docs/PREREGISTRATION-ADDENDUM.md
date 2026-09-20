# Addendum to the H1 pre-registration

Written 2026-09-19, after the eval run of 2026-09-18 and after the system
clustering analysis of 2026-09-19.

`docs/PREREGISTRATION.md` is a registration and **is not edited**. Nothing here
changes a hypothesis, a test, a threshold or a falsification condition. This
document does one thing: it states which quantity each clause of that document
refers to, because the clauses refer to three different quantities and have been
read as though they referred to one.

The ambiguity is not academic. It has produced two published errors already, the
"four attack reversals" and the "twelve of thirteen" count, both recorded in
the project log as retractions.

---

## 1. The three quantities

Let `A_p(G)` be the folded AUC separating bonafide from spoof using the phase
statistic `mgd_hf_std` over the slices in group `G`, and `A_m(G)` the same using
the magnitude statistic `mfcc_frame_spread`.

| name | definition | plain statement |
|---|---|---|
| **LEVEL** | `adv(SON) = A_p(SON) - A_m(SON)` | does phase beat magnitude in sonorants at all |
| **ASYMMETRY** | `adv(SON) - adv(OBS)` | does phase beat magnitude *more* in sonorants than in obstruents |
| **BARE SEPARATION** | `A_p(G)` | does the phase statistic separate at all in group `G`, magnitude not involved |

SON is NASALS, APPROXIMANTS, VOWELS_STEADY. OBS is FRICATIVES, PLOSIVES,
AFFRICATES. These are the groups named in the registration and they are not
changed here.

H1, the hypothesis the project claims, is about the **ASYMMETRY**. A mechanism
indexed to articulation predicts that the phase advantage is distributed
unevenly across manner classes. It does not predict any particular absolute
level, and the level is the quantity most exposed to which synthesis systems
happen to be in the corpus.

## 2. Which clause refers to which

| clause of `PREREGISTRATION.md` | quantity |
|---|---|
| Primary test, "the sonorant mean advantage exceeds the obstruent mean advantage with a 95% interval excluding zero" | **ASYMMETRY** |
| Falsification 1, "the sonorant advantage interval includes zero on eval" | **LEVEL** |
| Falsification 2, "the ordering reverses, so obstruents show the larger advantage" | **ASYMMETRY**, sign only |
| Falsification 3, "`mgd_hf_std` fails to clear chance in at least two of the three sonorant categories" | **BARE SEPARATION** |
| Falsification 4, present on train A01 to A06 but absent on eval | either, applied to both partitions |

Falsification 1 is therefore a condition on a quantity **that is not the one H1
asserts**. That was defensible when written: if phase does not beat magnitude in
sonorants at all, an asymmetry between classes is not worth interpreting. It is
a sanity condition, not a statement of the hypothesis. It must not be read as
one.

## 3. What was actually executed, which is not the primary test

`scripts/analyse_slices.py` implements the four falsification conditions and
prints them under the heading `PRE-REGISTERED TEST`. Its condition 2 is:

    [PASS] 2. sonorant advantage exceeds obstruent advantage
           +0.048 against -0.015

That is a comparison of **two point estimates**. It is a faithful implementation
of falsification condition 2, which is about ordering. It is **not** the primary
test, which requires a 95% interval on the difference, excluding zero. No
interval on the ASYMMETRY appears in the output of the registered path. The two
intervals it does print, `+0.048 [+0.039, +0.057]` and `-0.015 [-0.030, +0.001]`,
are marginal intervals on the two levels, and an interval on a difference does
not follow from two marginal intervals.

So the outcome recorded as **H1 SUPPORTED** rests on the four falsification
conditions passing. The primary test, as worded, has not been run on eval under
the registered clustering.

This is a statement about execution, not a retraction. The conclusion may well
survive: the two marginal intervals do not overlap and the estimates are of
opposite sign, which makes a difference interval excluding zero likely. Likely
is not tested. **The required action is to compute the ASYMMETRY with a paired
cluster bootstrap under the registered clustering and report its interval.**
Until that number exists, the primary test is untested rather than passed, and
the correct description of the result is "all four falsification conditions
passed", which is what the outcome table actually shows.

## 4. The contradiction under system clustering, and how to read it

The registration fixes the resampling unit inside the primary test itself:
"clustered by source utterance". **Utterance clustering is part of the
registered procedure.** Any other resampling unit is a different analysis.

The 2026-09-19 analysis that clusters the spoof side by synthesis system
therefore sits outside the registration. It reports the ASYMMETRY surviving,
+0.0630 [+0.0195, +0.0979], and the LEVEL not, [-0.0002, +0.0820]. Read through
the registration those two results say SUPPORTED and REJECTED at once.

They do not, because neither is a registered result. The rule, stated here so it
is not decided case by case afterwards:

**A robustness analysis outside the registration cannot trigger a registered
falsification condition, and cannot satisfy the registered primary test
either.** It bounds the scope of a confirmed claim. It does not adjudicate it.

The symmetry is the point and it cuts against the project's current summary.
Citing the system clustered ASYMMETRY as evidence that "H1 still stands" while
declining the system clustered LEVEL that would reject under falsification 1 is
selection after the fact, standing question 9. Either both are reported or
neither is. The honest form is:

> Under the registered utterance clustering, H1's falsification conditions all
> pass. A post hoc analysis that resamples the 13 synthesis systems instead
> widens every interval; under it the asymmetry remains clear of zero and the
> sonorant level does not. That analysis is not registered and is reported as a
> scope bound, not as a confirmation.

## 5. What the registered result licenses, and what it does not

Clustering by source utterance makes every registered interval a statement about
**more utterances drawn from these thirteen synthesis systems**. It is not a
statement about synthesis systems not in the corpus, and no reading of the
registration makes it one.

A claim that generalises across synthesis systems requires system level
resampling, which is not registered, or a corpus with more distinct systems. If
the project wants that claim it must register it prospectively and test it on
data that had no part in forming it. The existing eval data had a part in
forming it and cannot confirm it.

## 6. What this addendum does not do

It does not change the hypothesis, the primary test, the falsification
conditions, the thresholds, the groups or the statistics.

It does not declare H1 unsupported. Every falsification condition that was run,
passed.

It does not register the system clustered analysis. That analysis was chosen
after seeing the data and cannot be converted into a registered test by being
written down afterwards, which is the entire reason registrations exist.

## 7. Consequent actions

1. Compute the ASYMMETRY interval under registered utterance clustering and add
   it to the outcome record, so the primary test as worded has a result.
2. Report the system clustered LEVEL wherever the system clustered ASYMMETRY is
   reported.
3. Correct the session summary in the project log, which states the system
   clustered asymmetry as "H1 still stands" without the level beside it.
