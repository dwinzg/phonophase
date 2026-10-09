# Addendum to the H1 pre-registration

Written 2026-09-19, after the eval run of 2026-09-18 and after the system
clustering analysis of 2026-09-19.

`docs/PREREGISTRATION.md` is a registration and **is not edited**. Nothing here
changes a hypothesis, a test, a threshold or a falsification condition. This
document does one thing: it states which quantity each clause of that document
refers to, because the clauses refer to three different quantities and have been
read as though they referred to one.

The ambiguity is not academic. It has produced two published errors already, the
"four attack reversals" and the "twelve of thirteen" count, both recorded as
retractions in `docs/RESOURCES.md`.

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

### The fourth falsification condition is not the fourth falsification condition

The registration lists four. The script implements three of them and substitutes
a requirement taken from a different section of the document for the fourth.

| # | registered falsification bullet | what the script checks |
|---|---|---|
| 1 | sonorant advantage interval includes zero | same, inverted. implemented |
| 2 | the ordering reverses | same. implemented |
| 3 | phase fails to clear chance in 2 of 3 sonorants | same. implemented |
| 4 | **present on train A01 to A06 but absent on eval** | **200+ bonafide utterances per category** |

The fourth registered condition is the one that would catch the effect being a
property of six particular synthesis systems. It is **not implemented anywhere**.
What sits in its place is the "Required sample size" requirement, which is a
power precondition and not a falsification condition at all.

The script's own decision logic agrees. `analyse_slices.py:466` reads
`verdicts[:3]`, so it decides on three conditions and treats the fourth as a
qualifier. The Outcome table nonetheless prints all four under the heading
"Condition, fixed in advance".

So the outcome recorded as **H1 SUPPORTED** rests on **three** falsification
conditions passing, plus a power check. The primary test as worded has not been
run on eval under the registered clustering, and the fourth registered condition
has not been run at all.

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
therefore sits outside the registration.

**Correction, 2026-09-19b.** An earlier revision of this document said that
analysis reports the LEVEL spanning zero and therefore fires falsification
condition 1, putting the registration in direct contradiction with itself. That
was written from a single run of an ad hoc script at 600 draws. It is wrong, and
the reason is worth more than the correction.

Two implementations of the same registered quantity disagreed. Re-deriving both
independently at 2,000 draws shows why: they differ in **what a bootstrap draw
resamples**, not in arithmetic.

`analyse_slices.py:advantage_ci` prepares a cluster index **per category** and
draws each category independently. The registered LEVEL is the mean of three
per-category advantages, and one source utterance contributes slices to all
three. Drawing them independently discards that shared dependence and shrinks
the variance of their mean. `sonority_voicing.py` does the opposite by explicit
design: "One sequence of utterance resamples serves every cell and every
contrast." The two paths have never been reconciled.

Measured, seg=run, 2,000 draws, on `results/eval_slices.csv.gz`:

| quantity | clustering | draw | 95% interval | width |
|---|---|---|---|---|
| LEVEL | utterance | per category, as registered | [+0.0388, +0.0574] | 0.0186 |
| LEVEL | utterance | one shared draw | [+0.0348, +0.0613] | **0.0265** |
| LEVEL | system | per category, as registered | [+0.0162, +0.0726] | 0.0564 |
| LEVEL | system | one shared draw | [+0.0007, +0.0852] | **0.0845** |
| ASYMMETRY | utterance | per category | [+0.0453, +0.0805] | 0.0352 |
| ASYMMETRY | utterance | one shared draw | [+0.0456, +0.0813] | 0.0357 |
| ASYMMETRY | system | per category | [+0.0200, +0.0982] | 0.0783 |
| ASYMMETRY | system | one shared draw | [+0.0204, +0.1023] | 0.0819 |

Three things follow, and they matter more than the correction that produced them.

**The registered LEVEL interval is about 40 percent too narrow.** The published
+0.048 [+0.039, +0.057] should be nearer [+0.035, +0.061]. It still excludes
zero, comfortably, so this is a correction and not a retraction.

**The ASYMMETRY does not care.** Its interval moves by 0.0005 to 0.004 between
the two draw designs, because the shared draw widens the two group estimates and
pairs them at the same time, and the two effects cancel. **The quantity H1
actually asserts is the robust one.** The fragile quantity is the one the
registration's falsification condition 1 happens to name.

**Falsification condition 1 does not fire, and does not clearly pass either.**
Under system clustering with the shared draw the LEVEL lower bound is +0.0007 at
2,000 draws and -0.0002 at 600. The bootstrap cannot resolve which side of zero
it sits on. This project's own rule applies to its own registration:
**undecided is not rejected**. The honest statement is that the level is
unresolved at 13 clusters, not that it fails.

None of this is a registered result, because the registration fixes clustering
by source utterance. The rule, stated here so it is not decided case by case
afterwards:

**A robustness analysis outside the registration cannot trigger a registered
falsification condition, and cannot satisfy the registered primary test
either.** It bounds the scope of a confirmed claim. It does not adjudicate it.

The symmetry is the point. Either both quantities are reported or neither is.
The honest form is:

> Under the registered utterance clustering, the three implemented falsification
> conditions pass. A post hoc analysis that resamples the 13 synthesis systems
> instead widens every interval; under it the asymmetry remains clear of zero at
> +0.063 [+0.020, +0.102], while the sonorant level becomes unresolved, sitting
> on zero at [+0.001, +0.085]. That analysis is not registered and is reported
> as a scope bound, not as a confirmation.

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
   it to the outcome record, so the primary test as worded has a result. Done in
   `task/T15-preregistered-clustering`: **+0.063 [+0.045, +0.081]**, which
   satisfies the primary test as written.
2. Reconcile the two bootstrap draw designs, and re-issue every published
   interval on a multi-category advantage. The conclusions survive; the widths
   do not.
3. Implement the fourth registered falsification condition, or state in the
   outcome record that it was not run.
4. Report the system clustered LEVEL wherever the system clustered ASYMMETRY is
   reported.
5. Wherever the system clustered asymmetry is summarised as "H1 still
   stands", put the level beside it.
