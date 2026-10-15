# Plan

> **A historical plan. Note added 2026-10-14.** Written 2026-09-18 and kept as
> a record of how the work was sequenced. Phase 1 has since run: H1 is confirmed
> as an observation on ASVspoof 2019 LA eval (`docs/PREREGISTRATION.md`), and
> its cause is undecided, because a low quality resampler on genuine speech
> alone reproduces its signature (`docs/BAND-EDGE.md`, rule R1). No mechanism
> may be stated for it, so the articulatory reading in Phase 2 is a hypothesis
> only. Phase 3's phoneme conditioned fusion is out of scope (`docs/SCOPE.md`).

Written 2026-09-18. Framing is analysis, decided in `docs/RESEARCH.md` section 4.

**Push discipline**, project practice. Work stays local and is committed per step. A phase is pushed
only once its verification passes. Steps within a phase are grouped because
they share a verification, not because they happened at the same time.

---

## Phase 0, make the instrument trustworthy. Done, local.

The first probe run reported bare AUC from as few as 35 samples across roughly
49 tests with no interval, no null and no correction. That is not a measurement.

Built `src/phonophase/stats.py`, with stratified bootstrap intervals, a permutation
null, a paired bootstrap on the difference between two statistics measured on
the same slices, and Benjamini Hochberg FDR. 17 tests.

Rebuilt the probe on top of it, with a **pre-registered primary comparison**
fixed before running, duration control, and FDR across every exploratory test.

### What it found, and it is not what the first pass said

**The original result was selection bias.** Phase ahead in four of six
categories came from picking the best phase statistic and comparing it to the
best magnitude statistic. Under the pre-registered comparison on frame spread,
phase is ahead in **zero** of six categories, magnitude wins two, and four are
inconclusive.

**Duration was never a confound, for an unexpected reason.** Every slice is
exactly 60ms. The slicer takes a fixed window around a CTC peak frame rather
than the phoneme's actual boundaries, so longer phonemes are truncated and
shorter ones carry neighbouring context. That is a real limitation of the
segmentation and it weakens every phoneme conditioned claim built on it.

**Something narrower did survive.** `mgd_hf_std`, dispersion of Modified Group
Delay in the upper half of the spectrum, is the strongest statistic in three of
six categories, with intervals clear of chance and surviving FDR.

| Category | statistic | separation | 95% CI | q | direction |
|---|---|---|---|---|---|
| NASALS | mgd_hf_std | 0.718 | [0.661, 0.783] | 0.006 | bonafide higher |
| APPROXIMANTS | mgd_hf_std | 0.714 | [0.639, 0.777] | 0.006 | bonafide higher |
| VOWELS_STEADY | mgd_hf_std | 0.696 | [0.648, 0.743] | 0.006 | bonafide higher |
| FRICATIVES | mgd_iqr | 0.629 | [0.576, 0.679] | 0.006 | bonafide higher |
| PLOSIVES | mgd_std | 0.599 | [0.542, 0.661] | 0.011 | bonafide higher |
| VOWELS_MOVING | mfcc_std | 0.666 | [0.562, 0.758] | 0.008 | spoof higher |

The three strongest are all sonorants, all on the same statistic, all in the
direction the predecessor project called Spectral Collapse, with bonafide carrying greater phase dispersion.

**This is exploratory and cannot be claimed.** It was selected after seeing the
data. The pre-registered test does not support it. The correct next move is to
register it as a hypothesis and test it on data that had no part in finding it.

---

## Phase 1, confirm or kill it

One decision. Everything downstream depends on it.

1. **Register the hypothesis before looking at more data.** Write it down.
   `mgd_hf_std` separates bonafide from spoof more strongly in sonorants, nasals
   approximants and steady vowels, than in obstruents, with bonafide higher.
   State the falsification condition. Register `mgd_hf_std` against
   `mfcc_frame_spread` as the new primary pair.
2. **Fix the segmentation.** Use real phoneme boundaries rather than a fixed
   60ms window. Any result on fixed windows is about windows, not phonemes.
3. **Slice at scale.** All 2,580 bonafide training utterances rather than the
   300 used so far. GPU work, so Colab.
4. **Run the probe on the eval partition.** Attacks A07 to A19, none seen in
   training. If the effect is a property of A01 to A06 it will not survive.

Verification to push. The registered hypothesis either holds on eval with
intervals clear of chance, or it does not. Both outcomes are publishable and
both get pushed. **Commit group, `probe: confirmatory run on eval`.**

If it fails, stop here and write it up as a negative result. That is a real
contribution given how much of this literature reports only successes.

---

## Phase 2, the mechanism and the use case

Only if phase 1 holds.

5. **Cross lingual probe.** MLAAD, one additional language to start, sliced with
   the same IPA model so the phonetic space is shared by construction. Does the
   same statistic behave the same way on the same articulatory classes?

   This is the sharpest test. If the artifact is articulatory it should be
   language universal. If it is not, it was English specific and the cross
   lingual story dies cleanly.

6. **Then scale to many languages.** MLAAD carries 54. The question becomes
   whether per language transfer difficulty is predictable from that language's
   sonorant proportion.

Verification to push. Effect replicates in at least one additional language
with intervals clear of chance. **Commit group, `probe: cross lingual`.**

The use case sits here. If phase carries a language universal cue while
magnitude carries a language specific one, that is a deployment recipe. Weight
phase when going cross lingual, and estimate degradation from the target
language's phonetic inventory without needing labelled deepfakes in it. That is
the question a practitioner actually has, and nobody currently answers it.

---

## Phase 3, the model as a consequence

7. **Gates.** dev versus eval, the channel ablation, and the silence check. Run
   these before any model claim. Each can close a direction.
8. **Phoneme conditioned fusion.** Out of scope (`docs/SCOPE.md`); kept as
   originally planned. Weight channels by what phases 1 and 2
   measured rather than by tuning. Compare against naive equal weighting on
   identical data. The claim is beating our own ablation arms, not AASIST.

Verification to push. Conditioned fusion beats naive fusion on held out data
with a paired interval excluding zero. **Commit group, `model: conditioned
fusion`.**

---

## Phase 4, make it usable

9. **The probe as a standalone tool.** It already runs on any directory of
   sliced audio. Document the input contract, publish the statistics module
   separately, and include a worked example. This is the part other people can
   pick up regardless of whether our finding holds.
10. **Registries as artifacts.** `configs/datasets.yaml` and
    `configs/literature.yaml` are useful to anyone entering this area, including
    the licence verification flags.
11. **Write up, including what failed.** The selection bias correction in phase
    0 is itself worth reporting. A bare AUC table became a very different
    picture under intervals and correction, which is a common failure in this
    literature.

**Commit group, `docs: tool and write-up`.**

---

## Out of scope for this plan

Chase ASVspoof leaderboard performance. Covered in `docs/RESEARCH.md` section 4.

Add more corpora before phase 1 resolves. Nine are registered and none of them
help until the instrument has produced one confirmed result.

Report the phase 0 exploratory table as a finding anywhere. It is a hypothesis.
