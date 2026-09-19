# Research design

Written 2026-09-18. This document is deliberately unflattering. It exists so
that a reviewer's objections are met here first rather than in review.

Companion files. `configs/literature.yaml` records what has been checked and
what each paper forecloses. `configs/datasets.yaml` records every corpus with
its licence and source.

---

## 1. What the literature has already taken

Checked rather than assumed, and logged per paper in `configs/literature.yaml`.
Written so a reviewer's objections are met here first.

**Lightweight and accurate detection.** Closed. AASIST-L reaches 0.99% EER on
ASVspoof 2019 LA eval with 85,000 parameters and full AASIST reaches 0.83% with
297,000. Any detection framing here competes with that and loses.

**Phoneme level analysis for detection.** Closed. Zhang et al. AAAI 2025 for
features, Salvi et al. ICCV workshops 2025 for speaker conditioned analysis.

**Phoneme level explainability.** Closed. Taylor et al. 2026, from the ASVspoof
organisers.

**Which phoneme classes carry the strongest cues.** Closed. That fricatives,
plosives and diphthongs lead is published, across several papers, in the
magnitude and embedding domains.

**Cross lingual transfer, and zero shot cross lingual.** Closed. Ba et al. ACM
WWW 2023 for transfer, Kim et al. 2026 for the strict unseen setting, the latter
posted three days before this document was written.

**Vocoder attribution.** Closed. Model specific fingerprints in generated
waveforms are established, with residual based methods reportedly above 99
percent AUROC.

What no side of this has stated is **where in speech the phase deficit lands**.
The synthesis literature reports that phase is the weaker half of vocoding and
that some vocoders do not constrain it in their loss at all, so a deficit is
predicted rather than merely observed. Neither literature says which sounds
carry it, and none of the phoneme level work above operates in the phase domain.
That is the gap this occupies.

Caveat on all of the above. Searches cannot prove absence, and the search pass
behind this section has a knowledge boundary. One deliberate pass is required
before anything is written up.

---

## 2. A preliminary result that suggests one

`scripts/phoneme_probe.py` reduces each phoneme slice to a few scalars and
measures how separable bonafide is from spoof, with no model and no training.
Every phase statistic is paired with a magnitude statistic over identical
framing, so the comparison isolates the domain.

Run on the existing 300 utterance pilot slices, corrected MGD against MFCC.

| Category | best MGD AUC | best MFCC AUC | phase advantage |
|---|---|---|---|
| NASALS | 0.702 | 0.583 | **+0.120** |
| VOWELS_STEADY | 0.700 | 0.600 | **+0.099** |
| APPROXIMANTS | 0.713 | 0.647 | **+0.066** |
| PLOSIVES | 0.603 | 0.591 | +0.013 |
| FRICATIVES | 0.609 | 0.632 | −0.023 |
| VOWELS_MOVING | 0.536 | 0.689 | −0.153 |

Three things stand out.

**Phase carries signal beyond magnitude, in four of six categories.** That is the
gate. Had phase lost everywhere, the whole thesis would be dead regardless of
framing. It did not.

**The advantage sits in sonorants, not obstruents.** Nasals, approximants and
steady vowels are the phase wins. Fricatives, the class the February framing
named, is one of only two categories where magnitude wins.

There is a physical account for that, and it is the opposite of the project's
current story. Sonorants are voiced and periodic, so they carry coherent
harmonic phase structure that a vocoder has to reconstruct and generally
reconstructs badly. Fricatives are aperiodic broadband noise, where phase is
close to random in real speech too, so there is little structure for a synthesis
artifact to disturb. Magnitude, meanwhile, does discriminate on fricatives,
which is exactly what the published phoneme level work reports.

**The direction is consistent with Spectral Collapse.** Every MGD statistic runs
`bonafide_higher`, meaning real speech has greater phase dispersion than
synthetic. That is the claim the project has been making, now measured on a
feature that is actually phase.

### What this is not

Small. Bonafide counts range from 35 to 129 against 400 spoof, drawn from 300
source utterances of one corpus in one language. AFFRICATES was skipped at 10
bonafide. An AUC from 35 samples is not a number to build on.

Weak. AUCs run 0.53 to 0.71. These are scalar summaries, not a detector, but the
effect is not dramatic.

Unreplicated. One corpus, one language, no held out partition, no correction for
testing seven categories against each other.

Treat the table as a direction worth testing, not a result.

---

## 3. The candidate contribution

**Magnitude and phase carry complementary phoneme level information. Magnitude
discriminates on obstruents, phase on sonorants.**

If that survives scale, it is a physically grounded justification for fusing the
two, which is what the architecture already does but has never justified. It
differs from the published phoneme level work, which operates in the magnitude
and embedding domains and therefore cannot see the split. It also reframes the
fusion from an engineering choice into a prediction.

It suggests a cross lingual mechanism too. If the artifact is tied to
articulatory class rather than to language, a detector conditioned on
language universal phonetic categories should transfer where a whole utterance
model does not. That is a mechanistic explanation for the transfer failure
already observed, and it is falsifiable. The slicer already uses an IPA
outputting multilingual wav2vec 2.0 model, so the phonetic space is shared
across languages by construction.

### Honest assessment of its size

This is a thesis chapter or a workshop paper. It is not a headline result. It
will not be competitive as a detector contribution and should not be submitted
as one.

### What would kill it

The ablation showing corrected MGD adds nothing over MFCC at the model level,
even though the probe suggests otherwise at the statistic level.

The sonorant advantage failing to replicate at scale, or reversing on another
corpus, which small n makes entirely possible.

The silence confound explaining the split. Sonorants and obstruents differ in
duration and in surrounding silence, so an apparent phase effect could be a
segmentation artifact. This must be ruled out with `trim_silence`.

The effect disappearing once phase is measured on a held out partition with
unseen attacks, A07 to A19, rather than on training attacks.

---

## 4. Framing, decided

**Analysis, with a detector component that follows from it rather than leads.**

Decided 2026-09-18, when the project moved from Master's work to an open ended
one. Recorded here with the reasoning so it can be revisited rather than
re-argued.

### Why not a detector contribution

Not because of time. Time is no longer the constraint. Because the payoff is
poor and the competition is well resourced.

Beating AASIST-L means competing with EURECOM, NII, KAIST and Naver on their
benchmark, with their compute, for a marginal gain from 0.99% EER to something
slightly under it. That is a grind with no distinctive result at the end of it,
and it is the least interesting version of this work. Removing the deadline
makes that race possible in principle. It does not make it worth running.

### Why analysis

There is already a live preliminary finding in section 2 that nobody else has,
because nobody else is looking at phase conditioned on phoneme class. Following
it needs modest compute. The probe runs in minutes on CPU.

An open ended project is good at exactly this and bad at leaderboard chasing.
Curiosity into a gap is the comparative advantage here, not throughput.

An analysis contribution also survives a 10% EER. A detector contribution does
not. The bar for analysis is rigour, and rigour is affordable.

### The part that is not a binary

The choice above is not analysis instead of a model. It is analysis first, with
the model as a consequence.

If the sonorant and obstruent split in section 2 holds, the natural next step is
a phoneme conditioned fusion. Weight phase where phase discriminates, weight
magnitude where magnitude does, with the weighting derived from measurement
rather than tuned. If that beats naive equal weighting, the analysis has
produced a mechanism and the mechanism has produced an improvement. Each then
supports the other.

Note what that claim does not require. It does not require beating AASIST. It
requires beating **our own ablation arms**, which is a controlled, self
contained comparison on identical data and identical training. That is a claim
this project can actually make and defend.

### What this means in practice

Report EER, t-DCF and per attack breakdowns because they are needed for
credibility, not because the numbers are the contribution.

Compare against AASIST-L honestly as context, and say plainly that nothing here
is competitive as a general detector. Stating that first removes a reviewer's
main objection before they raise it.

Put the effort into measurement quality. Sample sizes, held out partitions,
multiple comparison correction, confound controls. That is where an analysis
contribution lives or dies.

Treat the repository as a deliverable in its own right. For an open ended
project the artifact existing, reproducible and usable by other people is a
substantial part of the value, in a way it is not for a thesis.

## 5. Experiment order

Gates first. Each of the first three can end a line of work, so run them before
investing in anything downstream. They are cheap.

1. **Resolve dev versus eval.** Until this is known there is no baseline. If the
   10.0% was on dev it is on attacks already seen and means much less.
2. **Ablation, MFCC against MGD against both.** `--channels` does this from one
   subset build. If phase adds nothing at the model level, sections 2 and 3 are
   void regardless of what the probe shows at the statistic level.
3. **Silence.** `trim_silence`, retrain, compare. Zhang et al. 2023 show TTS
   speech carries less silence than bonafide, so a detector can learn padding.
   Sonorants and obstruents also differ in duration and surrounding silence, so
   this is a direct confound for the section 2 split, not a general worry.

Then the analysis core. This is the contribution.

4. **Rerun the probe at scale.** Thousands of slices per class rather than 35 to
   129, on the eval partition with unseen attacks, with a correction for testing
   seven categories against each other.
5. **Cross lingual probe.** Do the same phoneme classes show the same phase
   behaviour in another language? This is the sharpest test of the mechanism. If
   the effect is articulatory it should be language universal, and the slicer
   already emits IPA from a multilingual model so the phonetic space is shared
   by construction. Promoted above the detector work because a positive result
   here is the finding, and a negative one closes the line honestly.
6. **Regenerate the Spectral Collapse artifacts** through corrected features with
   corpus level normalisation. The existing figures came through the ramp.

Then the model, as a consequence rather than a goal.

7. **Phoneme conditioned fusion.** Weight the channels by what steps 4 and 5
   measured. Compare against naive equal weighting on identical data. This is
   the claim the project can defend.
8. **Data efficiency curve** on Bengali or IndicSynth. Ten, fifty, one hundred,
   five hundred, a thousand target samples.

Then context, needed for credibility rather than as the contribution.

9. **Baselines and required metrics.** AASIST-L on the same splits, t-DCF, per
   attack EER.

## 6. The main risk

It changed when the deadline went away, and it got harder to notice rather than
easier.

The old risk was doing eight experiments shallowly before a deadline. The new
risk is drift. With no deadline, nothing forces a line of work to be closed, so
directions accumulate unfinished and the project becomes a collection of
half answered questions. Nine registered corpora and nine experiments is already
enough surface for that to happen.

The defence is the gates. Steps 1 to 3 are cheap and each can kill a direction
outright. Run them and act on the answers, including the unwelcome ones. If the
ablation shows phase adds nothing, section 3 should be deleted rather than
reframed.

Second risk, specific to an open ended project. Nobody is checking, so
measurement quality slips quietly. The probe result in section 2 is exactly the
kind of thing that is tempting to believe. It is built on 35 to 129 bonafide
samples per class from one corpus in one language with no correction for
multiple comparisons. Step 4 exists to find out whether it is real, and it has
to be run before the finding is repeated anywhere.
