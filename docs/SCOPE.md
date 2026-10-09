# Scope

Set 2026-09-18, after the decision not to import the UBC ARC files. That
decision is what makes this document necessary, because it changes what the
project can honestly claim.

> **Status, 2026-10-08.** The question below is the one this project set out
> with, and it presupposes more than has been shown. The pattern it predicts was
> confirmed. That the pattern is a *phase artifact of synthesis* was not: a
> sample rate conversion with no synthesis in it reproduces it, and a magnitude
> statistic tracks it. See `docs/BAND-EDGE.md` and `docs/ROBUSTNESS.md`. Read
> "the phase artifact" below as "the pattern", and its cause as open.

## The question, stated precisely

**Is the phase artifact left by speech synthesis a property of articulation
rather than of language?**

That is the whole project in one sentence, and the precision matters because two
looser versions of it are wrong in ways that would waste the work.

### It is not "can we detect deepfakes from phase"

That question is already answered here, and the answer is no, not on its own.
`mgd_hf_std` separates at 0.51 to 0.60 AUC depending on phoneme class. A single
statistic at 0.60 detects nothing useful, and AASIST-L reaches 0.99% EER with
85,000 parameters, so the detection framing is closed either way.

What is interesting is not the level, it is the **asymmetry**. The artifact is
not spread evenly through speech. It concentrates in sonorants, nasals,
approximants and steady vowels, and is close to absent in plosives. That
structure is the finding. It was read here as what a mechanism looks like; see
the status note above for why no mechanism is claimed now.

### It is not "generalise to all languages"

No study concludes anything about all languages, and that phrasing invites a
claim the evidence can never carry. The testable version is narrower and
sharper.

Sonorants are voiced and periodic in every language. A nasal is produced the
same way in Bengali as in German. So if the artifact is **articulatory**, the
same articulatory classes should carry it in a language absent from training,
because a vocoder is failing at the same physical reconstruction regardless of
what language is being spoken.

That is what language universal means here. Not "works on all languages", but
"indexed to how sounds are produced rather than to which language produces
them". It is falsifiable in a single additional language, which is H2.

### What each outcome means

**H2 positive.** The artifact is articulatory. That explains cross lingual
transfer mechanically rather than empirically, and it predicts which classes
carry the signal in a language nobody has labelled data for. That is the useful
version of this work.

**H2 negative.** The effect was English, or corpus specific, and the sonorant
result is a local fact about ASVspoof rather than about speech. Worth reporting
plainly, and it closes the line cleanly.

### A warning the existing data already gives

Within English, the sonorant advantage holds in only **8 of 13** attacks and
reverses on A08, A13, A16 and A19. So the effect does not even generalise across
synthesis systems in one language. Expect H2 to be partial at best. Anyone
setting out to show it holds everywhere is going to be disappointed, and should
instead be asking which systems and which languages break it, because that is
where the mechanism actually is.

## What the project is

**A measurement study of where phase information lives in speech deepfake
detection, with the instrument released.**

Not a detector. Not a benchmark entry. A measurement study.

## Why it is that and not something else

The detector framing is closed. AASIST-L reaches 0.99% EER on ASVspoof 2019 LA
eval with 85,000 parameters. Nothing here competes with that, and chasing it
means grinding against well resourced labs on their benchmark for a marginal
gain. See `docs/RESEARCH.md` section 4.

What this project claims is a measurement, and only what it can reproduce. `mgd_hf_std` separates bonafide
from spoofed speech more strongly in sonorants than in obstruents, confirmed on
held out attacks under a pre-registered test. That is small and narrow, but it
is real, and it is the only thing here that anyone else can verify.

## In scope

**The finding.** H1 is confirmed on ASVspoof 2019 LA eval. H2, whether it holds
in a language absent from training, is the open question and the one that
decides whether this is a mechanism or an English artifact.

**The instrument.** `phonophase.stats`, `phonophase.probe`, `scripts/phoneme_probe.py`,
`scripts/slice_and_probe.py`, `scripts/analyse_slices.py`. Corpus agnostic
through the manifest route, 78 tests. Useful to anyone measuring where a signal
lives across phoneme classes, whatever their detector.

**The methodology cautions.** Every one was verified here and every one is a
general failure mode rather than a local mistake.

- Group delay computed with a signal wide ramp instead of a per frame one,
  correlating 0.999999 with frame index
- Raw group delay presented as Modified Group Delay, with no cepstral smoothing
  or compression
- Per file normalisation dividing out the dispersion being measured
- Fixed width windows around CTC peaks described as phoneme segments
- Best of many statistics compared against best of many others, turning zero of
  six into four of six
- Slices from one utterance bootstrapped as independent observations, when 129
  came from 23 utterances

The instrument detects the last two directly.

## Out of scope, stated so it stays out

Competing on ASVspoof EER.

Zero shot cross lingual detection. Taken by Kim et al. 2026.

Building a Bengali detector.

Importing anything from UBC ARC.

## On adding machine learning methods

Decided 2026-09-19, because the question will keep coming up and the answer is
not the same in every direction.

A method earns its place here by making a measurement **more trustworthy**, not
by making the project sound more current. Applied to the three obvious
candidates, that rule gives three different answers.

**Where it would genuinely help.** `mgd_hf_std` is one hand chosen scalar, and
whether it is the right one is a standing open question. A cross validated
**linear probe** on the phase plane, fitted per phoneme class, would measure how
much phase information is present far more completely than a single statistic
can. Conditions if it is done: held out by **utterance** and not by slice, the
same probe family for every class so no class gets a better model than another,
and **registered before it is run**. `mgd_hf_std` was itself selected from
exploratory analysis, so quietly replacing it with something that scores better
is the best of many failure in a new costume.

**Where it would actively hurt.** Adapting the alignment model per language,
with LoRA or by any other means, for the cross linguistic test. H2 compares
languages. If the aligner is adapted for each one, alignment quality varies with
language, and alignment quality is then confounded with the effect being
measured. The correct move for H2 is the opposite of adaptation: one unadapted
multilingual model everywhere, so that whatever alignment error exists is
constant across the comparison and cannot masquerade as a language effect. Use
forced alignment to reduce that error, not adaptation.

**Where it is simply a different project.** Parameter efficient fine tuning for
a detector, a data efficiency curve, phoneme conditioned fusion. All are already
out of scope above, and nothing here changes that. They become interesting if
the mechanism holds, which is what H2 and H3 decide.

## Deliverables

1. H1, confirmed. Done.
2. H2 on one or two additional languages via MLAAD. One Colab run, tooling
   already supports it through `--manifest` and `--by-language`.
3. The attack heterogeneity question. The sonorant advantage holds in 8 of 13
   attacks and reverses on A08, A13, A16 and A19. Which systems break it, and
   what they share, is the most interesting loose thread.
4. Write up, including the methodology cautions and whatever H2 returns.
5. The instrument, documented for use on other corpora.

## Finish line

H2 resolves and the write up exists. That is a complete, self contained piece of
work whether H2 is positive or negative, and it is reachable.

Anything past that, phoneme conditioned fusion, data efficiency curves, more
corpora, is a second project. Do not start it before this one is finished. The
standing risk on an open ended project is drift, and this repository has already
changed direction three times.

## Out of the repository by design

The detector path is not here. A CNN and Transformer is not part of a
measurement study, and leaving it out keeps the premise legible. If phoneme
conditioned fusion later becomes relevant, it is a separate piece of work with
its own justification.

Everything in this repository is something we measured here and can reproduce
from what is committed.
