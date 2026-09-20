# H2 design, the cross linguistic test

Draft 2026-09-18, revised 2026-09-19. Not yet a pre-registration. That gets
written once the design is settled and before the run.

**Three things changed in the revision and each one weakens a claim this
document previously made.** The speaker diversity control does not exist,
because Spanish has 3 real speakers and not the 21 an unverified source
reported. The griffin_lim positive control is contradicted by ASVspoof's own
Griffin-Lim system. And the crossed sonority against voicing test cannot travel
to these languages, because none of them has phonemic voiceless sonorants.

None of that blocks H2. All of it would have been discovered after the run
rather than before it.

## What H2 asks

H1 established that in English, phase dispersion separates real from synthetic
more in sonorants than obstruents. H2 asks whether that is a fact about
**articulation** or a fact about **English**.

## The confound that shapes everything

The obvious design, measure language A and language B and compare, does not
work. Different languages are synthesised by different systems, so a language
difference and a vocoder difference are inseparable.

This is not hypothetical. Within English the sonorant advantage already holds in
only 8 of 13 attacks and reverses on four of them. Vocoder variation is large
and it would swamp any language effect in a naive comparison.

## The design that does work

MLAAD records `model_name`, `architecture`, `language` and
`is_original_language` per sample, and the same TTS systems appear across
multiple languages. So the comparison can be made **within a single synthesiser
across languages**, which holds the vocoder fixed and lets language vary.

    for each TTS system S that covers languages L1, L2, L3:
        for each language L:
            measure the articulatory feature profile of the
            real versus S-synthetic separation
        ask whether the profile is the same across L

Same model, same architecture, same training recipe. If the profile holds, the
artifact is articulatory. If it changes with language under a fixed synthesiser,
it is not.

That is a genuinely controlled comparison and it is the whole reason MLAAD is
the right corpus rather than a collection of per language datasets.

## The baseline is per language, not English

One correction to the obvious framing. English real speech cannot serve as the
reference for another language's synthetic speech, because the two would differ
in phone inventory, speaker set and recording conditions before synthesis is
considered at all. Separation is always measured **within** a language, real
against fake.

English keeps a different and more useful role. It is where the feature profile
was established, so it is the **prediction** being tested elsewhere, not the
baseline being subtracted.

MLAAD ships synthetic audio only and pairs with M-AILABS for the real side, so
languages are limited to those M-AILABS covers.

## Feasibility, verified 2026-09-18

The design is viable and better powered than assumed. Verified against the
HuggingFace tree API, not taken from the paper.

**18 models produce audio in all three target languages**, each with 1000
utterances per language. A further 20 span exactly two. English has 143 model
directories, French 42, Polish 21.

The 18 span a wide range of eras, from classical Griffin-Lim through GAN era
bark and xtts to recent LLM based TTS such as MOSS-TTS and Llasa. That spread
lets us ask whether the effect is architecture dependent, which connects
directly to the heterogeneity seen in English. Stated correctly, after T6: on
the eval attacks the asymmetry is clear and positive on 4, clear and negative on
1, and undecided on 8 after correction. The variation is real but most of it is
unresolved rather than contrary, and the classical parametric vocoders lead.

### griffin_lim is a positive control, and should be named as one

**Unresolved, and this section previously overstated it.** Griffin-Lim as an
algorithm reconstructs phase iteratively from magnitude alone, with no learned
phase model. If MLAAD's `griffin_lim` applies it to the *original* audio's
magnitude, then real and fake share speaker, content and magnitude and differ
essentially only in phase, which would be the cleanest test in the study. If it
instead vocodes a mel predicted by an acoustic model, it preserves nothing and
is merely the extreme phase case.

The metadata says `training_data: none` and `architecture: griffin_lim`, which
is consistent with the first reading but does not establish it. A duration test
did not discriminate, griffin_lim at mean 7.44s against Edge-TTS at 7.05 and
xtts_v2 at 7.30. Settle it by comparing magnitude spectrograms of one
griffin_lim file against the M-AILABS file its `original_file` names, and do not
rely on it as a control until then.

### The prediction this section used to make is already in trouble

**Updated 2026-09-19, and this is the most important change to the design.**

This section previously said griffin_lim should show the largest phase deficit
of any system, and that if a method with no phase model did not show the effect
most strongly, the account was wrong.

ASVspoof already contains that test and it came out the other way. A11 is
Tacotron 2 with Griffin-Lim, the extreme case of phase generated with no learned
phase model at all. Its asymmetry is **+0.034 with an interval spanning zero**,
well short of A09 Vocaine at +0.218 and A14 STRAIGHT at +0.122, which are
classical parametric source filter vocoders.

So iterative phase reconstruction from magnitude does **not** maximise the
effect in the one corpus where we can check. Two readings, and MLAAD's
griffin_lim discriminates between them rather than confirming either:

**If the mechanism is phase reconstruction as such**, griffin_lim should lead,
and A11 becomes a corpus specific anomaly needing its own explanation.

**If the mechanism is parametric excitation modelling**, the systems that impose
a pulse-or-noise source and a minimum phase envelope should lead, which is what
Vocaine and STRAIGHT are, and griffin_lim should sit mid table exactly as A11
does.

The second currently fits the evidence better. It is a hypothesis generated
after seeing A11, so it must be registered before it is tested and must not be
presented as a finding. Register the DIRECTION of griffin_lim relative to the
parametric systems, not the assumption that it leads.

### Access, resolved

MLAAD is gated and the account is now authorized. Reading gated `meta.csv`
returns http 200. One schema correction came out of it: the final field is
`speaker_similarity`, not `reference_speaker` as the paper and card state, and
it is empty in every row checked.

Also found: models do not share source files. Each draws its own 1000 M-AILABS
files with zero overlap between the models checked, so there is within model
real to fake pairing but no cross model pairing on identical utterances. That is
fine for a design that compares within a system, and it rules out comparing two
systems on the same sentences.

## The real side, and why it changed the language set

Verified 2026-09-18. M-AILABS supplies the genuine speech and is usable. Modified
BSD three clause licence, already mono 16 kHz, and MLAAD's `original_file` gives
utterance level pairing. The original host is dead, the live route is
`https://ics.tau-ceti.space/data/Training/stt_tts/{lang}.tgz`.

The problem is speaker counts, and the figures below are **measured from the
data** rather than taken from documentation, which is how the earlier version of
this section went wrong.

| language | named speakers |
|---|---|
| French | 5 |
| English | 4 |
| Spanish | 3 |
| Polish | 2 |

**Correction, 2026-09-19.** This section previously said Spanish has 21 real
speakers, and added Spanish to the language set specifically to create a speaker
diversity control ranging from 2 to 21. That figure came from Montreal Forced
Aligner documentation and was never verified. Counted from `original_file` in
1000 MLAAD rows per language, Spanish has **3**.

**So the speaker diversity control does not exist.** The real range is 2 to 5,
which is far too narrow to vary deliberately. Any claim in this design that
rests on contrasting a speaker rich language against a speaker poor one is
withdrawn.

**What controls speaker instead, and it is stronger.** Every MLAAD row carries
`original_file`, naming the specific M-AILABS recording it was derived from. So
a synthetic sample is compared against **its own source**: same speaker, same
text, same recording session. That is exact control rather than balancing, it is
available in all four languages, and it is now the load bearing element of the
design. A corpus proposed as a replacement for MLAAD has to be checked for an
equivalent pairing before it is adopted.

## Language selection

Four, chosen to vary both phonology and speaker diversity rather than for
convenience.

**English.** The anchor. The profile is already measured here.

**French.** Has phonemic nasal vowels, which are [+son, +nas]. Under an
articulatory account these should pattern with other sonorants. Under a language
account there is no reason they should. A sharp test, and it uses a contrast
English does not have.

**Polish.** A rich sibilant and affricate inventory, so it loads the obstruent
side where the hypothesis predicts little signal. If separation stays low there
under a fixed synthesiser, that is the negative half of the prediction holding.
Only 2 real speakers, so it is also the speaker poor end of the control.

**Spanish.** Its original justification, speaker diversity, was based on a wrong
figure and is withdrawn. It stays only on the weaker ground that the same 18
systems cover it at no extra cost and it adds a fourth, phonologically simple
point. If the run needs to be smaller, Spanish is the one to drop.

### What T7 adds, and the limit it runs into

T7 found that sonority and voicing both carry the effect, additively and at
similar size, and that English cannot separate them because its only voiceless
sonorants are the glottals. So the ideal H2 language is one that **breaks the
sonority and voicing collinearity**.

Checked against this language set, the news is half good.

**The voiced obstruent arm can be strengthened a lot.** Polish, and Russian or
Ukrainian if ever added, have far richer voiced obstruent inventories than
English, with paired sibilants and affricates. That populates the [-son, +voi]
cell properly instead of resting it on seven English phones.

**The voiceless sonorant arm cannot be strengthened at all.** None of the eight
M-AILABS languages has phonemic voiceless sonorants. The [+son, -voi] cell will
be glottals again in every one of them.

**Consequence for the design, stated now rather than discovered later.** The
crossed 2 by 2 test stays an English only analysis. H2 should test the ladder,
[+son,+voi] against [-son,+voi] against [-son,-voi], which all four languages
support and which Polish supports better than English does. Fully separating
sonority from voicing needs a language with voiceless sonorants, such as Welsh,
Icelandic or Burmese, and none is reachable through MLAAD paired with M-AILABS.
That is a separate corpus problem and should not hold up H2.

## Measurement

Per language and per system, the articulatory feature profile is the vector of
separations across PanPhon's 24 features, not the seven coarse categories. The
coarse map covers 41 of 387 phones and discards aspiration, nasalisation, length
and retroflexion, which is most of what distinguishes these languages.

Comparison is between profiles, by rank correlation across features, with
bootstraps clustered by source utterance as everywhere else.

## What would falsify it

The feature profile under a fixed synthesiser differs between languages by more
than it differs between synthesisers within a language. That would mean language
dominates articulation and the mechanism is not what the account claims.

Sonorants failing to lead in a language other than English, under a system that
shows the effect in English.

French nasal vowels patterning with obstruents rather than sonorants, which
would specifically contradict the articulatory reading.

## Open before this becomes a pre-registration

~~Confirm M-AILABS language coverage and that real and synthetic can be paired
per language.~~ **Resolved 2026-09-18.** All eight languages confirmed, modified
BSD licence, already mono 16 kHz, pairing available through `original_file`.

Decide how `is_original_language` is handled. MLAAD includes audio generated
from translated text, so the phonetics are of the target language but the text
distribution is not native. Either restrict to original language or carry it as
a covariate.

Count how many systems actually span the chosen languages with enough
utterances. The design needs at least a few, and if only one system spans them
the result rests on that system alone.

~~Decide whether `reference_speaker` needs controlling.~~ **Cannot be, as
documented.** The field is actually named `speaker_similarity`, not
`reference_speaker` as the paper and dataset card state, and it is empty in
every row checked across griffin_lim, xtts_v2 and Edge-TTS. The voice cloning
covariate is therefore not usable. Voice cloning reference does affect phase
directly, so this is a real uncontrolled variable and belongs in the writeup as
one.

Fix the tone digit handling in `phonophase.phonology` if any tonal language
enters later. It is not needed for English, French and Polish.
