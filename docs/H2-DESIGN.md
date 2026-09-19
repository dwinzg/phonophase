# H2 design, the cross linguistic test

Draft 2026-09-18, before any data is pulled. Not yet a pre-registration. That
gets written once the design is settled and before the run.

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
directly to the unexplained heterogeneity already seen in English where the
effect holds in 8 of 13 attacks.

### griffin_lim is a positive control, and should be named as one

Griffin-Lim reconstructs phase iteratively from magnitude alone. It has no
learned phase model at all. Under the hypothesis it should therefore show the
**largest** phase deficit of any system in the set, and the sonorant pattern
should be at its clearest there.

This is worth naming in the pre-registration as a prediction, because it is a
cheap and severe test. If a method with no phase model does not show the effect
most strongly, the account is wrong and no amount of averaging over the other
17 systems will rescue it.

### One access blocker

MLAAD is gated. Directory listings are public but everything under `fake/` needs
the account to be authorized. That blocks `architecture`, `is_original_language`
and `reference_speaker`, which are precisely the three covariates this design
needs to control. Resolve before the run, not during.

## The real side, and why it changed the language set

Verified 2026-09-18. M-AILABS supplies the genuine speech and is usable. Modified
BSD three clause licence, already mono 16 kHz, and MLAAD's `original_file` gives
utterance level pairing. The original host is dead, the live route is
`https://ics.tau-ceti.space/data/Training/stt_tts/{lang}.tgz`.

The problem is speaker counts. **Polish has 2 real speakers, English 3, French
unknown**, and M-AILABS documents none of this. With two voices on the real
side, a real versus synthetic separation risks separating those two particular
voices from TTS voices rather than genuine from synthetic speech.

**Spanish is added as a fourth language in response.** The identical 18 models
that span English, French and Polish also span Spanish, so coverage costs
nothing, and Spanish has 21 M-AILABS speakers against Polish's 2.

That turns the confound into a control rather than a caveat. Speaker diversity
now varies deliberately across the set, from 2 to 21. If the effect holds at the
same magnitude in Spanish as in Polish, speaker diversity is not driving it. If
it does not, that is itself a finding. Dropping Polish would have been weaker,
since it would have discarded the phonological contrast that justified it.

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

**Spanish.** Added for speaker diversity, 21 real speakers against Polish's 2,
with a comparatively simple phonology. It is the speaker rich end of the
control, and the same 18 systems cover it.

Between them these cover both arms of the phonological claim and both ends of
the speaker diversity control. Adding further languages before the design is
validated buys noise.

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

Confirm M-AILABS language coverage and that real and synthetic can be paired per
language.

Decide how `is_original_language` is handled. MLAAD includes audio generated
from translated text, so the phonetics are of the target language but the text
distribution is not native. Either restrict to original language or carry it as
a covariate.

Count how many systems actually span the chosen languages with enough
utterances. The design needs at least a few, and if only one system spans them
the result rests on that system alone.

Decide whether `reference_speaker` needs controlling, since voice cloning
reference affects phase directly.

Fix the tone digit handling in `phonophase.phonology` if any tonal language
enters later. It is not needed for English, French and Polish.
