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

## Language selection

Two or three, chosen to maximise contrast in the features the hypothesis is
about rather than for convenience.

**English.** The anchor. The profile is already measured here.

**French.** Has phonemic nasal vowels, which are [+son, +nas]. Under an
articulatory account these should pattern with other sonorants. Under a language
account there is no reason they should. A sharp test, and it uses a contrast
English does not have.

**Polish.** A rich sibilant and affricate inventory, so it loads the obstruent
side where the hypothesis predicts little signal. If separation stays low there
under a fixed synthesiser, that is the negative half of the prediction holding.

Between them these cover both arms of the claim. Adding more languages before
the design is validated buys noise.

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
