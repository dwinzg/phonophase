"""
IPA to articulatory feature vectors, via PanPhon.

Why this replaces the seven category map
    `01_phoneme_slicer.py` sorted phones into PLOSIVES, FRICATIVES, NASALS,
    APPROXIMANTS, AFFRICATES, VOWELS_STEADY and VOWELS_MOVING. That map covers
    41 of the 389 phone symbols the alignment model emits, so **89 percent of
    the inventory was discarded**, and what it discarded was not random. It
    threw away aspiration and breathy voice (bʰ, dʰ), palatalisation (bʲ),
    nasalised vowels (ã), length (aː) and retroflexion (ʈ, ɖ).

    Those are exactly the contrasts that distinguish languages. Bengali's
    defining property is a four way stop contrast including breathy voice, so
    the old map would have discarded most of Bengali before measuring anything.

    A coarse category is also the wrong unit for a cross linguistic claim.
    Comparing NASALS in English to NASALS in Bengali only works for categories
    the two share. Articulatory features are universal by construction, so every
    phone in every language decomposes into the same 24 dimensions and the
    comparison is well defined even when the inventories do not overlap.

Why decode at analysis time rather than during the GPU pass
    The slice tables record the phone symbol. Features are derived from it here,
    so the feature definition can be revised without re-running alignment, and
    the tables stay small.

Reference
    Mortensen, D. R., Littell, P., Bharadwaj, A., Goyal, K., Dyer, C. and
    Levin, L. (2016). PanPhon, a resource for mapping IPA segments to
    articulatory feature vectors. COLING 2016.
"""

from __future__ import annotations

import functools
import re
import unicodedata

import numpy as np

# The 24 PanPhon features, in the order word_to_vector_list returns them.
FEATURE_NAMES = [
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid",
    "voi", "sg", "cg", "ant", "cor", "distr", "lab", "hi", "lo",
    "back", "round", "velaric", "tense", "long", "hitone", "hireg",
]

# The espeak vocabulary carries entries that are not IPA at all. They are
# rejected rather than guessed at. Bare digits are tone markers with no segment
# attached, and parse_tone handles them, so they are not listed here.
_NON_IPA = {"<pad>", "<s>", "</s>", "<unk>", "??"}

# espeak writes lexical tone as a trailing digit, `a1`, `iɛ5`, `onɡ5`. Mandarin
# uses 1 to 5, and other tonal languages in the vocabulary follow the same
# convention.
_TONE = re.compile(r"^(.*?)([1-9])$")


def parse_tone(symbol: str):
    """
    Split a symbol into its segment and its lexical tone.

    Returns (base, tone). `base` is None when the symbol is a bare tone marker
    with no segment attached, which the vocabulary does contain, and inventing a
    vowel for one would fabricate a segment that was never emitted.

    Tone is returned separately rather than folded into the feature vector on
    purpose. Tone is suprasegmental, while PanPhon's `hitone` and `hireg`
    describe register on a segment. Writing one into the other would put two
    different things in one dimension and make the cross linguistic comparison
    this project exists to do meaningless.
    """
    if not symbol:
        return None, None
    match = _TONE.match(symbol)
    if not match:
        return symbol, None
    base, tone = match.group(1), int(match.group(2))
    return (base or None), tone


@functools.lru_cache(maxsize=1)
def _feature_table():
    import panphon
    return panphon.FeatureTable()


def _nfd(text: str) -> str:
    """PanPhon returns decomposed text, so a precomposed `ã` must not look partly consumed."""
    return unicodedata.normalize("NFD", text)


@functools.lru_cache(maxsize=8192)
def features(phone: str) -> tuple | None:
    """
    Return the 24 dimensional feature vector for a phone, or None.

    Values are -1, 0 or +1 for minus, unspecified and plus. A multi segment
    string, which the aligner occasionally emits, is averaged across its
    segments so it still contributes rather than being dropped.

    A trailing tone digit is stripped before parsing, so `a1` decodes to the
    features of /a/. Read the tone itself with `parse_tone`.
    """
    if not phone or phone in _NON_IPA:
        return None
    base, _ = parse_tone(phone)
    if base is None or base in _NON_IPA:
        return None
    # Outside the try: a missing or broken PanPhon must fail loudly, not turn
    # every phone into "unparseable" and leave the analyses running on nothing.
    table = _feature_table()
    try:
        # PanPhon drops characters it does not recognise without failing, so
        # `dZ` would come back as a plain /d/. A symbol it only partly consumed
        # is rejected rather than given the features of its prefix.
        if _nfd("".join(table.ipa_segs(base))) != _nfd(base):
            return None
        vectors = table.word_to_vector_list(base, numeric=True)
    except Exception:
        return None
    if not vectors:
        return None
    if len(vectors) == 1:
        return tuple(vectors[0])
    return tuple(np.asarray(vectors, dtype=float).mean(axis=0))


def is_parseable(phone: str) -> bool:
    """Whether this phone contributes to a feature based analysis."""
    return features(phone) is not None


def feature_matrix(phones) -> tuple[np.ndarray, list[str]]:
    """
    Stack feature vectors for a sequence of phones.

    Returns (matrix of shape (n, 24), list of the phones that parsed). Callers
    must use the returned phone list rather than the input, since unparseable
    entries are dropped.
    """
    rows, kept = [], []
    for phone in phones:
        vector = features(phone)
        if vector is not None:
            rows.append(vector)
            kept.append(phone)
    if not rows:
        return np.empty((0, len(FEATURE_NAMES))), []
    return np.asarray(rows, dtype=float), kept


def coverage(vocabulary) -> dict:
    """
    Report how much of a phone inventory the feature mapping recovers.

    `partial` is the important and least obvious field. PanPhon silently drops
    characters it does not recognise rather than failing, so a symbol can come
    back with the features of its recognised prefix and nothing says so. `dZ`,
    X-SAMPA for the affricate, decodes to a plain /d/. `aɪɚ` loses its rhotic.
    Each entry is (symbol, what PanPhon actually consumed).

    `features` rejects these, so they are counted as unparsed. They are listed
    separately because a symbol PanPhon half understood usually means the
    corpus uses a different transcription, which is worth knowing.
    """
    phones = [p for p in vocabulary if p not in _NON_IPA and not p.startswith("<")]
    parsed = [p for p in phones if is_parseable(p)]

    partial = []
    table = _feature_table()
    for phone in sorted(set(phones) - set(parsed)):
        base, _ = parse_tone(phone)
        if base is None:
            continue
        try:
            consumed = "".join(table.ipa_segs(base))
        except Exception:
            continue
        if consumed and _nfd(consumed) != _nfd(base):
            partial.append((phone, consumed))

    return {
        "total": len(phones),
        "parsed": len(parsed),
        "unparsed": sorted(set(phones) - set(parsed)),
        "partial": sorted(partial),
        "fraction": len(parsed) / len(phones) if phones else 0.0,
    }


# The contrast the sonorant hypothesis is about, in feature terms. Sonorants are
# [+son], obstruents [-son]. Stated here so the pre-registered grouping and the
# feature analysis are demonstrably the same claim rather than two similar ones.
SONORANT_FEATURE = "son"