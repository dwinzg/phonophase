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

import numpy as np

# The 24 PanPhon features, in the order word_to_vector_list returns them.
FEATURE_NAMES = [
    "syl", "son", "cons", "cont", "delrel", "lat", "nas", "strid",
    "voi", "sg", "cg", "ant", "cor", "distr", "lab", "hi", "lo",
    "back", "round", "velaric", "tense", "long", "hitone", "hireg",
]

# The espeak vocabulary carries entries that are not IPA at all, tone digits and
# transcription artifacts among them. They are rejected rather than guessed at.
_NON_IPA = {"<pad>", "<s>", "</s>", "<unk>", "??", "1", "2", "3", "4", "5"}


@functools.lru_cache(maxsize=1)
def _feature_table():
    import panphon
    return panphon.FeatureTable()


@functools.lru_cache(maxsize=8192)
def features(phone: str) -> tuple | None:
    """
    Return the 24 dimensional feature vector for a phone, or None.

    Values are -1, 0 or +1 for minus, unspecified and plus. A multi segment
    string, which the aligner occasionally emits, is averaged across its
    segments so it still contributes rather than being dropped.
    """
    if not phone or phone in _NON_IPA or any(c.isdigit() for c in phone):
        return None
    try:
        vectors = _feature_table().word_to_vector_list(phone, numeric=True)
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
    """Report how much of a phone inventory the feature mapping recovers."""
    phones = [p for p in vocabulary if p not in _NON_IPA and not p.startswith("<")]
    parsed = [p for p in phones if is_parseable(p)]
    return {
        "total": len(phones),
        "parsed": len(parsed),
        "unparsed": sorted(set(phones) - set(parsed)),
        "fraction": len(parsed) / len(phones) if phones else 0.0,
    }


# The contrast the sonorant hypothesis is about, in feature terms. Sonorants are
# [+son], obstruents [-son]. Stated here so the pre-registered grouping and the
# feature analysis are demonstrably the same claim rather than two similar ones.
SONORANT_FEATURE = "son"