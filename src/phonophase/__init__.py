"""
PhonoPhase.

Where does the phase artifact of speech synthesis live, and is that location a
property of articulation rather than of language?

The package is the measurement instrument.

    features    the corrected Modified Group Delay front end
    probe       per slice statistics, one definition shared by every caller
    phonology   IPA to PanPhon articulatory feature vectors
    stats       cluster bootstrap, permutation null, paired delta, FDR
    metrics     EER, t-DCF, per group breakdowns
    corpus      ASVspoof partition handling and utterance selection
    config      configuration loading anchored to the repository root

Successor to dwinzg/Deepfake-PADS-Net, which is archived. See docs/SCOPE.md for
what migrated and why, and docs/PREREGISTRATION.md for the confirmed result.
"""

__version__ = "0.1.0"

__all__ = ["config", "corpus", "features", "metrics", "phonology", "probe", "stats"]
