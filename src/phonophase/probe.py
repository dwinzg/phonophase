"""
Per-slice statistics shared by the probes.

One definition, used by both the offline probe over sliced wav files and the
fused Colab pass that never writes wavs at all. Three divergent feature
implementations is the mistake this repository already made once.

Every phase statistic is paired with a magnitude statistic computed over
identical framing, so any comparison isolates the domain rather than the
analysis settings. hf_energy and hf_lf_tilt are the confound controls, present
so a phase result can be checked against the possibility that it is really
about high frequency energy.
"""

from __future__ import annotations

import librosa
import numpy as np

from phonophase.features import modified_group_delay

SR = 16000
N_FFT = 512
HOP = 128
WIN = 400
N_MFCC = 40
MIN_SAMPLES = 512

STATISTIC_NAMES = [
    "mgd_std", "mgd_iqr", "mgd_frame_spread", "mgd_hf_std",
    "mfcc_std", "mfcc_iqr", "mfcc_frame_spread",
    "hf_energy", "hf_lf_tilt",
]


def _spread(plane: np.ndarray) -> tuple[float, float, float]:
    q75, q25 = np.percentile(plane, [75, 25])
    return (
        float(np.std(plane)),
        float(q75 - q25),
        # Mean over time of the per frame spread across frequency. The closest
        # scalar to the Spectral Collapse idea, that synthetic speech is
        # unnaturally smooth within a frame.
        float(np.mean(np.std(plane, axis=0))),
    )


def slice_statistics(audio: np.ndarray) -> dict | None:
    """
    Reduce one audio slice to the scalars in STATISTIC_NAMES, or None.

    `audio` must be mono at SR (16 kHz). Stereo would be framed along the
    channel axis and return finite nonsense, so it is refused.
    """
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError(f"slice_statistics needs mono audio, got shape {audio.shape}")
    if audio.size < MIN_SAMPLES:
        return None

    mgd = modified_group_delay(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP)
    mag = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN))

    if not (np.all(np.isfinite(mgd)) and np.all(np.isfinite(mfcc))):
        return None

    half = mgd.shape[0] // 2
    mgd_std, mgd_iqr, mgd_frame = _spread(mgd)
    mfcc_std, mfcc_iqr, mfcc_frame = _spread(mfcc)

    hf = float(np.mean(mag[half:] ** 2))
    lf = float(np.mean(mag[:half] ** 2))

    return {
        "mgd_std": mgd_std,
        "mgd_iqr": mgd_iqr,
        "mgd_frame_spread": mgd_frame,
        # Dispersion in the upper half of the spectrum. The statistic the
        # pre-registered hypothesis is about.
        "mgd_hf_std": float(np.std(mgd[half:])),
        "mfcc_std": mfcc_std,
        "mfcc_iqr": mfcc_iqr,
        "mfcc_frame_spread": mfcc_frame,
        "hf_energy": float(np.log(hf + 1e-12)),
        "hf_lf_tilt": float(np.log(hf / (lf + 1e-12) + 1e-12)),
    }
