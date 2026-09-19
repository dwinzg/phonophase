"""
Shared feature front end.

Why this file exists
    Three different feature implementations had drifted apart in this repo.
    02_feature_loader.compute_features produced (magnitude_dB, group_delay) at
    hop 160. dataset.extract_fused_features produced (MFCC, mel compressed
    absolute phase) at hop 128. utils.compute_mgd was a third variant that
    nothing imported. The published work describes one front end, so there
    should be one implementation. This is it.

Two defects in the earlier versions are corrected here, both documented inline
at the point of the fix.

    1. The ramp used for the group delay numerator was applied across the whole
       signal rather than within each analysis frame. That injected a linear
       time ramp into the feature which was far larger than the phase
       information itself. See modified_group_delay.

    2. What the earlier code computed was the raw group delay, not the Modified
       Group Delay. The modification is the whole point of MGD. It suppresses
       the spurious spikes that appear wherever the magnitude spectrum
       approaches zero, and without it the feature is dominated by those
       spikes. See cepstrally_smoothed_magnitude.

Reference for the MGD formulation
    Murthy, H. A. and Gadde, V. (2003). The modified group delay function and
    its application to phoneme recognition. ICASSP 2003.
    Hegde, R. M., Murthy, H. A. and Gadde, V. R. R. (2007). Significance of
    joint features derived from the modified group delay function in speech
    processing. EURASIP Journal on Audio, Speech, and Music Processing.

    tau(k) = sgn * | (X_R(k) Y_R(k) + X_I(k) Y_I(k)) / H(k)^(2*gamma) | ^ alpha

    X is the DFT of the windowed frame, Y is the DFT of that frame multiplied
    by the within frame sample index, H is the cepstrally smoothed magnitude
    spectrum, and sgn is the sign of the numerator. Typical values are
    0 < alpha <= 1 and 0 < gamma <= 1.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.fft import dct, idct

# Defaults chosen to match config.yaml so that behaviour is predictable when a
# caller does not pass anything. Override them from the config in real runs.
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_N_FFT = 512
DEFAULT_HOP_LENGTH = 128
DEFAULT_WIN_LENGTH = 400
DEFAULT_N_MELS = 40

# alpha and gamma are the MGD control parameters. Values in this range are the
# ones commonly reported for speech tasks. gamma below 1 is what actually
# suppresses the magnitude zeros, and alpha below 1 compresses the dynamic
# range so a handful of frames cannot dominate the feature.
DEFAULT_ALPHA = 0.4
DEFAULT_GAMMA = 0.9

# Number of cepstral coefficients kept when smoothing the magnitude spectrum.
# Keeping few coefficients keeps only the spectral envelope, which is what we
# want in the denominator.
DEFAULT_N_CEP = 24

_EPS = 1e-8


def cepstrally_smoothed_magnitude(mag: np.ndarray, n_cep: int = DEFAULT_N_CEP) -> np.ndarray:
    """
    Return the cepstrally smoothed magnitude spectrum H(k).

    This is the term that makes a Modified Group Delay modified. The raw group
    delay divides by |X(k)|^2, which explodes wherever the magnitude spectrum
    has a zero, producing the spurious spikes that MGD was introduced to fix.
    Dividing by a smooth envelope instead keeps the denominator well behaved.

    The smoothing is a DCT of the log magnitude, truncation to the lowest n_cep
    quefrency bins, then an inverse DCT. That keeps the spectral envelope and
    discards the fine harmonic structure.
    """
    log_mag = np.log(np.maximum(mag, _EPS))
    cep = dct(log_mag, type=2, axis=0, norm="ortho")
    cep[n_cep:, :] = 0.0
    return np.exp(idct(cep, type=2, axis=0, norm="ortho"))


def _framed_spectra(
    y: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (X, Y) where X is the DFT of each windowed frame and Y is the DFT of
    that same frame multiplied by the within frame sample index.

    This function exists because of a subtle bug in the earlier code. It did

        Y = librosa.stft(audio * np.arange(len(audio)))

    which ramps across the entire signal. Frame m then contains
    (m*hop + k) * x[k] rather than k * x[k], so

        Y_m = m*hop*X_m + Y_m_correct

    and the resulting group delay picks up an additive offset of m*hop in every
    frequency bin. With hop 128 and a four second clip that offset reaches
    64000 by the final frame, while a genuine group delay is bounded by the
    window length of 400. The feature was therefore mostly a linear time index.

    Framing explicitly and ramping inside the frame removes the offset.
    """
    # Reflect pad so the frame count matches librosa.stft with center=True,
    # which keeps this aligned frame for frame with the MFCC channel.
    pad = n_fft // 2
    if len(y) < pad + 1:
        y = np.pad(y, (0, pad + 1 - len(y)), mode="constant")
    y_padded = np.pad(y, pad, mode="reflect")

    window = librosa.filters.get_window("hann", win_length, fftbins=True)
    window = librosa.util.pad_center(window, size=n_fft)

    frames = librosa.util.frame(y_padded, frame_length=n_fft, hop_length=hop_length)
    windowed = frames * window[:, None]

    # The ramp is the sample index inside the frame, which is the whole point.
    ramp = np.arange(n_fft, dtype=np.float64)

    X = np.fft.rfft(windowed, axis=0)
    Y = np.fft.rfft(windowed * ramp[:, None], axis=0)
    return X, Y


def modified_group_delay(
    y: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    win_length: int = DEFAULT_WIN_LENGTH,
    alpha: float = DEFAULT_ALPHA,
    gamma: float = DEFAULT_GAMMA,
    n_cep: int = DEFAULT_N_CEP,
) -> np.ndarray:
    """
    Compute the Modified Group Delay spectrogram, shape (1 + n_fft/2, n_frames).

    Set alpha=1.0, gamma=1.0 and n_cep large to recover something close to the
    raw group delay, which is useful when comparing against the old behaviour.
    """
    X, Y = _framed_spectra(y, n_fft, hop_length, win_length)

    numerator = X.real * Y.real + X.imag * Y.imag
    H = cepstrally_smoothed_magnitude(np.abs(X), n_cep=n_cep)
    denominator = np.power(H, 2.0 * gamma) + _EPS

    tau = numerator / denominator

    # Sign preserving power compression. Without it a few high energy frames
    # dominate the dynamic range of the whole feature plane.
    return np.sign(tau) * np.power(np.abs(tau), alpha)


def mel_compress(spectrogram: np.ndarray, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    """
    Project a linear frequency plane onto n_mels bands.

    Used to bring the MGD plane to the same number of rows as the MFCC plane so
    the two can be stacked as channels. The mel basis is built once by the
    caller in hot paths, see FusedFeatureExtractor.
    """
    basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    return basis @ spectrogram


class FusedFeatureExtractor:
    """
    Builds the two channel input plane.

    Channel 0 is MFCC, standing in for phonetic and prosodic structure.
    Channel 1 is Modified Group Delay, standing in for signal phase physics.

    The mel filter bank is built once in __init__ rather than per call. The
    earlier dataset.py rebuilt it inside __getitem__, so it was reconstructed
    once per sample per epoch.

    Normalisation modes
        per_file  Reproduces the earlier behaviour. Each plane is z scored
                  using its own statistics. Note that this removes the absolute
                  magnitude of phase dispersion, which is the quantity the
                  project's Spectral Collapse hypothesis is actually about, so
                  it should not be used for variance analysis.
        global    Uses statistics supplied by the caller, normally computed
                  once over the training split and then reused unchanged for
                  dev, eval and any transfer language. This preserves relative
                  dispersion across files and is the correct default for a
                  variance based hypothesis.
        none      No normalisation.
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: int = DEFAULT_HOP_LENGTH,
        win_length: int = DEFAULT_WIN_LENGTH,
        n_mels: int = DEFAULT_N_MELS,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        n_cep: int = DEFAULT_N_CEP,
        normalisation: str = "global",
        stats: dict | None = None,
        trim_silence: bool = False,
        trim_top_db: float = 30.0,
    ):
        if normalisation not in {"per_file", "global", "none"}:
            raise ValueError(f"unknown normalisation mode {normalisation!r}")

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.alpha = alpha
        self.gamma = gamma
        self.n_cep = n_cep
        self.normalisation = normalisation
        self.stats = stats

        # Silence is a known confound in anti spoofing. Utterances vary a lot in
        # length and are then padded to a fixed duration, so a model can learn
        # the padding rather than the synthesis artifacts. Turning this on is
        # how you find out whether that is happening.
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db

        # Built once, reused for every call.
        self._mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels
        )

    @classmethod
    def from_config(cls, config: dict, **overrides):
        """Build an extractor from the features block of config.yaml."""
        f = config["features"]
        kwargs = dict(
            sample_rate=f["sample_rate"],
            n_fft=f["n_fft"],
            hop_length=f["hop_length"],
            win_length=f.get("win_length", DEFAULT_WIN_LENGTH),
            n_mels=f["n_mfcc"],
            alpha=f.get("mgd_alpha", DEFAULT_ALPHA),
            gamma=f.get("mgd_gamma", DEFAULT_GAMMA),
            n_cep=f.get("mgd_n_cep", DEFAULT_N_CEP),
            normalisation=f.get("normalisation", "global"),
            trim_silence=f.get("trim_silence", False),
            trim_top_db=f.get("trim_top_db", 30.0),
        )
        kwargs.update(overrides)
        return cls(**kwargs)

    def n_frames_for(self, n_samples: int) -> int:
        """Frame count a signal of n_samples produces. Useful for shape tests."""
        return 1 + n_samples // self.hop_length

    def _normalise(self, plane: np.ndarray, key: str) -> np.ndarray:
        if self.normalisation == "none":
            return plane
        if self.normalisation == "per_file":
            return (plane - plane.mean()) / (plane.std() + 1e-6)
        # global
        if not self.stats or key not in self.stats:
            raise RuntimeError(
                f"normalisation='global' needs stats[{key!r}]. Compute them once "
                "over the training split with compute_global_stats and pass them in."
            )
        mean, std = self.stats[key]
        return (plane - mean) / (std + 1e-6)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        """Return the fused plane, shape (2, n_mels, n_frames), float32."""
        if self.trim_silence:
            trimmed, _ = librosa.effects.trim(y, top_db=self.trim_top_db)
            # Guard against trimming everything away on a near silent clip.
            if trimmed.size >= self.n_fft:
                y = trimmed

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=self.sample_rate,
            n_mfcc=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        mgd = modified_group_delay(
            y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            alpha=self.alpha,
            gamma=self.gamma,
            n_cep=self.n_cep,
        )
        mgd = self._mel_basis @ mgd

        # Guard against frame count drift between the two front ends. librosa
        # and the explicit framing above should agree, but a mismatch here
        # would silently misalign the channels.
        n = min(mfcc.shape[1], mgd.shape[1])
        mfcc, mgd = mfcc[:, :n], mgd[:, :n]

        mfcc = self._normalise(mfcc, "mfcc")
        mgd = self._normalise(mgd, "mgd")

        return np.stack([mfcc, mgd], axis=0).astype(np.float32)


def compute_global_stats(extractor: "FusedFeatureExtractor", signals) -> dict:
    """
    Accumulate corpus level mean and standard deviation for each channel.

    Run this once over the training split, save the result next to the
    checkpoint, and reuse it unchanged for every other split and language.
    Recomputing stats per split would leak information across the boundary.
    """
    raw = FusedFeatureExtractor(
        sample_rate=extractor.sample_rate,
        n_fft=extractor.n_fft,
        hop_length=extractor.hop_length,
        win_length=extractor.win_length,
        n_mels=extractor.n_mels,
        alpha=extractor.alpha,
        gamma=extractor.gamma,
        n_cep=extractor.n_cep,
        normalisation="none",
    )

    # Running sums so this never holds the whole corpus in memory.
    totals = {"mfcc": [0.0, 0.0, 0], "mgd": [0.0, 0.0, 0]}
    for y in signals:
        plane = raw(y)
        for idx, key in enumerate(("mfcc", "mgd")):
            ch = plane[idx]
            totals[key][0] += float(ch.sum())
            totals[key][1] += float((ch.astype(np.float64) ** 2).sum())
            totals[key][2] += ch.size

    stats = {}
    for key, (s, sq, n) in totals.items():
        if n == 0:
            raise ValueError("no signals supplied")
        mean = s / n
        var = max(sq / n - mean * mean, 0.0)
        stats[key] = (mean, float(np.sqrt(var)))
    return stats
