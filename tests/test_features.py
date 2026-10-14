"""
Tests for src/features.py.

The important one here is test_legacy_group_delay_is_a_time_ramp, which pins
down the defect this module was written to fix.
"""

import os
import sys

import librosa
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from phonophase.features import (  # noqa: E402
    FusedFeatureExtractor,
    cepstrally_smoothed_magnitude,
    compute_global_stats,
    modified_group_delay,
)

SR = 16000
N_FFT = 512
HOP = 128
WIN = 400


def _stationary_signal(seconds=1.0, seed=0):
    """
    A signal with no systematic change over time, so any time trend that shows
    up in a feature came from the feature code and not from the audio.
    """
    rng = np.random.default_rng(seed)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    sig = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 440.0, 880.0, 1320.0))
    return (sig + 0.1 * rng.normal(size=n)).astype(np.float64)


def _legacy_group_delay(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN):
    """
    Exact reproduction of 02_feature_loader.compute_features, kept here so the
    defect it contains is demonstrated rather than merely asserted.
    """
    X = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, center=True)
    n_ramp = np.arange(len(audio))
    Y = librosa.stft(audio * n_ramp, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, center=True)
    X_mag_sq = np.abs(X) ** 2 + 1e-8
    return (X.real * Y.real + X.imag * Y.imag) / X_mag_sq


def _frame_index_correlation(plane):
    """Correlation between frame index and that frame's mean over frequency."""
    per_frame = plane.mean(axis=0)
    idx = np.arange(plane.shape[1], dtype=np.float64)
    return abs(float(np.corrcoef(idx, per_frame)[0, 1]))


def test_legacy_group_delay_is_a_time_ramp():
    # The ramp in the old code runs over the whole signal instead of within
    # each frame, so frame m picks up an additive offset of m * hop_length in
    # every bin. On a stationary input the feature should carry no time trend
    # at all, yet the correlation with frame index comes out near one.
    audio = _stationary_signal()
    corr = _frame_index_correlation(_legacy_group_delay(audio))
    assert corr > 0.99, (
        f"expected the legacy implementation to be dominated by a time ramp, "
        f"correlation was {corr:.4f}"
    )


def test_corrected_mgd_has_no_time_ramp():
    audio = _stationary_signal()
    corr = _frame_index_correlation(
        modified_group_delay(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    )
    assert corr < 0.5, f"corrected MGD still trends with frame index at {corr:.4f}"


def test_mgd_shape_matches_librosa_framing():
    # The MGD plane must line up frame for frame with the MFCC plane, otherwise
    # the two channels are misaligned in time.
    audio = _stationary_signal(seconds=0.75)
    mgd = modified_group_delay(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
    reference = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP,
                             win_length=WIN, center=True)
    assert mgd.shape == reference.shape, f"{mgd.shape} vs {reference.shape}"


def test_mgd_is_finite_including_on_silence():
    # Silence drives the magnitude spectrum to zero, which is precisely where
    # an unmodified group delay blows up.
    for audio in (_stationary_signal(), np.zeros(SR, dtype=np.float64)):
        mgd = modified_group_delay(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN)
        assert np.all(np.isfinite(mgd)), "MGD produced non finite values"


def test_compression_bounds_dynamic_range():
    # alpha below one should compress the tails. Compare against alpha of one.
    audio = _stationary_signal()
    common = dict(n_fft=N_FFT, hop_length=HOP, win_length=WIN, gamma=0.9)
    compressed = modified_group_delay(audio, alpha=0.4, **common)
    uncompressed = modified_group_delay(audio, alpha=1.0, **common)
    assert np.abs(compressed).max() < np.abs(uncompressed).max()


def test_cepstral_smoothing_is_positive_and_smoother():
    audio = _stationary_signal()
    mag = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP, win_length=WIN))
    smoothed = cepstrally_smoothed_magnitude(mag, n_cep=24)

    assert smoothed.shape == mag.shape
    assert np.all(smoothed > 0), "smoothed magnitude must stay strictly positive"

    # Roughness measured as mean absolute difference along the frequency axis.
    rough_in = np.abs(np.diff(np.log(mag + 1e-8), axis=0)).mean()
    rough_out = np.abs(np.diff(np.log(smoothed), axis=0)).mean()
    assert rough_out < rough_in, f"smoothing did not reduce roughness, {rough_out} vs {rough_in}"


def test_extractor_shape_and_dtype():
    audio = _stationary_signal(seconds=4.0)
    ex = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                               n_mels=40, normalisation="per_file")
    plane = ex(audio)

    assert plane.shape[0] == 2, "expected two channels"
    assert plane.shape[1] == 40, "expected forty mel bands"
    assert plane.shape[2] == ex.n_frames_for(len(audio)), \
        f"frame count {plane.shape[2]} does not match n_frames_for"
    assert plane.dtype == np.float32
    assert np.all(np.isfinite(plane))


def test_per_file_normalisation_standardises_each_channel():
    audio = _stationary_signal(seconds=2.0)
    ex = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                               n_mels=40, normalisation="per_file")
    plane = ex(audio)
    for ch in range(2):
        assert abs(float(plane[ch].mean())) < 1e-4
        assert abs(float(plane[ch].std()) - 1.0) < 1e-3


def test_global_normalisation_requires_stats():
    ex = FusedFeatureExtractor(normalisation="global")
    try:
        ex(_stationary_signal(seconds=0.5))
    except RuntimeError as exc:
        assert "compute_global_stats" in str(exc)
        return
    raise AssertionError("expected a RuntimeError when global stats are missing")


def test_global_stats_round_trip():
    # Statistics computed over a corpus and then applied back to it should
    # leave the pooled channels near zero mean and unit variance.
    signals = [_stationary_signal(seconds=1.0, seed=s) for s in range(4)]
    raw = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                                n_mels=40, normalisation="none")
    stats = compute_global_stats(raw, signals)

    assert set(stats) == {"mfcc", "mgd"}
    for key in stats:
        assert np.isfinite(stats[key]).all()
        assert stats[key][1] > 0, f"{key} standard deviation must be positive"

    ex = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                               n_mels=40, normalisation="global", stats=stats)
    pooled = np.concatenate([ex(s) for s in signals], axis=2)
    for ch in range(2):
        assert abs(float(pooled[ch].mean())) < 0.05
        assert abs(float(pooled[ch].std()) - 1.0) < 0.05


def test_silence_trimming_shortens_a_padded_signal():
    # Silence is a known confound here. Every utterance is padded to a fixed
    # duration, so a model can key on the padding instead of on synthesis
    # artifacts. This switch is how that gets tested.
    speech = _stationary_signal(seconds=1.0)
    padded = np.concatenate([np.zeros(SR), speech, np.zeros(SR)])

    keep = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                                 n_mels=40, normalisation="per_file")
    trim = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                                 n_mels=40, normalisation="per_file",
                                 trim_silence=True)

    assert trim(padded).shape[2] < keep(padded).shape[2], \
        "trimming should drop the padded frames"
    # Roughly a third of the padded signal is speech, so the trimmed version
    # should land near the untrimmed speech length rather than collapse.
    assert trim(padded).shape[2] > keep(speech).shape[2] * 0.7


def test_silence_trimming_leaves_a_near_silent_clip_usable():
    # Trimming an almost empty clip must not produce an empty plane.
    quiet = np.zeros(SR, dtype=np.float64)
    quiet[::4000] = 1e-4
    ex = FusedFeatureExtractor(n_fft=N_FFT, hop_length=HOP, win_length=WIN,
                               n_mels=40, normalisation="per_file",
                               trim_silence=True)
    plane = ex(quiet)
    assert plane.shape[0] == 2 and plane.shape[2] > 0
    assert np.all(np.isfinite(plane))


def test_unknown_normalisation_is_rejected():
    try:
        FusedFeatureExtractor(normalisation="whatever")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown normalisation mode")


def test_slice_statistics_refuses_stereo():
    from phonophase.probe import slice_statistics

    try:
        slice_statistics(np.zeros((4000, 2)))
    except ValueError:
        return
    raise AssertionError("stereo audio must be refused, not framed along the channels")


def test_load_config_looks_in_the_working_directory_first():
    import tempfile

    from phonophase.config import load_config

    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, "config.yaml"), "w") as fh:
            fh.write("marker: from_cwd\n")
        os.chdir(tmp)
        try:
            assert load_config()["marker"] == "from_cwd"
        finally:
            os.chdir(here)


if __name__ == "__main__":
    from _runner import main
    main(globals(), "test_features")
