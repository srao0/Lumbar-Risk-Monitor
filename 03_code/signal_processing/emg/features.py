"""
sEMG Feature Extraction
========================
Window-level features from filtered surface EMG signals.

All features are validated for use at 200 Hz sampling (Ganglion hardware
limit, Nyquist = 100 Hz). Spectral features (MPF, MDF, spectral moments)
are EXCLUDED, they require ≥ 1 kHz to be reliable and will be aliased
at 200 Hz. See spec §6.2 and §2.2 for the hardware constraint rationale.

Feature set (time-domain only):
    RMS (per channel)
        Root-mean-square amplitude over a window, standard amplitude
        estimator for EMG (De Luca, 1997). Units: mV.

    MAV (per channel)
        Mean absolute value: mean(|x|). Correlated with muscle force output.
        Computationally identical to the full-wave rectified mean; preferred
        in real-time systems due to its simplicity and low latency.

    ZCR (per channel)
        Zero-crossing rate: count of sign changes per sample. A proxy for
        motor unit firing rate and signal frequency content without requiring
        a full spectral transform.

    Asymmetry Index, AI (bilateral pairs)
        AI = (right - left) / (right + left)  ∈ [-1, +1]
        AI > 0 → right dominant; AI < 0 → left dominant.
        |AI| > 0.3 (AR > 1.5 equivalent) indicates potentially risky
        compensatory patterns (McGill, 2007; spec §7.1).

    Co-activation Index, CAI (bilateral pairs)
        CAI = min(RMS_L, RMS_R) / max(RMS_L, RMS_R)  ∈ [0, 1]
        1.0 = perfectly symmetric co-activation; 0.0 = complete asymmetry.
        Complements AI: both direction and magnitude of imbalance captured.

References:
    De Luca, C.J. (1997) "The use of surface electromyography in biomechanics."
    J Applied Biomechanics, 13(2), 135-163.

    McGill, S. (2007) Low Back Disorders (2nd ed.). Human Kinetics.

    Phinyomark, A. et al. (2012) "Feature reduction and selection for
    EMG signal classification." Expert Systems with Applications, 39(8).
"""

import numpy as np
from typing import Optional, Dict, List, Tuple


# Channel naming conventions

# Expected column order in (N, 4) sEMG arrays.
# LES  = Left Erector Spinae,  RES  = Right Erector Spinae,
# LOBL = Left Oblique,         ROBL = Right Oblique.
# NOTE: channels 3-4 are surface obliques (placed ~4 cm lateral at the L5
# level), NOT the deep multifidus. Earlier code mislabelled these as "MF"
# (multifidus); the electrodes never recorded multifidus. See session_converter.py.
DEFAULT_CHANNEL_NAMES = ["LES", "RES", "LOBL", "ROBL"]

# Bilateral pairs for asymmetry index: (left_idx, right_idx, label)
DEFAULT_BILATERAL_PAIRS = [
    (0, 1, "ES"),    # LES vs RES   — erector spinae pair
    (2, 3, "OBL"),   # LOBL vs ROBL — oblique pair
]


# RMS

def compute_rms(window: np.ndarray) -> float:
    """
    Root-mean-square amplitude of a 1-D window.

    Parameters
    ----------
    window : (M,) array, single EMG channel, one window

    Returns
    -------
    rms : scalar (same units as input)
    """
    return float(np.sqrt(np.mean(window ** 2)))


def rms_all_channels(window: np.ndarray) -> np.ndarray:
    """
    RMS amplitude for each channel in a (M, C) window.

    Parameters
    ----------
    window : (M, C) array, M samples, C channels

    Returns
    -------
    rms : (C,) array
    """
    if window.ndim == 1:
        return np.array([compute_rms(window)])
    return np.sqrt(np.mean(window ** 2, axis=0))


# MAV

def compute_mav(window: np.ndarray) -> float:
    """
    Mean absolute value (MAV) of a 1-D window.

    MAV = mean(|x|). Equivalent to the full-wave rectified mean.
    Correlated with muscular force output.

    Parameters
    ----------
    window : (M,) array, single EMG channel, one window

    Returns
    -------
    mav : scalar (same units as input)
    """
    return float(np.mean(np.abs(window)))


def mav_all_channels(window: np.ndarray) -> np.ndarray:
    """
    MAV for each channel in a (M, C) window.

    Parameters
    ----------
    window : (M, C) array

    Returns
    -------
    mav : (C,) array
    """
    if window.ndim == 1:
        return np.array([compute_mav(window)])
    return np.mean(np.abs(window), axis=0)


# ZCR

def compute_zcr(window: np.ndarray) -> float:
    """
    Zero-crossing rate (ZCR) of a 1-D window.

    ZCR = number of sign changes / (N - 1)

    A proxy for the dominant frequency content of the signal without
    requiring a spectral transform. Validated for use at 200 Hz.

    Parameters
    ----------
    window : (M,) array, single EMG channel, one window

    Returns
    -------
    zcr : float in [0, 1]
    """
    if len(window) < 2:
        return 0.0
    signs = np.sign(window)
    # Treat zero as positive to avoid spurious crossings at rest
    signs[signs == 0] = 1
    crossings = np.sum(np.diff(signs) != 0)
    return float(crossings / (len(window) - 1))


def zcr_all_channels(window: np.ndarray) -> np.ndarray:
    """
    ZCR for each channel in a (M, C) window.

    Parameters
    ----------
    window : (M, C) array

    Returns
    -------
    zcr : (C,) array
    """
    if window.ndim == 1:
        return np.array([compute_zcr(window)])
    return np.array([compute_zcr(window[:, c]) for c in range(window.shape[1])])


# Asymmetry Index

def asymmetry_index(left_rms: float, right_rms: float) -> float:
    """
    Bilateral asymmetry index.

        AI = (right - left) / (right + left)

    Returns NaN if both channels are silent (avoids 0/0).

    Parameters
    ----------
    left_rms  : RMS amplitude of the left channel
    right_rms : RMS amplitude of the right channel

    Returns
    -------
    ai : float in [-1, +1], or NaN
    """
    denom = left_rms + right_rms
    if denom < 1e-9:
        return np.nan
    return float((right_rms - left_rms) / denom)


def asymmetry_indices(
    rms_values: np.ndarray,
    pairs: List[Tuple[int, int, str]] = DEFAULT_BILATERAL_PAIRS,
) -> Dict[str, float]:
    """
    Compute asymmetry index for each bilateral pair.

    Parameters
    ----------
    rms_values : (C,) RMS amplitudes, indexed by DEFAULT_CHANNEL_NAMES
    pairs      : list of (left_idx, right_idx, label) tuples

    Returns
    -------
    dict mapping label → AI value.
    Example: {'ES': 0.12, 'OBL': -0.04}
    """
    result = {}
    for left_idx, right_idx, label in pairs:
        result[label] = asymmetry_index(
            float(rms_values[left_idx]),
            float(rms_values[right_idx]),
        )
    return result


# Co-activation Index

def coactivation_index(left_rms: float, right_rms: float) -> float:
    """
    Bilateral co-activation index.

        CAI = min(RMS_L, RMS_R) / max(RMS_L, RMS_R)  ∈ [0, 1]

    1.0 = perfectly symmetric; 0.0 = one side completely dominant.
    Complements AI by capturing magnitude of imbalance independent of
    direction.

    Returns NaN if both channels are silent.

    Parameters
    ----------
    left_rms  : RMS amplitude of the left channel
    right_rms : RMS amplitude of the right channel

    Returns
    -------
    cai : float in [0, 1], or NaN
    """
    mx = max(left_rms, right_rms)
    if mx < 1e-9:
        return np.nan
    return float(min(left_rms, right_rms) / mx)


def coactivation_indices(
    rms_values: np.ndarray,
    pairs: List[Tuple[int, int, str]] = DEFAULT_BILATERAL_PAIRS,
) -> Dict[str, float]:
    """
    Compute co-activation index for each bilateral pair.

    Parameters
    ----------
    rms_values : (C,) RMS amplitudes
    pairs      : list of (left_idx, right_idx, label) tuples

    Returns
    -------
    dict mapping label → CAI value.
    """
    result = {}
    for left_idx, right_idx, label in pairs:
        result[label] = coactivation_index(
            float(rms_values[left_idx]),
            float(rms_values[right_idx]),
        )
    return result


# Combined windowed feature extraction

def extract_window_features(
    window: np.ndarray,
    fs: float,
    channel_names: List[str] = DEFAULT_CHANNEL_NAMES,
    bilateral_pairs: List[Tuple[int, int, str]] = DEFAULT_BILATERAL_PAIRS,
) -> dict:
    """
    Extract all sEMG features from a single (M, C) window.

    All features are time-domain only, validated for 200 Hz sampling.
    MPF/MDF spectral features are excluded per spec §6.2 (Ganglion 200 Hz
    constraint; Nyquist = 100 Hz makes spectral features unreliable).

    Parameters
    ----------
    window          : (M, C) filtered EMG array, M samples, C channels
    fs              : sampling frequency in Hz (for documentation only;
                      all features here are sample-count-based)
    channel_names   : list of channel label strings (len C)
    bilateral_pairs : (left_idx, right_idx, label) tuples for AI/CAI

    Returns
    -------
    features : dict with keys:
        emg_rms_<channel>: per-channel RMS amplitude (mV)
        emg_mav_<channel>: per-channel mean absolute value (mV)
        emg_zcr_<channel>: per-channel zero-crossing rate [0,1]
        emg_ai_<label>: bilateral asymmetry index [-1, +1]
        emg_cai_<label>: co-activation index [0, 1]
    """
    features = {}

    if window.ndim == 1:
        window = window[:, np.newaxis]

    n_ch = window.shape[1]
    names = channel_names[:n_ch]

    # RMS
    rms_vals = rms_all_channels(window)
    for i, name in enumerate(names):
        features[f"emg_rms_{name}"] = float(rms_vals[i])

    # MAV
    mav_vals = mav_all_channels(window)
    for i, name in enumerate(names):
        features[f"emg_mav_{name}"] = float(mav_vals[i])

    # ZCR
    zcr_vals = zcr_all_channels(window)
    for i, name in enumerate(names):
        features[f"emg_zcr_{name}"] = float(zcr_vals[i])

    # Asymmetry Index
    ai_dict = asymmetry_indices(rms_vals, bilateral_pairs)
    for label, ai in ai_dict.items():
        features[f"emg_ai_{label}"] = ai

    # Co-activation Index
    cai_dict = coactivation_indices(rms_vals, bilateral_pairs)
    for label, cai in cai_dict.items():
        features[f"emg_cai_{label}"] = cai

    return features


# Sliding window feature extraction

def windowed_emg_features(
    emg_filtered: np.ndarray,
    fs: float,
    window_samples: int,
    step_samples: int,
    channel_names: List[str] = DEFAULT_CHANNEL_NAMES,
    bilateral_pairs: List[Tuple[int, int, str]] = DEFAULT_BILATERAL_PAIRS,
) -> List[dict]:
    """
    Extract sEMG features from all sliding windows.

    Parameters
    ----------
    emg_filtered   : (N,) or (N, C) filtered EMG array
    fs             : sampling frequency in Hz
    window_samples : number of samples per window
    step_samples   : stride between windows
    channel_names  : list of channel label strings
    bilateral_pairs: bilateral pairs for AI/CAI computation

    Returns
    -------
    List of feature dicts, one per window.
    """
    if emg_filtered.ndim == 1:
        emg_filtered = emg_filtered[:, np.newaxis]

    n_samples = emg_filtered.shape[0]
    results = []
    starts = range(0, n_samples - window_samples + 1, step_samples)
    for s in starts:
        window = emg_filtered[s: s + window_samples, :]
        feat = extract_window_features(
            window,
            fs=fs,
            channel_names=channel_names,
            bilateral_pairs=bilateral_pairs,
        )
        results.append(feat)
    return results


# Retained spectral utility (offline analysis only)
# NOTE: Do NOT use median_power_frequency in the real-time pipeline or in
# any feature matrix intended for hardware deployment. It is retained here
# for offline validation and literature comparison only.

def median_power_frequency(
    window: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None,
) -> float:
    """
    ⚠  OFFLINE USE ONLY, NOT valid at 200 Hz Ganglion sampling rate.

    Estimate the median power frequency (MDF) of a single EMG window.
    Requires fs ≥ 1000 Hz for reliable spectral estimation (De Luca, 1993).
    At 200 Hz, Nyquist = 100 Hz, spectral features are aliased and unreliable.

    Parameters
    ----------
    window  : (M,) filtered EMG signal (single channel)
    fs      : sampling frequency in Hz
    nperseg : Welch segment length

    Returns
    -------
    mdf : float in Hz, or NaN
    """
    from scipy import signal as sp_signal

    if np.max(np.abs(window)) < 1e-9:
        return np.nan
    if nperseg is None:
        nperseg = max(16, len(window) // 4)
    freqs, psd = sp_signal.welch(window, fs=fs, nperseg=nperseg)
    cumulative = np.cumsum(psd)
    if cumulative[-1] < 1e-12:
        return np.nan
    half_power = cumulative[-1] / 2.0
    idx = min(np.searchsorted(cumulative, half_power), len(freqs) - 1)
    return float(freqs[idx])
