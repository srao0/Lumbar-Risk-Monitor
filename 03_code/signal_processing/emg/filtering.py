"""
sEMG Filtering
==============
Bandpass and notch filtering for surface electromyography signals.

Filter chain (applied in order):
    1. Notch     50 Hz      (2nd-order IIR, Q=30, zero-phase)
    2. Bandpass  20-95 Hz   (4th-order Butterworth, zero-phase)

Design rationale, Ganglion board at 200 Hz (spec §2.2, §4.2):
    - The OpenBCI Ganglion samples at 200 Hz max. Nyquist = 100 Hz.
    - Bandpass upper cutoff set to 95 Hz (5 Hz headroom below Nyquist)
      to retain the usable sEMG frequency band without aliasing.
    - SENIAM (1999) recommends 10-500 Hz; we adopt 20-95 Hz to match
      the 200 Hz hardware constraint. This is explicitly documented in
      the methodology (spec §8, hardware constraints table).
    - 20 Hz high-pass removes DC offset and sub-20 Hz motion artefact.
    - 50 Hz notch removes UK mains powerline interference (mandatory).
    - Zero-phase (sosfiltfilt / filtfilt) removes group delay, critical
      for onset detection and RMS envelope accuracy.

    ⚠  Do NOT raise BANDPASS_HIGH_HZ above 99 Hz for 200 Hz input data.
       Frequencies above Nyquist are aliased and cannot be recovered.

References:
    De Luca, C.J. (1997) "The use of surface electromyography in biomechanics."
    J Applied Biomechanics, 13(2), 135-163.

    SENIAM guidelines (Hermens et al., 1999).

    OpenBCI Ganglion datasheet, 200 Hz default sample rate, 24-bit ADC.
"""

import numpy as np
from scipy import signal as sp_signal
from typing import Optional


# Default filter parameters

EMG_FS_DEFAULT    = 200.0    # Hz — OpenBCI Ganglion (200 Hz hardware limit)
BANDPASS_LOW_HZ   = 20.0     # Hz
BANDPASS_HIGH_HZ  = 95.0     # Hz — 5 Hz headroom below Nyquist (fs/2 = 100 Hz)
BANDPASS_ORDER    = 4        # Butterworth order (4th ≈ -80 dB/dec)
NOTCH_FREQ_HZ     = 50.0     # Hz — UK power line
NOTCH_Q           = 30.0     # Quality factor (~1.67 Hz -3 dB bandwidth)


# Filter design

def design_bandpass(
    fs: float = EMG_FS_DEFAULT,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BANDPASS_ORDER,
) -> np.ndarray:
    """
    Design a Butterworth bandpass filter in second-order-sections (SOS) form.

    Using SOS form avoids the numerical instability of transfer-function
    coefficients for higher-order filters.

    Parameters
    ----------
    fs      : sampling frequency in Hz
    low_hz  : lower cutoff (-3 dB) in Hz
    high_hz : upper cutoff (-3 dB) in Hz
    order   : filter order (applied to each half, so effective roll-off
              is order × 2 poles total)

    Returns
    -------
    sos : (n_sections, 6) array, SOS filter coefficients
    """
    nyq = fs / 2.0
    if high_hz >= nyq:
        raise ValueError(
            f"High cutoff {high_hz} Hz must be below Nyquist ({nyq} Hz). "
            f"Check your sampling frequency (fs={fs} Hz)."
        )
    sos = sp_signal.butter(
        order,
        [low_hz / nyq, high_hz / nyq],
        btype="bandpass",
        output="sos",
    )
    return sos


def design_notch(
    fs: float = EMG_FS_DEFAULT,
    notch_hz: float = NOTCH_FREQ_HZ,
    Q: float = NOTCH_Q,
) -> tuple:
    """
    Design a notch (band-stop) IIR filter using iirnotch.

    Returns b, a coefficients (not SOS).  Applied separately via
    filtfilt for zero-phase.

    Parameters
    ----------
    fs       : sampling frequency in Hz
    notch_hz : centre frequency of the notch in Hz
    Q        : quality factor (higher Q = narrower notch)

    Returns
    -------
    b, a : filter coefficients as 1-D arrays
    """
    b, a = sp_signal.iirnotch(notch_hz / (fs / 2.0), Q)
    return b, a


# Single-channel filtering

def filter_emg_channel(
    raw: np.ndarray,
    fs: float = EMG_FS_DEFAULT,
    apply_notch: bool = True,
    sos_bp: Optional[np.ndarray] = None,
    notch_ba: Optional[tuple] = None,
) -> np.ndarray:
    """
    Apply the full filter chain to a single sEMG channel.

    Parameters
    ----------
    raw         : (N,) raw sEMG signal in millivolts (or arbitrary units)
    fs          : sampling frequency in Hz
    apply_notch : if False, skip the 50 Hz notch (use when power line
                  interference is negligible or already removed)
    sos_bp      : pre-designed bandpass SOS (design_bandpass output).
                  Designed on-the-fly if not supplied.
    notch_ba    : pre-designed notch (b, a) tuple (design_notch output).
                  Designed on-the-fly if not supplied.

    Returns
    -------
    filtered : (N,) filtered sEMG signal, same units as input
    """
    if raw.ndim != 1:
        raise ValueError("filter_emg_channel expects a 1-D array.")

    # 1. Bandpass
    if sos_bp is None:
        sos_bp = design_bandpass(fs=fs)
    filtered = sp_signal.sosfiltfilt(sos_bp, raw)

    # 2. Notch
    if apply_notch:
        if notch_ba is None:
            notch_ba = design_notch(fs=fs)
        b, a = notch_ba
        filtered = sp_signal.filtfilt(b, a, filtered)

    return filtered


# Multi-channel filtering

def filter_emg_array(
    raw_array: np.ndarray,
    fs: float = EMG_FS_DEFAULT,
    apply_notch: bool = True,
) -> np.ndarray:
    """
    Filter all sEMG channels in a (N, C) array.

    Designs the filters once and reuses them across channels.

    Parameters
    ----------
    raw_array   : (N, C) array, N samples, C channels
    fs          : sampling frequency in Hz
    apply_notch : if False, skip the 50 Hz notch

    Returns
    -------
    filtered : (N, C) array of filtered signals
    """
    if raw_array.ndim == 1:
        return filter_emg_channel(raw_array, fs=fs, apply_notch=apply_notch)

    n_samples, n_channels = raw_array.shape
    sos_bp   = design_bandpass(fs=fs)
    notch_ba = design_notch(fs=fs) if apply_notch else None

    filtered = np.empty_like(raw_array)
    for c in range(n_channels):
        filtered[:, c] = filter_emg_channel(
            raw_array[:, c],
            fs=fs,
            apply_notch=apply_notch,
            sos_bp=sos_bp,
            notch_ba=notch_ba,
        )
    return filtered


# DataFrame-aware entry point

def filter_emg_dataframe(
    emg_df,
    channel_cols: list,
    fs: float = EMG_FS_DEFAULT,
    apply_notch: bool = True,
    suffix: str = "_filt",
) -> object:
    """
    Apply the filter chain to specified columns in a pandas DataFrame.
    Returns a copy with additional filtered columns named <col><suffix>.

    Parameters
    ----------
    emg_df       : DataFrame with raw sEMG columns
    channel_cols : list of column names to filter (e.g. ['LES', 'RES', 'LOBL', 'ROBL'])
    fs           : sampling frequency in Hz
    apply_notch  : apply 50 Hz notch
    suffix       : appended to original column name for the new filtered column

    Returns
    -------
    df : copy of emg_df with additional <col>_filt columns
    """
    import pandas as pd

    df = emg_df.copy()
    sos_bp   = design_bandpass(fs=fs)
    notch_ba = design_notch(fs=fs) if apply_notch else None

    for col in channel_cols:
        raw = df[col].to_numpy(dtype=float)
        df[col + suffix] = filter_emg_channel(
            raw,
            fs=fs,
            apply_notch=apply_notch,
            sos_bp=sos_bp,
            notch_ba=notch_ba,
        )
    return df


# Amplitude artefact rejection

# : Default rejection threshold: any sample exceeding this multiple of the
# : session baseline RMS is flagged as a motion artefact window.
ARTEFACT_THRESHOLD_FACTOR = 5.0

# : Minimum number of non-artefact samples required to estimate a stable
# : baseline RMS. Sessions shorter than this are rejected entirely.
MIN_BASELINE_SAMPLES = 100


def estimate_baseline_rms(
    emg_array: np.ndarray,
    baseline_fraction: float = 0.10,
) -> np.ndarray:
    """
    Estimate the per-channel baseline RMS from the quietest portion of a session.

    Rather than trusting the first N seconds (which may contain setup noise),
    we use the lowest-amplitude fraction of the recording, more robust when
    the baseline recording phase is not available.

    Parameters
    ----------
    emg_array         : (N,) or (N, C) filtered sEMG array in mV
    baseline_fraction : fraction of the session to use (default 0.10 = lowest
                        10% of per-channel RMS windows, each 200 samples)

    Returns
    -------
    baseline_rms : (C,) array, per-channel baseline RMS in mV
    """
    if emg_array.ndim == 1:
        emg_array = emg_array[:, np.newaxis]

    n_samples, n_ch = emg_array.shape

    # Compute RMS in 200-sample (1 s at 200 Hz) non-overlapping chunks
    chunk = 200
    n_chunks = n_samples // chunk
    if n_chunks < 1:
        return np.sqrt(np.mean(emg_array ** 2, axis=0))

    rms_chunks = np.array([
        np.sqrt(np.mean(emg_array[i*chunk:(i+1)*chunk] ** 2, axis=0))
        for i in range(n_chunks)
    ])   # (n_chunks, n_ch)

    # Take the mean of the lowest baseline_fraction of chunks per channel
    n_keep = max(1, int(np.ceil(n_chunks * baseline_fraction)))
    baseline_rms = np.zeros(n_ch)
    for c in range(n_ch):
        sorted_rms   = np.sort(rms_chunks[:, c])
        baseline_rms[c] = float(np.mean(sorted_rms[:n_keep]))

    # Floor at 1 µV to avoid division by zero on silent channels
    baseline_rms = np.maximum(baseline_rms, 1e-3)
    return baseline_rms


def flag_artefact_windows(
    emg_array: np.ndarray,
    window_starts: np.ndarray,
    window_size: int,
    baseline_rms: Optional[np.ndarray] = None,
    threshold_factor: float = ARTEFACT_THRESHOLD_FACTOR,
) -> np.ndarray:
    """
    Flag EMG windows containing motion artefacts based on peak amplitude.

    A window is flagged as an artefact if ANY sample in ANY channel exceeds
    `threshold_factor × baseline_rms[channel]`.

    This is more conservative than a mean-based check and is appropriate for
    wearable EMG where artefacts are typically brief but very high amplitude
    (e.g. electrode lift during fast bending, belt contact artefacts).

    Parameters
    ----------
    emg_array        : (N,) or (N, C) FILTERED sEMG array in mV
                       Must be post-bandpass/notch filtering.
    window_starts    : (M,) array of window start indices (samples)
    window_size      : number of samples per window
    baseline_rms     : (C,) per-channel baseline RMS in mV.
                       If None, estimated automatically from emg_array.
    threshold_factor : multiplier applied to baseline_rms (default 5×).
                       Windows with |sample| > factor × baseline_rms are rejected.

    Returns
    -------
    artefact_mask : (M,) boolean array, True = window contains artefact,
                    features should be set to NaN for this window

    Notes
    -----
    The default threshold of 5× baseline RMS corresponds to roughly 5 standard
    deviations above the resting noise floor.  For typical sEMG at 200 Hz:
        - Resting noise: 0.005-0.020 mV RMS (electrode + Ganglion amplifier)
        - MVC amplitude: ~0.3-0.6 mV RMS (erector spinae during flexion)
        - Artefact amplitude: often > 1-2 mV (belt contact, cable tug)
    A 5× threshold (0.025-0.10 mV peak) will catch artefacts while preserving
    vigorous voluntary contraction signals.

    Reference: De Luca et al. (2010). Filtering the surface EMG signal.
    J Electromyography and Kinesiology, 20(2), 235-244.
    """
    if emg_array.ndim == 1:
        emg_array = emg_array[:, np.newaxis]

    n_samples, n_ch = emg_array.shape

    if baseline_rms is None:
        baseline_rms = estimate_baseline_rms(emg_array)

    # Threshold per channel in absolute amplitude (mV)
    thresholds = baseline_rms * threshold_factor   # (C,)

    artefact_mask = np.zeros(len(window_starts), dtype=bool)

    for i, start in enumerate(window_starts):
        end = min(start + window_size, n_samples)
        if end <= start:
            artefact_mask[i] = True
            continue
        window = emg_array[start:end, :]   # (W, C)
        peak_per_channel = np.max(np.abs(window), axis=0)  # (C,)
        if np.any(peak_per_channel > thresholds):
            artefact_mask[i] = True

    n_artefact = int(artefact_mask.sum())
    n_total    = len(window_starts)
    pct        = 100.0 * n_artefact / max(n_total, 1)
    print(f"  EMG artefact rejection: {n_artefact}/{n_total} windows flagged "
          f"({pct:.1f}%) at {threshold_factor:.0f}× baseline RMS threshold")
    if pct > 30.0:
        print(f"  [WARNING] >30% of windows flagged — check electrode contact "
              f"and cable routing. Threshold factor: {threshold_factor}×.")

    return artefact_mask


def compute_artefact_free_rms(
    emg_window: np.ndarray,
    baseline_rms: np.ndarray,
    threshold_factor: float = ARTEFACT_THRESHOLD_FACTOR,
) -> Optional[np.ndarray]:
    """
    Compute per-channel RMS for a single window, returning None if the window
    is flagged as an artefact.

    Lightweight wrapper for use inside the sliding-window feature loop when
    window-by-window artefact gating is preferred over batch flagging.

    Parameters
    ----------
    emg_window      : (W,) or (W, C) filtered EMG window
    baseline_rms    : (C,) per-channel baseline RMS (from estimate_baseline_rms)
    threshold_factor: rejection threshold multiplier

    Returns
    -------
    rms : (C,) per-channel RMS, or None if window is an artefact
    """
    if emg_window.ndim == 1:
        emg_window = emg_window[:, np.newaxis]
    thresholds  = baseline_rms * threshold_factor
    peak        = np.max(np.abs(emg_window), axis=0)
    if np.any(peak > thresholds):
        return None
    return np.sqrt(np.mean(emg_window ** 2, axis=0))


# Diagnostic: frequency response

def plot_filter_response(
    fs: float = EMG_FS_DEFAULT,
    apply_notch: bool = True,
    ax=None,
) -> None:
    """
    Plot the combined frequency response of the filter chain.
    Useful for verification during development.

    Parameters
    ----------
    fs          : sampling frequency in Hz
    apply_notch : include notch in the cascade
    ax          : matplotlib Axes (created if None)
    """
    import matplotlib.pyplot as plt

    worN = 4096
    freqs = np.linspace(0, fs / 2, worN)

    # Bandpass response
    sos_bp = design_bandpass(fs=fs)
    _, h_bp = sp_signal.sosfreqz(sos_bp, worN=worN, fs=fs)
    h_combined = np.abs(h_bp) ** 2   # zero-phase → amplitude squared = sosfiltfilt

    # Notch response (if requested)
    if apply_notch:
        b, a = design_notch(fs=fs)
        _, h_notch = sp_signal.freqz(b, a, worN=worN, fs=fs)
        h_combined *= np.abs(h_notch) ** 2

    h_db = 10 * np.log10(np.maximum(h_combined, 1e-12))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(freqs, h_db, lw=1.5)
    ax.axhline(-3, color="gray", linestyle="--", lw=0.8, label="-3 dB")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Gain (dB)")
    ax.set_title("sEMG filter chain — combined frequency response (zero-phase)")
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(-80, 5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
