"""Physiological signal utilities: quality checks and interpretable ECG features.

These are engineering heuristics for synthetic-data validation, not clinical measurements.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal


def bandpass(x: np.ndarray, fs: int, low: float = 0.5, high: float = 40.0, order: int = 3) -> np.ndarray:
    high = min(high, fs / 2 - 1)
    b, a = signal.butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return signal.filtfilt(b, a, x, axis=-1)


def detect_r_peaks(x: np.ndarray, fs: int) -> np.ndarray:
    """Robust-ish R peak detector on a single lead."""
    y = bandpass(x, fs, 5, min(25, fs / 2 - 2))
    energy = np.diff(y) ** 2
    win = max(1, int(0.08 * fs))
    energy = np.convolve(energy, np.ones(win) / win, mode="same")
    thresh = np.percentile(energy, 92)
    distance = int(0.28 * fs)
    peaks, props = signal.find_peaks(energy, height=thresh, distance=distance)
    # refine on the filtered signal
    refined = []
    for p in peaks:
        lo, hi = max(0, p - int(0.1 * fs)), min(len(x), p + int(0.1 * fs))
        if hi > lo:
            refined.append(lo + int(np.argmax(np.abs(y[lo:hi]))))
    return np.array(sorted(set(refined)), dtype=int)


def quality_check(signals: np.ndarray, fs: int) -> dict:
    """Return aggregate + per-record quality flags (NaN runs, flatline, clipping, saturation)."""
    if signals.ndim == 2:
        signals = signals[None, ...]
    n, leads, length = signals.shape
    per = []
    for i in range(n):
        rec = signals[i]
        nan_runs = 0
        for lead in rec:
            isn = np.isnan(lead)
            if isn.any():
                padded = np.r_[False, isn, False]
                nan_runs = max(nan_runs, int(np.sum(np.diff(padded.astype(int)) == 1)))
        flat = any(np.nanstd(lead) < 1e-9 for lead in rec)
        clip = float(np.mean(np.abs(rec) >= 4.9))
        sat = float(np.max(np.abs(rec[np.isfinite(rec)]))) if np.isfinite(rec).any() else 0.0
        per.append(
            {
                "nan_runs": nan_runs,
                "flatline": bool(flat),
                "clipping_fraction": clip,
                "max_abs_mv": sat,
                "ok": bool((nan_runs == 0) and (not flat) and clip < 0.001),
            }
        )
    df = pd.DataFrame(per)
    return {
        "per_record": per,
        "records": n,
        "ok_rate": float(df["ok"].mean()) if n else 0.0,
        "flatline_records": int(df["flatline"].sum()) if n else 0,
        "nan_records": int((df["nan_runs"] > 0).sum()) if n else 0,
        "clipping_records": int((df["clipping_fraction"] > 0.001).sum()) if n else 0,
    }


def select_lead(signals: np.ndarray, lead_names: list[str] | None = None) -> int:
    if lead_names:
        for preferred in ("II", "Lead II", "MLII", "lead_ii"):
            if preferred in lead_names:
                return lead_names.index(preferred)
    # most energetic lead
    energy = np.nanmean(np.abs(signals), axis=-1)
    return int(np.argmax(energy)) if energy.ndim == 1 else 0


def extract_ecg_features(
    signals: np.ndarray, fs: int, *, lead_names: list[str] | None = None, max_records: int | None = None
) -> pd.DataFrame:
    """Per-record HR, RR variability, QRS width, QT, P-wave presence and ST level."""
    if signals.ndim == 2:
        signals = signals[None, ...]
    idx = select_lead(signals[0] if len(signals) else np.zeros((1, 2)), lead_names)
    rows = []
    limit = len(signals) if max_records is None else min(len(signals), max_records)
    for i in range(limit):
        rec = signals[i]
        x = np.nan_to_num(rec[idx], nan=0.0)
        peaks = detect_r_peaks(x, fs)
        if len(peaks) >= 3:
            rr = np.diff(peaks) / fs
            hr = 60.0 / float(np.median(rr))
            rr_std = float(np.std(rr))
            rmssd = float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) > 1 else 0.0
            # QRS width from average beat
            half = int(0.08 * fs)
            beats = [x[max(0, p - half) : p + half] for p in peaks if p - half > 0 and p + half < len(x)]
            qrs_ms = float("nan")
            qt_ms = float("nan")
            p_present = float("nan")
            st_level = float("nan")
            if beats:
                avg = np.mean(beats, axis=0)
                center = half
                base = np.median(avg[: max(2, int(0.03 * fs))])
                amp = avg - base
                r_amp = float(np.max(np.abs(amp)))
                threshold = 0.15 * max(r_amp, 1e-6)
                above = np.flatnonzero(np.abs(amp) > threshold)
                if len(above):
                    qrs_ms = float((above[-1] - above[0]) / fs * 1000)
                # ST level: 80-160 ms after R, relative to baseline
                a, b = center + int(0.08 * fs), center + int(0.16 * fs)
                st_level = float(np.mean(amp[a:b])) if b < len(amp) else float("nan")
                # T wave end: last crossing above 12% of max T amplitude in 0.15-0.5s
                ta, tb = center + int(0.15 * fs), min(len(amp), center + int(0.5 * fs))
                if tb > ta:
                    tseg = amp[ta:tb]
                    tmax = np.max(tseg)
                    if tmax > 0.02:
                        crossings = np.flatnonzero(tseg > 0.12 * tmax)
                        t_end = ta + (crossings[-1] if len(crossings) else 0)
                        qt_ms = float((t_end - center) / fs * 1000)
                # P wave presence: a positive bump in the 100-280ms before R
                pa, pb = max(0, center - int(0.28 * fs)), max(1, center - int(0.09 * fs))
                pseg = amp[pa:pb]
                p_amp = float(np.max(pseg)) if len(pseg) else 0.0
                p_present = float(p_amp > max(0.025, 0.08 * r_amp))
        else:
            hr = rr_std = rmssd = qrs_ms = qt_ms = p_present = st_level = float("nan")
        rows.append(
            {
                "record": i,
                "hr_bpm": hr,
                "rr_std_s": rr_std,
                "rmssd_s": rmssd,
                "qrs_width_ms": qrs_ms,
                "qt_ms": qt_ms,
                "p_wave_present": p_present,
                "st_level_mv": st_level,
                "n_beats": int(len(peaks)),
            }
        )
    return pd.DataFrame(rows)


def summarize_features(features: pd.DataFrame) -> dict:
    out: dict[str, float] = {}
    for col in features.columns:
        if col == "record":
            continue
        vals = pd.to_numeric(features[col], errors="coerce")
        if vals.notna().any():
            out[f"{col}_median"] = float(vals.median())
            out[f"{col}_mean"] = float(vals.mean())
    return out
