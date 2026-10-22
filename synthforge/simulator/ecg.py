"""Seeded 12-lead ECG simulator with McSharry-style PQRST morphology.

Classes approximate common patterns (normal, brady, tachy, AFib-like, PVC-like, ST-elevation-like)
but are *not* clinical data and must not be used for diagnosis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from synthforge.media.physio import quality_check
from synthforge.simulator.common import rng

LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SCALE = np.array([0.72, 1.0, 0.48, -0.62, 0.34, 0.62, -0.22, 0.32, 0.62, 0.9, 0.84, 0.7])
LEAD_T_SCALE = np.array([0.9, 1.0, 0.75, -0.8, 0.5, 0.8, -0.5, 0.4, 0.7, 0.95, 0.9, 0.8])
CLASSES = ["normal", "brady", "tachy", "afib_like", "pvc_like", "st_elevation_like"]


def gaussian(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _beat(
    fs: int,
    rr: float,
    *,
    cls: str,
    p_present: bool = True,
    wide: bool = False,
    st_elevation: float = 0.0,
) -> tuple[np.ndarray, int]:
    length = max(int(rr * fs), int(0.5 * fs))
    t = np.arange(length) / fs
    beat = np.zeros(length)
    center = 0.35
    if p_present and not wide:
        beat += gaussian(t, center - 0.20, 0.025, 0.11)
    qw, rw, sw = (0.028, 0.035, 0.03) if wide else (0.012, 0.014, 0.014)
    qamp = -0.18 if not wide else -0.28
    ramp = 1.0 if not wide else 0.75
    beat += gaussian(t, center - 0.035, qw, qamp)
    beat += gaussian(t, center, rw, ramp)
    beat += gaussian(t, center + 0.040, sw, -0.28 if not wide else -0.45)
    if st_elevation:
        a, b = center + 0.055, center + 0.20
        plateau = np.clip((t - a) / 0.02, 0, 1) * np.clip((b - t) / 0.03, 0, 1)
        beat += st_elevation * plateau
    beat += gaussian(t, center + 0.30, 0.07, 0.28)
    if wide:
        beat += gaussian(t, center + 0.38, 0.08, -0.18)
    return beat, center


def synthesize_ecg(
    r: np.random.Generator,
    *,
    fs: int = 250,
    duration: float = 10.0,
    leads: int = 12,
    cls: str = "normal",
    hr: float | None = None,
    hrv: float = 0.04,
    noise: float = 0.025,
    st_elevation: float | None = None,
) -> np.ndarray:
    if cls == "brady":
        hr = hr or r.uniform(40, 55)
    elif cls == "tachy":
        hr = hr or r.uniform(112, 150)
    elif cls in ("normal", "afib_like", "pvc_like", "st_elevation_like"):
        hr = hr or r.uniform(62, 92)
    if st_elevation is None:
        st_elevation = 0.16 if cls == "st_elevation_like" else 0.0

    n = int(fs * duration)
    t = np.arange(n) / fs
    signal_out = np.zeros((leads, n))
    pos = 0
    beat_idx = 0
    while pos < n:
        if cls == "afib_like":
            rr = float(np.clip(r.lognormal(np.log(60.0 / hr), 0.22), 0.35, 1.8))
            p_present = False
        else:
            rr = 60.0 / hr * (1 + hrv * np.sin(2 * np.pi * 0.22 * beat_idx) + r.normal(0, hrv / 2))
            rr = float(np.clip(rr, 0.3, 2.0))
            p_present = True
        wide = cls == "pvc_like" and beat_idx % 5 == 2
        if wide:
            rr = rr * 0.72
        beat, center = _beat(
            fs,
            rr,
            cls=cls,
            p_present=p_present,
            wide=wide,
            st_elevation=st_elevation if cls == "st_elevation_like" else 0.0,
        )
        end = min(n, pos + len(beat))
        signal_out[:, pos:end] += beat[: end - pos]
        pos += int(round(rr * fs))
        beat_idx += 1

    # lead projection and axis: lead II is the reference morphology
    for li in range(leads):
        scale = LEAD_SCALE[li % 12] * np.exp(r.normal(0, 0.06))
        tscale = LEAD_T_SCALE[li % 12]
        # approximate T scaling by reshaping is expensive; apply a mild low-frequency correction
        signal_out[li] *= scale
        signal_out[li] += 0.025 * tscale * np.sin(2 * np.pi * 0.33 * t + r.uniform(0, 6.28))
        signal_out[li] += 0.12 * np.sin(2 * np.pi * 0.07 * t + r.uniform(0, 6.28))  # respiration
        signal_out[li] += 0.02 * np.sin(2 * np.pi * 50 * t + r.uniform(0, 6.28))  # powerline
        signal_out[li] += r.normal(0, noise, n)
        if cls == "afib_like":
            signal_out[li] += r.normal(0, 0.035, n) * np.sin(2 * np.pi * 6.0 * t + r.uniform(0, 6.28))
        # occasional baseline wander
        if r.random() < 0.35:
            signal_out[li] += r.uniform(0.05, 0.25) * np.sin(2 * np.pi * r.uniform(0.15, 0.4) * t + r.uniform(0, 6.28))
    # clip extreme values like a real ADC would
    signal_out = np.clip(signal_out, -5.0, 5.0).astype(np.float32)
    return signal_out


def simulate_ecg_cohort(
    n: int = 80,
    seed: int = 21,
    *,
    fs: int = 250,
    duration: float = 10.0,
    leads: int = 12,
    classes: list[str] | None = None,
    class_probs: list[float] | None = None,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, list[str]]:
    r = rng(seed)
    classes = classes or CLASSES
    probs = class_probs or [0.5, 0.08, 0.1, 0.1, 0.08, 0.14]
    cls = r.choice(classes, size=n, p=np.array(probs) / np.sum(probs))
    signals = np.zeros((n, leads, int(fs * duration)), dtype=np.float32)
    labels = []
    for i in range(n):
        age = int(np.clip(r.normal(58, 17), 18, 95))
        sex = str(r.choice(["F", "M"], p=[0.48, 0.52]))
        hypertension = int(r.random() < (0.15 + 0.004 * age))
        c = str(cls[i])
        base_hr = 60 + (hypertension * 4) + r.normal(0, 3)
        sig = synthesize_ecg(r, fs=fs, duration=duration, leads=leads, cls=c, hr=base_hr)
        signals[i] = sig
        labels.append(
            {
                "subject_id": f"ECG{i:05d}",
                "patient_id": f"P{i // 2 + 1:05d}",
                "class": c,
                "age": age,
                "sex": sex,
                "hypertension": hypertension,
                "source": "seeded_mcsharry_simulator",
            }
        )
    labels_df = pd.DataFrame(labels)
    quality = quality_check(signals, fs)
    quality_df = pd.DataFrame(quality["per_record"])
    quality_df.insert(0, "subject_id", labels_df["subject_id"].values)
    lead_names = LEADS_12[:leads]
    return signals, labels_df, quality_df, lead_names


def write_wfdb_cohort(
    root: str | Path,
    n: int = 40,
    seed: int = 21,
    *,
    fs: int = 250,
    duration: float = 10.0,
    leads: int = 12,
) -> Path:
    import wfdb

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signals, labels, _, lead_names = simulate_ecg_cohort(n=n, seed=seed, fs=fs, duration=duration, leads=leads)
    rows = []
    for i in range(n):
        name = f"ecg_{i:04d}"
        wfdb.wrsamp(
            name,
            fs=fs,
            units=["mV"] * leads,
            sig_name=lead_names,
            p_signal=signals[i].T,
            fmt=["16"] * leads,
            write_dir=str(root),
        )
        rows.append({**labels.iloc[i].to_dict(), "file": f"{name}.hea"})
    meta = pd.DataFrame(rows)
    meta.to_csv(root / "labels.csv", index=False)
    return root


def _edf_field(value: str, width: int) -> bytes:
    return value[:width].ljust(width).encode("ascii", errors="replace")


def write_edf(
    path: str | Path,
    signals: np.ndarray,
    fs: int,
    lead_names: list[str],
    *,
    duration: float = 10.0,
    patient_id: str = "SYNTHETIC",
    recording_id: str = "SynthForge simulated ECG",
) -> Path:
    """Minimal EDF writer (single data record, int16). Keeps no real identifiers."""
    path = Path(path)
    x = np.asarray(signals, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    ns, n_samples = x.shape
    if " " in patient_id:
        patient_id = patient_id.replace(" ", "_")
    header_bytes = 256 + ns * 256
    rec_duration = duration
    h = bytearray()
    h += _edf_field("0", 8)
    h += _edf_field(f"X X X {patient_id}", 80)
    h += _edf_field(recording_id, 80)
    h += _edf_field("01.01.00", 8)
    h += _edf_field("00.00.00", 8)
    h += _edf_field(str(header_bytes), 8)
    h += _edf_field("", 44)
    h += _edf_field("1", 8)
    h += _edf_field(str(rec_duration), 8)
    h += _edf_field(str(ns), 4)
    phys_ranges = []
    field_values: dict[str, list[bytes]] = {k: [] for k in ("label", "transducer", "dimension", "pmin", "pmax", "dmin", "dmax", "prefilter", "spr", "reserved")}
    for i in range(ns):
        v = x[i]
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-6:
            lo, hi = -1.0, 1.0
        pad = (hi - lo) * 0.02 + 1e-6
        lo, hi = lo - pad, hi + pad
        phys_ranges.append((lo, hi))
        field_values["label"].append(_edf_field(lead_names[i] if i < len(lead_names) else f"ch{i}", 16))
        field_values["transducer"].append(_edf_field("", 80))
        field_values["dimension"].append(_edf_field("mV", 8))
        field_values["pmin"].append(_edf_field(f"{lo:.6f}", 8))
        field_values["pmax"].append(_edf_field(f"{hi:.6f}", 8))
        field_values["dmin"].append(_edf_field("-32768", 8))
        field_values["dmax"].append(_edf_field("32767", 8))
        field_values["prefilter"].append(_edf_field("", 80))
        field_values["spr"].append(_edf_field(str(n_samples), 8))
        field_values["reserved"].append(_edf_field("", 32))
    for key in ("label", "transducer", "dimension", "pmin", "pmax", "dmin", "dmax", "prefilter", "spr", "reserved"):
        h += b"".join(field_values[key])
    data = bytearray()
    for i in range(ns):
        lo, hi = phys_ranges[i]
        v = np.nan_to_num(x[i], nan=(lo + hi) / 2)
        dig = (v - lo) / (hi - lo) * 65535.0 - 32768.0
        dig = np.clip(np.round(dig), -32768, 32767).astype("<i2")
        data += dig.tobytes()
    path.write_bytes(bytes(h) + bytes(data))
    return path


def write_edf_cohort(root: str | Path, n: int = 12, seed: int = 31, *, fs: int = 250, duration: float = 10.0) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signals, labels, _, lead_names = simulate_ecg_cohort(n=n, seed=seed, fs=fs, duration=duration)
    rows = []
    for i in range(n):
        p = root / f"ecg_{i:04d}.edf"
        write_edf(p, signals[i], fs, lead_names, duration=duration, patient_id="SYNTH_PATIENT")
        rows.append({**labels.iloc[i].to_dict(), "file": p.name})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    (root / "README.txt").write_text(
        "SYNTHETIC SOURCE: generated by synthforge.simulator.ecg. Not real patient data.\n",
        encoding="utf-8",
    )
    return root


def write_ecg_csv_cohort(root: str | Path, n: int = 8, seed: int = 41, *, fs: int = 250, duration: float = 10.0) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    signals, labels, _, lead_names = simulate_ecg_cohort(n=n, seed=seed, fs=fs, duration=duration)
    rows = []
    for i in range(n):
        df = pd.DataFrame(signals[i].T, columns=lead_names)
        df.insert(0, "time_s", np.arange(len(df)) / fs)
        p = root / f"ecg_{i:04d}.csv"
        df.to_csv(p, index=False)
        rows.append({**labels.iloc[i].to_dict(), "file": p.name})
    pd.DataFrame(rows).to_csv(root / "labels.csv", index=False)
    return root
