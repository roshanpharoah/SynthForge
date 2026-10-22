"""ECG/time-series ingestion: WFDB, EDF, CSV and NumPy with resampling, windowing and QC."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import signal as sps

from synthforge.core.errors import IngestError
from synthforge.core.types import Dataset, SignalCohort
from synthforge.ingest.profile import profile_dataset
from synthforge.media.physio import quality_check

LEAD_ALIASES = {
    "MLII": "II",
    "MLI": "I",
    "LEAD II": "II",
    "LEAD I": "I",
    "LEAD III": "III",
    "AVR": "aVR",
    "AVL": "aVL",
    "AVF": "aVF",
    "ECG1": "I",
    "ECG2": "II",
    "ECG3": "III",
}
SUPPORTED_ECG = {".hea", ".edf", ".csv", ".tsv", ".npy", ".npz"}


def _canonical_lead(name: str) -> str:
    n = str(name).strip()
    return LEAD_ALIASES.get(n.upper(), n)


def _resample(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return x.astype(np.float32)
    frac = Fraction(fs_out / fs_in).limit_denominator(200)
    if frac.numerator == frac.denominator:
        return x.astype(np.float32)
    y = sps.resample_poly(x, frac.numerator, frac.denominator, axis=-1)
    return np.asarray(y, dtype=np.float32)


def _fit_length(x: np.ndarray, n: int) -> tuple[np.ndarray, bool]:
    if x.shape[-1] == n:
        return x, False
    if x.shape[-1] > n:
        start = max(0, (x.shape[-1] - n) // 2)
        return x[..., start : start + n], False
    pad = n - x.shape[-1]
    return np.pad(x, ((0, 0), (0, pad)), mode="constant"), True


def _normalize(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.nan_to_num(x.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == "mv":
        return np.clip(x, -5.0, 5.0)
    if mode == "zscore":
        mu = np.mean(x, axis=-1, keepdims=True)
        sd = np.std(x, axis=-1, keepdims=True)
        return ((x - mu) / np.maximum(sd, 1e-6)).astype(np.float32)
    if mode == "minmax":
        lo = np.min(x, axis=-1, keepdims=True)
        hi = np.max(x, axis=-1, keepdims=True)
        return (2 * (x - lo) / np.maximum(hi - lo, 1e-6) - 1).astype(np.float32)
    raise IngestError(f"unknown normalization mode {mode!r}", hint="Use mv, zscore or minmax.")


def read_edf(path: str | Path) -> dict[str, Any]:
    """Minimal EDF/EDF+ reader (uncompressed). Returns signals, fs, lead names and stripped header info."""
    p = Path(path)
    raw = p.read_bytes()
    if len(raw) < 256:
        raise IngestError("EDF file is shorter than its 256-byte header", hint="Verify the file is a valid EDF.")
    try:
        header_bytes = int(raw[184:192].decode("ascii").strip())
        reserved = raw[192:236]
        n_records = int(raw[236:244].decode("ascii").strip())
        duration = float(raw[244:252].decode("ascii").strip())
        ns = int(raw[252:256].decode("ascii").strip())
    except ValueError as exc:
        raise IngestError("invalid EDF header numeric fields", hint="This may not be an EDF file.") from exc
    if ns < 1 or header_bytes < 256 + ns * 256:
        raise IngestError("invalid EDF signal count", hint="Verify the file is uncompressed EDF.")
    pos = 256
    fields: dict[str, list[str]] = {}
    widths = {
        "label": 16,
        "transducer": 80,
        "dimension": 8,
        "pmin": 8,
        "pmax": 8,
        "dmin": 8,
        "dmax": 8,
        "prefilter": 80,
        "spr": 8,
        "reserved": 32,
    }
    for key, width in widths.items():
        vals = []
        for _ in range(ns):
            vals.append(raw[pos : pos + width].decode("ascii", errors="replace").strip())
            pos += width
        fields[key] = vals
    try:
        pmin = np.array([float(v) for v in fields["pmin"]])
        pmax = np.array([float(v) for v in fields["pmax"]])
        dmin = np.array([float(v) for v in fields["dmin"]])
        dmax = np.array([float(v) for v in fields["dmax"]])
        spr = np.array([int(float(v)) for v in fields["spr"]])
    except ValueError as exc:
        raise IngestError("invalid EDF scaling fields", hint="The EDF header is corrupted.") from exc
    data = np.frombuffer(raw[header_bytes:], dtype="<i2")
    total_samples = int(spr.sum()) * max(n_records if n_records > 0 else 1, 1)
    if data.size < total_samples:
        # allow truncated final record
        n_records = max(1, data.size // int(spr.sum()))
    per_signal = []
    offset = 0
    for r in range(max(n_records, 1)):
        for i in range(ns):
            count = int(spr[i])
            chunk = data[offset : offset + count].astype(np.float64)
            offset += count
            if chunk.size < count and r == 0:
                raise IngestError("EDF data is truncated", hint="Re-export the EDF file.")
            per_signal.append(chunk)
    signals = []
    fs_list = []
    for i in range(ns):
        chunks = [per_signal[r * ns + i] for r in range(max(n_records, 1)) if r * ns + i < len(per_signal)]
        arr = np.concatenate([c for c in chunks if c.size]) if chunks else np.array([])
        if arr.size == 0:
            continue
        dig = arr
        phys = (dig - dmin[i]) / max(dmax[i] - dmin[i], 1e-9) * (pmax[i] - pmin[i]) + pmin[i]
        signals.append(phys)
        fs_list.append(spr[i] / max(duration, 1e-6))
    if not signals:
        raise IngestError("EDF has no readable signal samples")
    target_fs = int(round(float(np.median(fs_list))))
    common_len = int(min(len(s) for s in signals))
    resampled = []
    for s, fs_i in zip(signals, fs_list, strict=False):
        s = s[:common_len]
        resampled.append(_resample(s, fs_i, target_fs))
    min_len = min(len(s) for s in resampled)
    matrix = np.vstack([s[:min_len] for s in resampled]).astype(np.float32)
    # EDF physical dimension: convert microvolts to millivolts
    units = [u.strip().lower() for u in fields["dimension"]]
    for i, u in enumerate(units):
        if u in ("uv", "µv", "microvolt", "microvolts"):
            matrix[i] /= 1000.0
    return {
        "signals": matrix,
        "fs": target_fs,
        "lead_names": [_canonical_lead(x) for x in fields["label"]],
        "header_identifiers_removed": True,
        "metadata": {
            "format": "EDF",
            "n_records": n_records,
            "record_duration_s": duration,
            "reserved_field_present": bool(reserved.strip(b"\x00 ")),
            "patient_field_removed": True,
            "recording_field_removed": True,
            "synthetic_source": False,
        },
    }


def read_wfdb(path: str | Path) -> dict[str, Any]:
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise IngestError("wfdb is required for WFDB records", hint="pip install wfdb") from exc
    p = Path(path)
    record = p.with_suffix("") if p.suffix.lower() == ".hea" else p
    try:
        rec = wfdb.rdrecord(str(record))
    except Exception as exc:
        raise IngestError(
            f"WFDB read failed: {exc}",
            hint="Ensure the .hea and .dat files share a name and the record is uncompressed or supported.",
        ) from exc
    x = np.asarray(rec.p_signal, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    x = x.T
    fs = int(round(float(rec.fs)))
    units = [str(u).lower() for u in (rec.units or [])]
    for i, u in enumerate(units):
        if u in ("uv", "µv", "microvolt", "microvolts") and i < x.shape[0]:
            x[i] /= 1000.0
    names = [_canonical_lead(n) for n in (rec.sig_name or [f"lead_{i}" for i in range(x.shape[0])])]
    return {
        "signals": x,
        "fs": fs,
        "lead_names": names,
        "header_identifiers_removed": True,
        "metadata": {
            "format": "WFDB",
            "comments_removed": True,
            "n_samples": int(x.shape[1]),
            "units": units,
            "synthetic_source": False,
        },
    }


def read_ecg_csv(path: str | Path, default_fs: int = 250) -> dict[str, Any]:
    p = Path(path)
    try:
        df = pd.read_csv(p, sep=None, engine="python")
    except Exception as exc:
        raise IngestError(f"could not read ECG CSV: {exc}", hint="Use numeric lead columns and an optional time column.") from exc
    if df.empty:
        raise IngestError("ECG CSV is empty")
    time_col = next((c for c in df.columns if str(c).lower() in ("time", "t", "time_s", "timestamp", "sec", "seconds")), None)
    if time_col is not None:
        t = pd.to_numeric(df[time_col], errors="coerce").to_numpy()
        dt = np.nanmedian(np.diff(t))
        fs = int(round(1.0 / dt)) if dt and np.isfinite(dt) and dt > 0 else default_fs
        lead_cols = [c for c in df.columns if c != time_col]
    else:
        fs = default_fs
        lead_cols = list(df.columns)
    numeric = [c for c in lead_cols if pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        raise IngestError("ECG CSV has no numeric lead columns", hint="Remove text columns or convert values to numbers.")
    x = df[numeric].to_numpy(dtype=np.float32).T
    return {
        "signals": x,
        "fs": fs,
        "lead_names": [_canonical_lead(str(c)) for c in numeric],
        "header_identifiers_removed": False,
        "metadata": {"format": "CSV", "synthetic_source": False},
    }


def read_ecg_numpy(path: str | Path, default_fs: int = 250) -> dict[str, Any]:
    p = Path(path)
    if p.suffix.lower() == ".npz":
        z = np.load(p, allow_pickle=False)
        key = "signals" if "signals" in z else next(iter(z.files))
        x = np.asarray(z[key], dtype=np.float32)
        fs = int(z["fs"]) if "fs" in z else default_fs
    else:
        x = np.asarray(np.load(p, allow_pickle=False), dtype=np.float32)
        fs = default_fs
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 2 and x.shape[0] > x.shape[1]:
        x = x.T
    if x.ndim != 2:
        raise IngestError(f"unsupported NumPy ECG shape {x.shape}", hint="Expected (leads, samples) or (samples, leads).")
    return {
        "signals": x,
        "fs": fs,
        "lead_names": [f"lead_{i}" for i in range(x.shape[0])],
        "header_identifiers_removed": False,
        "metadata": {"format": "NumPy", "synthetic_source": False},
    }


def _discover(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise IngestError(f"ECG path not found: {path}", hint="Pass a file or a directory of ECG records.")
    files = [
        p
        for p in sorted(path.iterdir())
        if p.suffix.lower() in SUPPORTED_ECG and p.name.lower() not in ("labels.csv",)
    ]
    if not files:
        raise IngestError(
            "no ECG files found",
            hint="Supported: .hea/.dat (WFDB), .edf, .csv/.tsv, .npy/.npz.",
        )
    return files


def load_ecg(
    path: str | Path,
    *,
    name: str | None = None,
    labels_path: str | Path | None = None,
    target_fs: int = 250,
    window_seconds: float = 10.0,
    normalize: str = "mv",
    max_records: int | None = None,
    profile: bool = True,
    synthetic_source: bool = False,
) -> Dataset:
    p = Path(path).resolve()
    files = _discover(p)
    if max_records:
        files = files[:max_records]
    labels = None
    lab_path = Path(labels_path) if labels_path else (p / "labels.csv" if p.is_dir() else None)
    if lab_path and lab_path.exists():
        try:
            labels = pd.read_csv(lab_path)
        except Exception as exc:
            raise IngestError(f"could not read ECG labels: {exc}", hint="Use a CSV with a 'file' column.") from exc
        if "file" in labels.columns:
            labels["_file_key"] = labels["file"].astype(str).str.replace(r"\.(hea|edf|csv|npy|npz)$", "", regex=True)
        else:
            labels["_file_key"] = [f.stem for f in files[: len(labels)]]
        labels = labels.set_index("_file_key", drop=False)
    target_len = int(round(target_fs * window_seconds))
    signals = []
    lead_names: list[str] = []
    rows = []
    warnings: list[str] = []
    deid_notes: list[dict] = []
    for idx, f in enumerate(files):
        suffix = f.suffix.lower()
        try:
            if suffix in (".edf",):
                rec = read_edf(f)
            elif suffix in (".hea",):
                rec = read_wfdb(f)
            elif suffix in (".csv", ".tsv"):
                rec = read_ecg_csv(f, default_fs=target_fs)
            elif suffix in (".npy", ".npz"):
                rec = read_ecg_numpy(f, default_fs=target_fs)
            else:
                raise IngestError(f"unsupported ECG format {suffix}")
        except IngestError:
            raise
        except Exception as exc:
            raise IngestError(
                f"failed reading {f.name}: {exc}",
                hint="Check the file is a valid WFDB/EDF/CSV record.",
            ) from exc
        x = _resample(rec["signals"], float(rec["fs"]), target_fs)
        x, padded = _fit_length(x, target_len)
        if padded:
            warnings.append(f"{f.name} was shorter than {window_seconds}s and was zero-padded.")
        x = _normalize(x, normalize)
        if not lead_names:
            lead_names = rec["lead_names"]
        elif len(rec["lead_names"]) == len(lead_names) and lead_names != rec["lead_names"]:
            warnings.append(f"{f.name} lead order differed; reordered by name where possible.")
            order = []
            for ln in lead_names:
                if ln in rec["lead_names"]:
                    order.append(rec["lead_names"].index(ln))
                else:
                    order.append(None)
            if all(o is not None for o in order):
                x = x[[int(o) for o in order if o is not None]]
        if x.shape[0] != len(lead_names):
            raise IngestError(
                f"lead count mismatch in {f.name}: {x.shape[0]} vs {len(lead_names)}",
                hint="All records in one ingestion must have the same leads.",
            )
        signals.append(x)
        key = f.stem
        row = {"subject_id": key, "source_file": f.name}
        if labels is not None and key in labels.index:
            row.update({k: v for k, v in labels.loc[key].to_dict().items() if not str(k).startswith("_")})
        elif labels is not None and idx < len(labels):
            row.update({k: v for k, v in labels.iloc[idx].to_dict().items() if not str(k).startswith("_")})
        rows.append(row)
        if rec.get("header_identifiers_removed"):
            deid_notes.append({"file": f.name, "removed": ["patient_field", "recording_field", "comments"]})
    if not signals:
        raise IngestError("no ECG records were read")
    arr = np.stack(signals)
    quality = quality_check(arr, target_fs)
    qdf = pd.DataFrame(quality["per_record"])
    lab_df = pd.DataFrame(rows)
    if "subject_id" in lab_df:
        qdf.insert(0, "subject_id", lab_df["subject_id"].values)
    if "class" not in lab_df.columns:
        lab_df["class"] = "unlabeled"
        warnings.append("No class labels found. Conditional ECG generation will use 'unlabeled' only.")
    cohort = SignalCohort(
        signals=arr,
        sampling_rate=target_fs,
        lead_names=lead_names,
        labels=lab_df,
        quality=qdf,
        source_paths=[str(f) for f in files],
        metadata={
            "format": "mixed",
            "deidentification": deid_notes,
            "header_identifiers_removed": bool(deid_notes),
            "synthetic_source": synthetic_source,
            "normalization": normalize,
        },
    )
    dataset = Dataset(
        name=name or p.stem,
        kind="timeseries",
        signals=cohort,
        source={"type": "ecg", "path": str(p), "format": sorted({f.suffix.lower() for f in files})},
        warnings=sorted(set(warnings)),
        provenance={
            "loader": "synthforge.ingest.timeseries",
            "synthetic_source": synthetic_source,
            "deid_report": {"records_with_header_fields_removed": len(deid_notes), "notes": deid_notes},
        },
    )
    dataset.profile = profile_dataset(dataset) if profile else {}
    return dataset
