"""Medical-modality evaluation: ECG physiology and imaging fidelity (FID-lite, diversity, masks)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.types import ImageCohort, SignalCohort
from synthforge.media.physio import extract_ecg_features, quality_check, summarize_features


def ecg_physiology(real: SignalCohort, syn: SignalCohort, *, fs: int | None = None) -> dict[str, Any]:
    fs_r = fs or real.sampling_rate
    fs_s = fs or syn.sampling_rate
    fr = extract_ecg_features(real.signals, fs_r, lead_names=real.lead_names)
    fsyn = extract_ecg_features(syn.signals, fs_s, lead_names=syn.lead_names)
    metrics: dict[str, Any] = {
        "real_summary": summarize_features(fr),
        "synthetic_summary": summarize_features(fsyn),
        "real_quality": {k: v for k, v in quality_check(real.signals, fs_r).items() if k != "per_record"},
        "synthetic_quality": {k: v for k, v in quality_check(syn.signals, fs_s).items() if k != "per_record"},
    }
    pairs = {}
    for col in ("hr_bpm", "rr_std_s", "qrs_width_ms", "qt_ms", "n_beats"):
        a = pd.to_numeric(fr[col], errors="coerce").dropna()
        b = pd.to_numeric(fsyn[col], errors="coerce").dropna()
        if len(a) and len(b):
            pairs[col] = {
                "real_median": float(a.median()),
                "synthetic_median": float(b.median()),
                "median_ratio": float(b.median() / a.median()) if a.median() else float("nan"),
                "std_real": float(a.std()),
                "std_synthetic": float(b.std()),
            }
    metrics["feature_comparison"] = pairs
    # distributional agreement over interpretable features
    from scipy import stats

    dist = {}
    for col in ("hr_bpm", "qrs_width_ms", "qt_ms"):
        a = pd.to_numeric(fr[col], errors="coerce").dropna().to_numpy()
        b = pd.to_numeric(fsyn[col], errors="coerce").dropna().to_numpy()
        if len(a) > 3 and len(b) > 3:
            dist[col] = {"ks": float(stats.ks_2samp(a, b).statistic), "wasserstein": float(stats.wasserstein_distance(a, b))}
    metrics["feature_distance"] = dist
    plausible = {}
    fsyn_hr = pd.to_numeric(fsyn["hr_bpm"], errors="coerce")
    plausible["hr_in_range_fraction"] = float(((fsyn_hr >= 30) & (fsyn_hr <= 220)).mean())
    plausible["nonzero_beats_fraction"] = float((pd.to_numeric(fsyn["n_beats"], errors="coerce") >= 2).mean())
    plausible["quality_ok_rate"] = float(quality_check(syn.signals, fs_s)["ok_rate"])
    metrics["plausibility"] = plausible
    metrics["note"] = "Engineering heuristics only; not a clinical validation."
    return metrics


def _image_stats(images: np.ndarray) -> dict[str, float]:
    counts, _ = np.histogram(images, bins=64)
    p = counts / max(counts.sum(), 1)
    entropy = float(-np.sum(p[p > 0] * np.log10(p[p > 0])))
    return {
        "mean": float(np.mean(images)),
        "std": float(np.std(images)),
        "min": float(np.min(images)),
        "max": float(np.max(images)),
        "entropy": entropy,
    }


def _feature_vector(images: np.ndarray, *, bins: int = 16) -> np.ndarray:
    feats = []
    flat = images.reshape(len(images), -1)
    for row in flat:
        hist, _ = np.histogram(row, bins=bins, range=(float(flat.min()), float(flat.max()) + 1e-9), density=True)
        # simple gradient stats
        gx = np.abs(np.diff(row))
        feats.append(np.concatenate([hist, [float(np.mean(gx)), float(np.std(gx)), float(np.percentile(row, 90) - np.percentile(row, 10))]]))
    return np.asarray(feats, dtype=float)


def fid_lite(real: ImageCohort, syn: ImageCohort, *, max_n: int = 400) -> dict[str, Any]:
    r = real.images[:max_n]
    s = syn.images[:max_n]
    fr = _feature_vector(r)
    fs = _feature_vector(s)
    mu_r, mu_s = fr.mean(axis=0), fs.mean(axis=0)
    cr = np.cov(fr, rowvar=False) + np.eye(fr.shape[1]) * 1e-6
    cs = np.cov(fs, rowvar=False) + np.eye(fs.shape[1]) * 1e-6
    diff = mu_r - mu_s
    try:
        from scipy.linalg import sqrtm

        covmean = sqrtm(cr @ cs)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = float(diff @ diff + np.trace(cr + cs - 2 * covmean))
    except Exception:
        fid = float(diff @ diff)
    return {
        "fid_lite": fid,
        "metric": "Fréchet distance on hand-crafted histogram/gradient features (not Inception FID)",
        "real_stats": _image_stats(r),
        "synthetic_stats": _image_stats(s),
        "n_real": int(len(r)),
        "n_synthetic": int(len(s)),
    }


def precision_recall_diversity(real: ImageCohort, syn: ImageCohort, *, k: int = 5, max_n: int = 300) -> dict[str, Any]:
    fr = _feature_vector(real.images[:max_n])
    fs = _feature_vector(syn.images[:max_n])
    if len(fr) < k + 1 or len(fs) < k + 1:
        return {"precision": float("nan"), "recall": float("nan"), "diversity": float("nan")}
    scale = fr.std(axis=0) + 1e-9
    fr = fr / scale
    fs = fs / scale

    def radii(X: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        return np.sort(d, axis=1)[:, k - 1]

    rr = radii(fr)
    rs = radii(fs)
    d_rs = np.linalg.norm(fs[:, None, :] - fr[None, :, :], axis=2)
    precision = float(np.mean(np.any(d_rs <= rr[None, :], axis=1)))
    d_sr = np.linalg.norm(fr[:, None, :] - fs[None, :, :], axis=2)
    recall = float(np.mean(np.any(d_sr <= rs[None, :], axis=1)))
    diversity = float(np.mean(rs))
    return {"precision": precision, "recall": recall, "diversity": diversity, "k": k}


def mask_consistency(real: ImageCohort, syn: ImageCohort) -> dict[str, Any]:
    if real.masks is None or syn.masks is None:
        return {"available": False, "reason": "masks not present for one of the cohorts"}
    r = real.masks
    s = syn.masks
    rbin = (r > 0.5).astype(np.float32)
    sbin = (s > 0.5).astype(np.float32)
    inter = (rbin * sbin).sum()
    dice = float(2 * inter / max(rbin.sum() + sbin.sum(), 1e-9))
    iou = float(inter / max((rbin + sbin - rbin * sbin).sum(), 1e-9))
    from scipy import stats

    return {
        "available": True,
        "dice": dice,
        "iou": iou,
        "mean_real_area": float(rbin.mean()),
        "mean_synthetic_area": float(sbin.mean()),
        "area_ks": float(stats.ks_2samp(rbin.sum(axis=tuple(range(1, rbin.ndim))), sbin.sum(axis=tuple(range(1, sbin.ndim)))).statistic),
    }


def imaging_report(real: ImageCohort, syn: ImageCohort) -> dict[str, Any]:
    return {
        "fid_lite": fid_lite(real, syn),
        "precision_recall_diversity": precision_recall_diversity(real, syn),
        "mask_consistency": mask_consistency(real, syn),
        "shape": {"real": list(real.images.shape), "synthetic": list(syn.images.shape)},
        "note": "Feature-based image metrics; they are not clinical or perceptual-equivalence claims.",
    }


__all__ = ["ecg_physiology", "imaging_report", "fid_lite", "precision_recall_diversity", "mask_consistency"]
