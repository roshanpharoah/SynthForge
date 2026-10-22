"""Fidelity metrics: distributional distances between real and synthetic tables.

Every metric returns a value and a direction ("lower_better" / "higher_better") so the scorecard can
normalise without hand-tuned constants. All metrics are computed from the actual frames handed in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from synthforge.core.types import TableSchema


def _numeric_columns(schema: TableSchema, frame: pd.DataFrame) -> list[str]:
    out = []
    for c in schema.columns:
        if c.name in frame.columns and c.kind in ("integer", "float") and not c.dropped:
            out.append(c.name)
    return out


def _categorical_columns(schema: TableSchema, frame: pd.DataFrame) -> list[str]:
    out = []
    for c in schema.columns:
        if c.name in frame.columns and c.kind in ("categorical", "boolean") and not c.dropped:
            out.append(c.name)
    return out


def ks_statistic(real: pd.Series, syn: pd.Series) -> float:
    a = pd.to_numeric(real, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(syn, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(stats.ks_2samp(a, b).statistic)


def wasserstein(real: pd.Series, syn: pd.Series) -> float:
    a = pd.to_numeric(real, errors="coerce").dropna().to_numpy(dtype=float)
    b = pd.to_numeric(syn, errors="coerce").dropna().to_numpy(dtype=float)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    scale = float(np.std(np.concatenate([a, b])) or 1.0)
    return float(stats.wasserstein_distance(a, b) / scale)


def chi_square_tvd(real: pd.Series, syn: pd.Series) -> dict[str, float]:
    a = real.astype("object").where(real.notna(), "<MISSING>").astype(str)
    b = syn.astype("object").where(syn.notna(), "<MISSING>").astype(str)
    cats = sorted(set(a.unique()) | set(b.unique()))
    ca = a.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    cb = b.value_counts().reindex(cats, fill_value=0).to_numpy(dtype=float)
    pa = ca / max(ca.sum(), 1)
    pb = cb / max(cb.sum(), 1)
    tvd = float(0.5 * np.abs(pa - pb).sum())
    chi2 = float("nan")
    p = float("nan")
    if ca.sum() > 0 and cb.sum() > 0 and len(cats) > 1:
        try:
            table = np.vstack([ca, cb])
            chi2, p, _, _ = stats.chi2_contingency(table + 1e-9)
            chi2, p = float(chi2), float(p)
        except Exception:
            pass
    return {"tvd": tvd, "chi2": chi2, "p_value": p, "js": float(jensenshannon(pa, pb, base=2))}


def correlation_difference(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    cols = [c for c in cols if c in real.columns and c in syn.columns]
    if len(cols) < 2:
        return {"mean_abs_diff": float("nan"), "max_abs_diff": float("nan"), "columns": cols}
    r = real[cols].apply(pd.to_numeric, errors="coerce").corr()
    s = syn[cols].apply(pd.to_numeric, errors="coerce").corr()
    diff = (r - s).abs()
    mask = ~np.eye(len(cols), dtype=bool)
    vals = diff.to_numpy()[mask]
    vals = vals[np.isfinite(vals)]
    return {
        "mean_abs_diff": float(np.mean(vals)) if vals.size else float("nan"),
        "max_abs_diff": float(np.max(vals)) if vals.size else float("nan"),
        "columns": cols,
    }


def mutual_information_difference(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    from sklearn.metrics import mutual_info_score

    cols = [c for c in cols if c in real.columns and c in syn.columns]
    if len(cols) < 2:
        return {"mean_abs_diff": float("nan"), "pairs": []}
    diffs = []
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            ra = real[a].astype("object").where(real[a].notna(), "<M>").astype(str)
            rb = real[b].astype("object").where(real[b].notna(), "<M>").astype(str)
            sa = syn[a].astype("object").where(syn[a].notna(), "<M>").astype(str)
            sb = syn[b].astype("object").where(syn[b].notna(), "<M>").astype(str)
            mi_r = mutual_info_score(ra.factorize()[0], rb.factorize()[0])
            mi_s = mutual_info_score(sa.factorize()[0], sb.factorize()[0])
            diffs.append(abs(mi_r - mi_s))
            pairs.append({"pair": f"{a}~{b}", "real": float(mi_r), "synthetic": float(mi_s)})
    return {"mean_abs_diff": float(np.mean(diffs)) if diffs else float("nan"), "pairs": pairs[:50]}


def _prepare_matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = []
    for c in cols:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            v = np.where(np.isfinite(v), v, np.nan)
            parts.append(v[:, None])
        else:
            codes = pd.Categorical(s.astype("object").where(s.notna(), "<M>").astype(str)).codes.astype(float)
            parts.append(codes[:, None])
    if not parts:
        return np.zeros((len(frame), 0))
    return np.concatenate(parts, axis=1)


def discriminator_auc(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str], *, seed: int = 0) -> dict[str, Any]:
    """Train a classifier to tell real from synthetic. 0.5 means indistinguishable."""
    cols = [c for c in cols if c in real.columns and c in syn.columns]
    if not cols or len(real) < 20 or len(syn) < 20:
        return {"auc": float("nan"), "accuracy": float("nan")}
    Xr = _prepare_matrix(real, cols)
    Xs = _prepare_matrix(syn, cols)
    X = np.vstack([Xr, Xs])
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xs))])
    col_med = np.nanmedian(X, axis=0)
    col_med = np.where(np.isfinite(col_med), col_med, 0.0)
    X = np.where(np.isfinite(X), X, col_med)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    models = {
        "logistic": LogisticRegression(max_iter=1000),
        "hgb": HistGradientBoostingClassifier(max_iter=200, random_state=seed),
    }
    out: dict[str, Any] = {}
    for name, model in models.items():
        try:
            model.fit(X_tr, y_tr)
            proba = model.predict_proba(X_te)[:, 1]
            out[name] = {
                "auc": float(roc_auc_score(y_te, proba)),
                "accuracy": float((model.predict(X_te) == y_te).mean()),
            }
        except Exception as exc:
            out[name] = {"auc": float("nan"), "accuracy": float("nan"), "error": str(exc)}
    best = max(
        (v["auc"] for v in out.values() if v.get("auc") is not None and np.isfinite(v.get("auc", float("nan")))),
        default=float("nan"),
    )
    out["auc"] = best
    return out


def sliced_wasserstein(real: pd.DataFrame, syn: pd.DataFrame, cols: list[str], *, n_projections: int = 64, seed: int = 0) -> float:
    cols = [c for c in cols if c in real.columns and c in syn.columns]
    if not cols:
        return float("nan")
    Xr = _prepare_matrix(real, cols)
    Xs = _prepare_matrix(syn, cols)
    med = np.nanmedian(np.vstack([Xr, Xs]), axis=0)
    Xr = np.where(np.isfinite(Xr), Xr, med)
    Xs = np.where(np.isfinite(Xs), Xs, med)
    mean = Xr.mean(axis=0)
    std = Xr.std(axis=0) + 1e-9
    Xr = (Xr - mean) / std
    Xs = (Xs - mean) / std
    rng = np.random.default_rng(seed)
    d = Xr.shape[1]
    dirs = rng.normal(size=(n_projections, d))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
    pr = Xr @ dirs.T
    ps = Xs @ dirs.T
    vals = [stats.wasserstein_distance(pr[:, i], ps[:, i]) for i in range(n_projections)]
    return float(np.mean(vals))


def column_summary(real: pd.DataFrame, syn: pd.DataFrame, schema: TableSchema, *, seed: int = 0) -> dict[str, Any]:
    num = _numeric_columns(schema, real)
    cat = _categorical_columns(schema, real)
    out: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for c in num:
        if c not in syn.columns:
            continue
        out["numeric"][c] = {
            "ks": ks_statistic(real[c], syn[c]),
            "wasserstein_normalised": wasserstein(real[c], syn[c]),
            "real_mean": float(pd.to_numeric(real[c], errors="coerce").mean()),
            "synthetic_mean": float(pd.to_numeric(syn[c], errors="coerce").mean()),
            "real_std": float(pd.to_numeric(real[c], errors="coerce").std()),
            "synthetic_std": float(pd.to_numeric(syn[c], errors="coerce").std()),
            "real_missing_rate": float(real[c].isna().mean()),
            "synthetic_missing_rate": float(syn[c].isna().mean()),
        }
    for c in cat:
        if c not in syn.columns:
            continue
        out["categorical"][c] = chi_square_tvd(real[c], syn[c])
    ks_vals = [v["ks"] for v in out["numeric"].values() if np.isfinite(v["ks"])]
    tvd_vals = [v["tvd"] for v in out["categorical"].values() if np.isfinite(v["tvd"])]
    out["summary"] = {
        "mean_ks": float(np.mean(ks_vals)) if ks_vals else float("nan"),
        "max_ks": float(np.max(ks_vals)) if ks_vals else float("nan"),
        "mean_tvd": float(np.mean(tvd_vals)) if tvd_vals else float("nan"),
        "max_tvd": float(np.max(tvd_vals)) if tvd_vals else float("nan"),
    }
    return out


def fidelity_report(real: pd.DataFrame, syn: pd.DataFrame, schema: TableSchema, *, seed: int = 0) -> dict[str, Any]:
    all_cols = [c.name for c in schema.columns if c.name in real.columns and c.name in syn.columns and not c.dropped]
    num = _numeric_columns(schema, real)
    return {
        "columns": column_summary(real, syn, schema, seed=seed),
        "correlation": correlation_difference(real, syn, num),
        "mutual_information": mutual_information_difference(real, syn, all_cols[:12]),
        "discriminator": discriminator_auc(real, syn, all_cols, seed=seed),
        "sliced_wasserstein": sliced_wasserstein(real, syn, all_cols, seed=seed),
        "n_real": int(len(real)),
        "n_synthetic": int(len(syn)),
    }


__all__ = [
    "fidelity_report",
    "ks_statistic",
    "wasserstein",
    "chi_square_tvd",
    "correlation_difference",
    "mutual_information_difference",
    "discriminator_auc",
    "sliced_wasserstein",
    "column_summary",
]
