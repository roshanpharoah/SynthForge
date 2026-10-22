"""Empirical privacy attack suite: membership inference and attribute inference.

These attacks produce *evidence*, not guarantees. A low attack advantage is consistent with privacy but
does not prove it; a high advantage is a concrete warning that the synthetic data leaks information
about training rows. Results are always reported alongside the formal DP claim (if any).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from synthforge.core.types import TableSchema


def _encode(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    parts = []
    for c in columns:
        s = frame[c]
        if s is None:
            continue
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            v = np.where(np.isfinite(v), v, np.nan)
            parts.append(v[:, None])
        else:
            codes = pd.Categorical(s.astype("object").where(s.notna(), "<M>").astype(str)).codes.astype(float)
            parts.append(codes[:, None])
    if not parts:
        return np.zeros((len(frame), 0))
    X = np.concatenate(parts, axis=1)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.where(np.isfinite(X), X, med)


def nearest_neighbour_distances(real: np.ndarray, syn: np.ndarray, *, max_real: int = 3000) -> dict[str, float]:
    if real.size == 0 or syn.size == 0:
        return {"min": float("nan"), "median": float("nan"), "d5": float("nan"), "exact_matches": 0.0}
    r = real[:max_real]
    s = syn[:max_real]
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(r)
        d, _ = tree.query(s, k=1)
    except Exception:
        d = np.array(
            [np.min(np.linalg.norm(r - row, axis=1)) for row in s[: 400]]
        )
    scale = float(np.std(r, axis=0).mean() or 1.0)
    d = np.asarray(d) / scale
    return {
        "min": float(np.min(d)),
        "median": float(np.median(d)),
        "d5": float(np.quantile(d, 0.05)),
        "exact_matches": float(np.mean(d < 1e-9)),
        "scale": scale,
    }


def _membership_dataset(
    train: np.ndarray, holdout: np.ndarray, syn: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Build (features, is_member) from the distance to the closest synthetic row."""
    rows = []
    labels = []
    for member, X in ((1, train), (0, holdout)):
        for row in X:
            if syn.size == 0:
                continue
            dist = np.linalg.norm(syn - row, axis=1)
            d_sorted = np.sort(dist)
            feats = [
                float(d_sorted[0]),
                float(np.mean(d_sorted[: min(5, len(d_sorted))])),
                float(np.median(d_sorted)),
                float(d_sorted[min(10, len(d_sorted) - 1)]),
            ]
            rows.append(feats)
            labels.append(member)
    if not rows:
        return np.zeros((0, 4)), np.zeros(0, dtype=int)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=int)


def membership_inference(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: TableSchema,
    *,
    seed: int = 0,
    max_rows: int = 4000,
    test_fraction: float = 0.3,
) -> dict[str, Any]:
    """Distance-based + model-based membership inference (report only, no guarantee)."""
    cols = [c.name for c in schema.columns if c.name in real.columns and c.name in syn.columns and not c.dropped and c.role not in ("primary_key",)]
    if not cols or len(real) < 10 or len(syn) < 5:
        return {"error": "insufficient data for a membership attack", "advantage": float("nan")}
    real = real.head(max_rows).reset_index(drop=True)
    syn_df = syn.head(max_rows).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real))
    n_hold = max(int(len(real) * test_fraction), 3)
    train_df = real.iloc[idx[n_hold:]]
    holdout_df = real.iloc[idx[:n_hold]]
    X_train = _encode(train_df, cols)
    X_hold = _encode(holdout_df, cols)
    X_syn = _encode(syn_df, cols)
    nn = nearest_neighbour_distances(X_train, X_syn, max_real=min(len(X_train), 2000))
    dd = {}
    for name, X in (("train", X_train), ("holdout", X_hold)):
        if len(X):
            d = np.array([np.min(np.linalg.norm(X_syn - row, axis=1)) for row in X[:800]])
            dd[name] = {"median_distance": float(np.median(d)), "min_distance": float(np.min(d))}
    # threshold attack: classify as member if closer than the union median
    X_m, y_m = _membership_dataset(X_train[:1200], X_hold[:1200], X_syn[:1500], rng)
    metrics: dict[str, Any] = {"nn_distances": nn, "distance_by_group": dd}
    if len(X_m) > 20 and len(np.unique(y_m)) == 2:
        Xtr, Xte, ytr, yte = train_test_split(X_m, y_m, test_size=0.4, random_state=seed, stratify=y_m)
        clf = RandomForestClassifier(n_estimators=200, random_state=seed, min_samples_leaf=2)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        acc = float(accuracy_score(yte, (proba > 0.5).astype(int)))
        metrics["model_attack"] = {
            "auc": float(roc_auc_score(yte, proba)),
            "accuracy": acc,
            "advantage": float(2 * acc - 1),
            "baseline_accuracy": float(max(np.mean(yte), 1 - np.mean(yte))),
            "baseline_advantage": float(2 * max(np.mean(yte), 1 - np.mean(yte)) - 1),
        }
        metrics["advantage"] = metrics["model_attack"]["advantage"]
    else:
        metrics["advantage"] = float("nan")
    # similarity-based threshold attack (exact/near duplicates)
    d_train = np.array([np.min(np.linalg.norm(X_syn - row, axis=1)) for row in X_train[:1200]]) if len(X_train) else np.array([])
    d_hold = np.array([np.min(np.linalg.norm(X_syn - row, axis=1)) for row in X_hold[:1200]]) if len(X_hold) else np.array([])
    if d_train.size and d_hold.size:
        thr = float(np.median(np.concatenate([d_train, d_hold])))
        tpr = float(np.mean(d_train <= thr))
        fpr = float(np.mean(d_hold <= thr))
        metrics["threshold_attack"] = {"threshold": thr, "tpr": tpr, "fpr": fpr, "advantage": tpr - fpr}
    n_cols = len(cols)
    metrics["nearest_neighbour_ratio"] = float(nn.get("d5", float("nan")) / max(nn.get("median", 1.0), 1e-9))
    metrics["exact_match_rate"] = float(np.mean(np.min(np.abs(X_syn[:, None] - X_train[:200][None, :]), axis=2) < 1e-9)) if n_cols and len(X_train) else float("nan")
    metrics["note"] = "Empirical attack results are evidence, not a privacy proof."
    return metrics


def attribute_inference(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: TableSchema,
    *,
    target: str,
    known: list[str] | None = None,
    seed: int = 0,
    max_rows: int = 4000,
) -> dict[str, Any]:
    """Predict a sensitive attribute for *real* rows using only synthetic data as training signal."""
    if target not in real.columns or target not in syn.columns:
        return {"error": f"attribute {target!r} missing"}
    known = known or [c.name for c in schema.columns if c.name in real.columns and c.name != target and c.kind in ("categorical", "boolean", "integer")]
    known = [c for c in known if c in syn.columns]
    if not known:
        return {"error": "no known attributes"}
    real = real.head(max_rows)
    Xr = _encode(real, known)
    yr = pd.Categorical(real[target].astype("object").where(real[target].notna(), "<M>").astype(str)).codes
    syn_v = syn.head(max_rows)
    Xs = _encode(syn_v, known)
    ys = pd.Categorical(syn_v[target].astype("object").where(syn_v[target].notna(), "<M>").astype(str)).codes
    mapping = {lbl: i for i, lbl in enumerate(pd.Categorical(real[target].astype("object").where(real[target].notna(), "<M>").astype(str)).categories)}
    if len(Xr) < 10 or len(Xs) < 10 or len(np.unique(yr)) < 2 or len(np.unique(ys)) < 2:
        return {"error": "insufficient data"}
    clf = RandomForestClassifier(n_estimators=200, random_state=seed, min_samples_leaf=2)
    clf.fit(Xs, ys)
    preds = clf.predict(Xr)
    acc = float((preds == yr).mean())
    majority = float(pd.Series(yr).value_counts(normalize=True).iloc[0])
    return {
        "target": target,
        "known_attributes": known,
        "accuracy": acc,
        "majority_baseline": majority,
        "advantage": float(acc - majority),
        "categories": mapping,
        "note": "Attribute inference uses synthetic rows as the only training signal.",
    }


def run_attack_suite(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: TableSchema | None = None,
    *,
    sensitive: str | None = None,
    known: list[str] | None = None,
    seed: int = 0,
    max_rows: int = 4000,
) -> dict[str, Any]:
    if schema is None:
        raise ValueError("schema is required for the attack suite")
    out: dict[str, Any] = {"seed": seed, "max_rows": max_rows}
    out["membership_inference"] = membership_inference(real, syn, schema, seed=seed, max_rows=max_rows)
    if sensitive and sensitive in real.columns:
        out["attribute_inference"] = attribute_inference(real, syn, schema, target=sensitive, known=known, seed=seed, max_rows=max_rows)
    from synthforge.privacy.kanon import k_anonymity

    qi = [c.name for c in schema.columns if c.sensitivity == "quasi_identifier"]
    if qi:
        out["k_anonymity_real"] = k_anonymity(real, qi)
        out["k_anonymity_synthetic"] = k_anonymity(syn, qi)
    out["interpretation"] = (
        "Attack advantage near 0 with overlapping distance distributions is consistent with privacy; "
        "it is not proof. Always read this together with the formal DP claim and its stated scope."
    )
    return out


__all__ = ["run_attack_suite", "membership_inference", "attribute_inference", "nearest_neighbour_distances"]
