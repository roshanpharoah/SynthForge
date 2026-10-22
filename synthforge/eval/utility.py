"""Utility evaluation: TSTR vs TRTR across classifiers with bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from synthforge.core.types import TableSchema


def _feature_kinds(schema: TableSchema, frame: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    num, cat = [], []
    for c in schema.columns:
        if c.name == target or c.name not in frame.columns or c.dropped:
            continue
        if c.role in ("primary_key", "foreign_key", "ignore"):
            continue
        if c.kind in ("integer", "float"):
            num.append(c.name)
        elif c.kind in ("categorical", "boolean", "text") and frame[c.name].nunique(dropna=True) <= 50:
            cat.append(c.name)
    return num, cat


def _make_pipeline(kind: str, num: list[str], cat: list[str], seed: int) -> Pipeline:
    transformers = []
    if num:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num))
    if cat:
        transformers.append(
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)
        )
    pre = ColumnTransformer(transformers, remainder="drop")
    if kind == "logistic":
        clf: Any = LogisticRegression(max_iter=2000, random_state=seed)
    elif kind == "hgb":
        clf = HistGradientBoostingClassifier(max_iter=250, random_state=seed)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=600, random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def _scores(clf: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    pred = clf.predict(X)
    out = {"accuracy": float(accuracy_score(y, pred))}
    if hasattr(clf.named_steps["clf"], "predict_proba"):
        try:
            proba = clf.predict_proba(X)
            if proba.shape[1] == 2:
                out["auc"] = float(roc_auc_score(y, proba[:, 1]))
                out["f1"] = float(f1_score(y, pred))
        except Exception:
            pass
    return out


def _bootstrap_ci(
    fn: Callable[[], float], n: int, *, iterations: int, seed: int, confidence: float = 0.95
) -> dict[str, float]:
    if iterations <= 0:
        val = fn()
        return {"value": val, "low": float("nan"), "high": float("nan"), "n_boot": 0}
    vals = []
    for _ in range(iterations):
        vals.append(fn())
    base = fn()
    arr = np.asarray([v for v in vals if np.isfinite(v)])
    if arr.size < 5:
        return {"value": base, "low": float("nan"), "high": float("nan"), "n_boot": int(arr.size)}
    alpha = (1 - confidence) / 2
    return {
        "value": base,
        "low": float(np.quantile(arr, alpha)),
        "high": float(np.quantile(arr, 1 - alpha)),
        "n_boot": int(arr.size),
    }


def tstr_trtr(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: TableSchema,
    target: str,
    *,
    classifiers: list[str] | None = None,
    test_fraction: float = 0.3,
    seed: int = 0,
    bootstrap: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Train-on-synthetic-test-on-real vs train-and-test-on-real.

    TRTR is the reference ceiling on the same held-out real test split.
    """
    if target not in real.columns:
        return {"error": f"target column {target!r} not present in real data"}
    if target not in syn.columns:
        return {"error": f"target column {target!r} not present in synthetic data"}
    classifiers = classifiers or ["logistic", "hgb", "mlp"]
    y_all = pd.to_numeric(real[target], errors="coerce")
    valid = y_all.notna() & real[target].notna()
    real_v = real.loc[valid].reset_index(drop=True)
    y_v = y_all[valid].reset_index(drop=True)
    if y_v.nunique() < 2:
        return {"error": "target has fewer than two classes in real data"}
    if y_v.nunique() > 10:
        # binarise at the median for regression-like targets
        y_v = (y_v > y_v.median()).astype(int)
    num, cat = _feature_kinds(schema, real_v, target)
    if not num and not cat:
        return {"error": "no usable feature columns"}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(real_v))
    n_test = max(int(len(real_v) * test_fraction), 5)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    Xtr_r, ytr = real_v.loc[train_idx], y_v[train_idx]
    Xte, yte = real_v.loc[test_idx], y_v[test_idx]
    syn_v = syn[syn[target].notna()]
    ys = pd.to_numeric(syn_v[target], errors="coerce")
    if ys.nunique() > 10:
        ys = (ys > ys.median()).astype(int)
    keep = ys.notna()
    Xtr_s, ytr_s = syn_v.loc[keep], ys[keep]
    results: dict[str, Any] = {"target": target, "features": {"numeric": num, "categorical": cat}, "classifiers": {}}
    for kind in classifiers:
        entry: dict[str, Any] = {}
        try:
            clf_s = _make_pipeline(kind, num, cat, seed).fit(Xtr_s, ytr_s)
            tstr = _scores(clf_s, Xte, yte)
        except Exception as exc:
            entry["tstr"] = {"error": str(exc)}
            tstr = {}
        try:
            clf_r = _make_pipeline(kind, num, cat, seed).fit(Xtr_r, ytr)
            trtr = _scores(clf_r, Xte, yte)
        except Exception as exc:
            entry["trtr"] = {"error": str(exc)}
            trtr = {}
        entry["tstr"] = tstr
        entry["trtr"] = trtr
        if "accuracy" in tstr and "accuracy" in trtr:
            entry["accuracy_gap"] = float(trtr["accuracy"] - tstr["accuracy"])
            entry["utility_ratio"] = float(tstr["accuracy"] / trtr["accuracy"]) if trtr["accuracy"] else float("nan")
        if bootstrap and "accuracy" in tstr:
            def _boot_tstr(clf_s: Pipeline = clf_s, rng: np.random.Generator = rng) -> float:
                ii = rng.integers(0, len(Xte), size=len(Xte))
                Xb, yb = Xte.iloc[ii], yte.iloc[ii]
                if yb.nunique() < 2:
                    return float("nan")
                return accuracy_score(yb, clf_s.predict(Xb))

            entry["tstr_accuracy_ci"] = _bootstrap_ci(_boot_tstr, len(Xte), iterations=bootstrap, seed=seed + 1, confidence=confidence)
        results["classifiers"][kind] = entry
    accs = [v["tstr"]["accuracy"] for v in results["classifiers"].values() if isinstance(v.get("tstr"), dict) and "accuracy" in v["tstr"]]
    results["mean_tstr_accuracy"] = float(np.mean(accs)) if accs else float("nan")
    return results


def multi_seed_utility(
    real: pd.DataFrame,
    syn: pd.DataFrame,
    schema: TableSchema,
    target: str,
    *,
    seeds: int = 3,
    classifiers: list[str] | None = None,
    test_fraction: float = 0.3,
    bootstrap: int = 0,
) -> dict[str, Any]:
    """Repeat the TSTR/TRTR experiment over several splits and report mean ± spread."""
    runs = [
        tstr_trtr(real, syn, schema, target, classifiers=classifiers, test_fraction=test_fraction, seed=1000 + s, bootstrap=bootstrap)
        for s in range(max(1, seeds))
    ]
    accs = [r.get("mean_tstr_accuracy") for r in runs if r.get("mean_tstr_accuracy") is not None]
    accs = [a for a in accs if np.isfinite(a)]
    trtr_accs = []
    for r in runs:
        vals = [v["trtr"]["accuracy"] for v in r.get("classifiers", {}).values() if isinstance(v.get("trtr"), dict) and "accuracy" in v["trtr"]]
        if vals:
            trtr_accs.append(float(np.mean(vals)))
    return {
        "target": target,
        "seeds": len(runs),
        "mean_tstr_accuracy": float(np.mean(accs)) if accs else float("nan"),
        "std_tstr_accuracy": float(np.std(accs)) if accs else float("nan"),
        "mean_trtr_accuracy": float(np.mean(trtr_accs)) if trtr_accs else float("nan"),
        "std_trtr_accuracy": float(np.std(trtr_accs)) if trtr_accs else float("nan"),
        "runs": runs,
    }


def rare_class_coverage(real: pd.DataFrame, syn: pd.DataFrame, schema: TableSchema, *, threshold: float = 0.05) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in schema.columns:
        if c.name not in real.columns or c.name not in syn.columns:
            continue
        if c.kind not in ("categorical", "boolean"):
            continue
        counts = real[c.name].astype("object").where(real[c.name].notna(), "<M>").astype(str).value_counts(normalize=True)
        rare = counts[counts < threshold]
        if rare.empty:
            continue
        syn_vals = syn[c.name].astype("object").where(syn[c.name].notna(), "<M>").astype(str)
        recovered = {str(v): int((syn_vals == v).sum()) for v in rare.index}
        out[c.name] = {
            "rare_categories": {str(k): float(v) for k, v in rare.items()},
            "synthetic_counts": recovered,
            "recovered": [k for k, v in recovered.items() if v > 0],
            "missing": [k for k, v in recovered.items() if v == 0],
        }
    return out


__all__ = ["tstr_trtr", "multi_seed_utility", "rare_class_coverage"]
