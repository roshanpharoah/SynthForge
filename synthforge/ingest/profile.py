"""Automatic data profiling shown before training. Aggregate statistics only, no raw row logs."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.types import Dataset, Table
from synthforge.media.physio import extract_ecg_features, summarize_features


def _safe_float(x: Any) -> float | None:
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(a, b)
    if tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return 0.0
    try:
        chi2, _, _, _ = chi2_contingency(tab, correction=False)
    except ValueError:
        return 0.0
    n = float(tab.values.sum())
    phi2 = chi2 / max(n, 1)
    r, k = tab.shape
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def profile_table(table: Table, *, seed: int = 0) -> dict[str, Any]:
    df = table.frame
    schema = table.schema
    n = len(df)
    cols: dict[str, Any] = {}
    for spec in schema.columns:
        if spec.name not in df.columns:
            continue
        s = df[spec.name]
        entry: dict[str, Any] = {
            "kind": spec.kind,
            "role": spec.role,
            "sensitivity": spec.sensitivity,
            "missing_rate": _safe_float(s.isna().mean()) or 0.0,
            "cardinality": int(s.nunique(dropna=True)),
            "dropped": spec.dropped,
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna()
            if len(v):
                q1, q3 = v.quantile(0.25), v.quantile(0.75)
                iqr = q3 - q1
                outliers = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
                entry.update(
                    {
                        "min": _safe_float(v.min()),
                        "max": _safe_float(v.max()),
                        "mean": _safe_float(v.mean()),
                        "std": _safe_float(v.std(ddof=0)),
                        "median": _safe_float(v.median()),
                        "skew": _safe_float(v.skew()),
                        "kurtosis": _safe_float(v.kurtosis()),
                        "outliers": outliers,
                        "outlier_rate": outliers / len(v),
                        "zeros": int((v == 0).sum()),
                        "negatives": int((v < 0).sum()),
                        "sentinels": spec.stats.get("sentinel"),
                    }
                )
                entry["histogram"] = _histogram(v)
        elif spec.kind in ("categorical", "boolean"):
            counts = s.astype(str).value_counts(dropna=False).head(20)
            entry["top"] = {str(k): int(v) for k, v in counts.items()}
        elif spec.kind == "datetime":
            v = pd.to_datetime(s, errors="coerce").dropna()
            if len(v):
                entry["min"] = str(v.min())
                entry["max"] = str(v.max())
        if spec.sensitivity == "direct_identifier":
            entry["note"] = "Direct identifier: excluded from modelling and never returned in previews."
        cols[spec.name] = entry

    # missingness patterns (column-presence masks; no values)
    patterns = []
    if n:
        mask = df[schema.column_names].isna() if all(c in df for c in schema.column_names) else df.isna()
        pattern_keys = mask.apply(lambda row: "".join("1" if v else "0" for v in row), axis=1)
        top = pattern_keys.value_counts().head(5)
        for key, count in top.items():
            patterns.append({"missing_columns": [c for c, f in zip(mask.columns, key, strict=False) if f == "1"], "rows": int(count), "rate": count / n})

    # numeric correlations
    num = df.select_dtypes(include=[np.number]).copy()
    corr_pairs = []
    if num.shape[1] >= 2 and n >= 10:
        corr = num.corr(method="pearson")
        for a, b in combinations(corr.columns, 2):
            v = corr.loc[a, b]
            if pd.notna(v) and abs(v) >= 0.45:
                corr_pairs.append({"a": str(a), "b": str(b), "pearson": float(v)})
        corr_pairs.sort(key=lambda x: abs(x["pearson"]), reverse=True)

    # categorical associations
    cat_cols = [c for c in df.columns if df[c].dtype == object or pd.api.types.is_bool_dtype(df[c])]
    assoc = []
    if 2 <= len(cat_cols) <= 12:
        for a, b in combinations(cat_cols, 2):
            if df[a].nunique() > 40 or df[b].nunique() > 40:
                continue
            v = _cramers_v(df[a].fillna("<NULL>"), df[b].fillna("<NULL>"))
            if v >= 0.2:
                assoc.append({"a": a, "b": b, "cramers_v": v})
        assoc.sort(key=lambda x: x["cramers_v"], reverse=True)

    # quasi-identifier screening
    qi = [c.name for c in schema.columns if c.sensitivity == "quasi_identifier" and c.name in df.columns]
    risk = _reid_risk(df, qi)
    target = next((c.name for c in schema.columns if c.role == "target"), None)
    balance = None
    if target and target in df:
        vc = df[target].value_counts(dropna=False)
        balance = {
            "target": target,
            "classes": {str(k): int(v) for k, v in vc.head(20).items()},
            "minority_rate": float(vc.min() / max(vc.sum(), 1)),
        }
    warnings = []
    if table.schema.primary_key:
        pk = table.schema.primary_key[0]
        if pk in df and df[pk].duplicated().any():
            warnings.append(f"primary key {pk} is not unique in the source; it will be regenerated when synthesising")
    if any(c.get("missing_rate", 0) > 0.4 for c in cols.values() if isinstance(c, dict)):
        warnings.append("One or more columns are >40% missing.")
    if balance and balance["minority_rate"] < 0.02:
        warnings.append("Target minority class is below 2%; generative models may under-represent it.")
    return {
        "name": table.name,
        "rows": n,
        "columns": len(df.columns),
        "column_profiles": cols,
        "missingness_patterns": patterns,
        "correlations": corr_pairs[:25],
        "categorical_associations": assoc[:15],
        "quasi_identifiers": qi,
        "reidentification_risk": risk,
        "target_balance": balance,
        "warnings": warnings,
        "schema": schema.model_dump(mode="json"),
    }


def _histogram(v: pd.Series, bins: int = 20) -> dict:
    counts, edges = np.histogram(v.to_numpy(), bins=bins)
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


def _reid_risk(df: pd.DataFrame, qi: list[str]) -> dict[str, Any]:
    if not qi or len(df) == 0:
        return {
            "level": "not_assessed",
            "unique_rows": 0,
            "unique_fraction": 0.0,
            "min_group_size": None,
            "basis": "No quasi-identifier columns were tagged. Tag them in the schema editor for screening.",
        }
    sub = df[qi].astype(str).fillna("<NULL>")
    groups = sub.groupby(qi, dropna=False).size()
    unique = int((groups == 1).sum())
    frac = unique / len(groups)
    min_group = int(groups.min()) if len(groups) else 0
    if frac > 0.2 or min_group == 1:
        level = "high"
    elif frac > 0.05 or min_group < 5:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "unique_rows": unique,
        "unique_fraction": float(frac),
        "min_group_size": min_group,
        "quasi_identifiers": qi,
        "basis": "Source-only screening heuristic over tagged quasi-identifiers; not a formal re-identification proof.",
    }


def profile_dataset(dataset: Dataset, *, seed: int = 0, max_ecg_features: int = 120) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "kind": dataset.kind,
        "rows": dataset.n_rows,
        "tables": {},
        "signals": None,
        "images": None,
        "warnings": list(dataset.warnings),
        "data_fingerprint": dataset.fingerprint(),
        "synthetic_source": bool(dataset.provenance.get("synthetic_source", dataset.synthetic)),
    }
    for name, table in dataset.tables.items():
        profile["tables"][name] = profile_table(table, seed=seed)
    if dataset.signals is not None:
        sg = dataset.signals
        feats = extract_ecg_features(sg.signals, sg.sampling_rate, lead_names=sg.lead_names, max_records=max_ecg_features)
        quality = sg.quality.to_dict(orient="records") if len(sg.quality) else []
        classes = {}
        if "class" in sg.labels:
            classes = {str(k): int(v) for k, v in sg.labels["class"].value_counts().items()}
        profile["signals"] = {
            "records": int(sg.signals.shape[0]),
            "leads": int(sg.signals.shape[1]),
            "samples": int(sg.signals.shape[2]),
            "sampling_rate": sg.sampling_rate,
            "duration_seconds": float(sg.signals.shape[2] / sg.sampling_rate),
            "lead_names": sg.lead_names,
            "classes": classes,
            "quality": quality,
            "quality_ok_rate": float(np.mean([q.get("ok", False) for q in quality])) if quality else None,
            "physiology": summarize_features(feats),
            "physiology_note": "Heuristic features for synthetic validation; not clinical measurements.",
        }
    if dataset.images is not None:
        im = dataset.images
        flat = im.images.reshape(len(im.images), -1)
        classes = {}
        if "contrast" in im.labels:
            classes = {str(k): int(v) for k, v in im.labels["contrast"].value_counts().items()}
        lesions = int(im.labels["lesion"].sum()) if "lesion" in im.labels else None
        profile["images"] = {
            "records": int(im.images.shape[0]),
            "shape": list(im.images.shape[1:]),
            "is_3d": im.is_3d,
            "has_masks": im.masks is not None,
            "mask_fraction": float(im.masks.mean()) if im.masks is not None else None,
            "intensity_mean": float(np.mean(flat)),
            "intensity_std": float(np.std(flat)),
            "contrast_counts": classes,
            "lesion_count": lesions,
            "spacing": list(im.spacing),
            "deidentification": im.deid_report,
            "source": "SYNTHETIC PHANTOM" if im.metadata.get("synthetic") else "user_upload",
        }
    if profile["tables"]:
        all_warn = []
        for t in profile["tables"].values():
            all_warn.extend(t.get("warnings", []))
        profile["warnings"] = sorted(set(profile["warnings"] + all_warn))
    return profile
