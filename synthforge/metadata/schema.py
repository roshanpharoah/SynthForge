"""Schema inference, sentinel detection and automatic constraint discovery."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.types import ColumnSpec, ForeignKey, TableSchema, infer_semantics
from synthforge.metadata.constraints import (
    InequalityConstraint,
    NotNullConstraint,
    UniqueConstraint,
)

SENTINELS = [-999, -9999, -99999, 9999, 99999, -1, 999]

PII_PATTERNS = {
    "email": r"^[\w.+-]+@[\w-]+\.[\w.-]+$",
    "phone": r"^\+?[\d ().-]{7,20}$",
    "ssn": r"^\d{3}-\d{2}-\d{4}$",
    "url": r"^https?://",
}


def detect_value_pii(s: pd.Series) -> str | None:
    """Detect high-confidence value-level direct identifiers in a string column."""
    import re

    vals = s.dropna().astype(str).head(200)
    if len(vals) < 5:
        return None
    for name, pattern in PII_PATTERNS.items():
        matches = sum(1 for v in vals if re.match(pattern, v.strip(), flags=re.IGNORECASE))
        if matches / len(vals) >= 0.8 and (name != "phone" or vals.nunique() > len(vals) * 0.5):
            return name
    return None


def _try_datetime(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return None
    non_null = s.dropna().astype(str)
    if len(non_null) == 0:
        return None
    sample = non_null.head(500)
    date_like = sample.str.contains(r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}", regex=True).mean()
    if date_like < 0.6:
        return None
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed", dayfirst=False)
        except (ValueError, TypeError):
            parsed = pd.to_datetime(sample, errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return pd.to_datetime(s, errors="coerce", format="mixed")
    return None


def detect_sentinels(df: pd.DataFrame, candidates: list[float] | None = None) -> dict[str, dict[str, Any]]:
    candidates = candidates or SENTINELS
    out: dict[str, dict[str, Any]] = {}
    for col in df.columns:
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if len(vals) < 10:
            continue
        med = float(vals.median())
        mad = float((vals - med).abs().median()) or 1e-9
        for c in candidates:
            if c not in set(vals.unique()):
                continue
            count = int((vals == c).sum())
            rate = count / len(vals)
            if rate < 0.002 or count < 2:
                continue
            extreme = abs(c - med) > 10 * mad
            nonneg_exception = c == -1 and float(vals.min()) >= 0
            if extreme or nonneg_exception:
                out[str(col)] = {
                    "sentinel": c,
                    "count": count,
                    "rate": float(rate),
                    "action": "flagged",
                }
                break
    return out


def infer_column(
    s: pd.Series,
    name: str,
    *,
    max_categories: int = 50,
    treat_sentinels: bool = True,
    sentinel: dict[str, Any] | None = None,
) -> tuple[ColumnSpec, pd.Series]:
    s = s.copy()
    if sentinel and treat_sentinels:
        s = s.mask(pd.to_numeric(s, errors="coerce") == sentinel["sentinel"])
    n = len(s)
    non_null = s.dropna()
    semantic, sensitivity = infer_semantics(name)
    kind = "unknown"
    stats: dict[str, Any] = {}
    if pd.api.types.is_bool_dtype(s):
        kind = "boolean"
    elif pd.api.types.is_integer_dtype(s):
        kind = "integer"
        if len(non_null):
            stats = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "std": float(non_null.std(ddof=0)),
                "q25": float(non_null.quantile(0.25)),
                "q50": float(non_null.median()),
                "q75": float(non_null.quantile(0.75)),
                "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
            }
    elif pd.api.types.is_float_dtype(s):
        kind = "float"
        if len(non_null):
            stats = {
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "mean": float(non_null.mean()),
                "std": float(non_null.std(ddof=0)),
                "q01": float(non_null.quantile(0.01)),
                "q25": float(non_null.quantile(0.25)),
                "q50": float(non_null.median()),
                "q75": float(non_null.quantile(0.75)),
                "q99": float(non_null.quantile(0.99)),
                "skew": float(non_null.skew()) if len(non_null) > 2 else 0.0,
            }
    else:
        parsed = _try_datetime(s)
        if parsed is not None and (name.lower().endswith(("date", "_ts", "timestamp", "time")) or parsed.notna().mean() > 0.98):
            kind = "datetime"
            s = parsed
            non_null = s.dropna()
            if len(non_null):
                stats = {
                    "min": str(non_null.min()),
                    "max": str(non_null.max()),
                    "span_days": float((non_null.max() - non_null.min()).total_seconds() / 86400),
                }
        else:
            nun = int(non_null.nunique())
            if nun <= max_categories or (n and nun / max(len(non_null), 1) < 0.05):
                kind = "categorical"
                counts = non_null.astype(str).value_counts().head(20)
                stats = {
                    "top": {str(k): int(v) for k, v in counts.items()},
                    "unique": nun,
                    "entropy": float(_entropy(non_null.astype(str).value_counts(normalize=True).to_numpy())),
                }
            else:
                kind = "text"
    if kind == "text" or (kind == "categorical" and sensitivity == "unknown"):
        pii = detect_value_pii(non_null)
        if pii is not None:
            sensitivity = "direct_identifier"
            semantic = f"detected_{pii}"
    spec = ColumnSpec(
        name=str(name),
        kind=kind,  # type: ignore[arg-type]
        dtype=str(s.dtype),
        nullable=bool(s.isna().any()),
        cardinality=int(non_null.nunique()) if kind != "text" else int(non_null.nunique()),
        missing_rate=float(s.isna().mean()) if n else 0.0,
        sensitivity=sensitivity,
        semantic=semantic,
        stats=stats,
    )
    return spec, s


def _entropy(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def infer_table_schema(
    df: pd.DataFrame,
    name: str,
    *,
    primary_key: list[str] | None = None,
    foreign_keys: list[ForeignKey] | None = None,
    target_column: str | None = None,
    max_categories: int = 50,
    treat_sentinels: bool = True,
    drop_direct_identifiers: bool = True,
    sentinel_candidates: list[float] | None = None,
) -> tuple[TableSchema, pd.DataFrame, dict[str, Any]]:
    """Infer a schema and return a cleaned (sentinels → NaN) copy plus a report."""
    frame = df.copy()
    frame.columns = [str(c).strip() for c in frame.columns]
    if frame.columns.duplicated().any():
        dupes = frame.columns[frame.columns.duplicated()].tolist()
        raise ValueError(f"duplicate column names after trimming: {dupes}; rename them before ingestion")
    detected = detect_sentinels(frame, sentinel_candidates)
    specs: list[ColumnSpec] = []
    cleaned = pd.DataFrame(index=frame.index)
    for col in frame.columns:
        spec, s = infer_column(
            frame[col], col, max_categories=max_categories, treat_sentinels=treat_sentinels,
            sentinel=detected.get(col),
        )
        if detected.get(col):
            spec.stats["sentinel"] = detected[col]
            spec.stats["sentinel_action"] = "replaced_with_null" if treat_sentinels else "flagged_only"
        specs.append(spec)
        cleaned[col] = s

    # primary key
    explicit_pk = list(primary_key or [])
    if explicit_pk:
        missing = [c for c in explicit_pk if c not in cleaned.columns]
        if missing:
            raise ValueError(f"primary key columns not found: {missing}")
        for spec in specs:
            if spec.name in explicit_pk:
                spec.role = "primary_key"
                spec.dropped = True
    else:
        candidates = []
        for spec in specs:
            s = cleaned[spec.name]
            if s.notna().all() and len(s) and s.nunique() == len(s):
                n = spec.name.lower()
                if n.endswith("_id") or n == "id" or n.endswith("_key") or n.endswith("_ref"):
                    candidates.append(spec.name)
        if candidates:
            explicit_pk = [candidates[0]]
            for spec in specs:
                if spec.name == explicit_pk[0]:
                    spec.role = "primary_key"
                    spec.dropped = True

    # roles
    for spec in specs:
        if spec.name in explicit_pk:
            continue
        if target_column and spec.name == target_column:
            spec.role = "target"
        elif spec.sensitivity == "direct_identifier":
            spec.role = "ignore"
            spec.dropped = drop_direct_identifiers
        elif foreign_keys and any(spec.name in fk.columns for fk in foreign_keys):
            spec.role = "foreign_key"
            spec.dropped = True
        elif spec.kind == "text":
            spec.role = "ignore"
            spec.dropped = True
        elif spec.kind == "categorical" and spec.cardinality and spec.cardinality == len(cleaned) and len(cleaned) > 50:
            # high-cardinality free text that looks like an identifier is not modelled
            spec.role = "ignore"
            spec.dropped = True
    # target should not be dropped even if name heuristic says sensitive
    if target_column:
        for spec in specs:
            if spec.name == target_column:
                spec.role = "target"
                spec.dropped = False

    constraints: list[Any] = []
    if explicit_pk:
        constraints.append(UniqueConstraint(columns=explicit_pk, description="primary key uniqueness"))
        constraints.append(NotNullConstraint(column=explicit_pk[0], description="primary key not null"))
    # obvious temporal ordering
    lower = {c.lower() for c in cleaned.columns}
    for a, b in [
        ("start_date", "end_date"), ("start", "end"), ("admit_date", "discharge_date"),
        ("opened_date", "closed_date"), ("created_at", "updated_at"),
    ]:
        if a in lower and b in lower and pd.api.types.is_datetime64_any_dtype(cleaned[a]):
            constraints.append(InequalityConstraint(left=a, op="<=", right=b, description=f"{b} after {a}"))

    schema = TableSchema(
        name=name,
        columns=specs,
        primary_key=explicit_pk,
        foreign_keys=list(foreign_keys or []),
        constraints=constraints,
        row_count=int(len(frame)),
        notes=[],
    )
    report = {
        "sentinels": detected,
        "dropped_columns": [s.name for s in specs if s.dropped],
        "inferred_primary_key": explicit_pk,
        "warnings": [],
    }
    if detected:
        report["warnings"].append(
            f"Detected sentinel values in {', '.join(detected)}; they were treated as missing."
            if treat_sentinels
            else f"Detected sentinel values in {', '.join(detected)}; flagged but left unchanged."
        )
    if report["dropped_columns"]:
        report["warnings"].append(
            "Columns not modelled: "
            + ", ".join(report["dropped_columns"])
            + " (direct identifier, free text, foreign key, or unique ID)."
        )
    return schema, cleaned, report


def apply_user_schema_overrides(
    schema: TableSchema,
    overrides: dict[str, Any],
) -> TableSchema:
    """Apply validated API edits: roles, sensitivity, target, constraints, dropped flags."""
    new = schema.model_copy(deep=True)
    cols = overrides.get("columns", {})
    for name, patch in cols.items():
        try:
            spec = new.column(name)
        except KeyError as exc:
            raise ValueError(f"unknown column in schema override: {name}") from exc
        for key in ("role", "sensitivity", "semantic", "dropped", "kind"):
            if key in patch:
                setattr(spec, key, patch[key])
    if "primary_key" in overrides:
        pk = list(overrides["primary_key"])
        for c in pk:
            new.column(c)
        new.primary_key = pk
        for c in new.columns:
            if c.name in pk:
                c.role = "primary_key"
                c.dropped = True
    if "constraints" in overrides:
        # use discriminated union through TableSchema itself for validation
        tmp = new.model_copy(deep=True)
        tmp.constraints = overrides["constraints"]
        new.constraints = TableSchema.model_validate(tmp.model_dump()).constraints
    if "foreign_keys" in overrides:
        new.foreign_keys = [ForeignKey.model_validate(fk) for fk in overrides["foreign_keys"]]
    return new


def validate_frame_against_schema(df: pd.DataFrame, schema: TableSchema) -> list[str]:
    problems = []
    missing = [c.name for c in schema.columns if c.name not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    unknown = [c for c in df.columns if c not in schema.column_names]
    if unknown:
        problems.append(f"unexpected columns: {unknown}")
    if schema.primary_key and any(c not in df.columns for c in schema.primary_key):
        problems.append("primary key missing from frame")
    return problems


def dataframe_to_schema_json(schema: TableSchema) -> dict[str, Any]:
    return schema.model_dump(mode="json")


def summarize_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "memory_bytes": int(df.memory_usage(deep=True).sum()),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
        }
