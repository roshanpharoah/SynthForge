"""Validity checks: constraints, ranges, sentinels, schema conformance, duplicates."""

from __future__ import annotations

from typing import Any

import pandas as pd

from synthforge.core.types import TableSchema
from synthforge.metadata.constraints import validate_table


def constraint_validity(frame: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    report = validate_table(frame, schema.constraints)
    return report


def schema_conformance(frame: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    expected = [c.name for c in schema.columns if not c.dropped]
    present = [c for c in expected if c in frame.columns]
    missing = [c for c in expected if c not in frame.columns]
    extra = [c for c in frame.columns if c not in expected]
    type_issues = []
    for c in schema.columns:
        if c.name not in frame.columns:
            continue
        s = frame[c.name]
        if c.kind in ("integer", "float") and not pd.api.types.is_numeric_dtype(s):
            coerced = pd.to_numeric(s, errors="coerce")
            bad = int(coerced.isna().sum() - s.isna().sum())
            if bad > 0:
                type_issues.append({"column": c.name, "expected": c.kind, "uncoercible": bad})
    return {
        "expected_columns": expected,
        "present": present,
        "missing": missing,
        "extra": extra,
        "type_issues": type_issues,
        "ok": not missing and not type_issues,
    }


def range_checks(frame: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in schema.columns:
        if c.name not in frame.columns or c.kind not in ("integer", "float"):
            continue
        s = pd.to_numeric(frame[c.name], errors="coerce").dropna()
        if s.empty:
            out[c.name] = {"empty": True}
            continue
        rmin, rmax = c.stats.get("min"), c.stats.get("max")
        entry = {"synthetic_min": float(s.min()), "synthetic_max": float(s.max())}
        if rmin is not None and rmax is not None and rmax > rmin:
            span = float(rmax) - float(rmin)
            below = float(((s < float(rmin) - 0.5 * span)).mean())
            above = float(((s > float(rmax) + 0.5 * span)).mean())
            entry.update(
                {
                    "real_min": float(rmin),
                    "real_max": float(rmax),
                    "fraction_far_below": below,
                    "fraction_far_above": above,
                    "ok": bool(below + above < 0.01),
                }
            )
        out[c.name] = entry
    return out


def sentinel_and_missing(frame: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    out: dict[str, Any] = {"sentinel_columns": {}, "missing_rates": {}}
    for c in schema.columns:
        if c.name not in frame.columns:
            continue
        out["missing_rates"][c.name] = float(frame[c.name].isna().mean())
        sents = c.stats.get("sentinels") or []
        if sents:
            s = pd.to_numeric(frame[c.name], errors="coerce")
            counts = {str(v): int((s == v).sum()) for v in sents}
            out["sentinel_columns"][c.name] = counts
    return out


def duplicate_analysis(frame: pd.DataFrame, schema: TableSchema) -> dict[str, Any]:
    feature_cols = [c.name for c in schema.columns if c.name in frame.columns and not c.dropped and c.role not in ("primary_key",)]
    if not feature_cols:
        return {"duplicate_rows": 0, "duplicate_rate": 0.0}
    dup = int(frame.duplicated(subset=feature_cols, keep=False).sum())
    return {
        "duplicate_rows": dup,
        "duplicate_rate": float(dup / max(len(frame), 1)),
        "distinct_rows": int(len(frame.drop_duplicates(subset=feature_cols))),
    }


def plausibility_rules(frame: pd.DataFrame, rules: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply user-defined plausibility rules (column/op/value) and report violation rates."""
    out = []
    for rule in rules:
        col, op, val = rule.get("column"), rule.get("op", ">="), rule.get("value")
        if col not in frame.columns:
            out.append({"rule": rule, "error": "column missing"})
            continue
        s = pd.to_numeric(frame[col], errors="coerce")
        mask = pd.Series(True, index=frame.index)
        try:
            if op in (">=", ">", "<=", "<"):
                mask = {">=": s >= val, ">": s > val, "<=": s <= val, "<": s < val}[op]
            elif op == "not_null":
                mask = frame[col].notna()
            elif op == "in":
                mask = frame[col].isin(val)
            elif op == "non_negative":
                mask = s >= 0
        except Exception as exc:
            out.append({"rule": rule, "error": str(exc)})
            continue
        out.append({"rule": rule, "violations": int((~mask.fillna(False)).sum()), "violation_rate": float((~mask.fillna(False)).mean())})
    return {"rules": out, "violations": int(sum(r.get("violations", 0) for r in out))}


def validity_report(frame: pd.DataFrame, schema: TableSchema, *, rules: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "constraints": constraint_validity(frame, schema),
        "schema": schema_conformance(frame, schema),
        "ranges": range_checks(frame, schema),
        "sentinels_missing": sentinel_and_missing(frame, schema),
        "duplicates": duplicate_analysis(frame, schema),
        "plausibility": plausibility_rules(frame, rules or []),
    }


__all__ = [
    "validity_report",
    "constraint_validity",
    "schema_conformance",
    "range_checks",
    "sentinel_and_missing",
    "duplicate_analysis",
    "plausibility_rules",
]
