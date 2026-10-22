"""k-anonymity and re-identification screening helpers.

These are data-minimisation diagnostics, not privacy proofs. They complement the DP accountant with
empirical, attack-oriented evidence.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def k_anonymity(
    frame: pd.DataFrame,
    quasi_identifiers: list[str],
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    cols = [c for c in quasi_identifiers if c in frame.columns]
    if not cols:
        return {"error": "no usable quasi-identifier columns", "k": 0, "groups": 0}
    df = frame[cols].astype("object").where(frame[cols].notna(), "<MISSING>").astype(str)
    if max_rows:
        df = df.head(max_rows)
    sizes = df.groupby(cols, dropna=False).size()
    return {
        "quasi_identifiers": cols,
        "groups": int(len(sizes)),
        "rows": int(len(df)),
        "k": int(sizes.min()) if len(sizes) else 0,
        "k_median": float(sizes.median()) if len(sizes) else 0.0,
        "unique_groups": int((sizes == 1).sum()),
        "unique_fraction": float((sizes == 1).sum() / max(len(sizes), 1)),
        "at_risk_fraction": float((sizes < 5).sum() / max(len(sizes), 1)),
    }


def equivalence_class_report(
    frame: pd.DataFrame,
    quasi_identifiers: list[str],
    *,
    sensitive: str | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    cols = [c for c in quasi_identifiers if c in frame.columns]
    if not cols:
        return {"error": "no usable quasi-identifier columns"}
    df = frame if max_rows is None else frame.head(max_rows)
    grouped = df.groupby(cols, dropna=False)
    sizes = grouped.size()
    out: dict[str, Any] = {
        "groups": int(len(sizes)),
        "smallest": int(sizes.min()) if len(sizes) else 0,
        "largest": int(sizes.max()) if len(sizes) else 0,
        "median": float(sizes.median()) if len(sizes) else 0.0,
    }
    if sensitive and sensitive in df.columns:
        nunique = grouped[sensitive].nunique()
        out["sensitive_attributes"] = {
            "column": sensitive,
            "groups_with_multiple_values": int((nunique > 1).sum()),
            "worst_case_distinct": int(nunique.max()) if len(nunique) else 0,
        }
        if len(nunique):
            out["sensitive_attributes"]["l_diversity_min"] = int(nunique.min())
    return out


__all__ = ["k_anonymity", "equivalence_class_report"]
