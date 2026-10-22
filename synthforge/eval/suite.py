"""Evaluation orchestration: run every applicable check for a modality and assemble the report."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.config import EvalConfig
from synthforge.core.types import Dataset, TableSchema
from synthforge.eval import fidelity as fidelity_mod
from synthforge.eval import medical as medical_mod
from synthforge.eval import utility as utility_mod
from synthforge.eval import validity as validity_mod
from synthforge.eval.report import datasheet_markdown, model_card_markdown, write_report
from synthforge.eval.scorecard import build_scorecard
from synthforge.privacy.attacks import run_attack_suite


def _schema_for(dataset: Dataset, table: str | None) -> tuple[str, TableSchema] | None:
    if not dataset.tables:
        return None
    name = table or next(iter(dataset.tables))
    return name, dataset.tables[name].schema


def quantify_reid_risk(real: pd.DataFrame, synthetic: pd.DataFrame, quasi_identifiers: list[str], *, max_rows: int = 2000) -> dict[str, Any]:
    """Risk screening: exact and near-duplicate overlap between real and synthetic rows on QIs."""
    qi = [c for c in quasi_identifiers if c in real.columns and c in synthetic.columns]
    out: dict[str, Any] = {"quasi_identifiers": qi, "basis": "quasi-identifier overlap screening"}
    if not qi:
        return {**out, "note": "no shared quasi-identifier columns"}
    r = real[qi].head(max_rows).astype("object").where(real[qi].head(max_rows).notna(), "<M>").astype(str)
    s = synthetic[qi].head(max_rows).astype("object").where(synthetic[qi].head(max_rows).notna(), "<M>").astype(str)
    r_join = r.agg("|".join, axis=1)
    s_join = s.agg("|".join, axis=1)
    real_keys = set(r_join)
    collisions = s_join.isin(real_keys)
    out.update(
        {
            "synthetic_rows_matching_real": int(collisions.sum()),
            "match_rate": float(collisions.mean()) if len(s_join) else 0.0,
            "real_unique_keys": int(len(real_keys)),
            "synthetic_unique_keys": int(s_join.nunique()),
        }
    )
    summary = {}
    for c in qi:
        summary[c] = {
            "real_unique": int(r[c].nunique()),
            "synthetic_unique": int(s[c].nunique()),
            "overlap": int(len(set(r[c]) & set(s[c]))),
        }
    out["per_column"] = summary
    return out


def evaluate_dataset(
    real: Dataset,
    synthetic: Dataset,
    *,
    config: EvalConfig | None = None,
    table: str | None = None,
    schema: TableSchema | None = None,
    privacy: dict[str, Any] | None = None,
    seed: int = 42,
    attacks: bool = True,
) -> dict[str, Any]:
    """Full evaluation for tabular/relational modalities."""
    cfg = config or EvalConfig()
    out: dict[str, Any] = {"modality": real.kind, "tables": {}}
    if real.tables and synthetic.tables:
        names = [n for n in real.tables if n in synthetic.tables]
        if table:
            names = [n for n in names if n == table]
        for name in names:
            schema_t = schema or real.tables[name].schema
            real_df = real.tables[name].frame
            syn_df = synthetic.tables[name].frame
            entry: dict[str, Any] = {
                "rows": {"real": int(len(real_df)), "synthetic": int(len(syn_df))},
                "fidelity": fidelity_mod.fidelity_report(real_df, syn_df, schema_t, seed=seed),
                "validity": validity_mod.validity_report(syn_df, schema_t),
            }
            qi = [c.name for c in schema_t.columns if c.sensitivity == "quasi_identifier"]
            if qi:
                entry["reidentification_screening"] = quantify_reid_risk(real_df, syn_df, qi, max_rows=cfg.attack_max_rows)
            target = cfg.target_column or (schema_t.target_columns[0] if schema_t.target_columns else None)
            if target and target in real_df.columns and target in syn_df.columns:
                if cfg.seeds > 1:
                    entry["utility"] = utility_mod.multi_seed_utility(
                        real_df, syn_df, schema_t, target, seeds=cfg.seeds, classifiers=cfg.classifiers, bootstrap=cfg.bootstrap
                    )
                else:
                    entry["utility"] = utility_mod.tstr_trtr(
                        real_df, syn_df, schema_t, target, classifiers=cfg.classifiers, bootstrap=cfg.bootstrap
                    )
                entry["rare_classes"] = utility_mod.rare_class_coverage(real_df, syn_df, schema_t)
            if attacks:
                sensitive = cfg.thresholds.get("sensitive")
                sensitive_col = None
                if isinstance(sensitive, str) and sensitive in real_df.columns:
                    sensitive_col = sensitive
                else:
                    for c in schema_t.columns:
                        if c.sensitivity == "sensitive" and c.name in real_df.columns:
                            sensitive_col = c.name
                            break
                entry["attacks"] = run_attack_suite(
                    real_df, syn_df, schema_t, sensitive=sensitive_col, seed=seed, max_rows=cfg.attack_max_rows
                )
                if entry.get("utility"):
                    entry["utility"]["accuracy_ratio"] = _ratio_from_utility(entry["utility"])
            entry["scorecard"] = build_scorecard(
                entry["fidelity"],
                entry.get("utility"),
                entry["validity"],
                entry.get("attacks"),
                privacy=privacy,
                thresholds=cfg.thresholds,
            )
            out["tables"][name] = entry
    if real.kind in ("timeseries", "multimodal") and real.signals is not None and synthetic.signals is not None:
        out["signals"] = medical_mod.ecg_physiology(real.signals, synthetic.signals)
    if real.kind in ("imaging", "multimodal") and real.images is not None and synthetic.images is not None:
        out["imaging"] = medical_mod.imaging_report(real.images, synthetic.images)
    out["privacy"] = privacy or {}
    out["disclaimer"] = (
        "Synthetic data only. Metrics describe statistical resemblance and empirical leakage evidence; "
        "they are not clinical, legal or regulatory validation."
    )
    return out


def _ratio_from_utility(utility: dict[str, Any]) -> float:
    if "accuracy_ratio" in utility and np.isfinite(utility.get("accuracy_ratio", float("nan"))):
        return float(utility["accuracy_ratio"])
    per = [v.get("utility_ratio") for v in utility.get("classifiers", {}).values() if isinstance(v, dict) and v.get("utility_ratio")]
    if per:
        return float(np.mean([float(p) for p in per if p is not None]))
    mt = utility.get("mean_tstr_accuracy")
    mr = utility.get("mean_trtr_accuracy")
    if mt and mr and np.isfinite(mt) and np.isfinite(mr) and mr > 0:
        return float(mt / mr)
    # single-split shape: mean over classifiers of tstr/trtr accuracy
    ratios: list[float] = []
    for v in utility.get("classifiers", {}).values():
        tstr = (v.get("tstr") or {}).get("accuracy")
        trtr = (v.get("trtr") or {}).get("accuracy")
        if tstr and trtr:
            ratios.append(tstr / trtr)
    return float(np.mean(ratios)) if ratios else float("nan")


def summarize_comparison(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Compare several models' scorecards on the same dataset."""
    rows = []
    for model, res in results.items():
        entry = next(iter(res.get("tables", {}).values()), {}) if res.get("tables") else {}
        sc = entry.get("scorecard", {})
        rows.append(
            {
                "model": model,
                "overall": sc.get("overall"),
                "overall_score": sc.get("overall_score"),
                "mean_ks": (entry.get("fidelity", {}).get("columns", {}).get("summary", {}) or {}).get("mean_ks"),
                "discriminator_auc": (entry.get("fidelity", {}).get("discriminator", {}) or {}).get("auc"),
                "constraint_violation_rate": (entry.get("validity", {}).get("constraints", {}) or {}).get("violation_rate"),
                "membership_advantage": ((entry.get("attacks", {}) or {}).get("membership_inference", {}) or {}).get("advantage"),
                "utility_ratio": _ratio_from_utility(entry.get("utility", {}) or {}),
            }
        )
    ranked = sorted(rows, key=lambda r: (-(r["overall_score"] or 0), r["model"]))
    return {"models": rows, "ranking": [r["model"] for r in ranked]}


__all__ = ["evaluate_dataset", "summarize_comparison", "write_report", "model_card_markdown", "datasheet_markdown"]
