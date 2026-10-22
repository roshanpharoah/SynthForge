"""Scorecard aggregation: combine fidelity, utility, validity and privacy evidence into a graded card.

Grades are computed from explicit thresholds so nothing is a hardcoded marketing number. Every entry
carries the raw metric that produced it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_THRESHOLDS = {
    "mean_ks_excellent": 0.05,
    "mean_ks_good": 0.10,
    "mean_ks_fair": 0.20,
    "mean_tvd_excellent": 0.05,
    "mean_tvd_good": 0.10,
    "mean_tvd_fair": 0.20,
    "discriminator_auc_excellent": 0.60,
    "discriminator_auc_good": 0.75,
    "discriminator_auc_fair": 0.90,
    "accuracy_ratio_good": 0.95,
    "accuracy_ratio_fair": 0.85,
    "constraint_violation_good": 0.001,
    "constraint_violation_fair": 0.01,
    "duplicate_rate_fair": 0.01,
    "attack_advantage_good": 0.05,
    "attack_advantage_fair": 0.15,
}


def _grade(value: float, thresholds: tuple[float, float, float], *, lower_better: bool = True) -> str:
    e, g, f = thresholds
    if not np.isfinite(value):
        return "unknown"
    if lower_better:
        if value <= e:
            return "excellent"
        if value <= g:
            return "good"
        if value <= f:
            return "fair"
        return "poor"
    if value >= e:
        return "excellent"
    if value >= g:
        return "good"
    if value >= f:
        return "fair"
    return "poor"


def build_scorecard(
    fidelity: dict[str, Any],
    utility: dict[str, Any] | None = None,
    validity: dict[str, Any] | None = None,
    attacks: dict[str, Any] | None = None,
    *,
    privacy: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    t = dict(DEFAULT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    fidelity_summary = fidelity.get("columns", {}).get("summary", {}) if fidelity else {}
    mean_ks = float(fidelity_summary.get("mean_ks", float("nan")))
    mean_tvd = float(fidelity_summary.get("mean_tvd", float("nan")))
    disc = fidelity.get("discriminator", {}) if fidelity else {}
    disc_auc = float(disc.get("auc", float("nan")))
    sw = float(fidelity.get("sliced_wasserstein", float("nan"))) if fidelity else float("nan")
    corr = fidelity.get("correlation", {}) if fidelity else {}
    corr_diff = float(corr.get("mean_abs_diff", float("nan")))
    candidates = {
        "univariate_ks": {
            "value": mean_ks,
            "grade": _grade(mean_ks, (t["mean_ks_excellent"], t["mean_ks_good"], t["mean_ks_fair"])),
            "detail": fidelity_summary,
            "direction": "lower_better",
        },
        "categorical_tvd": {
            "value": mean_tvd,
            "grade": _grade(mean_tvd, (t["mean_tvd_excellent"], t["mean_tvd_good"], t["mean_tvd_fair"])),
            "direction": "lower_better",
        },
        "discriminator_auc": {
            "value": disc_auc,
            "grade": _grade(disc_auc, (t["discriminator_auc_excellent"], t["discriminator_auc_good"], t["discriminator_auc_fair"])),
            "direction": "lower_better",
            "detail": {k: v for k, v in disc.items() if k != "auc"},
        },
        "sliced_wasserstein": {"value": sw, "grade": _grade(sw, (0.05, 0.15, 0.35)), "direction": "lower_better"},
        "correlation_difference": {"value": corr_diff, "grade": _grade(corr_diff, (0.05, 0.12, 0.25)), "direction": "lower_better"},
    }
    if utility:
        ratio = float(utility.get("accuracy_ratio", utility.get("mean_accuracy_ratio", float("nan")))) if utility else float("nan")
        if not np.isfinite(ratio) and utility.get("classifiers"):
            vals = [v.get("utility_ratio") for v in utility["classifiers"].values() if isinstance(v, dict) and v.get("utility_ratio")]
            ratio = float(np.mean(vals)) if vals else float("nan")
        if not np.isfinite(ratio):
            mt, mr = float(utility.get("mean_tstr_accuracy", float("nan"))), float(utility.get("mean_trtr_accuracy", float("nan")))
            if np.isfinite(mt) and np.isfinite(mr) and mr > 0:
                ratio = mt / mr
        gap = utility.get("mean_accuracy_gap", float("nan"))
        candidates["tstr_utility_ratio"] = {
            "value": ratio,
            "grade": _grade(ratio, (0.98, t["accuracy_ratio_good"], t["accuracy_ratio_fair"]), lower_better=False),
            "direction": "higher_better",
            "accuracy_gap": gap,
        }
    if validity:
        cviol = validity.get("constraints", {}).get("violation_rate", float("nan"))
        dup = validity.get("duplicates", {}).get("duplicate_rate", float("nan"))
        schema_ok = validity.get("schema", {}).get("ok", None)
        candidates["constraint_violation_rate"] = {
            "value": cviol,
            "grade": _grade(cviol, (0.0, t["constraint_violation_good"], t["constraint_violation_fair"])),
            "direction": "lower_better",
        }
        candidates["duplicate_rate"] = {
            "value": dup,
            "grade": _grade(dup, (0.0, t["duplicate_rate_fair"], 0.05)),
            "direction": "lower_better",
        }
        candidates["schema_conformance"] = {
            "value": 1.0 if schema_ok else 0.0,
            "grade": "excellent" if schema_ok else "poor",
            "direction": "higher_better",
        }
    if attacks:
        adv = float(attacks.get("membership_inference", {}).get("advantage", attacks.get("advantage", float("nan"))))
        candidates["membership_attack_advantage"] = {
            "value": adv,
            "grade": _grade(adv, (0.0, t["attack_advantage_good"], t["attack_advantage_fair"])),
            "direction": "lower_better",
            "disclaimer": "Empirical evidence only; not a privacy guarantee.",
        }
        ai = attacks.get("attribute_inference") or {}
        if ai and "advantage" in ai:
            candidates["attribute_inference_advantage"] = {
                "value": float(ai["advantage"]),
                "grade": _grade(float(ai["advantage"]), (0.0, 0.05, 0.15)),
                "direction": "lower_better",
                "disclaimer": "Empirical evidence only; not a privacy guarantee.",
            }
    grades = [c["grade"] for c in candidates.values() if c["grade"] != "unknown"]
    rank = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
    overall_score = float(np.mean([rank[g] for g in grades])) if grades else float("nan")
    overall = "unknown"
    if grades:
        overall = min(grades, key=lambda g: rank[g])
        if overall_score >= 3.5:
            overall = "excellent"
        elif overall_score >= 2.5:
            overall = "good" if overall not in ("poor", "fair") else overall
    return {
        "metrics": candidates,
        "overall": overall,
        "overall_score": overall_score,
        "worst_component": min(candidates, key=lambda k: rank.get(candidates[k]["grade"], 5)) if candidates else None,
        "thresholds": t,
        "privacy": privacy or {},
        "notes": [
            "Grades are computed from measured metrics with explicit thresholds; they are not claims of fitness.",
            "Privacy grades reflect empirical attack evidence and the stated DP scope, never a guarantee of anonymity.",
        ],
    }


__all__ = ["build_scorecard", "DEFAULT_THRESHOLDS"]
