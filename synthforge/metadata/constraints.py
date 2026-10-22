"""Declarative constraints. All are serialisable, validatable and repairable."""

from __future__ import annotations

import operator
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class ConstraintBase(BaseModel):
    kind: str
    description: str | None = None


class RangeConstraint(ConstraintBase):
    kind: Literal["range"] = "range"
    column: str
    min: float | None = None
    max: float | None = None
    include_min: bool = True
    include_max: bool = True

    @model_validator(mode="after")
    def _sane(self) -> RangeConstraint:
        if self.min is None and self.max is None:
            raise ValueError("range constraint requires min and/or max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("range min exceeds max")
        return self


class AllowedValuesConstraint(ConstraintBase):
    kind: Literal["allowed_values"] = "allowed_values"
    column: str
    values: list[str | int | float | bool]


class NotNullConstraint(ConstraintBase):
    kind: Literal["not_null"] = "not_null"
    column: str


class UniqueConstraint(ConstraintBase):
    kind: Literal["unique"] = "unique"
    columns: list[str] = Field(min_length=1)


class InequalityConstraint(ConstraintBase):
    kind: Literal["inequality"] = "inequality"
    left: str
    op: Literal[">=", ">", "<=", "<", "==", "!="] = ">="
    right: str


class FunctionalDependencyConstraint(ConstraintBase):
    kind: Literal["functional_dependency"] = "functional_dependency"
    determinant: str
    dependent: str


class CategoricalImplicationConstraint(ConstraintBase):
    kind: Literal["categorical_implication"] = "categorical_implication"
    if_column: str
    if_value: str | int | float | bool
    then_column: str
    then_values: list[str | int | float | bool]


class ForeignKeyConstraint(ConstraintBase):
    kind: Literal["foreign_key"] = "foreign_key"
    columns: list[str] = Field(min_length=1)
    ref_table: str
    ref_columns: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _arity(self) -> ForeignKeyConstraint:
        if len(self.columns) != len(self.ref_columns):
            raise ValueError("foreign key column arity must match referenced columns")
        return self


Constraint = Annotated[
    RangeConstraint | AllowedValuesConstraint | NotNullConstraint | UniqueConstraint | InequalityConstraint | FunctionalDependencyConstraint | CategoricalImplicationConstraint | ForeignKeyConstraint,
    Field(discriminator="kind"),
]

COMPARATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(col)
    return df[col]


def _violation_mask(constraint: ConstraintBase, df: pd.DataFrame) -> pd.Series:
    """Boolean mask of rows that violate the constraint (FK handled separately)."""
    if isinstance(constraint, RangeConstraint):
        s = _as_series(df, constraint.column)
        mask = pd.Series(False, index=df.index)
        if constraint.min is not None:
            mask |= (s < constraint.min) if constraint.include_min else (s <= constraint.min)
        if constraint.max is not None:
            mask |= (s > constraint.max) if constraint.include_max else (s >= constraint.max)
        return mask.fillna(False)
    if isinstance(constraint, AllowedValuesConstraint):
        s = _as_series(df, constraint.column)
        return ~s.isin(constraint.values) & s.notna()
    if isinstance(constraint, NotNullConstraint):
        return _as_series(df, constraint.column).isna()
    if isinstance(constraint, UniqueConstraint):
        return df.duplicated(subset=constraint.columns, keep=False)
    if isinstance(constraint, InequalityConstraint):
        a, b = _as_series(df, constraint.left), _as_series(df, constraint.right)
        ok = COMPARATORS[constraint.op](a, b)
        return ~ok.fillna(False)
    if isinstance(constraint, FunctionalDependencyConstraint):
        det, dep = constraint.determinant, constraint.dependent
        grouped = df.groupby(det, dropna=False)[dep].nunique(dropna=False)
        bad_keys = set(grouped[grouped > 1].index.tolist())
        return df[det].isin(bad_keys)
    if isinstance(constraint, CategoricalImplicationConstraint):
        cond = df[constraint.if_column] == constraint.if_value
        bad = ~df[constraint.then_column].isin(constraint.then_values)
        return cond & bad
    if isinstance(constraint, ForeignKeyConstraint):
        # Note: single-table FK validation requires the referenced table and is done at Dataset level.
        s = df[constraint.columns[0]]
        return s.isna()
    return pd.Series(False, index=df.index)


def validate_table(df: pd.DataFrame, constraints: list[ConstraintBase]) -> dict:
    """Return a structured violation report. Never mutates the frame."""
    details = []
    total_violations = 0
    for c in constraints:
        if isinstance(c, ForeignKeyConstraint):
            continue
        mask = _violation_mask(c, df)
        n = int(mask.sum())
        total_violations += n
        details.append(
            {
                "constraint": c.model_dump(mode="json"),
                "violations": n,
                "rate": float(n / max(len(df), 1)),
                "rows": np.flatnonzero(mask.to_numpy())[:20].tolist(),
            }
        )
    return {
        "rows": int(len(df)),
        "constraints": len([c for c in constraints if not isinstance(c, ForeignKeyConstraint)]),
        "violations": total_violations,
        "violation_rate": float(total_violations / max(len(df), 1)),
        "details": details,
        "ok": total_violations == 0,
    }


def repair_table(
    df: pd.DataFrame,
    constraints: list[ConstraintBase],
    *,
    seed: int = 0,
    max_passes: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Deterministically repair generated rows, reporting exactly what was changed."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    changed_rows: set[int] = set()
    n_passes = 0
    for n_passes in range(1, max_passes + 1):  # noqa: B007 - n_passes reported below
        before = out.copy(deep=True)
        for c in constraints:
            if isinstance(c, ForeignKeyConstraint):
                continue
            out = _repair_one(out, c, rng)
        diff = (out.fillna("<NA>") != before.fillna("<NA>")).any(axis=1)
        changed_rows |= set(np.flatnonzero(diff.to_numpy()).tolist())
        if not diff.any():
            break
    return out, {
        "repaired_rows": len(changed_rows),
        "repair_rate": float(len(changed_rows) / max(len(df), 1)),
        "passes": n_passes,
    }


def _repair_one(df: pd.DataFrame, c: ConstraintBase, rng: np.random.Generator) -> pd.DataFrame:
    if isinstance(c, RangeConstraint):
        col = c.column
        if col not in df.columns:
            return df
        s = pd.to_numeric(df[col], errors="coerce")
        if c.min is not None:
            s = s.where(s >= c.min if c.include_min else s > c.min, c.min)
        if c.max is not None:
            s = s.where(s <= c.max if c.include_max else s < c.max, c.max)
        df[col] = s
        return df
    if isinstance(c, AllowedValuesConstraint):
        s = df[c.column]
        bad = ~s.isin(c.values) & s.notna()
        if bad.any():
            df.loc[bad, c.column] = rng.choice(c.values, size=int(bad.sum()))
        return df
    if isinstance(c, NotNullConstraint):
        col = c.column
        if col in df.columns and df[col].isna().any():
            mode = df[col].mode(dropna=True)
            fill = mode.iloc[0] if not mode.empty else (0 if pd.api.types.is_numeric_dtype(df[col]) else "UNKNOWN")
            df[col] = df[col].fillna(fill)
        return df
    if isinstance(c, UniqueConstraint):
        cols = c.columns
        if any(col not in df.columns for col in cols):
            return df
        dup = df.duplicated(subset=cols, keep="first")
        if not dup.any():
            return df
        for col in cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                nums = pd.to_numeric(df[col], errors="coerce")
                step = max(1.0, float(nums.std(ddof=0) or 1.0) * 0.001)
                df.loc[dup, col] = nums[dup] + rng.integers(1, 1000, size=int(dup.sum())) * step
            else:
                base = df.loc[dup, col].astype(str)
                df.loc[dup, col] = base + "_" + pd.Series(
                    [str(x) for x in rng.integers(1, 1_000_000, size=len(base))], index=base.index
                )
        return df
    if isinstance(c, InequalityConstraint):
        a, b = c.left, c.right
        if a not in df.columns or b not in df.columns:
            return df
        av, bv = pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")
        op = c.op
        if op in (">=", ">"):
            delta = 0.0 if op == ">=" else 1e-6
            bad = ~(av > bv + delta - 1e-12) if op == ">" else ~(av >= bv)
            df.loc[bad, b] = av[bad] - delta
        elif op in ("<=", "<"):
            delta = 0.0 if op == "<=" else 1e-6
            bad = ~(av < bv - delta + 1e-12) if op == "<" else ~(av <= bv)
            df.loc[bad, a] = bv[bad] - delta
        else:
            bad = ~COMPARATORS[op](av, bv)
            df.loc[bad, b] = av[bad]
        return df
    if isinstance(c, FunctionalDependencyConstraint):
        det, dep = c.determinant, c.dependent
        if det not in df.columns or dep not in df.columns:
            return df
        for _, idx in df.groupby(det, dropna=False).groups.items():
            vals = df.loc[idx, dep]
            mode = vals.mode(dropna=True)
            if not mode.empty and vals.nunique(dropna=False) > 1:
                df.loc[idx, dep] = mode.iloc[0]
        return df
    if isinstance(c, CategoricalImplicationConstraint):
        cond = df[c.if_column] == c.if_value
        bad = cond & ~df[c.then_column].isin(c.then_values)
        if bad.any():
            df.loc[bad, c.then_column] = rng.choice(c.then_values, size=int(bad.sum()))
        return df
    return df
