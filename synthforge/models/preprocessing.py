"""Reversible mixed-type preprocessing with mode-specific normalization.

Non-private mode uses per-column variational-style Gaussian mixtures (selected by BIC).
DP mode uses exponential-mechanism bounds and Laplace-noised clipped moments, with no
data-derived mixture parameters. Category vocabularies are treated as schema metadata; this
assumption is stated in every privacy model card.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from synthforge.core.errors import ModelError
from synthforge.core.types import ColumnSpec, TableSchema
from synthforge.privacy.dp_stats import DPStats


@dataclass
class CategoricalGroup:
    name: str
    start: int
    size: int
    categories: list[str]
    frequencies: np.ndarray
    original_values: list[Any]
    is_boolean: bool = False


@dataclass
class NumericTransform:
    name: str
    kind: str
    bounds: tuple[float, float]
    mean: float
    std: float
    nullable: bool = False
    missing_indicator: bool = False
    component_means: list[float] = field(default_factory=list)
    component_stds: list[float] = field(default_factory=list)
    component_weights: list[float] = field(default_factory=list)
    dim: int = 1
    start: int = 0
    dp: bool = False

    @property
    def n_components(self) -> int:
        return len(self.component_means) if self.component_means else 1

    def encode(self, s: pd.Series) -> np.ndarray:
        x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        null = ~np.isfinite(x)
        out = []
        if self.missing_indicator:
            out.append(null.astype(np.float32)[:, None])
        if self.n_components > 1:
            comp = np.zeros((len(x), self.n_components), dtype=np.float32)
            for k in range(self.n_components):
                z = (x - self.component_means[k]) / self.component_stds[k]
                logp = -0.5 * z**2 - np.log(self.component_stds[k])
                comp[:, k] = logp
            comp = comp - comp.max(axis=1, keepdims=True)
            comp = np.exp(comp)
            comp /= np.maximum(comp.sum(axis=1, keepdims=True), 1e-12)
            comp[null] = 0.0
            comp[null, 0] = 1.0
            out.append(comp)
        comp_idx = np.zeros(len(x), dtype=int)
        if self.n_components > 1:
            comp_idx = np.argmax(out[-1], axis=1)
        means = np.asarray(self.component_means or [self.mean])[comp_idx]
        stds = np.asarray(self.component_stds or [self.std])[comp_idx]
        z = (x - means) / np.maximum(stds, 1e-9)
        z = np.clip(z, -10.0, 10.0) if self.dp else z
        z[null] = 0.0
        out.append(z.astype(np.float32)[:, None])
        return np.concatenate(out, axis=1)

    def decode(self, x: np.ndarray) -> pd.Series:
        pos = 0
        null = np.zeros(len(x), dtype=bool)
        if self.missing_indicator:
            null = x[:, pos] > 0.5
            pos += 1
        if self.n_components > 1:
            comp = np.argmax(x[:, pos : pos + self.n_components], axis=1)
            pos += self.n_components
        else:
            comp = np.zeros(len(x), dtype=int)
        means = np.asarray(self.component_means or [self.mean])[comp]
        stds = np.asarray(self.component_stds or [self.std])[comp]
        z = x[:, pos]
        vals = np.clip(z * stds + means, self.bounds[0], self.bounds[1])
        if self.kind == "integer":
            vals = np.round(vals)
        if self.kind == "datetime":
            vals = pd.to_datetime(vals, unit="s", errors="coerce")
        out = pd.Series(vals)
        out[null] = np.nan
        return out


@dataclass
class CategoricalTransform:
    name: str
    categories: list[str]
    original_values: list[Any]
    frequencies: np.ndarray
    is_boolean: bool = False
    dim: int = 1
    start: int = 0

    def encode(self, s: pd.Series) -> np.ndarray:
        str_vals = s.astype("object").where(s.notna(), "<MISSING>").astype(str)
        idx = pd.Categorical(str_vals, categories=self.categories).codes
        unknown = idx < 0
        if unknown.any():
            miss = self.categories.index("<MISSING>") if "<MISSING>" in self.categories else 0
            idx[unknown] = miss
        out = np.zeros((len(s), len(self.categories)), dtype=np.float32)
        out[np.arange(len(s)), idx] = 1.0
        return out

    def decode(self, x: np.ndarray) -> pd.Series:
        idx = np.argmax(x[:, : self.dim], axis=1)
        vals = [self.original_values[i] for i in idx]
        return pd.Series(vals, dtype="object")

    def condition_index(self, value: Any) -> int:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            if "<MISSING>" not in self.categories:
                raise ModelError(f"column {self.name!r} has no missing category")
            return self.categories.index("<MISSING>")
        s = str(value)
        if s in self.categories:
            return self.categories.index(s)
        raise ModelError(
            f"condition {self.name}={value!r} is not a known category",
            hint=f"Use one of: {', '.join(self.categories[:25])}",
        )


class TabularPreprocessor:
    def __init__(self) -> None:
        self.transforms: list[NumericTransform | CategoricalTransform] = []
        self.feature_names: list[str] = []
        self.schema: TableSchema | None = None
        self.dp_details: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.fitted = False

    @property
    def dim(self) -> int:
        return sum(t.dim for t in self.transforms)

    @property
    def numeric(self) -> list[NumericTransform]:
        return [t for t in self.transforms if isinstance(t, NumericTransform)]

    @property
    def categorical(self) -> list[CategoricalTransform]:
        return [t for t in self.transforms if isinstance(t, CategoricalTransform)]

    @property
    def condition_groups(self) -> list[CategoricalGroup]:
        return [
            CategoricalGroup(t.name, t.start, t.dim, t.categories, t.frequencies, t.original_values, t.is_boolean)
            for t in self.categorical
        ]

    @property
    def condition_dim(self) -> int:
        return sum(t.dim for t in self.categorical)

    def fit(
        self,
        frame: pd.DataFrame,
        schema: TableSchema,
        *,
        seed: int = 42,
        dp_epsilon: float = 0.0,
        dp_mode: bool = False,
        max_components: int = 5,
    ) -> TabularPreprocessor:
        self.schema = schema
        self.feature_names = []
        self.transforms = []
        self.warnings = []
        modeled = [
            c
            for c in schema.columns
            if c.name in frame.columns and (not c.dropped) and c.role not in ("primary_key", "foreign_key", "ignore")
        ]
        if not modeled:
            raise ModelError(
                "no modelable columns remain after excluding identifiers and free text",
                hint="Tag at least one column as feature/target in the schema editor.",
            )
        numeric_cols = [c for c in modeled if c.kind in ("integer", "float", "datetime")]
        n_numeric = max(len(numeric_cols), 1)
        per_col_eps = (dp_epsilon / n_numeric) if (dp_mode and dp_epsilon > 0) else 0.0
        dp_stats = DPStats(epsilon=dp_epsilon, seed=seed) if dp_mode else None
        start = 0
        for spec in modeled:
            if spec.kind in ("integer", "float", "datetime"):
                values = self._numeric_values(frame[spec.name], spec.kind)
                finite = values[np.isfinite(values)]
                if finite.size == 0:
                    self.warnings.append(f"column {spec.name!r} has no finite values; excluded")
                    continue
                if dp_mode:
                    bounds = dp_stats.bounds(finite, per_col_eps * 0.5) if dp_stats else (float(np.min(finite)), float(np.max(finite)))
                    mean, std = dp_stats.clipped_mean_std(finite, bounds, per_col_eps * 0.5) if dp_stats else (float(np.mean(finite)), float(np.std(finite) or 1.0))
                    means, stds, weights = [mean], [max(std, (bounds[1] - bounds[0]) * 1e-3, 1e-9)], [1.0]
                else:
                    bounds = (float(np.min(finite)), float(np.max(finite)))
                    means, stds, weights = self._fit_vgm(finite, seed, max_components)
                    mean, std = means[0], stds[0]
                missing = frame[spec.name].isna().any()
                t = NumericTransform(
                    name=spec.name,
                    kind=spec.kind,
                    bounds=bounds,
                    mean=mean,
                    std=max(std, 1e-9),
                    nullable=bool(missing or spec.nullable),
                    missing_indicator=bool(missing),
                    component_means=means,
                    component_stds=stds,
                    component_weights=weights,
                    dim=(1 if missing else 0) + (len(means) if len(means) > 1 else 0) + 1,
                    start=start,
                    dp=dp_mode,
                )
            else:
                s = frame[spec.name]
                str_vals = s.astype("object").where(s.notna(), "<MISSING>").astype(str)
                counts = str_vals.value_counts(dropna=False)
                if dp_mode:
                    frequencies = np.ones(len(counts), dtype=float) / max(len(counts), 1)
                else:
                    frequencies = (counts / counts.sum()).to_numpy(dtype=float)
                original = {}
                for raw in s.dropna().unique():
                    original[str(raw)] = raw
                categories = list(counts.index)
                original_values = [None if c == "<MISSING>" else original.get(c, c) for c in categories]
                t = CategoricalTransform(
                    name=spec.name,
                    categories=categories,
                    original_values=original_values,
                    frequencies=frequencies,
                    is_boolean=spec.kind == "boolean",
                    dim=len(categories),
                    start=start,
                )
            self.transforms.append(t)
            self.feature_names.extend(self._feature_names_for(t))
            start += t.dim
        if not self.transforms:
            raise ModelError("preprocessing produced no features")
        self.fitted = True
        self.dp_details = (
            dp_stats.to_dict()
            if dp_stats is not None
            else {"epsilon": 0.0, "mechanism": "non-private", "details": []}
        )
        if dp_mode:
            self.warnings.append(
                "DP preprocessing privatised numeric bounds and moments. Category vocabularies and column "
                "names are assumed to be public schema metadata; category frequencies were not used for "
                "training sampling in DP mode."
            )
        return self

    @staticmethod
    def _numeric_values(s: pd.Series, kind: str) -> np.ndarray:
        if kind == "datetime":
            return pd.to_datetime(s, errors="coerce").astype("int64").to_numpy(dtype=np.float64) / 1e9
        return pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)

    @staticmethod
    def _fit_vgm(values: np.ndarray, seed: int, max_components: int) -> tuple[list[float], list[float], list[float]]:
        x = values.reshape(-1, 1)
        n_unique = len(np.unique(values))
        if len(values) < 80 or n_unique < 12:
            return [float(np.mean(values))], [float(max(np.std(values), 1e-9))], [1.0]
        max_k = int(min(max_components, max(1, n_unique // 25), len(values) // 30))
        if max_k <= 1:
            return [float(np.mean(values))], [float(max(np.std(values), 1e-9))], [1.0]
        best: GaussianMixture | None = None
        best_bic = np.inf
        for k in range(1, max_k + 1):
            try:
                gm = GaussianMixture(n_components=k, covariance_type="full", n_init=1, random_state=seed, reg_covar=1e-6, max_iter=120)
                gm.fit(x)
                bic = gm.bic(x)
                if bic < best_bic:
                    best_bic, best = bic, gm
            except Exception:
                continue
        if best is None:
            return [float(np.mean(values))], [float(max(np.std(values), 1e-9))], [1.0]
        means = best.means_.reshape(-1).astype(float).tolist()
        stds = np.sqrt(best.covariances_.reshape(-1)).astype(float)
        stds = np.maximum(stds, (values.max() - values.min()) * 1e-3 + 1e-9).tolist()
        weights = best.weights_.astype(float).tolist()
        return means, stds, weights

    def _feature_names_for(self, t: NumericTransform | CategoricalTransform) -> list[str]:
        if isinstance(t, NumericTransform):
            names = []
            if t.missing_indicator:
                names.append(f"{t.name}__null")
            if t.n_components > 1:
                names.extend(f"{t.name}__mode{k}" for k in range(t.n_components))
            names.append(f"{t.name}__value")
            return names
        return [f"{t.name}__{c}" for c in t.categories]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ModelError("preprocessor is not fitted")
        parts = [t.encode(frame[t.name]) for t in self.transforms]
        return np.concatenate(parts, axis=1).astype(np.float32)

    def inverse(self, x: np.ndarray) -> pd.DataFrame:
        if not self.fitted:
            raise ModelError("preprocessor is not fitted")
        x = np.asarray(x)
        out: dict[str, pd.Series] = {}
        for t in self.transforms:
            block = x[:, t.start : t.start + t.dim]
            out[t.name] = t.decode(block)
        df = pd.DataFrame(out)
        for spec in (self.schema.columns if self.schema else []):
            if spec.name in df.columns:
                try:
                    if spec.kind == "integer":
                        df[spec.name] = pd.to_numeric(df[spec.name], errors="coerce").round().astype("Int64")
                    elif spec.kind == "boolean":
                        df[spec.name] = df[spec.name].map(lambda v: None if pd.isna(v) else bool(v)).astype("boolean")
                except Exception:
                    pass
        return df

    def condition_vector(self, conditions: dict[str, Any] | None = None) -> np.ndarray:
        vec = np.zeros(self.condition_dim, dtype=np.float32)
        if not conditions:
            return vec
        groups = {t.name: t for t in self.categorical}
        for name, value in conditions.items():
            if name not in groups:
                continue
            t = groups[name]
            vec[t.start - self.categorical_offset(t) + t.condition_index(value)] = 1.0
        return vec

    def categorical_offset(self, target: CategoricalTransform) -> int:
        # Categorical blocks are not contiguous in X; condition vectors use their own concatenation.
        offset = 0
        for t in self.categorical:
            if t is target:
                return offset
            offset += t.dim
        return offset

    def condition_indices(self, conditions: dict[str, Any] | None = None) -> dict[str, int]:
        groups = {t.name: t for t in self.categorical}
        out = {}
        for name, value in (conditions or {}).items():
            if name in groups:
                t = groups[name]
                out[name] = self.categorical_offset(t) + t.condition_index(value)
        return out

    def sample_condition(self, rng: np.random.Generator, *, dp_mode: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        cats = self.categorical
        vec = np.zeros(self.condition_dim, dtype=np.float32)
        if not cats:
            return vec, {}
        if dp_mode:
            weights = np.ones(len(cats)) / len(cats)
        else:
            weights = np.array([max(float(t.frequencies.sum()), 1e-9) for t in cats], dtype=float)
            weights = weights / weights.sum()
        group = cats[int(rng.choice(len(cats), p=weights))]
        if dp_mode:
            probs = np.ones(group.dim) / group.dim
        else:
            probs = group.frequencies / max(group.frequencies.sum(), 1e-12)
        idx = int(rng.choice(group.dim, p=probs))
        vec[self.categorical_offset(group) + idx] = 1.0
        return vec, {group.name: group.original_values[idx]}

    def raw_column_dims(self) -> dict[str, tuple[int, int]]:
        return {t.name: (t.start, t.dim) for t in self.transforms}

    def schema_columns(self) -> list[ColumnSpec]:
        return [c for c in (self.schema.columns if self.schema else []) if c.name in [t.name for t in self.transforms]]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return p

    @classmethod
    def load(cls, path: str | Path) -> TabularPreprocessor:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, TabularPreprocessor):
            raise ModelError("preprocessor artifact is not a TabularPreprocessor")
        return obj

    def metadata(self) -> dict[str, Any]:
        return {
            "dim": self.dim,
            "features": len(self.feature_names),
            "columns": [
                {
                    "name": t.name,
                    "dim": t.dim,
                    "type": "numeric" if isinstance(t, NumericTransform) else "categorical",
                    "components": t.n_components if isinstance(t, NumericTransform) else len(t.categories),
                }
                for t in self.transforms
            ],
            "condition_dim": self.condition_dim,
            "dp_preprocessing": self.dp_details,
            "warnings": self.warnings,
        }
