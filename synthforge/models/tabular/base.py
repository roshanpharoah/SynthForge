"""Shared tabular training/sampling logic: conditional filtering, constraint repair, provenance."""

from __future__ import annotations

import contextlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.config import RunConfig
from synthforge.core.errors import ModelError, PrivacyError
from synthforge.core.protocol import (
    BaseSynthesizer,
    CancelCheck,
    FitResult,
    PrivacyReport,
    ProgressCallback,
    TrainHistory,
)
from synthforge.core.types import Dataset, Table
from synthforge.metadata.constraints import repair_table
from synthforge.models.preprocessing import TabularPreprocessor


class BaseTabularSynthesizer(BaseSynthesizer):
    modality = "tabular"

    def __init__(self, config: RunConfig | None = None):
        super().__init__(config)
        self.preprocessor: TabularPreprocessor | None = None
        self.train_rows = 0
        self.table_name = "table"
        self.fit_warnings: list[str] = []
        self.constraint_report: dict[str, Any] = {}

    # ---------------------------------------------------------------- helpers
    def _prepare(
        self, dataset: Dataset, config: RunConfig, *, table: str | None = None
    ) -> tuple[pd.DataFrame, Table]:
        if not dataset.tables:
            raise ModelError(
                f"{self.model_name} requires a tabular dataset",
                hint="Use an ECG model for waveforms or an imaging model for volumes.",
            )
        name = table or next(iter(dataset.tables))
        if name not in dataset.tables:
            raise ModelError(f"table {name!r} not found", hint=f"Available: {list(dataset.tables)}")
        tbl = dataset.tables[name]
        self.table_name = name
        self.train_rows = len(tbl.frame)
        if self.train_rows < 10:
            raise ModelError(
                "at least 10 rows are required to train a tabular generator",
                hint="Ingest more rows or use a baseline model for a smoke test.",
            )
        dp_eps_prep = 0.0
        if config.privacy.enabled and config.privacy.dp_preprocessing and config.privacy.epsilon is not None:
            dp_eps_prep = 0.1 * float(config.privacy.epsilon)
        elif config.privacy.enabled and config.privacy.dp_preprocessing and config.privacy.noise_multiplier is not None:
            self.fit_warnings.append(
                "DP preprocessing was requested with noise_multiplier only; no total epsilon exists to split, "
                "so preprocessing is non-private and is excluded from the claim."
            )
        self.preprocessor = TabularPreprocessor().fit(
            tbl.frame,
            tbl.schema,
            seed=config.seed,
            dp_epsilon=dp_eps_prep,
            dp_mode=bool(config.privacy.enabled and config.privacy.dp_preprocessing and dp_eps_prep > 0),
        )
        self.fit_warnings.extend(self.preprocessor.warnings)
        return tbl.frame, tbl

    def _training_epsilon_cap(self, config: RunConfig) -> float | None:
        if not config.privacy.enabled:
            return None
        if config.privacy.epsilon is None:
            return None
        prep = float(self.preprocessor.dp_details.get("epsilon", 0.0)) if self.preprocessor else 0.0
        cap = float(config.privacy.epsilon) - prep
        if cap <= 0:
            raise PrivacyError(
                "the DP preprocessing allocation consumed the entire epsilon budget",
                hint="Increase epsilon or disable dp_preprocessing.",
            )
        return cap

    def _privacy_report_common(
        self,
        *,
        mechanism: str,
        epsilon_training: float | None,
        delta: float | None,
        steps: int,
        sampling_rate: float,
        clip_norm: float | None,
        noise_multiplier: float | None,
        covered: list[str],
        excluded: list[str],
        dp_preprocessing: bool,
        composition: list[dict[str, Any]] | None = None,
    ) -> PrivacyReport:
        prep = float(self.preprocessor.dp_details.get("epsilon", 0.0)) if self.preprocessor else 0.0
        total = (epsilon_training or 0.0) + prep
        if total > 0:
            statement = (
                f"{mechanism}. ε={total:.4f}, δ={delta}. This covers: {'; '.join(covered)}. "
                f"Excluded: {'; '.join(excluded)}. Empirical attacks are complementary evidence, not a proof."
            )
        else:
            statement = "Non-private run: no differential-privacy guarantee is claimed."
        comp = list(composition or [])
        if prep > 0:
            comp.insert(
                0,
                {
                    "component": "DP preprocessing",
                    "mechanism": self.preprocessor.dp_details.get("mechanism") if self.preprocessor else "",
                    "epsilon": prep,
                    "delta": 0.0,
                    "composition": "basic composition (conservative)",
                },
            )
        if epsilon_training:
            comp.append(
                {
                    "component": "training",
                    "mechanism": mechanism,
                    "epsilon": epsilon_training,
                    "delta": delta,
                    "steps": steps,
                    "sampling_rate": sampling_rate,
                }
            )
        return PrivacyReport(
            mechanism=mechanism,
            epsilon=total if total > 0 else None,
            delta=delta,
            covered=covered,
            excluded=excluded,
            assumptions=[
                "Each source row contributes to at most one training example.",
                "Poisson subsampling is used for the DP training steps (unless stated otherwise).",
                "Category vocabularies and column names are treated as public schema metadata."
                if dp_preprocessing
                else "Preprocessing statistics were computed on real data without DP noise.",
                "The accountant uses integer Rényi orders and basic composition with DP preprocessing.",
            ],
            steps=steps,
            sampling_rate=sampling_rate,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            dp_preprocessing=dp_preprocessing,
            statement=statement,
            composition=comp,
        )

    def _add_placeholders(self, df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, list[str]]:
        """Add synthetic IDs/placeholders for columns intentionally not modelled."""
        n = len(df)
        warnings: list[str] = []
        schema = self.preprocessor.schema if self.preprocessor else None
        if schema is None:
            return df, warnings
        out = df.copy()
        for spec in schema.columns:
            if spec.name in out.columns:
                continue
            if spec.name in schema.primary_key:
                if spec.kind == "integer":
                    out[spec.name] = np.arange(1, n + 1, dtype=np.int64)
                else:
                    out[spec.name] = [f"SYN-{i + 1:08d}" for i in range(n)]
            elif spec.role == "foreign_key":
                warnings.append(f"Foreign key {spec.name!r} is not generated by a single-table model; use relational synthesis.")
            elif spec.sensitivity == "direct_identifier":
                out[spec.name] = [f"SYNTHETIC-{i + 1:06d}" for i in range(n)]
            elif spec.kind == "text":
                warnings.append(f"Free-text column {spec.name!r} was excluded from modelling and output.")
            elif spec.dropped:
                warnings.append(f"Column {spec.name!r} was excluded from modelling.")
        ordered = [c.name for c in schema.columns if c.name in out.columns]
        out = out[ordered]
        return out, warnings

    def _conditioned_frame(
        self,
        n: int,
        conditions: dict[str, Any] | None,
        seed: int,
        *,
        max_oversample: int = 6,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        assert self.preprocessor is not None
        conditions = self._conditions_dict(conditions, self._conditions)
        cat_groups = {t.name: t for t in self.preprocessor.categorical}
        cond_vec = self.preprocessor.condition_vector({k: v for k, v in conditions.items() if k in cat_groups})
        numeric_conditions = {
            k: v
            for k, v in conditions.items()
            if k not in cat_groups and self.preprocessor.schema and any(c.name == k for c in self.preprocessor.schema.columns)
        }
        unknown = [k for k in conditions if k not in cat_groups and k not in numeric_conditions]
        if unknown:
            raise ModelError(
                f"unknown condition columns: {unknown}",
                hint="Condition on a modelled column. Primary keys, foreign keys and dropped text columns are not conditional.",
            )
        target = n
        oversample = max_oversample if (conditions and len(numeric_conditions) == 0 and cat_groups) else 1
        frames = []
        attempts = 0
        info: dict[str, Any] = {"conditions": conditions, "oversample": oversample, "attempts": 0}
        while attempts < 4 and sum(len(f) for f in frames) < target:
            attempts += 1
            m = n if not frames else max(n - sum(len(f) for f in frames), int(n * 0.5))
            x = self._generate_transformed(m * oversample, cond_vec, seed + attempts * 7919)
            frame = self.preprocessor.inverse(x)
            if conditions:
                mask = np.ones(len(frame), dtype=bool)
                for k, v in conditions.items():
                    if k in cat_groups:
                        mask &= frame[k].astype(str).to_numpy() == str(v)
                    else:
                        vals = pd.to_numeric(frame[k], errors="coerce").to_numpy(dtype=float)
                        target_v = float(v)
                        span = self._numeric_span(k)
                        mask &= np.abs(vals - target_v) <= max(span * 0.05, 1e-9)
                frame = frame.loc[mask]
            frames.append(frame)
        out = pd.concat(frames, ignore_index=True) if frames else self.preprocessor.inverse(
            self._generate_transformed(n, cond_vec, seed)
        )
        if len(out) < target:
            # fill by nearest available or exact repair, and disclose
            if conditions:
                fill = out.sample(max(target - len(out), 0), replace=True, random_state=seed) if len(out) else pd.DataFrame()
                if len(fill):
                    for k, v in conditions.items():
                        if k not in cat_groups:
                            fill[k] = v
                    out = pd.concat([out, fill], ignore_index=True)
            else:
                x = self._generate_transformed(target - len(out), cond_vec, seed + 99)
                out = pd.concat([out, self.preprocessor.inverse(x)], ignore_index=True)
            info["conditioning_fallback"] = True
        out = out.head(target).reset_index(drop=True)
        info["attempts"] = attempts
        # enforce constraints on the modelled frame
        if self.preprocessor.schema is not None:
            out, report = repair_table(out, self.preprocessor.schema.constraints, seed=seed)
            self.constraint_report = report
            info["constraint_repair"] = report
        out, warns = self._add_placeholders(out, seed)
        if warns:
            info["placeholder_warnings"] = warns
        return out, info

    def _numeric_span(self, column: str) -> float:
        assert self.preprocessor and self.preprocessor.schema is not None
        for t in self.preprocessor.numeric:
            if t.name == column:
                return float(t.bounds[1] - t.bounds[0])
        return 1.0

    def _finish_sample(self, frame: pd.DataFrame, n: int, seed: int, info: dict[str, Any]) -> Dataset:
        assert self.preprocessor is not None and self.preprocessor.schema is not None
        schema = self.preprocessor.schema.model_copy(deep=True)
        for spec in schema.columns:
            if spec.name in frame.columns and spec.kind == "integer" and spec.name not in schema.primary_key:
                with contextlib.suppress(Exception):
                    frame[spec.name] = pd.to_numeric(frame[spec.name], errors="coerce").round().astype("Int64")
        table = Table(name=self.table_name, frame=frame, schema=schema)
        return Dataset(
            name=f"{self.table_name}_synthetic",
            kind="tabular",
            tables={self.table_name: table},
            synthetic=True,
            provenance={
                "model": self.model_name,
                "model_params": self.config.model_params,
                "seed": seed,
                "n": n,
                "conditions": info.get("conditions"),
                "conditioning": info,
                "privacy": self.report.model_dump(mode="json"),
                "constraint_report": self.constraint_report,
                "note": "Synthetic rows generated from a fitted model; not real records.",
            },
        )

    def sample(
        self,
        n: int,
        *,
        conditions: dict[str, Any] | None = None,
        seed: int | None = None,
        table: str | None = None,
    ) -> Dataset:
        if not self.is_fitted or self.preprocessor is None:
            raise ModelError(
                f"{self.model_name} is not fitted",
                hint="Call fit() or load a trained model first.",
            )
        if n < 1:
            raise ModelError("n must be >= 1")
        use_seed = int(seed if seed is not None else self.config.seed + 1)
        frame, info = self._conditioned_frame(n, conditions, use_seed)
        return self._finish_sample(frame, n, use_seed, info)

    def _generate_transformed(self, n: int, condition_vector: np.ndarray, seed: int) -> np.ndarray:
        raise NotImplementedError

    def _base_save(self, path: Path) -> None:
        assert self.preprocessor is not None
        self.preprocessor.save(path / "preprocessor.pkl")

    def _base_load(self, path: Path) -> None:
        self.preprocessor = TabularPreprocessor.load(path / "preprocessor.pkl")
        self.is_fitted = True

    def model_card(self) -> dict[str, Any]:
        card = super().model_card()
        card.update(
            {
                "model": self.model_name,
                "table": self.table_name,
                "training_rows": self.train_rows,
                "preprocessing": self.preprocessor.metadata() if self.preprocessor else {},
                "excluded_columns": [
                    c.name
                    for c in (self.preprocessor.schema.columns if self.preprocessor and self.preprocessor.schema else [])
                    if c.dropped
                ],
                "conditioning": "Categorical columns via a conditional vector; numeric conditions via disclosed post-filtering.",
                "constraints": [c.model_dump(mode="json") for c in (self.preprocessor.schema.constraints if self.preprocessor and self.preprocessor.schema else [])],
                "warnings": list(dict.fromkeys(self.warnings + self.fit_warnings + (self.preprocessor.warnings if self.preprocessor else []))),
                "intended_use": "Downstream development, testing, augmentation and research on tabular data.",
                "out_of_scope": [
                    "Clinical/legal/regulatory decisions",
                    "Claims that synthetic rows are anonymous",
                    "Use as a substitute for real-data validation",
                ],
            }
        )
        return card

    def fit_placeholder(self, message: str) -> None:
        raise ModelError(message)


def make_history(started: float, losses: dict[str, list[float]], steps: int, epochs: int, early: bool = False) -> TrainHistory:
    return TrainHistory(
        steps=steps,
        epochs=epochs,
        losses={k: [float(x) for x in v] for k, v in losses.items()},
        early_stopped=early,
        best_step=steps,
        duration_seconds=time.time() - started,
    )


__all__ = ["BaseTabularSynthesizer", "FitResult", "ProgressCallback", "CancelCheck"]
