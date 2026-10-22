"""Gaussian-copula baseline over the mixed-type transformed space.

The copula captures linear dependence between encoded features. It is deliberately simple: it cannot
represent multimodal conditionals or higher-order interactions. Its purpose is to provide an honest
baseline and a failure case for the benchmark, not to rival CTGAN.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from synthforge.core.config import RunConfig
from synthforge.core.errors import ModelError
from synthforge.core.protocol import CancelCheck, FitResult, ProgressCallback, TrainHistory
from synthforge.core.types import Dataset
from synthforge.models.preprocessing import NumericTransform
from synthforge.models.tabular.base import BaseTabularSynthesizer


class GaussianCopulaSynthesizer(BaseTabularSynthesizer):
    model_name = "copula"

    def __init__(self, config: RunConfig | None = None):
        super().__init__(config)
        self.mean: np.ndarray | None = None
        self.cov: np.ndarray | None = None
        self._blocks: list[dict[str, Any]] = []
        self.train_matrix: np.ndarray | None = None

    def _build_blocks(self) -> None:
        assert self.preprocessor is not None
        self._blocks = []
        for t in self.preprocessor.transforms:
            if isinstance(t, NumericTransform):
                pos = t.start
                b: dict[str, Any] = {"kind": "numeric"}
                if t.missing_indicator:
                    b["null"] = pos
                    pos += 1
                if t.n_components > 1:
                    b["modes"] = (pos, pos + t.n_components)
                    pos += t.n_components
                b["value"] = pos
                self._blocks.append(b)
            else:
                self._blocks.append(
                    {"kind": "categorical", "slice": (t.start, t.start + t.dim)}
                )

    def _activate(self, sample: np.ndarray) -> np.ndarray:
        out = sample.copy()
        for b in self._blocks:
            if b["kind"] == "numeric":
                if "null" in b:
                    out[:, b["null"]] = (sample[:, b["null"]] > 0).astype(np.float32)
                if "modes" in b:
                    a, c = b["modes"]
                    out[:, a:c] = 0.0
                    out[np.arange(len(out)), a + np.argmax(sample[:, a:c], axis=1)] = 1.0
            else:
                a, c = b["slice"]
                out[:, a:c] = 0.0
                out[np.arange(len(out)), a + np.argmax(sample[:, a:c], axis=1)] = 1.0
        return out

    def fit(
        self,
        dataset: Dataset,
        config: RunConfig,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult:
        started = time.time()
        self.config = config
        frame, _ = self._prepare(dataset, config)
        assert self.preprocessor is not None
        if config.privacy.enabled:
            raise ModelError(
                "the copula baseline has no accounted DP path",
                hint="Use model='ctgan' for DP runs.",
            )
        self.fit_warnings = []
        self.warnings = []
        self._build_blocks()
        x = self.preprocessor.transform(frame).astype(np.float64)
        n, d = x.shape
        self.train_matrix = x
        self.mean = x.mean(axis=0)
        if n > 2:
            cov = np.cov(x, rowvar=False)
        else:
            cov = np.eye(d)
        cov = np.atleast_2d(cov)
        # shrink toward a diagonal to keep the covariance well-conditioned in small samples
        lam = 1.0 / max(n, 1)
        cov = (1 - lam) * cov + lam * np.diag(np.maximum(np.diag(cov), 1e-6))
        vals, vecs = np.linalg.eigh(cov)
        vals = np.maximum(vals, 1e-8)
        self.cov = (vecs * vals) @ vecs.T
        self.data_dim = d
        self.condition_dim = self.preprocessor.condition_dim
        self.is_fitted = True
        self.train_rows = n
        steps = 1
        if progress:
            progress({"phase": "train", "step": 1, "max_steps": 1, "progress": 0.95})
        self.history = TrainHistory(steps=steps, epochs=1, duration_seconds=time.time() - started)
        self.metadata = {
            "architecture": "Gaussian copula on mixed-type encoded features",
            "shrinkage": lam,
            "data_dim": d,
            "likelihood": "multivariate normal on one-hot / VGM encoded space",
        }
        self.report = self._privacy_report_common(
            mechanism="non-private",
            epsilon_training=None,
            delta=None,
            steps=1,
            sampling_rate=0.0,
            clip_norm=None,
            noise_multiplier=None,
            covered=[],
            excluded=["No differential privacy: the copula parameters are computed from real data."],
            dp_preprocessing=False,
        )
        self.warnings = list(dict.fromkeys(self.warnings + self.fit_warnings))
        if progress:
            progress({"phase": "done", "progress": 1.0, "epsilon": None, "step": 1})
        return FitResult(history=self.history, n_train_rows=n, warnings=self.warnings, metadata=self.metadata)

    def _generate_transformed(self, n: int, condition_vector: np.ndarray, seed: int) -> np.ndarray:
        if self.mean is None or self.cov is None:
            raise ModelError("copula is not fitted")
        rng = np.random.default_rng(seed)
        raw = rng.multivariate_normal(self.mean, self.cov, size=n, method="cholesky")
        out = self._activate(raw)
        # condition columns are not part of the copula generation; the shared sampler applies
        # categorical filtering and numeric post-filtering, which keeps this baseline honest.
        return out.astype(np.float32)

    def _save_state(self, path: Path) -> None:
        self._base_save(path)
        np.savez_compressed(
            path / "copula.npz",
            mean=self.mean,
            cov=self.cov,
            train_matrix=self.train_matrix,
            history=np.asarray([self.history.model_dump_json()], dtype=object),
            report=np.asarray([self.report.model_dump_json()], dtype=object),
            table_name=np.asarray([self.table_name], dtype=object),
            train_rows=np.asarray([self.train_rows]),
        )

    def _load_state(self, path: Path) -> None:
        self._base_load(path)
        z = np.load(path / "copula.npz", allow_pickle=True)
        self.mean = z["mean"]
        self.cov = z["cov"]
        self.train_matrix = z["train_matrix"]
        self.history = TrainHistory.model_validate_json(str(z["history"][0]))
        self.report = type(self.report).model_validate_json(str(z["report"][0]))
        self.table_name = str(z["table_name"][0])
        self.train_rows = int(z["train_rows"][0])
        self.data_dim = int(self.mean.shape[0])
        self.condition_dim = self.preprocessor.condition_dim if self.preprocessor else 0
        self._build_blocks()
        self.is_fitted = True

    @classmethod
    def load(cls, path: str | Path) -> GaussianCopulaSynthesizer:
        obj = cls()
        obj._load_state(Path(path))
        cfg = Path(path) / "config.json"
        if cfg.exists():
            obj.config = RunConfig.model_validate_json(cfg.read_text(encoding="utf-8"))
        return obj

    def model_card(self) -> dict[str, Any]:
        card = super().model_card()
        card.update(
            {
                "architecture": {
                    "type": "Gaussian copula",
                    "data_dim": self.data_dim,
                    "limitations": "linear dependence only; no conditional mode structure",
                },
                "privacy_claim": self.report.model_dump(mode="json"),
                "deviations": ["No DP path."],
            }
        )
        return card


class ResampleNoiseSynthesizer(BaseTabularSynthesizer):
    """Non-private baseline: bootstrap source rows and perturb continuous values."""

    model_name = "resample_noise"

    def __init__(self, config: RunConfig | None = None):
        super().__init__(config)
        self.source = None
        self.noise_scale = 0.05

    def fit(
        self,
        dataset: Dataset,
        config: RunConfig,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult:
        started = time.time()
        self.config = config
        frame, _ = self._prepare(dataset, config)
        if config.privacy.enabled:
            raise ModelError(
                "the resample+noise baseline has no accounted DP path",
                hint="Use model='ctgan' for DP runs.",
            )
        self.source = self.preprocessor.transform(frame) if self.preprocessor else None
        self.is_fitted = True
        self.train_rows = len(frame)
        self.history = TrainHistory(steps=1, epochs=1, duration_seconds=time.time() - started)
        self.metadata = {
            "architecture": "row bootstrap with Gaussian jitter on continuous value dimensions",
            "noise_scale_fraction": self.noise_scale,
            "warning": "This baseline can reproduce source rows closely. It is included as a negative control.",
        }
        self.report = self._privacy_report_common(
            mechanism="non-private",
            epsilon_training=None,
            delta=None,
            steps=1,
            sampling_rate=0.0,
            clip_norm=None,
            noise_multiplier=None,
            covered=[],
            excluded=["No differential privacy; the baseline samples directly from training rows."],
            dp_preprocessing=False,
        )
        self.warnings = ["resample_noise is a negative-control baseline and may expose training rows."]
        if progress:
            progress({"phase": "done", "progress": 1.0, "epsilon": None, "step": 1})
        return FitResult(history=self.history, n_train_rows=self.train_rows, warnings=self.warnings, metadata=self.metadata)

    def _generate_transformed(self, n: int, condition_vector: np.ndarray, seed: int) -> np.ndarray:
        if self.source is None or self.preprocessor is None:
            raise ModelError("baseline is not fitted")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, len(self.source), size=n)
        out = self.source[idx].copy()
        for t in self.preprocessor.numeric:
            pos = t.start
            if t.missing_indicator:
                pos += 1
            if t.n_components > 1:
                pos += t.n_components
            scale = max(float(t.std), 1e-9) * self.noise_scale
            out[:, pos] = out[:, pos] + rng.normal(0, scale, size=n)
        return out.astype(np.float32)

    def _save_state(self, path: Path) -> None:
        self._base_save(path)
        np.savez_compressed(
            path / "baseline.npz",
            source=self.source,
            table_name=np.asarray([self.table_name], dtype=object),
            train_rows=np.asarray([self.train_rows]),
            report=np.asarray([self.report.model_dump_json()], dtype=object),
            history=np.asarray([self.history.model_dump_json()], dtype=object),
        )

    def _load_state(self, path: Path) -> None:
        self._base_load(path)
        z = np.load(path / "baseline.npz", allow_pickle=True)
        self.source = z["source"]
        self.table_name = str(z["table_name"][0])
        self.train_rows = int(z["train_rows"][0])
        self.report = type(self.report).model_validate_json(str(z["report"][0]))
        self.history = TrainHistory.model_validate_json(str(z["history"][0]))
        self.is_fitted = True

    @classmethod
    def load(cls, path: str | Path) -> ResampleNoiseSynthesizer:
        obj = cls()
        obj._load_state(Path(path))
        cfg = Path(path) / "config.json"
        if cfg.exists():
            obj.config = RunConfig.model_validate_json(cfg.read_text(encoding="utf-8"))
        return obj

    def model_card(self) -> dict[str, Any]:
        card = super().model_card()
        card.update(
            {
                "architecture": {"type": "row bootstrap + noise", "noise_scale_fraction": self.noise_scale},
                "privacy_claim": self.report.model_dump(mode="json"),
                "deviations": ["Negative-control baseline. Not private; may reproduce source rows."],
            }
        )
        return card


__all__ = ["GaussianCopulaSynthesizer", "ResampleNoiseSynthesizer"]
