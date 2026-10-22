"""The model contract shared by every modality and by the registry/API."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from synthforge.core.config import RunConfig
from synthforge.core.types import Dataset

ProgressCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


class TrainHistory(BaseModel):
    steps: int = 0
    epochs: int = 0
    losses: dict[str, list[float]] = Field(default_factory=dict)
    early_stopped: bool = False
    best_step: int | None = None
    duration_seconds: float = 0.0


class FitResult(BaseModel):
    history: TrainHistory
    device: str = "cpu"
    n_train_rows: int = 0
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PrivacyReport(BaseModel):
    mechanism: str
    epsilon: float | None = None
    delta: float | None = None
    covered: list[str] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    steps: int | None = None
    sampling_rate: float | None = None
    clip_norm: float | None = None
    noise_multiplier: float | None = None
    dp_preprocessing: bool = False
    statement: str = "Non-private run: no differential-privacy guarantee is claimed."
    composition: list[dict[str, Any]] = Field(default_factory=list)


@runtime_checkable
class Synthesizer(Protocol):
    model_name: str

    def fit(
        self,
        dataset: Dataset,
        config: RunConfig,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult: ...

    def sample(
        self,
        n: int,
        *,
        conditions: dict[str, Any] | None = None,
        seed: int | None = None,
        table: str | None = None,
    ) -> Dataset: ...

    def condition(self, **conditions: Any) -> Synthesizer: ...

    def save(self, path: str | Path) -> Path: ...

    def privacy_report(self) -> PrivacyReport: ...

    def model_card(self) -> dict[str, Any]: ...


class BaseSynthesizer(ABC):
    """Shared serialisation helpers. Subclasses implement architecture and training."""

    model_name = "base"
    modality = "tabular"

    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        self.is_fitted = False
        self.history = TrainHistory()
        self.report = PrivacyReport(mechanism="none")
        self.warnings: list[str] = []
        self.metadata: dict[str, Any] = {}
        self._conditions: dict[str, Any] = {}

    def condition(self, **conditions: Any) -> BaseSynthesizer:
        self._conditions.update({k: v for k, v in conditions.items() if v is not None})
        return self

    @abstractmethod
    def fit(
        self,
        dataset: Dataset,
        config: RunConfig,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult: ...

    @abstractmethod
    def sample(
        self,
        n: int,
        *,
        conditions: dict[str, Any] | None = None,
        seed: int | None = None,
        table: str | None = None,
    ) -> Dataset: ...

    @abstractmethod
    def _save_state(self, path: Path) -> None: ...

    @abstractmethod
    def _load_state(self, path: Path) -> None: ...

    def save(self, path: str | Path) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        self._save_state(root)
        (root / "config.json").write_text(self.config.model_dump_json(indent=2), encoding="utf-8")
        (root / "privacy.json").write_text(self.report.model_dump_json(indent=2), encoding="utf-8")
        (root / "model_card.json").write_text(
            __import__("json").dumps(self.model_card(), indent=2, default=str), encoding="utf-8"
        )
        return root

    def privacy_report(self) -> PrivacyReport:
        return self.report

    def model_card(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "modality": self.modality,
            "version": "0.1.0",
            "intended_use": "Synthetic data generation for development, testing and research.",
            "out_of_scope": [
                "Clinical or diagnostic decision making",
                "Regulatory compliance certification",
                "Replacing validation on real data",
            ],
            "training": self.history.model_dump(mode="json"),
            "privacy_claim": self.report.model_dump(mode="json"),
            "limitations": self.warnings,
            "metadata": self.metadata,
        }

    @staticmethod
    def _conditions_dict(conditions: dict[str, Any] | None, fallback: dict[str, Any]) -> dict[str, Any]:
        out = dict(fallback)
        if conditions:
            out.update(conditions)
        return out


def labels_frame(n: int, conditions: dict[str, Any]) -> pd.DataFrame:
    if not conditions:
        return pd.DataFrame(index=pd.RangeIndex(n))
    data = {}
    for k, v in conditions.items():
        if isinstance(v, np.ndarray):
            data[k] = v
        elif isinstance(v, (list, tuple)):
            data[k] = list(v)
        else:
            data[k] = [v] * n
    return pd.DataFrame(data)
