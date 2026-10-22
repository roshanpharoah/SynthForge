"""Run configuration. Profiles are explicit and applied deterministically."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class Profile(str, Enum):
    FAST = "fast"
    FULL = "full"


class TrainingConfig(BaseModel):
    epochs: int = Field(default=40, ge=1, le=100_000)
    batch_size: int = Field(default=256, ge=4, le=100_000)
    lr: float = Field(default=2e-4, gt=0)
    weight_decay: float = Field(default=1e-6, ge=0)
    latent_dim: int = Field(default=64, ge=2)
    hidden_dim: int = Field(default=256, ge=8)
    pac: int = Field(default=2, ge=1, le=16)
    early_stop_patience: int = Field(default=15, ge=1)
    checkpoint_every: int = Field(default=5, ge=1)
    max_steps: int | None = Field(default=None, ge=1)
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    num_threads: int | None = Field(default=None, ge=1)

    # media specific
    resolution: int = Field(default=32, ge=8, le=256)
    signal_length: int = Field(default=2500, ge=64)
    signal_fs: int = Field(default=250, ge=25)
    leads: int = Field(default=12, ge=1, le=32)
    image_channels: int = Field(default=1, ge=1, le=4)


class PrivacyConfig(BaseModel):
    """Differential privacy configuration.

    When enabled, every number needed to reproduce epsilon is recorded in the run manifest.
    """

    enabled: bool = False
    epsilon: float | None = Field(default=None, gt=0)
    delta: float = Field(default=1e-5, gt=0, lt=1)
    clip_norm: float = Field(default=1.0, gt=0)
    noise_multiplier: float | None = Field(default=None, gt=0)
    accountant: Literal["rdp"] = "rdp"
    poisson_sampling: bool = True
    dp_preprocessing: bool = True
    target_epsilon: float | None = Field(default=None, gt=0)
    max_steps: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _require_target(self) -> PrivacyConfig:
        if self.enabled and self.epsilon is None and self.target_epsilon is None:
            raise ValueError("DP enabled but neither epsilon nor target_epsilon was supplied")
        if self.epsilon is not None and self.target_epsilon is not None:
            raise ValueError("Supply epsilon or target_epsilon, not both")
        return self

    @property
    def target(self) -> float:
        assert self.epsilon is not None or self.target_epsilon is not None
        return float(self.epsilon if self.epsilon is not None else self.target_epsilon)


class EvalConfig(BaseModel):
    target_column: str | None = None
    classifiers: list[str] = Field(default_factory=lambda: ["logistic", "hgb", "mlp"])
    bootstrap: int = Field(default=200, ge=0, le=5000)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    train_fraction: float = Field(default=0.7, gt=0.2, lt=0.9)
    seeds: int = Field(default=3, ge=1, le=20)
    discriminator_auc: bool = True
    privacy_attacks: bool = True
    attack_max_rows: int = Field(default=4000, ge=100)
    thresholds: dict[str, float] = Field(default_factory=dict)


class RunConfig(BaseModel):
    model_config = {"protected_namespaces": ()}

    model: str = "ctgan"
    profile: Profile = Profile.FAST
    seed: int = 42
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    model_params: dict[str, Any] = Field(default_factory=dict)
    conditions: dict[str, Any] | None = None
    target_column: str | None = None
    notes: str | None = None

    @field_validator("profile", mode="before")
    @classmethod
    def _profile(cls, v: Any) -> Any:
        return Profile(v) if isinstance(v, str) else v

    def apply_profile(self) -> RunConfig:
        """Return a copy with profile defaults. Explicit user values win only where safe.

        The fast profile is a real training run with fewer steps and smaller networks. It is not a
        stub: tests and the demo assert non-degenerate outputs.
        """
        data = self.model_dump()
        if self.profile == Profile.FAST:
            training = data["training"]
            training.update(
                {
                    "epochs": min(training["epochs"], 8),
                    "batch_size": min(training["batch_size"], 128),
                    "latent_dim": min(training["latent_dim"], 32),
                    "hidden_dim": min(training["hidden_dim"], 128),
                    "early_stop_patience": min(training["early_stop_patience"], 4),
                    "checkpoint_every": min(training["checkpoint_every"], 2),
                    "resolution": min(training["resolution"], 32),
                    "signal_length": min(training["signal_length"], 500),
                    "signal_fs": min(training["signal_fs"], 125),
                }
            )
            data["eval"]["seeds"] = min(int(data["eval"]["seeds"]), 2)
            data["eval"]["bootstrap"] = min(int(data["eval"]["bootstrap"]), 80)
            data["eval"]["attack_max_rows"] = min(int(data["eval"]["attack_max_rows"]), 1500)
        else:
            data["eval"]["seeds"] = max(int(data["eval"]["seeds"]), 3)
        return RunConfig.model_validate(data)

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if p.suffix.lower() in {".yaml", ".yml"} else __import__("json").loads(text)
        return cls.model_validate(data or {})

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)


class JobSpec(BaseModel):
    kind: Literal["train", "sample", "evaluate", "attack", "frontier", "bundle", "reproduce", "relational"]
    project_id: str
    dataset_id: str | None = None
    run_id: str | None = None
    config: RunConfig = Field(default_factory=RunConfig)
    params: dict[str, Any] = Field(default_factory=dict)
    max_attempts: int = Field(default=2, ge=1, le=10)
