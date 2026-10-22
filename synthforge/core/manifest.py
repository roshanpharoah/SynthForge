"""Run manifests: the reproducibility contract for every trained model and report."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from synthforge.core.config import RunConfig
from synthforge.core.determinism import stable_hash
from synthforge.core.types import Dataset


def code_version() -> str:
    from synthforge import __version__

    version = __version__
    try:
        sha = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True)
            .strip()
        )
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return f"{version}+{sha}{'.dirty' if dirty else ''}"
    except Exception:
        return version


def library_versions() -> dict[str, str]:
    import importlib.metadata as md

    names = [
        "numpy", "pandas", "scipy", "scikit-learn", "torch", "pydantic", "sqlalchemy",
        "pyarrow", "duckdb", "nibabel", "pydicom", "wfdb", "fastapi",
    ]
    out = {}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "not-installed"
    return out


class ArtifactRef(BaseModel):
    name: str
    path: str
    sha256: str
    size: int
    kind: str = "file"


class PrivacyClaim(BaseModel):
    mechanism: str
    epsilon: float | None
    delta: float | None
    covered: list[str]
    excluded: list[str]
    assumptions: list[str]
    steps: int | None = None
    sampling_rate: float | None = None
    clip_norm: float | None = None
    noise_multiplier: float | None = None
    dp_preprocessing: bool = False
    statement: str


class RunManifest(BaseModel):
    schema_version: int = 1
    run_id: str
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "completed"
    project_id: str | None = None
    dataset_id: str | None = None
    dataset_name: str | None = None
    data_fingerprint: str | None = None
    config: dict[str, Any]
    config_hash: str
    code_version: str
    git_sha: str | None = None
    python: str
    platform: str
    library_versions: dict[str, str]
    seed: int
    device: str = "cpu"
    model: str
    model_params: dict[str, Any] = Field(default_factory=dict)
    training: dict[str, Any] = Field(default_factory=dict)
    privacy: PrivacyClaim | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactRef] = Field(default_factory=list)
    notes: str | None = None
    command: str | None = None
    warnings: list[str] = Field(default_factory=list)

    def reproduce_key(self) -> str:
        """Deterministic identity excluding timestamps and artifact paths."""
        return stable_hash(
            {
                "config_hash": self.config_hash,
                "data_fingerprint": self.data_fingerprint,
                "seed": self.seed,
                "code_version": self.code_version,
                "model": self.model,
                "model_params": self.model_params,
                "privacy": self.privacy.model_dump(mode="json") if self.privacy else None,
            }
        )


def build_manifest(
    *,
    run_id: str,
    config: RunConfig,
    dataset: Dataset | None,
    model: str,
    model_params: dict[str, Any] | None = None,
    training: dict[str, Any] | None = None,
    privacy: PrivacyClaim | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: list[ArtifactRef] | None = None,
    device: str = "cpu",
    notes: str | None = None,
    project_id: str | None = None,
    dataset_id: str | None = None,
    command: str | None = None,
    warnings: list[str] | None = None,
) -> RunManifest:
    config_hash = stable_hash(config.model_dump(mode="json"))
    cfg = config.model_dump(mode="json")
    return RunManifest(
        run_id=run_id,
        project_id=project_id,
        dataset_id=dataset_id,
        dataset_name=dataset.name if dataset else None,
        data_fingerprint=dataset.fingerprint() if dataset else None,
        config=cfg,
        config_hash=config_hash,
        code_version=code_version(),
        git_sha=_git_sha(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        library_versions=library_versions(),
        seed=config.seed,
        device=device,
        model=model,
        model_params=model_params or config.model_params,
        training=training or {},
        privacy=privacy,
        metrics=metrics or {},
        artifacts=artifacts or [],
        notes=notes if notes is not None else config.notes,
        command=command,
        warnings=warnings or [],
    )


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def save_manifest(manifest: RunManifest, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_manifest(path: str | Path) -> RunManifest:
    return RunManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
