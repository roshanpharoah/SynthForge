"""SynthForge: local-first enterprise synthetic data platform.

Public surface (SDK): load_* → fit → sample → evaluate. Nothing here requires network access.
"""

from __future__ import annotations

from typing import Any

from synthforge.core.config import PrivacyConfig, Profile, RunConfig, TrainingConfig

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DP",
    "RunConfig",
    "TrainingConfig",
    "PrivacyConfig",
    "Profile",
    "load_csv",
    "load_parquet",
    "load_excel",
    "load_jsonl",
    "load_database",
    "load_csv_folder",
    "load_ecg",
    "load_nifti",
    "load_dicom",
    "load_dataset",
    "fit",
    "sample",
    "evaluate",
    "attack",
    "SynthForge",
]


def DP(
    epsilon: float | None = None,
    *,
    delta: float = 1e-5,
    clip_norm: float = 1.0,
    target_epsilon: float | None = None,
    dp_preprocessing: bool = True,
    noise_multiplier: float | None = None,
) -> PrivacyConfig:
    """Build a differential-privacy configuration.

    ``epsilon`` is the target for the *training mechanism* and (if enabled) DP preprocessing,
    composed by the RDP accountant. See the generated model card for the exact covered scope.
    """
    return PrivacyConfig(
        enabled=True,
        epsilon=epsilon,
        target_epsilon=target_epsilon,
        delta=delta,
        clip_norm=clip_norm,
        dp_preprocessing=dp_preprocessing,
        noise_multiplier=noise_multiplier,
    )


def load_csv(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.tabular import load_csv as _load

    return _load(path, **kwargs)


def load_parquet(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.tabular import load_parquet as _load

    return _load(path, **kwargs)


def load_excel(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.tabular import load_excel as _load

    return _load(path, **kwargs)


def load_jsonl(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.tabular import load_jsonl as _load

    return _load(path, **kwargs)


def load_database(url: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.database import load_database as _load

    return _load(url, **kwargs)


def load_csv_folder(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.database import load_csv_folder as _load

    return _load(path, **kwargs)


def load_ecg(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.timeseries import load_ecg as _load

    return _load(path, **kwargs)


def load_nifti(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.imaging import load_imaging

    return load_imaging(path, **kwargs)


def load_dicom(path: str, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.ingest.imaging import load_imaging

    return load_imaging(path, **kwargs)


def load_dataset(path: str, *, profile: bool = True):  # type: ignore[no-untyped-def]
    from synthforge.core.types import Dataset
    from synthforge.ingest.profile import profile_dataset

    ds = Dataset.load(path)
    if not ds.profile and profile:
        ds.profile = profile_dataset(ds)
    return ds


def fit(dataset, model: str = "ctgan", *, privacy: PrivacyConfig | None = None, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.sdk.local import fit as _fit

    return _fit(dataset, model=model, privacy=privacy, **kwargs)


def sample(model, n: int, **kwargs: Any):  # type: ignore[no-untyped-def]
    return model.sample(n, **kwargs)


def evaluate(real, synthetic, target: str | None = None, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.sdk.local import evaluate as _eval

    return _eval(real, synthetic, target=target, **kwargs)


def attack(real, synthetic, **kwargs: Any):  # type: ignore[no-untyped-def]
    from synthforge.privacy.attacks import run_attack_suite

    return run_attack_suite(real, synthetic, **kwargs)


def SynthForge(root: str | None = None):  # type: ignore[no-untyped-def]
    from synthforge.service import SynthForge as _Service

    return _Service(root=root)
