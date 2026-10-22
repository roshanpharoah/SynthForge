"""Model factory and artifact loader. Names are stable public identifiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from synthforge.core.config import RunConfig
from synthforge.core.errors import ModelError
from synthforge.core.protocol import BaseSynthesizer

TABULAR_ALIASES = {
    "ctgan": "ctgan",
    "tvae": "tvae",
    "copula": "copula",
    "gaussian_copula": "copula",
    "noise": "resample_noise",
    "resample_noise": "resample_noise",
    "baseline": "resample_noise",
}


def available_models() -> list[dict[str, Any]]:
    return [
        {"name": "ctgan", "modality": "tabular", "dp": True, "description": "Conditional GAN (WGAN-GP + PacGAN); DP-SGD path."},
        {"name": "tvae", "modality": "tabular", "dp": False, "description": "Conditional VAE with a proper ELBO."},
        {"name": "copula", "modality": "tabular", "dp": False, "description": "Gaussian copula baseline."},
        {"name": "resample_noise", "modality": "tabular", "dp": False, "description": "Row bootstrap + noise negative control."},
        {"name": "relational", "modality": "relational", "dp": False, "description": "Hierarchical multi-table synthesis with PK/FK integrity."},
        {"name": "ecg", "modality": "timeseries", "dp": False, "description": "Conditional 12-lead ECG generator with physiological checks."},
        {"name": "diffusion", "modality": "imaging", "dp": False, "description": "DiffuseGAN (diffusion-inspired GAN) for 2D/3D images and masks."},
        {"name": "gan", "modality": "imaging", "dp": False, "description": "Plain DCGAN baseline for imaging."},
        {"name": "multimodal", "modality": "multimodal", "dp": False, "description": "Patient bundles: tabular + signal + image."},
    ]


def build_model(name: str, config: RunConfig | None = None) -> BaseSynthesizer:
    key = (name or "ctgan").lower()
    if key in TABULAR_ALIASES:
        target = TABULAR_ALIASES[key]
        if target == "ctgan":
            from synthforge.models.tabular.ctgan import CTGANSynthesizer

            return CTGANSynthesizer(config)
        if target == "tvae":
            from synthforge.models.tabular.tvae import TVAESynthesizer

            return TVAESynthesizer(config)
        if target == "copula":
            from synthforge.models.tabular.copula import GaussianCopulaSynthesizer

            return GaussianCopulaSynthesizer(config)
        from synthforge.models.tabular.copula import ResampleNoiseSynthesizer

        return ResampleNoiseSynthesizer(config)
    if key in ("relational", "hierarchical"):
        from synthforge.models.relational.hierarchical import RelationalSynthesizer

        return RelationalSynthesizer(config)
    if key in ("ecg", "timeseries", "signal"):
        from synthforge.models.timeseries.ecg import ECGSynthesizer

        m = ECGSynthesizer(config)
        return m  # type: ignore[return-value,assignment]
    if key in ("diffusion", "diffusegan", "imaging", "mri"):
        from synthforge.models.imaging.diffusion import DiffuseGANSynthesizer

        m2 = DiffuseGANSynthesizer(config)
        return m2  # type: ignore[return-value]
    if key in ("gan", "dcgan"):
        from synthforge.models.imaging.diffusion import PlainGANSynthesizer

        m3 = PlainGANSynthesizer(config)
        return m3  # type: ignore[return-value]
    if key in ("multimodal", "bundle"):
        from synthforge.models.multimodal import MultimodalSynthesizer

        return MultimodalSynthesizer(config)
    raise ModelError(
        f"unknown model {name!r}",
        hint=f"Available: {', '.join(m['name'] for m in available_models())}",
    )


def load_model(path: str | Path) -> BaseSynthesizer:
    root = Path(path)
    card_path = root / "model_card.json"
    model_name = ""
    if card_path.exists():
        try:
            model_name = json.loads(card_path.read_text(encoding="utf-8")).get("model", "")
        except Exception:
            model_name = ""
    if not model_name:
        # infer from artifact signatures
        if (root / "weights.pt").exists() and (root / "preprocessor.pkl").exists():
            model_name = "ctgan"  # caller can override; TVAE stores the same weights file
        elif (root / "copula.npz").exists():
            model_name = "copula"
        elif (root / "baseline.npz").exists():
            model_name = "resample_noise"
        elif (root / "relational.json").exists():
            model_name = "relational"
        elif (root / "ecg.pt").exists():
            model_name = "ecg"
        elif (root / "images.pt").exists():
            model_name = "diffusion"
    if model_name in ("ctgan", "tvae"):
        if model_name == "ctgan":
            from synthforge.models.tabular.ctgan import CTGANSynthesizer

            return CTGANSynthesizer.load(root)
        from synthforge.models.tabular.tvae import TVAESynthesizer

        return TVAESynthesizer.load(root)
    if model_name == "copula":
        from synthforge.models.tabular.copula import GaussianCopulaSynthesizer

        return GaussianCopulaSynthesizer.load(root)
    if model_name == "resample_noise":
        from synthforge.models.tabular.copula import ResampleNoiseSynthesizer

        return ResampleNoiseSynthesizer.load(root)
    if model_name == "relational":
        from synthforge.models.relational.hierarchical import RelationalSynthesizer

        return RelationalSynthesizer.load(root)
    if model_name == "ecg":
        from synthforge.models.timeseries.ecg import ECGSynthesizer

        m = ECGSynthesizer.load(root)
        return m  # type: ignore[return-value]
    if model_name in ("diffusion", "gan"):
        from synthforge.models.imaging.diffusion import DiffuseGANSynthesizer, PlainGANSynthesizer

        m = DiffuseGANSynthesizer.load(root) if model_name == "diffusion" else PlainGANSynthesizer.load(root)  # type: ignore[assignment]
        return m  # type: ignore[return-value]
    if model_name == "multimodal":
        from synthforge.models.multimodal import MultimodalSynthesizer

        return MultimodalSynthesizer.load(root)
    raise ModelError(
        f"cannot determine model type for artifact at {root}",
        hint="Pass the model name explicitly or regenerate the artifact with model_card.json.",
    )
