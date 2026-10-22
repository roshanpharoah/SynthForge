"""Conditional ECG synthesiser: multi-lead convolutional generator over discretised features.

Rows are encoded as (leads, samples) float arrays. The generator emits raw values per lead/sample; a
small critic scores real vs fake. Class labels (normal, brady, tachy, afib_like, pvc_like,
st_elevation_like) act as conditioning inputs, enabling targeted, rare-class augmentation.

Every cohort produced here is validated with the physiological feature extractor so the platform can
report, from a real run, whether the generated beats are plausible engineering-wise. This is not a
clinical claim.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from synthforge.core.config import RunConfig
from synthforge.core.errors import CancelledError, ModelError
from synthforge.core.protocol import CancelCheck, FitResult, ProgressCallback, TrainHistory
from synthforge.core.types import Dataset, SignalCohort
from synthforge.media.physio import extract_ecg_features, quality_check

CLASSES = ["normal", "brady", "tachy", "afib_like", "pvc_like", "st_elevation_like"]
MAX_LEADS = 32


def _class_vector(labels: pd.DataFrame, column: str, classes: list[str]) -> np.ndarray:
    out = np.zeros((len(labels), len(classes)), dtype=np.float32)
    if column in labels.columns:
        for i, v in enumerate(labels[column].astype(str)):
            if v in classes:
                out[i, classes.index(v)] = 1.0
            else:
                out[i, 0] = 1.0
    else:
        out[:, 0] = 1.0
    return out


class _Gen(nn.Module):
    def __init__(self, noise: int, cond: int, leads: int, samples: int, hidden: int, base: int = 32):
        super().__init__()
        self.leads = leads
        self.samples = samples
        self.seed_len = max(8, samples // 32)
        self.fc = nn.Linear(noise + cond, hidden * self.seed_len)
        self.hidden = hidden
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(hidden, hidden // 2, 4, 2, 1),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.ConvTranspose1d(hidden // 2, hidden // 4, 4, 2, 1),
            nn.BatchNorm1d(hidden // 4),
            nn.ReLU(),
            nn.Conv1d(hidden // 4, leads, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        h = self.fc(z).view(n, self.hidden, self.seed_len)
        out = self.conv(h)
        if out.shape[-1] != self.samples:
            out = torch.nn.functional.interpolate(out, size=self.samples, mode="linear", align_corners=False)
        return out


class _Disc(nn.Module):
    def __init__(self, leads: int, samples: int, hidden: int, cond: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(leads, hidden // 4, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden // 4, hidden // 2, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(hidden // 2, hidden, 5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden + cond, 1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.net(x).squeeze(-1)
        return self.head(torch.cat([h, cond], dim=1))


class ECGSynthesizer:
    model_name = "ecg"
    modality = "timeseries"

    def __init__(self, config: RunConfig | None = None):
        self.config = config or RunConfig()
        self.generator: _Gen | None = None
        self.discriminator: _Disc | None = None
        self.history = TrainHistory()
        self.report = None
        self.warnings: list[str] = []
        self.metadata: dict[str, Any] = {}
        self.is_fitted = False
        self.classes = list(CLASSES)
        self.label_columns: list[str] = []
        self.label_stats: dict[str, pd.Series] = {}
        self.leads = 12
        self.samples = 500
        self.fs = 250
        self.mean = 0.0
        self.std = 1.0
        self.device = torch.device("cpu")
        self.quality: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cohort_from_dataset(dataset: Dataset) -> SignalCohort:
        if dataset.signals is None:
            raise ModelError(
                "the ECG model requires signal data",
                hint="Ingest a WFDB/EDF/CSV ECG source first (synthforge ingest --format ecg).",
            )
        return dataset.signals

    def _prepare(self, cohort: SignalCohort, config: RunConfig) -> np.ndarray:
        x = np.asarray(cohort.signals, dtype=np.float32)
        if x.ndim != 3:
            raise ModelError("signals must be (n, leads, samples)")
        if x.shape[0] < 8:
            raise ModelError("at least 8 ECG records are required", hint="Ingest more records.")
        target_len = int(config.training.signal_length)
        if x.shape[2] != target_len:
            idx = np.linspace(0, x.shape[2] - 1, target_len).round().astype(int)
            x = x[:, :, idx]
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        self.mean = float(np.mean(x))
        self.std = float(np.std(x) or 1.0)
        self.leads = x.shape[1]
        self.samples = x.shape[2]
        return (x - self.mean) / self.std

    def _cond(self, n: int, cls: np.ndarray, extra: np.ndarray | None = None, seed: int = 0) -> torch.Tensor:
        _ = seed  # deterministic condition vector; no randomness needed
        parts = [cls]
        if extra is not None and extra.shape[1]:
            parts.append(extra)
        return torch.tensor(np.concatenate(parts, axis=1), dtype=torch.float32)

    def _label_matrix(
        self, labels: pd.DataFrame, *, for_sampling: bool, seed: int = 0, n: int | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        n = int(n) if n is not None else len(labels)
        rng = np.random.default_rng(seed)
        cls_counts = labels["class"].astype(str).value_counts() if "class" in labels.columns else pd.Series(dtype=int)
        probs = np.array([cls_counts.get(c, 0) for c in self.classes], dtype=float)
        if probs.sum() <= 0:
            probs = np.ones(len(self.classes))
        probs = probs / probs.sum()
        cls_idx = rng.choice(len(self.classes), size=n, p=probs) if for_sampling else np.array(
            [
                self.classes.index(v) if v in self.classes else 0
                for v in (labels["class"].astype(str) if "class" in labels.columns else pd.Series(["normal"] * n))
            ]
        )
        cls_vec = np.zeros((n, len(self.classes)), dtype=np.float32)
        cls_vec[np.arange(n), cls_idx] = 1.0
        extras = []
        for col in self.label_columns:
            stats = self.label_stats[col]
            if for_sampling:
                vals = stats.sample(n, replace=True, random_state=int(rng.integers(0, 1 << 31))).to_numpy()
            else:
                vals = pd.to_numeric(labels[col], errors="coerce").fillna(stats.mean()).to_numpy()
            mu, sd = float(stats.mean()), float(stats.std() or 1.0)
            extras.append(((vals - mu) / sd).astype(np.float32)[:, None])
        extra = np.concatenate(extras, axis=1) if extras else np.zeros((n, 0), dtype=np.float32)
        return cls_vec, extra

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        dataset: Dataset,
        config: RunConfig,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult:
        from synthforge.core.protocol import PrivacyReport

        started = time.time()
        self.config = config
        cohort = self._cohort_from_dataset(dataset)
        if config.privacy.enabled:
            self.warnings.append(
                "The ECG generator has no accounted DP path; the run is recorded as non-private."
            )
        x_all = self._prepare(cohort, config)
        n = x_all.shape[0]
        labels = cohort.labels.reset_index(drop=True)
        for col in ("age", "hypertension"):
            if col in labels.columns:
                self.label_columns.append(col)
                self.label_stats[col] = pd.to_numeric(labels[col], errors="coerce").dropna()
        self.device = torch.device("cuda" if (config.training.device in ("auto", "cuda") and torch.cuda.is_available()) else "cpu")
        noise = int(config.training.latent_dim)
        hidden = int(config.training.hidden_dim)
        cond_dim = len(self.classes) + len(self.label_columns)
        data = torch.tensor(x_all, dtype=torch.float32)
        self.generator = _Gen(noise, cond_dim, self.leads, self.samples, hidden).to(self.device)
        self.discriminator = _Disc(self.leads, self.samples, hidden, cond_dim).to(self.device)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=float(config.training.lr), betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=float(config.training.lr), betas=(0.5, 0.999))
        torch.manual_seed(config.seed)
        batch = min(int(config.training.batch_size), n)
        steps_per_epoch = max(1, n // batch)
        max_steps = int(config.training.max_steps or config.training.epochs * steps_per_epoch)
        losses: dict[str, list[float]] = {"discriminator": [], "generator": []}
        cls_np, extra_np = self._label_matrix(labels, for_sampling=False)
        cls_t = torch.tensor(cls_np, dtype=torch.float32)
        extra_t = torch.tensor(extra_np, dtype=torch.float32)
        steps_done = 0
        for epoch in range(int(config.training.epochs)):
            ep_d, ep_g = [], []
            steps_this = min(steps_per_epoch, max_steps - steps_done)
            if steps_this <= 0:
                break
            for step in range(steps_this):
                if should_cancel and should_cancel():
                    self.history = self._history(losses, steps_done, epoch + 1, started, True)
                    raise CancelledError("training cancelled by request")
                seed = config.seed * 7919 + epoch * 131 + step
                rng = np.random.default_rng(seed)
                idx = torch.as_tensor(rng.integers(0, n, size=batch))
                real = data[idx].to(self.device)
                cond = torch.cat([cls_t[idx], extra_t[idx]], dim=1).to(self.device)
                z = torch.randn(batch, noise, device=self.device)
                fake = self.generator(torch.cat([z, cond], dim=1))
                # discriminator
                opt_d.zero_grad(set_to_none=True)
                real_score = self.discriminator(real, cond)
                fake_score = self.discriminator(fake.detach(), cond)
                gp_alpha = torch.rand(batch, 1, 1, device=self.device)
                interp = (gp_alpha * real + (1 - gp_alpha) * fake.detach()).requires_grad_(True)
                iscore = self.discriminator(interp, cond)
                grads = torch.autograd.grad(iscore.sum(), interp, create_graph=True)[0]
                gp = ((grads.flatten(1).norm(2, dim=1) - 1) ** 2).mean()
                d_loss = fake_score.mean() - real_score.mean() + 8.0 * gp
                d_loss.backward()
                opt_d.step()
                # generator
                opt_g.zero_grad(set_to_none=True)
                z2 = torch.randn(batch, noise, device=self.device)
                fake2 = self.generator(torch.cat([z2, cond], dim=1))
                g_loss = -self.discriminator(fake2, cond).mean()
                g_loss.backward()
                opt_g.step()
                steps_done += 1
                ep_d.append(float(d_loss.item()))
                ep_g.append(float(g_loss.item()))
                if progress and step % max(1, steps_per_epoch // 8) == 0:
                    progress(
                        {
                            "phase": "train",
                            "epoch": epoch + 1,
                            "epochs": int(config.training.epochs),
                            "step": steps_done,
                            "max_steps": max_steps,
                            "discriminator_loss": float(np.mean(ep_d)),
                            "generator_loss": float(np.mean(ep_g)),
                            "progress": min(0.98, 0.05 + 0.92 * steps_done / max(max_steps, 1)),
                            "device": str(self.device),
                        }
                    )
            losses["discriminator"].append(float(np.mean(ep_d)))
            losses["generator"].append(float(np.mean(ep_g)))
            if steps_done >= max_steps:
                break
            if checkpoint_dir:
                self._checkpoint(Path(checkpoint_dir) / "checkpoint.pt", epoch, noise, hidden, cond_dim)
        self.is_fitted = True
        self.history = self._history(losses, steps_done, int(config.training.epochs), started, False)
        self.metadata = {
            "architecture": "conditional 1D transposed-conv generator + WGAN-GP critic",
            "leads": self.leads,
            "samples": self.samples,
            "sampling_rate": cohort.sampling_rate,
            "classes": self.classes,
            "label_columns": self.label_columns,
            "device": str(self.device),
        }
        self.report = PrivacyReport(
            mechanism="non-private",
            statement="Non-private run: the ECG generator is trained directly on real waveforms; no DP guarantee.",
            excluded=["No differential privacy."],
            covered=[],
        )
        if progress:
            progress({"phase": "done", "progress": 1.0, "epsilon": None, "step": steps_done})
        return FitResult(history=self.history, device=str(self.device), n_train_rows=n, warnings=self.warnings, metadata=self.metadata)

    def _checkpoint(self, path: Path, epoch: int, noise: int, hidden: int, cond_dim: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict() if self.generator else None,
                "discriminator": self.discriminator.state_dict() if self.discriminator else None,
                "epoch": epoch,
                "noise": noise,
                "hidden": hidden,
                "cond_dim": cond_dim,
                "history": self.history.model_dump(mode="json"),
            },
            str(path),
        )

    def _history(self, losses: dict[str, list[float]], steps: int, epochs: int, started: float, early: bool) -> TrainHistory:
        return TrainHistory(
            steps=int(steps),
            epochs=int(epochs),
            losses={k: [float(x) for x in v] for k, v in losses.items()},
            early_stopped=early,
            duration_seconds=time.time() - started,
        )

    # ------------------------------------------------------------------ sampling
    def sample(
        self,
        n: int,
        *,
        conditions: dict[str, Any] | None = None,
        seed: int | None = None,
        label_frame: pd.DataFrame | None = None,
    ) -> Dataset:
        if not self.is_fitted or self.generator is None:
            raise ModelError("ECG generator is not fitted")
        conditions = dict(conditions or {})
        use_seed = int(seed if seed is not None else self.config.seed + 1)
        wanted_class = conditions.get("class")
        cls_idx = np.full(n, 0, dtype=int)
        if wanted_class is not None:
            if str(wanted_class) not in self.classes:
                raise ModelError(
                    f"unknown ECG class {wanted_class!r}",
                    hint=f"Available: {', '.join(self.classes)}",
                )
            cls_idx[:] = self.classes.index(str(wanted_class))
        else:
            if label_frame is not None and "class" in label_frame.columns:
                cls_vec, extra = self._label_matrix(label_frame.reset_index(drop=True), for_sampling=True, seed=use_seed, n=n)
            else:
                cls_vec, extra = self._label_matrix(pd.DataFrame(), for_sampling=True, seed=use_seed, n=n)
            cls_idx = cls_vec.argmax(axis=1)
        cls_vec = np.zeros((n, len(self.classes)), dtype=np.float32)
        cls_vec[np.arange(n), cls_idx] = 1.0
        if label_frame is not None and "class" in label_frame.columns:
            _, extra_f = self._label_matrix(label_frame.reset_index(drop=True), for_sampling=True, seed=use_seed, n=n)
        else:
            _, extra_f = self._label_matrix(pd.DataFrame(), for_sampling=True, seed=use_seed, n=n)
        extra = extra_f
        cond = torch.tensor(np.concatenate([cls_vec, extra], axis=1), dtype=torch.float32, device=self.device)
        noise = int(self.config.training.latent_dim)
        outs = []
        gen = torch.Generator().manual_seed(use_seed)
        with torch.no_grad():
            for start in range(0, n, 256):
                m = min(256, n - start)
                z = torch.randn(m, noise, generator=gen).to(self.device)
                raw = self.generator(torch.cat([z, cond[start : start + m]], dim=1))
                outs.append(raw.cpu().numpy())
        signals = np.concatenate(outs, axis=0)[:, : self.leads, : self.samples]
        signals = (signals * self.std + self.mean).astype(np.float32)
        signals = np.clip(signals, -5.0, 5.0)
        labels = pd.DataFrame(
            {
                "subject_id": [f"SYN-ECG{i:05d}" for i in range(n)],
                "class": [self.classes[i] for i in cls_idx],
                "synthetic": True,
            }
        )
        for j, col in enumerate(self.label_columns):
            stats = self.label_stats[col]
            labels[col] = (extra[:, j] * float(stats.std() or 1.0) + float(stats.mean())).round(2)
        quality = quality_check(signals, int(self.metadata.get("sampling_rate", 250)))
        features = extract_ecg_features(signals, int(self.metadata.get("sampling_rate", 250)))
        self.quality = {
            "quality": {k: v for k, v in quality.items() if k != "per_record"},
            "features": {
                "hr_bpm_median": float(features["hr_bpm"].median()),
                "qrs_width_ms_median": float(features["qrs_width_ms"].median()),
                "n_beats_median": float(features["n_beats"].median()),
            },
        }
        return Dataset(
            name="ecg_synthetic",
            kind="timeseries",
            signals=SignalCohort(
                signals=signals,
                sampling_rate=int(self.metadata.get("sampling_rate", 250)),
                lead_names=list(self.metadata.get("lead_names", []) or [f"lead_{i}" for i in range(self.leads)]),
                labels=labels,
                quality=pd.DataFrame(quality["per_record"]),
            ),
            synthetic=True,
            provenance={
                "model": "ecg",
                "seed": use_seed,
                "n": n,
                "conditions": conditions,
                "physiology": self.quality,
                "note": "Synthetic waveforms from a fitted generator; not clinically interpreted.",
            },
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        if not self.is_fitted:
            raise ModelError("cannot save an unfitted model")
        torch.save(
            {
                "generator": self.generator.state_dict() if self.generator else None,
                "discriminator": self.discriminator.state_dict() if self.discriminator else None,
                "leads": self.leads,
                "samples": self.samples,
                "mean": self.mean,
                "std": self.std,
                "classes": self.classes,
                "label_columns": self.label_columns,
                "label_stats": {k: v.to_dict() for k, v in self.label_stats.items()},
                "metadata": self.metadata,
                "history": self.history.model_dump(mode="json"),
                "quality": self.quality,
                "warnings": self.warnings,
            },
            str(root / "ecg.pt"),
        )
        (root / "config.json").write_text(self.config.model_dump_json(indent=2), encoding="utf-8")
        if self.report is not None:
            (root / "privacy.json").write_text(self.report.model_dump_json(indent=2), encoding="utf-8")
        (root / "model_card.json").write_text(
            __import__("json").dumps(self.model_card(), indent=2, default=str), encoding="utf-8"
        )
        return root

    @classmethod
    def load(cls, path: str | Path) -> ECGSynthesizer:
        root = Path(path)
        obj = cls()
        state = torch.load(str(root / "ecg.pt"), map_location="cpu", weights_only=False)
        obj.leads = int(state["leads"])
        obj.samples = int(state["samples"])
        obj.mean = float(state["mean"])
        obj.std = float(state["std"])
        obj.classes = list(state["classes"])
        obj.label_columns = list(state["label_columns"])
        obj.label_stats = {k: pd.Series(v) for k, v in state["label_stats"].items()}
        obj.metadata = state.get("metadata", {})
        obj.history = TrainHistory.model_validate(state["history"])
        obj.quality = state.get("quality", {})
        obj.warnings = state.get("warnings", [])
        cfg = root / "config.json"
        if cfg.exists():
            obj.config = RunConfig.model_validate_json(cfg.read_text(encoding="utf-8"))
        from synthforge.core.protocol import PrivacyReport

        priv = root / "privacy.json"
        if priv.exists():
            obj.report = PrivacyReport.model_validate_json(priv.read_text(encoding="utf-8"))
        cond_dim = len(obj.classes) + len(obj.label_columns)
        hidden = int(obj.config.training.hidden_dim)
        noise = int(obj.config.training.latent_dim)
        obj.generator = _Gen(noise, cond_dim, obj.leads, obj.samples, hidden)
        obj.generator.load_state_dict(state["generator"])
        obj.discriminator = _Disc(obj.leads, obj.samples, hidden, cond_dim)
        if state.get("discriminator") is not None:
            obj.discriminator.load_state_dict(state["discriminator"])
        obj.is_fitted = True
        return obj

    def privacy_report(self):  # type: ignore[no-untyped-def]
        from synthforge.core.protocol import PrivacyReport

        return self.report or PrivacyReport(mechanism="none")

    def model_card(self) -> dict[str, Any]:
        return {
            "model": "ecg",
            "modality": "timeseries",
            "version": "0.1.0",
            "intended_use": "Synthetic waveform generation for development, testing and signal-processing research.",
            "out_of_scope": [
                "Clinical interpretation or diagnosis",
                "Regulatory submission or device validation",
                "Any claim that these waveforms are anatomically exact",
            ],
            "classes": self.classes,
            "training": self.history.model_dump(mode="json"),
            "privacy_claim": (self.report.model_dump(mode="json") if self.report else {}),
            "limitations": self.warnings,
            "metadata": self.metadata,
        }


__all__ = ["ECGSynthesizer", "CLASSES"]
