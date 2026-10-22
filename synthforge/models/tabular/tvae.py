"""TVAE: a conditional variational autoencoder with a proper ELBO.

The decoder emits raw logits; one-hot / mode blocks are scored with cross-entropy, null indicators with
binary cross-entropy, and continuous values with a Gaussian likelihood (MSE + log-variance term).
The KL term is the analytic Gaussian KL against the standard normal prior.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from synthforge.core.config import RunConfig
from synthforge.core.errors import CancelledError, ModelError
from synthforge.core.protocol import CancelCheck, FitResult, ProgressCallback, TrainHistory
from synthforge.core.types import Dataset
from synthforge.models.preprocessing import CategoricalTransform, NumericTransform
from synthforge.models.tabular.base import BaseTabularSynthesizer


class _Encoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, latent: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.mu(h), self.logvar(h).clamp(-12.0, 8.0)


class _Decoder(nn.Module):
    def __init__(self, latent: int, cond_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent + cond_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=1) if cond.shape[1] else z)


class TVAESynthesizer(BaseTabularSynthesizer):
    model_name = "tvae"

    def __init__(self, config: RunConfig | None = None):
        super().__init__(config)
        self.encoder: _Encoder | None = None
        self.decoder: _Decoder | None = None
        self.data_dim = 0
        self.condition_dim = 0
        self.latent_dim = 32
        self.hidden_dim = 128
        self.device = torch.device("cpu")
        self._blocks: list[dict[str, Any]] = []
        self._loss_weights: dict[str, torch.Tensor] = {}
        self._cond_groups: list[CategoricalTransform] = []

    def _resolve_device(self) -> torch.device:
        d = self.config.training.device
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        if d == "cuda" and not torch.cuda.is_available():
            self.fit_warnings.append("CUDA requested but unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(d)

    def _build_blocks(self) -> None:
        assert self.preprocessor is not None
        self._blocks = []
        for t in self.preprocessor.transforms:
            if isinstance(t, NumericTransform):
                pos = t.start
                b: dict[str, Any] = {"kind": "numeric", "name": t.name}
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
                    {"kind": "categorical", "name": t.name, "slice": (t.start, t.start + t.dim)}
                )

    def _activate(self, raw: torch.Tensor, *, hard: bool = True, generator: torch.Generator | None = None) -> torch.Tensor:
        out = raw.clone()
        for b in self._blocks:
            if b["kind"] == "numeric":
                if "null" in b:
                    out[:, b["null"]] = torch.sigmoid(raw[:, b["null"]])
                if "modes" in b:
                    a, c = b["modes"]
                    out[:, a:c] = F.softmax(raw[:, a:c], dim=1) if not hard else F.one_hot(
                        torch.argmax(raw[:, a:c], dim=1), c - a
                    ).float()
            else:
                a, c = b["slice"]
                out[:, a:c] = F.softmax(raw[:, a:c], dim=1) if not hard else F.one_hot(
                    torch.argmax(raw[:, a:c], dim=1), c - a
                ).float()
        return out

    def _reconstruction(self, out: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for b in self._blocks:
            if b["kind"] == "numeric":
                if "null" in b:
                    losses.append(
                        F.binary_cross_entropy_with_logits(out[:, b["null"]], target[:, b["null"]], reduction="mean")
                    )
                if "modes" in b:
                    a, c = b["modes"]
                    losses.append(
                        F.cross_entropy(out[:, a:c], target[:, a:c].argmax(dim=1), reduction="mean")
                    )
                v = b["value"]
                losses.append(F.mse_loss(out[:, v], target[:, v], reduction="mean"))
            else:
                a, c = b["slice"]
                losses.append(F.cross_entropy(out[:, a:c], target[:, a:c].argmax(dim=1), reduction="mean"))
        return torch.stack(losses).mean() if losses else torch.zeros((), device=out.device)

    def _sample_conditions(self, k: int, seed: int, *, dp_mode: bool) -> np.ndarray:
        assert self.preprocessor is not None
        rng = np.random.default_rng(seed)
        vec = np.zeros((k, self.condition_dim), dtype=np.float32)
        if not self._cond_groups or self.condition_dim == 0:
            return vec
        offsets = []
        off = 0
        for g in self._cond_groups:
            offsets.append(off)
            off += g.dim
        for i in range(k):
            gi = int(rng.integers(0, len(self._cond_groups)))
            g = self._cond_groups[gi]
            probs = np.ones(g.dim) / g.dim if dp_mode else g.frequencies / max(g.frequencies.sum(), 1e-12)
            vec[i, offsets[gi] + int(rng.choice(g.dim, p=probs))] = 1.0
        return vec

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
        self.data_dim = self.preprocessor.dim
        self.condition_dim = self.preprocessor.condition_dim
        self._cond_groups = self.preprocessor.categorical
        self._build_blocks()
        self.hidden_dim = int(config.training.hidden_dim)
        self.latent_dim = int(config.training.latent_dim)
        self.device = self._resolve_device()
        if config.training.num_threads:
            torch.set_num_threads(int(config.training.num_threads))
        torch.manual_seed(config.seed)
        self.fit_warnings = []
        self.warnings = []
        if config.privacy.enabled:
            raise ModelError(
                "TVAE does not have an accounted DP training path in this build",
                hint="Use model='ctgan' with DP enabled. A non-private TVAE is available with privacy disabled.",
            )
        self.encoder = _Encoder(self.data_dim + self.condition_dim, self.hidden_dim, self.latent_dim).to(self.device)
        self.decoder = _Decoder(self.latent_dim, self.condition_dim, self.hidden_dim, self.data_dim).to(self.device)
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=float(config.training.lr),
            weight_decay=float(config.training.weight_decay),
        )
        x_np = self.preprocessor.transform(frame)
        x_all = torch.tensor(x_np, dtype=torch.float32)
        n = len(x_all)
        batch = int(config.training.batch_size)
        steps_per_epoch = max(1, math.ceil(n / batch))
        max_steps = int(config.training.max_steps or config.training.epochs * steps_per_epoch)
        losses: dict[str, list[float]] = {"total": [], "reconstruction": [], "kl": []}
        steps_done = 0
        start_epoch = 0
        ckpt = Path(checkpoint_dir) / "checkpoint.pt" if checkpoint_dir else None
        if ckpt and ckpt.exists():
            try:
                state = torch.load(str(ckpt), map_location=self.device, weights_only=False)
                self.encoder.load_state_dict(state["encoder"])
                self.decoder.load_state_dict(state["decoder"])
                start_epoch = int(state.get("epoch", 0)) + 1
                self.history = TrainHistory.model_validate(state.get("history", {}))
                self.fit_warnings.append(f"Resumed from checkpoint at epoch {start_epoch}.")
            except Exception as exc:
                self.fit_warnings.append(f"Checkpoint could not be restored ({exc}); training from scratch.")
        for epoch in range(start_epoch, int(config.training.epochs)):
            self.encoder.train()
            self.decoder.train()
            ep_total, ep_rec, ep_kl = [], [], []
            steps_this = min(steps_per_epoch, max_steps - steps_done)
            if steps_this <= 0:
                break
            for step in range(steps_this):
                if should_cancel and should_cancel():
                    if ckpt:
                        self._checkpoint(ckpt, epoch)
                    self.history = self._history(losses, steps_done, epoch + 1, started, True)
                    raise CancelledError("training cancelled by request")
                seed = config.seed * 1_000_003 + epoch * 1009 + step
                idx = torch.as_tensor(np.random.default_rng(seed).integers(0, n, size=batch))
                real = x_all[idx].to(self.device)
                cond_np = self._sample_conditions(len(real), seed + 3, dp_mode=False)
                cond = torch.tensor(cond_np, dtype=torch.float32, device=self.device)
                xin = torch.cat([real, cond], dim=1) if self.condition_dim else real
                mu, logvar = self.encoder(xin)
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
                out = self.decoder(z, cond)
                rec = self._reconstruction(out, real)
                kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean() / self.data_dim
                loss = rec + 0.15 * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()), 10.0
                )
                optimizer.step()
                steps_done += 1
                ep_total.append(float(loss.item()))
                ep_rec.append(float(rec.item()))
                ep_kl.append(float(kl.item()))
                if progress and step % max(1, steps_per_epoch // 8) == 0:
                    progress(
                        {
                            "phase": "train",
                            "epoch": epoch + 1,
                            "epochs": int(config.training.epochs),
                            "step": steps_done,
                            "max_steps": max_steps,
                            "loss": float(np.mean(ep_total)),
                            "reconstruction": float(np.mean(ep_rec)),
                            "kl": float(np.mean(ep_kl)),
                            "progress": min(0.98, 0.05 + 0.92 * steps_done / max(max_steps, 1)),
                            "device": str(self.device),
                        }
                    )
            losses["total"].append(float(np.mean(ep_total)))
            losses["reconstruction"].append(float(np.mean(ep_rec)))
            losses["kl"].append(float(np.mean(ep_kl)))
            if ckpt and (epoch + 1) % int(config.training.checkpoint_every) == 0:
                self._checkpoint(ckpt, epoch)
            if steps_done >= max_steps:
                break
        self.is_fitted = True
        self.history = self._history(losses, steps_done, int(config.training.epochs), started, False)
        self.train_rows = len(frame)
        self.metadata = {
            "architecture": "conditional VAE",
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "data_dim": self.data_dim,
            "condition_dim": self.condition_dim,
            "objective": "ELBO: reconstruction + 0.15 * KL (per-dimension)",
            "device": str(self.device),
        }
        self.report = self._privacy_report_common(
            mechanism="non-private",
            epsilon_training=None,
            delta=None,
            steps=steps_done,
            sampling_rate=0.0,
            clip_norm=None,
            noise_multiplier=None,
            covered=[],
            excluded=["No differential privacy: this run provides no formal privacy guarantee."],
            dp_preprocessing=False,
        )
        self.warnings = list(dict.fromkeys(self.warnings + self.fit_warnings))
        if progress:
            progress({"phase": "done", "progress": 1.0, "epsilon": None, "step": steps_done})
        return FitResult(
            history=self.history,
            device=str(self.device),
            n_train_rows=len(frame),
            warnings=self.warnings,
            metadata=self.metadata,
        )

    def _checkpoint(self, path: Path, epoch: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "encoder": self.encoder.state_dict() if self.encoder else None,
                "decoder": self.decoder.state_dict() if self.decoder else None,
                "epoch": epoch,
                "history": self.history.model_dump(mode="json") if self.history else {},
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

    def _generate_transformed(self, n: int, condition_vector: np.ndarray, seed: int) -> np.ndarray:
        if self.decoder is None:
            raise ModelError("TVAE decoder is not initialized")
        gen = torch.Generator().manual_seed(seed)
        cond = torch.tensor(condition_vector, dtype=torch.float32, device=self.device).view(1, -1)
        outs = []
        with torch.no_grad():
            for start in range(0, n, 2048):
                m = min(2048, n - start)
                z = torch.randn(m, self.latent_dim, generator=gen).to(self.device)
                c = cond.expand(m, -1) if cond.shape[1] else torch.empty(m, 0, device=self.device)
                outs.append(self._activate(self.decoder(z, c), hard=True).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    def _save_state(self, path: Path) -> None:
        if not self.is_fitted:
            raise ModelError("cannot save an unfitted model")
        self._base_save(path)
        torch.save(
            {
                "encoder": self.encoder.state_dict() if self.encoder else None,
                "decoder": self.decoder.state_dict() if self.decoder else None,
                "data_dim": self.data_dim,
                "condition_dim": self.condition_dim,
                "latent_dim": self.latent_dim,
                "hidden_dim": self.hidden_dim,
                "history": self.history.model_dump(mode="json"),
                "report": self.report.model_dump(mode="json"),
                "metadata": self.metadata,
                "warnings": self.warnings,
                "fit_warnings": self.fit_warnings,
                "table_name": self.table_name,
                "train_rows": self.train_rows,
            },
            str(path / "weights.pt"),
        )

    def _load_state(self, path: Path) -> None:
        self._base_load(path)
        state = torch.load(str(path / "weights.pt"), map_location="cpu", weights_only=False)
        self.data_dim = int(state["data_dim"])
        self.condition_dim = int(state["condition_dim"])
        self.latent_dim = int(state["latent_dim"])
        self.hidden_dim = int(state["hidden_dim"])
        self.encoder = _Encoder(self.data_dim + self.condition_dim, self.hidden_dim, self.latent_dim)
        self.decoder = _Decoder(self.latent_dim, self.condition_dim, self.hidden_dim, self.data_dim)
        self.encoder.load_state_dict(state["encoder"])
        self.decoder.load_state_dict(state["decoder"])
        self._build_blocks()
        self.history = TrainHistory.model_validate(state["history"])
        self.report = type(self.report).model_validate(state["report"])
        self.metadata = state.get("metadata", {})
        self.warnings = state.get("warnings", [])
        self.fit_warnings = state.get("fit_warnings", [])
        self.table_name = state.get("table_name", "table")
        self.train_rows = int(state.get("train_rows", 0))
        self.is_fitted = True

    @classmethod
    def load(cls, path: str | Path) -> TVAESynthesizer:
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
                    "type": "conditional variational autoencoder",
                    "latent_dim": self.latent_dim,
                    "hidden_dim": self.hidden_dim,
                    "elbo": "Gaussian reconstruction (mixed-type) + 0.15 * analytic KL",
                },
                "privacy_claim": self.report.model_dump(mode="json"),
                "deviations": ["TVAE in this build has no accounted DP path; it refuses DP-enabled runs."],
            }
        )
        return card


__all__ = ["TVAESynthesizer"]
