"""CTGAN from scratch: conditional generator, training-by-sampling, PacGAN, WGAN-GP.

DP mode uses per-sample BCE on the critic (WGAN-GP's gradient penalty is not per-example and is
therefore incompatible with the accounting claim), Poisson subsampling, and the in-repo DP-SGD
engine. The generator is trained only on fake samples and the DP critic, so it is post-processing.
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
from synthforge.core.errors import CancelledError, ModelError, PrivacyError
from synthforge.core.protocol import CancelCheck, FitResult, ProgressCallback, TrainHistory
from synthforge.core.types import Dataset
from synthforge.models.preprocessing import CategoricalTransform, NumericTransform
from synthforge.models.tabular.base import BaseTabularSynthesizer
from synthforge.privacy.dp_sgd import (
    DPSGDConfig,
    PrivacyEngine,
    clip_and_noise,
    make_functional_params,
    per_sample_gradients,
)


class Generator(nn.Module):
    def __init__(self, noise_dim: int, cond_dim: int, data_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim + cond_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, cond], dim=1) if cond.shape[1] else z)


class Critic(nn.Module):
    def __init__(self, input_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CTGANSynthesizer(BaseTabularSynthesizer):
    model_name = "ctgan"

    def __init__(self, config: RunConfig | None = None):
        super().__init__(config)
        self.generator: Generator | None = None
        self.critic: Critic | None = None
        self.noise_dim = 32
        self.hidden_dim = 128
        self.pac = 2
        self.data_dim = 0
        self.condition_dim = 0
        self.device = torch.device("cpu")
        self.dp_result: dict[str, Any] | None = None
        self._blocks: list[dict[str, Any]] = []
        self._cond_groups: list[CategoricalTransform] = []
        self._start_epoch = 0

    # ------------------------------------------------------------------ setup
    def _resolve_device(self) -> torch.device:
        d = self.config.training.device
        if d == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
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
                block: dict[str, Any] = {"kind": "numeric", "name": t.name, "start": t.start}
                if t.missing_indicator:
                    block["null"] = pos
                    pos += 1
                if t.n_components > 1:
                    block["modes"] = (pos, pos + t.n_components)
                    pos += t.n_components
                block["value"] = pos
                self._blocks.append(block)
            elif isinstance(t, CategoricalTransform):
                self._blocks.append(
                    {"kind": "categorical", "name": t.name, "slice": (t.start, t.start + t.dim)}
                )

    def _activate(self, raw: torch.Tensor, *, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        out = raw.clone()
        for b in self._blocks:
            if b["kind"] == "numeric":
                if "null" in b:
                    out[:, b["null"]] = torch.sigmoid(raw[:, b["null"]])
                if "modes" in b:
                    a, c = b["modes"]
                    out[:, a:c] = F.gumbel_softmax(raw[:, a:c], tau=tau, hard=hard)
            else:
                a, c = b["slice"]
                out[:, a:c] = F.gumbel_softmax(raw[:, a:c], tau=tau, hard=hard)
        return out

    def _activate_inference(self, raw: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
        out = raw.clone()
        for b in self._blocks:
            if b["kind"] == "numeric":
                if "null" in b:
                    p = torch.sigmoid(raw[:, b["null"]])
                    out[:, b["null"]] = torch.bernoulli(p, generator=gen)
                if "modes" in b:
                    a, c = b["modes"]
                    probs = torch.softmax(raw[:, a:c], dim=1)
                    out[:, a:c] = F.one_hot(torch.multinomial(probs, 1, generator=gen).squeeze(1), c - a).float()
            else:
                a, c = b["slice"]
                probs = torch.softmax(raw[:, a:c], dim=1)
                out[:, a:c] = F.one_hot(torch.multinomial(probs, 1, generator=gen).squeeze(1), c - a).float()
        return out

    def _sample_conditions(self, k: int, seed: int, *, dp_mode: bool) -> np.ndarray:
        assert self.preprocessor is not None
        rng = np.random.default_rng(seed)
        vec = np.zeros((k, self.condition_dim), dtype=np.float32)
        groups = self._cond_groups
        if not groups or self.condition_dim == 0:
            return vec
        if dp_mode:
            gweights = np.ones(len(groups)) / len(groups)
        else:
            sizes = np.array([len(g.categories) for g in groups], dtype=float)
            gweights = np.log(sizes + 1.0)
            gweights = gweights / gweights.sum()
        which = rng.choice(len(groups), size=k, p=gweights)
        offset = 0
        offsets = []
        for g in groups:
            offsets.append(offset)
            offset += g.dim
        for i, gi in enumerate(which):
            g = groups[int(gi)]
            if dp_mode:
                probs = np.ones(g.dim) / g.dim
            else:
                probs = g.frequencies / max(g.frequencies.sum(), 1e-12)
            cat = int(rng.choice(g.dim, p=probs))
            vec[i, offsets[int(gi)] + cat] = 1.0
        return vec

    def _pack(self, x: torch.Tensor, cond: torch.Tensor, pac: int) -> torch.Tensor:
        aug = torch.cat([x, cond], dim=1) if cond.shape[1] else x
        if pac == 1:
            return aug
        b = aug.shape[0] // pac
        return aug[: b * pac].reshape(b, pac * aug.shape[1])

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
        started = time.time()
        self.config = config
        frame, _ = self._prepare(dataset, config)
        assert self.preprocessor is not None
        self.data_dim = self.preprocessor.dim
        self.condition_dim = self.preprocessor.condition_dim
        self._cond_groups = self.preprocessor.categorical
        self._build_blocks()
        self.hidden_dim = int(config.training.hidden_dim)
        self.noise_dim = int(config.training.latent_dim)
        self.pac = int(config.training.pac)
        self.device = self._resolve_device()
        if config.training.num_threads:
            torch.set_num_threads(int(config.training.num_threads))
        torch.manual_seed(config.seed)
        self.warnings = []
        self.fit_warnings = []

        dp_mode = bool(config.privacy.enabled)
        if dp_mode and self.pac > 1:
            self.fit_warnings.append(
                "PacGAN was disabled in DP mode because each critic input must correspond to one source row "
                "for per-sample DP accounting. DP mode uses a per-example BCE critic objective."
            )
            self.pac = 1
        epochs = int(config.training.epochs)
        steps_per_epoch = max(1, math.ceil(len(frame) / int(config.training.batch_size)))
        max_steps = int(config.training.max_steps or (epochs * steps_per_epoch))
        self.generator = Generator(self.noise_dim, self.condition_dim, self.data_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.pac * (self.data_dim + self.condition_dim), self.hidden_dim).to(self.device)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=float(config.training.lr), betas=(0.5, 0.999), weight_decay=float(config.training.weight_decay))
        opt_c = torch.optim.Adam(self.critic.parameters(), lr=float(config.training.lr), betas=(0.5, 0.999), weight_decay=float(config.training.weight_decay))

        engine: PrivacyEngine | None = None
        training_cap: float | None = None
        if dp_mode:
            training_cap = self._training_epsilon_cap(config)
            dp_config = DPSGDConfig(
                enabled=True,
                clip_norm=float(config.privacy.clip_norm),
                noise_multiplier=config.privacy.noise_multiplier,
                epsilon=training_cap,
                delta=float(config.privacy.delta),
                batch_size=int(config.training.batch_size),
                dataset_size=len(frame),
                steps=max_steps,
                poisson_sampling=bool(config.privacy.poisson_sampling),
                seed=config.seed,
            )
            try:
                engine = PrivacyEngine(dp_config)
            except Exception as exc:
                raise PrivacyError(
                    f"could not initialize the DP accountant: {exc}",
                    hint="Increase epsilon, reduce epochs, or increase batch size.",
                ) from exc
            if training_cap is not None and engine.epsilon() > training_cap * 1.02:
                raise PrivacyError(
                    f"target epsilon {training_cap:.4g} is infeasible for {max_steps} steps at q={dp_config.sampling_rate:.3g}",
                    hint="Reduce epochs/batch steps, increase epsilon, or supply a larger noise multiplier budget.",
                )

        # resume from checkpoint if present
        start_epoch = 0
        ckpt_path = Path(checkpoint_dir) / "checkpoint.pt" if checkpoint_dir else None
        if ckpt_path and ckpt_path.exists():
            try:
                state = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
                self.generator.load_state_dict(state["generator"])
                self.critic.load_state_dict(state["critic"])
                start_epoch = int(state.get("epoch", 0)) + 1
                self.history = TrainHistory.model_validate(state.get("history", {}))
                if engine is not None and state.get("privacy_history"):
                    engine.accountant.history = state["privacy_history"]
                self.fit_warnings.append(f"Resumed from checkpoint at epoch {start_epoch}.")
            except Exception as exc:
                self.fit_warnings.append(f"Checkpoint could not be restored ({exc}); training from scratch.")

        x_np = self.preprocessor.transform(frame)
        x_all = torch.tensor(x_np, dtype=torch.float32)
        n = len(x_all)
        losses: dict[str, list[float]] = {"critic": [], "generator": [], "privacy_epsilon": []}
        gen_ema = float("inf")
        stale = 0
        steps_done = 0
        device = self.device
        pac = self.pac

        for epoch in range(start_epoch, epochs):
            self.generator.train()
            self.critic.train()
            epoch_c, epoch_g = [], []
            steps_this = min(steps_per_epoch, max_steps - steps_done)
            if steps_this <= 0:
                break
            for step in range(steps_this):
                if should_cancel and should_cancel():
                    if ckpt_path:
                        self._save_checkpoint(ckpt_path, epoch, engine)
                    history = self._history(losses, steps_done, epoch + 1, started, early=True)
                    self.history = history
                    raise CancelledError("training cancelled by request")
                if engine is not None and engine.should_stop():
                    self.fit_warnings.append(
                        f"Stopped early at the requested epsilon ({engine.epsilon():.3f}) after {steps_done} steps."
                    )
                    break
                seed = config.seed * 100_000 + epoch * 1000 + step
                if engine is not None:
                    idx = engine.sample_lot(n)
                    real = x_all[idx].to(device)
                    k = len(idx)
                else:
                    k = int(config.training.batch_size)
                    idx_np = np.random.default_rng(seed).integers(0, n, size=k)
                    real = x_all[torch.as_tensor(idx_np)].to(device)
                cond_np = self._sample_conditions(len(real), seed + 17, dp_mode=engine is not None)
                cond = torch.tensor(cond_np, dtype=torch.float32, device=device)
                fake_raw = self.generator(torch.randn(len(real), self.noise_dim, device=device), cond)
                fake = self._activate(fake_raw, tau=1.0, hard=False)
                if engine is None:
                    real_pack = self._pack(real, cond, pac)
                    fake_pack = self._pack(fake, cond, pac)
                    # WGAN-GP critic
                    opt_c.zero_grad(set_to_none=True)
                    real_score = self.critic(real_pack)
                    fake_score = self.critic(fake_pack.detach())
                    alpha = torch.rand(real_pack.shape[0], 1, device=device)
                    interp = (alpha * real_pack + (1 - alpha) * fake_pack).detach().requires_grad_(True)
                    interp_score = self.critic(interp)
                    grads = torch.autograd.grad(
                        outputs=interp_score.sum(), inputs=interp, create_graph=True, retain_graph=True
                    )[0]
                    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
                    c_loss = fake_score.mean() - real_score.mean() + 10.0 * gp
                    c_loss.backward()
                    opt_c.step()
                    # generator
                    opt_g.zero_grad(set_to_none=True)
                    fake_raw2 = self.generator(torch.randn(len(real), self.noise_dim, device=device), cond)
                    fake2 = self._activate(fake_raw2, tau=1.0, hard=False)
                    g_loss = -self.critic(self._pack(fake2, cond, pac)).mean()
                    g_loss.backward()
                    opt_g.step()
                    step_c, step_g = float(c_loss.item()), float(g_loss.item())
                else:
                    # DP critic: per-sample real grads + data-independent fake gradient
                    if self.critic is None:
                        raise ModelError("DP critic not initialized")
                    critic = self.critic
                    params = make_functional_params(critic)

                    def loss_single(p: dict[str, torch.Tensor], single: torch.Tensor) -> torch.Tensor:
                        logit = torch.func.functional_call(critic, p, (single.unsqueeze(0),)).squeeze(0)
                        return F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))

                    real_pack = self._pack(real, cond, 1)
                    per_grads = per_sample_gradients(critic, params, loss_single, real_pack)
                    agg, stats = clip_and_noise(
                        per_grads,
                        clip_norm=float(config.privacy.clip_norm),
                        noise_multiplier=engine.noise_multiplier,
                        expected_lot_size=engine.expected_lot_size(n),
                        generator=engine.generator,
                    )
                    fake_pack = self._pack(fake, cond, 1)
                    fake_logits = self.critic(fake_pack.detach())
                    fake_loss = F.binary_cross_entropy_with_logits(
                        fake_logits, torch.zeros_like(fake_logits), reduction="sum"
                    ) / engine.expected_lot_size(n)
                    fake_loss.backward()
                    for name, p in self.critic.named_parameters():
                        if p.grad is None:
                            p.grad = torch.zeros_like(p)
                        if name in agg:
                            p.grad = p.grad + agg[name].to(p.grad.dtype)
                    opt_c.step()
                    engine.step()
                    steps_done += 1
                    # generator post-processing
                    opt_g.zero_grad(set_to_none=True)
                    fake_raw2 = self.generator(torch.randn(len(real), self.noise_dim, device=device), cond)
                    fake2 = self._activate(fake_raw2, tau=1.0, hard=False)
                    g_loss = -self.critic(self._pack(fake2, cond, 1)).mean()
                    g_loss.backward()
                    opt_g.step()
                    step_c, step_g = float(fake_loss.item()), float(g_loss.item())
                    losses.setdefault("critic_grad_norm", []).append(stats["mean_grad_norm"])
                    losses.setdefault("clipped_fraction", []).append(stats["clipped_fraction"])
                if engine is None:
                    steps_done += 1
                epoch_c.append(step_c)
                epoch_g.append(step_g)
                if engine is not None:
                    losses["privacy_epsilon"].append(engine.epsilon())
                if step % max(1, steps_per_epoch // 8) == 0 and progress:
                        payload: dict[str, Any] = {
                            "phase": "train",
                            "epoch": epoch + 1,
                            "epochs": epochs,
                            "step": steps_done,
                            "max_steps": max_steps,
                            "critic_loss": float(np.mean(epoch_c)) if epoch_c else None,
                            "generator_loss": float(np.mean(epoch_g)) if epoch_g else None,
                            "progress": min(0.98, 0.05 + 0.9 * steps_done / max(max_steps, 1)),
                            "device": str(device),
                        }
                        if engine is not None:
                            payload["epsilon"] = engine.epsilon()
                            payload["noise_multiplier"] = engine.noise_multiplier
                            payload["sampling_rate"] = engine.config.sampling_rate
                        progress(payload)
                if engine is not None and engine.should_stop():
                    break
            c_mean = float(np.mean(epoch_c)) if epoch_c else float("nan")
            g_mean = float(np.mean(epoch_g)) if epoch_g else float("nan")
            losses["critic"].append(c_mean)
            losses["generator"].append(g_mean)
            if np.isfinite(g_mean):
                if g_mean >= gen_ema * 0.999:
                    stale += 1
                else:
                    stale = 0
                    gen_ema = g_mean
            if ckpt_path and ((epoch + 1) % int(config.training.checkpoint_every) == 0):
                self._save_checkpoint(ckpt_path, epoch, engine)
            if engine is not None and engine.should_stop():
                break
            if stale >= int(config.training.early_stop_patience) and epoch >= 2:
                self.fit_warnings.append(f"Early stopping at epoch {epoch + 1} (generator loss plateau).")
                break
            if steps_done >= max_steps:
                break

        self.is_fitted = True
        self.history = self._history(losses, steps_done, epochs, started, early=False)
        self.train_rows = len(frame)
        self.metadata = {
            "device": str(device),
            "data_dim": self.data_dim,
            "condition_dim": self.condition_dim,
            "noise_dim": self.noise_dim,
            "hidden_dim": self.hidden_dim,
            "pac": pac,
            "objective": "DP per-example BCE" if engine is not None else "WGAN-GP + PacGAN",
        }
        if engine is not None:
            dp_result = engine.result()
            self.dp_result = dp_result.to_dict()
            covered = [
                "DP-SGD optimization of the critic over source rows",
                "generator output, by post-processing of the DP critic",
                "DP preprocessing bounds and moments (if dp_preprocessing was enabled)",
            ]
            excluded = [
                "source data profiling and evaluation statistics (they access real data directly; not part of the released model)",
                "empirical attack results (complementary evidence, not a privacy proof)",
                "any non-DP baseline run",
            ]
            if not config.privacy.dp_preprocessing:
                excluded.append("preprocessing statistics (min/max, GMM fits, category frequencies) were non-private")
            self.report = self._privacy_report_common(
                mechanism="DP-SGD on the CTGAN critic (Poisson-subsampled Gaussian)",
                epsilon_training=dp_result.epsilon,
                delta=float(config.privacy.delta),
                steps=dp_result.steps,
                sampling_rate=dp_result.sampling_rate,
                clip_norm=float(config.privacy.clip_norm),
                noise_multiplier=dp_result.noise_multiplier,
                covered=covered,
                excluded=excluded,
                dp_preprocessing=bool(config.privacy.dp_preprocessing),
                composition=[dp_result.to_dict()],
            )
            if training_cap is not None and dp_result.epsilon > training_cap * 1.02:
                self.fit_warnings.append(
                    f"Reported training epsilon {dp_result.epsilon:.3f} exceeded the allocated cap {training_cap:.3f}; "
                    "check the accountant configuration."
                )
        else:
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
            progress({"phase": "done", "progress": 1.0, "epsilon": self.report.epsilon, "step": steps_done})
        return FitResult(
            history=self.history,
            device=str(device),
            n_train_rows=len(frame),
            warnings=self.warnings,
            metadata=self.metadata,
        )

    def _history(self, losses: dict[str, list[float]], steps: int, epochs: int, started: float, *, early: bool) -> TrainHistory:
        return TrainHistory(
            steps=int(steps),
            epochs=int(epochs),
            losses={k: [float(x) for x in v] for k, v in losses.items()},
            early_stopped=early,
            duration_seconds=time.time() - started,
        )

    def _save_checkpoint(self, path: Path, epoch: int, engine: PrivacyEngine | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "generator": self.generator.state_dict() if self.generator else None,
                "critic": self.critic.state_dict() if self.critic else None,
                "epoch": epoch,
                "history": self.history.model_dump(mode="json") if self.history else {},
                "privacy_history": engine.accountant.history if engine else [],
            },
            str(path),
        )

    # ------------------------------------------------------------------ sampling
    def _generate_transformed(self, n: int, condition_vector: np.ndarray, seed: int) -> np.ndarray:
        if self.generator is None:
            raise ModelError("CTGAN generator is not initialized")
        device = self.device
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        cond = torch.tensor(condition_vector, dtype=torch.float32, device=device).view(1, -1)
        outs = []
        with torch.no_grad():
            for start in range(0, n, 2048):
                m = min(2048, n - start)
                z = torch.randn(m, self.noise_dim, generator=gen).to(device)
                c = cond.expand(m, -1) if cond.shape[1] else torch.empty(m, 0, device=device)
                raw = self.generator(z, c)
                outs.append(self._activate_inference(raw, gen).cpu().numpy())
        return np.concatenate(outs, axis=0).astype(np.float32)

    # ------------------------------------------------------------------ persistence
    def _save_state(self, path: Path) -> None:
        if not self.is_fitted:
            raise ModelError("cannot save an unfitted model")
        self._base_save(path)
        torch.save(
            {
                "state": "ctgan",
                "generator": self.generator.state_dict() if self.generator else None,
                "critic": self.critic.state_dict() if self.critic else None,
                "noise_dim": self.noise_dim,
                "hidden_dim": self.hidden_dim,
                "pac": self.pac,
                "data_dim": self.data_dim,
                "condition_dim": self.condition_dim,
                "history": self.history.model_dump(mode="json"),
                "report": self.report.model_dump(mode="json"),
                "metadata": self.metadata,
                "warnings": self.warnings,
                "fit_warnings": self.fit_warnings,
                "table_name": self.table_name,
                "train_rows": self.train_rows,
                "dp_result": self.dp_result,
            },
            str(path / "weights.pt"),
        )

    def _load_state(self, path: Path) -> None:
        self._base_load(path)
        state = torch.load(str(path / "weights.pt"), map_location="cpu", weights_only=False)
        self.noise_dim = int(state["noise_dim"])
        self.hidden_dim = int(state["hidden_dim"])
        self.pac = int(state["pac"])
        self.data_dim = int(state["data_dim"])
        self.condition_dim = int(state["condition_dim"])
        self.generator = Generator(self.noise_dim, self.condition_dim, self.data_dim, self.hidden_dim)
        self.critic = Critic(self.pac * (self.data_dim + self.condition_dim), self.hidden_dim)
        self.generator.load_state_dict(state["generator"])
        self.critic.load_state_dict(state["critic"])
        self._build_blocks()
        self.history = TrainHistory.model_validate(state["history"])
        self.report = type(self.report).model_validate(state["report"])
        self.metadata = state.get("metadata", {})
        self.warnings = state.get("warnings", [])
        self.fit_warnings = state.get("fit_warnings", [])
        self.table_name = state.get("table_name", "table")
        self.train_rows = int(state.get("train_rows", 0))
        self.dp_result = state.get("dp_result")
        self.is_fitted = True

    @classmethod
    def load(cls, path: str | Path) -> CTGANSynthesizer:
        obj = cls()
        obj._load_state(Path(path))
        cfg_path = Path(path) / "config.json"
        if cfg_path.exists():
            obj.config = RunConfig.model_validate_json(cfg_path.read_text(encoding="utf-8"))
        return obj

    def model_card(self) -> dict[str, Any]:
        card = super().model_card()
        card.update(
            {
                "architecture": {
                    "generator": "MLP with conditional vector and Gumbel-softmax outputs",
                    "critic": f"MLP, PacGAN pack={self.pac}, {'BCE (DP mode)' if self.dp_result else 'WGAN-GP'}",
                    "noise_dim": self.noise_dim,
                    "hidden_dim": self.hidden_dim,
                },
                "privacy_claim": self.report.model_dump(mode="json"),
                "dp_accounting": self.dp_result,
                "deviations": [
                    "DP mode replaces WGAN-GP with a per-example BCE critic objective because the gradient penalty is not per-example decomposable.",
                    "DP mode disables PacGAN and training-by-sampling frequencies; conditions are sampled uniformly.",
                    "The generator itself is not noised; privacy follows from post-processing the DP critic.",
                ],
            }
        )
        return card
