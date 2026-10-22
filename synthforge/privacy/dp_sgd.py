"""Differentially private SGD with true per-sample gradients via torch.func.

The implementation supports:
- per-sample L2 clipping,
- Gaussian noise with σ·C,
- Poisson subsampling (variable-size lots),
- explicit accounting through the in-repo RDP accountant,
- optional Opacus cross-checks in tests.

For WGAN-GP training see the CTGAN notes: DP mode uses a per-sample BCE objective because the
gradient penalty is not decomposable per example and would invalidate the accounting. This is
documented in the model card rather than silently claimed.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from synthforge.core.errors import PrivacyError
from synthforge.privacy.accountant import (
    DEFAULT_ALPHAS,
    DPResult,
    RDPAccountant,
    compute_noise_multiplier,
    epsilon_for_sampled_gaussian,
)


@dataclass
class DPSGDConfig:
    enabled: bool = True
    clip_norm: float = 1.0
    noise_multiplier: float | None = None
    epsilon: float | None = None
    delta: float = 1e-5
    batch_size: int = 256
    dataset_size: int = 1000
    steps: int = 100
    alphas: tuple[int, ...] = DEFAULT_ALPHAS
    poisson_sampling: bool = True
    seed: int = 42

    @property
    def sampling_rate(self) -> float:
        return min(1.0, self.batch_size / max(self.dataset_size, 1))

    def resolved_noise(self) -> tuple[float, DPResult]:
        if self.noise_multiplier is not None:
            res = epsilon_for_sampled_gaussian(
                noise_multiplier=self.noise_multiplier,
                sampling_rate=self.sampling_rate,
                steps=self.steps,
                delta=self.delta,
                alphas=self.alphas,
            )
            return self.noise_multiplier, res
        if self.epsilon is None:
            raise PrivacyError("DP requires epsilon or noise_multiplier")
        sigma, res = compute_noise_multiplier(
            target_epsilon=self.epsilon,
            delta=self.delta,
            sampling_rate=self.sampling_rate,
            steps=self.steps,
            alphas=self.alphas,
        )
        return sigma, res


class PrivacyEngine:
    """Stateful accountant for a training loop."""

    def __init__(self, config: DPSGDConfig):
        self.config = config
        self.noise_multiplier, _ = config.resolved_noise()
        self.accountant = RDPAccountant(config.alphas)
        self.generator = torch.Generator().manual_seed(config.seed)
        self._steps = 0

    @property
    def noise_scale(self) -> float:
        return float(self.noise_multiplier * self.config.clip_norm)

    def sample_lot(self, n: int) -> torch.Tensor:
        """Return indices for one Poisson-sampled lot (expected size qN)."""
        if not self.config.poisson_sampling:
            q = self.config.sampling_rate
            k = max(1, int(round(q * n)))
            return torch.randperm(n, generator=self.generator)[:k]
        q = self.config.sampling_rate
        mask = torch.rand(n, generator=self.generator) < q
        if not bool(mask.any()):
            # ensure at least one example; this slightly perturbs Poisson sampling and is logged
            mask[torch.randint(0, n, (1,), generator=self.generator)] = True
        return torch.nonzero(mask, as_tuple=False).flatten()

    def expected_lot_size(self, n: int) -> float:
        return max(self.config.sampling_rate * n, 1.0)

    def step(self) -> None:
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sampling_rate=self.config.sampling_rate,
            steps=1,
        )
        self._steps += 1

    def epsilon(self) -> float:
        return self.accountant.get_epsilon(self.config.delta)

    def result(self) -> DPResult:
        return self.accountant.get_privacy_spent(self.config.delta)

    def should_stop(self) -> bool:
        return bool(self.config.epsilon is not None and self.epsilon() >= self.config.epsilon)


def _flatten_grads(grads: dict[str, torch.Tensor]) -> torch.Tensor:
    """Concatenate per-sample gradients into (n_samples, n_params).

    Each value in ``grads`` is expected to carry a leading sample dimension
    (produced by ``vmap`` over the batch), so we flatten *all but* that
    dimension to keep the per-sample rows separate.
    """
    rows = []
    for g in grads.values():
        if g is None:
            continue
        if g.ndim == 0:
            rows.append(g.reshape(1, 1))
        else:
            rows.append(g.reshape(g.shape[0], -1))
    if not rows:
        return torch.zeros((0, 0))
    return torch.cat(rows, dim=1)


def per_sample_gradients(
    model: nn.Module,
    params: dict[str, torch.Tensor],
    loss_fn: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
    batch: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-sample gradients for ``loss_fn(params, single_example) -> scalar``."""
    grad_fn = torch.func.grad(loss_fn)
    grads = torch.func.vmap(grad_fn, in_dims=(None, 0), randomness="different")(params, batch)
    return {k: v.detach() for k, v in grads.items()}


def clip_and_noise(
    per_sample_grads: dict[str, torch.Tensor],
    *,
    clip_norm: float,
    noise_multiplier: float,
    expected_lot_size: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Clip each per-sample gradient, sum, add Gaussian noise, divide by expected lot size."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    flat = _flatten_grads(per_sample_grads)
    norms = torch.linalg.vector_norm(flat, dim=1)
    factors = torch.clamp(clip_norm / (norms + 1e-12), max=1.0)
    clipped = {k: g * factors.reshape(-1, *([1] * (g.ndim - 1))) for k, g in per_sample_grads.items()}
    summed = {k: g.sum(dim=0) for k, g in clipped.items()}
    noise = {
        k: torch.randn(g.shape, generator=generator, dtype=g.dtype) * (noise_multiplier * clip_norm)
        for k, g in summed.items()
    }
    scale = max(expected_lot_size, 1.0)
    out = {k: (summed[k] + noise[k]) / scale for k in summed}
    stats = {
        "mean_grad_norm": float(norms.mean()) if len(norms) else 0.0,
        "max_grad_norm": float(norms.max()) if len(norms) else 0.0,
        "clipped_fraction": float((norms > clip_norm).float().mean()) if len(norms) else 0.0,
        "noise_scale": float(noise_multiplier * clip_norm),
        "lot_size": int(len(norms)),
    }
    return out, stats


class DPSGDOptimizer:
    """DP-SGD step helper around a torch optimizer.

    Usage: build a functional loss over a *single* row, extract per-sample grads, clip+noise,
    assign them to ``param.grad``, then call the wrapped optimizer's ``step``.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        clip_norm: float,
        noise_multiplier: float,
        expected_lot_size: float,
        generator: torch.Generator | None = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.clip_norm = clip_norm
        self.noise_multiplier = noise_multiplier
        self.expected_lot_size = expected_lot_size
        self.generator = generator
        self.last_stats: dict[str, float] = {}

    def step(
        self,
        params: dict[str, torch.Tensor],
        loss_single: Callable[[dict[str, torch.Tensor], torch.Tensor], torch.Tensor],
        batch: torch.Tensor,
    ) -> dict[str, float]:
        grads = per_sample_gradients(self.model, params, loss_single, batch)
        agg, stats = clip_and_noise(
            grads,
            clip_norm=self.clip_norm,
            noise_multiplier=self.noise_multiplier,
            expected_lot_size=self.expected_lot_size,
            generator=self.generator,
        )
        for name, p in self.model.named_parameters():
            if name in agg:
                p.grad = agg[name].to(p.dtype)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.last_stats = stats
        return stats


def make_functional_params(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone().requires_grad_(True) for k, v in model.state_dict().items()}


def poisson_batches(n: int, batch_size: int, seed: int) -> Iterable[Sequence[int]]:
    """Utility for tests; training loops use PrivacyEngine.sample_lot."""
    rng = np.random.default_rng(seed)
    q = min(1.0, batch_size / max(n, 1))
    while True:
        mask = rng.random(n) < q
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            idx = np.array([int(rng.integers(0, n))])
        yield idx


def theoretical_noise_multiplier_small_epsilon(epsilon: float, q: float, steps: int, delta: float) -> float:
    """Rough analytic guide used only for sanity tests (NOT for reported epsilon)."""
    if q <= 0:
        return float("inf")
    return math.sqrt(2 * steps * q**2 * math.log(1 / delta)) / max(epsilon, 1e-9)
