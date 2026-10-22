"""Rényi-DP accountant for Poisson-subsampled Gaussian mechanisms.

Implements the integer-alpha RDP bound from Mironov, Talwar & Zhang (2019) for the sampled
Gaussian mechanism and the standard RDP→(ε,δ) conversion. Pure-DP preprocessing is composed by
basic composition. No claim is made for mechanisms not accounted for here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.special import logsumexp

DEFAULT_ALPHAS: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256)


def _log_comb(n: int, k: int) -> float:
    return float(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def gaussian_rdp(sigma: float, alpha: float) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if alpha <= 1:
        raise ValueError("alpha must be > 1")
    return alpha / (2.0 * sigma**2)


def sampled_gaussian_rdp(sigma: float, alpha: int, q: float) -> float:
    """RDP of the Poisson-subsampled Gaussian mechanism at integer alpha."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0,1]")
    if q == 1:
        return gaussian_rdp(sigma, alpha)
    if q == 0:
        return 0.0
    alpha = int(alpha)
    k = np.arange(alpha + 1)
    log_terms = (
        np.array([_log_comb(alpha, int(i)) for i in k])
        + (alpha - k) * math.log(1 - q)
        + k * math.log(q)
        + (k**2 - k) / (2.0 * sigma**2)
    )
    return float(logsumexp(log_terms) / (alpha - 1))


def rdp_to_epsilon(rdp: float, alpha: float, delta: float) -> float:
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0,1)")
    if rdp == 0:
        return 0.0
    return float(rdp + math.log(1.0 / delta) / (alpha - 1))


def epsilon_from_rdp(rdp_by_alpha: dict[float, float], delta: float) -> float:
    vals = [rdp_to_epsilon(v, a, delta) for a, v in rdp_by_alpha.items() if a > 1]
    return float(min(vals)) if vals else float("inf")


def mean_rdp(rdp_by_alpha: dict[float, float]) -> float:
    return float(np.mean(list(rdp_by_alpha.values()))) if rdp_by_alpha else 0.0


@dataclass
class DPResult:
    epsilon: float
    delta: float
    noise_multiplier: float
    sampling_rate: float
    steps: int
    alphas: tuple[int, ...]
    rdp: dict[float, float]
    mechanism: str = "Poisson-subsampled Gaussian (DP-SGD)"

    def covers(self) -> str:
        return (
            f"{self.mechanism}: {self.steps} steps, sampling rate q={self.sampling_rate:.6g}, "
            f"σ={self.noise_multiplier:.6g}. ε={self.epsilon:.4f}, δ={self.delta:g}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "sampling_rate": self.sampling_rate,
            "steps": self.steps,
            "alphas": list(self.alphas),
            "rdp": {str(k): v for k, v in self.rdp.items()},
            "mechanism": self.mechanism,
            "statement": self.covers(),
        }


class RDPAccountant:
    def __init__(self, alphas: tuple[int, ...] | None = None):
        self.alphas = tuple(sorted(set(alphas or DEFAULT_ALPHAS)))
        self.history: list[dict[str, Any]] = []

    def step(self, *, noise_multiplier: float, sampling_rate: float, steps: int = 1) -> None:
        if steps <= 0:
            return
        rdp = {
            float(a): steps * sampled_gaussian_rdp(noise_multiplier, a, sampling_rate)
            for a in self.alphas
        }
        self.history.append(
            {
                "mechanism": "sampled_gaussian",
                "noise_multiplier": noise_multiplier,
                "sampling_rate": sampling_rate,
                "steps": steps,
                "rdp": rdp,
            }
        )

    def add_pure_dp(self, epsilon: float, mechanism: str = "laplace") -> None:
        if epsilon <= 0:
            return
        # Pure ε-DP is ε-DP at every Rényi order; conversion is conservative but valid.
        self.history.append({"mechanism": mechanism, "pure_epsilon": epsilon, "steps": 1})

    def current_rdp(self) -> dict[float, float]:
        out = {float(a): 0.0 for a in self.alphas}
        for entry in self.history:
            if "rdp" in entry:
                for a, v in entry["rdp"].items():
                    out[float(a)] = out.get(float(a), 0.0) + float(v)
        return out

    def get_epsilon(self, delta: float) -> float:
        eps = epsilon_from_rdp(self.current_rdp(), delta)
        pure = sum(float(e.get("pure_epsilon", 0.0)) for e in self.history)
        return float(eps + pure)

    def get_privacy_spent(self, delta: float) -> DPResult:
        rdp = self.current_rdp()
        eps = self.get_epsilon(delta)
        sampled = [h for h in self.history if h["mechanism"] == "sampled_gaussian"]
        total_steps = sum(int(h["steps"]) for h in sampled)
        sigma = sampled[-1]["noise_multiplier"] if sampled else float("nan")
        q = sampled[-1]["sampling_rate"] if sampled else float("nan")
        return DPResult(
            epsilon=eps,
            delta=delta,
            noise_multiplier=sigma,
            sampling_rate=q,
            steps=total_steps,
            alphas=self.alphas,
            rdp=rdp,
        )


def epsilon_for_sampled_gaussian(
    *, noise_multiplier: float, sampling_rate: float, steps: int, delta: float,
    alphas: tuple[int, ...] | None = None,
) -> DPResult:
    acc = RDPAccountant(alphas)
    acc.step(noise_multiplier=noise_multiplier, sampling_rate=sampling_rate, steps=steps)
    return acc.get_privacy_spent(delta)


def compute_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    sampling_rate: float,
    steps: int,
    alphas: tuple[int, ...] | None = None,
    tolerance: float = 0.01,
    max_iter: int = 100,
) -> tuple[float, DPResult]:
    """Binary-search σ so the composed mechanism is within ``target_epsilon``."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive")
    alphas = alphas or DEFAULT_ALPHAS
    lo, hi = 0.05, 10_000.0
    best_sigma = hi
    best_res = epsilon_for_sampled_gaussian(
        noise_multiplier=best_sigma, sampling_rate=sampling_rate, steps=steps, delta=delta, alphas=alphas
    )
    if best_res.epsilon > target_epsilon:
        # Even heavy noise cannot meet an extremely small target at this step count.
        return best_sigma, best_res
    for _ in range(max_iter):
        mid = math.sqrt(lo * hi)
        res = epsilon_for_sampled_gaussian(
            noise_multiplier=mid, sampling_rate=sampling_rate, steps=steps, delta=delta, alphas=alphas
        )
        if res.epsilon <= target_epsilon:
            hi = mid
            best_sigma, best_res = mid, res
            if abs(res.epsilon - target_epsilon) / target_epsilon <= tolerance:
                break
        else:
            lo = mid
    return float(best_sigma), best_res


# ------------------------------------------------------------------ pure DP utilities
def laplace_epsilon_for_mean(bounds: tuple[float, float], n: int, scale: float) -> float:
    """Pure ε for a clipped mean with Laplace noise of ``scale`` (sensitivity (hi-lo)/n)."""
    sens = (bounds[1] - bounds[0]) / max(n, 1)
    return float(sens / max(scale, 1e-300))


def exponential_quantile(values: np.ndarray, target_rank: int, epsilon: float, rng: np.random.Generator) -> float:
    """DP quantile via the exponential mechanism over data values.

    Utility is negative rank distance; sensitivity of the rank under adding/removing one record is 1,
    so the exponential mechanism with score/(2Δ) is ε-DP.
    """
    vals = np.sort(np.asarray(values, dtype=float))
    if vals.size == 0:
        return float("nan")
    target_rank = int(np.clip(target_rank, 0, vals.size - 1))
    scores = -np.abs(np.arange(vals.size) - target_rank).astype(float)
    logits = epsilon * scores / 2.0
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    idx = int(rng.choice(vals.size, p=probs))
    return float(vals[idx])


def summarize_epsilon(result: DPResult) -> str:
    return result.covers()
