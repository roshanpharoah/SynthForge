"""Differentially private preprocessing statistics (bounds, means, standard deviations).

Uses the exponential mechanism for quantile bounds and Laplace noise for clipped moments. The
pure-DP budget used here is composed by basic composition with the RDP-accounted training step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synthforge.privacy.accountant import exponential_quantile


@dataclass
class DPStats:
    epsilon: float
    seed: int = 0
    details: list[dict] = field(default_factory=list)
    spent: float = 0.0

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def bounds(self, values: np.ndarray, epsilon: float) -> tuple[float, float]:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return 0.0, 1.0
        if epsilon <= 0:
            return float(np.min(v)), float(np.max(v))
        lo = exponential_quantile(v, int(0.005 * v.size), epsilon / 2, self._rng)
        hi = exponential_quantile(v, int(0.995 * v.size), epsilon / 2, self._rng)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(np.min(v)), float(np.max(v))
        self.spent += epsilon
        self.details.append({"stat": "bounds", "epsilon": epsilon, "lo": lo, "hi": hi})
        return float(lo), float(hi)

    def clipped_mean_std(self, values: np.ndarray, bounds: tuple[float, float], epsilon: float) -> tuple[float, float]:
        v = np.asarray(values, dtype=float)
        v = v[np.isfinite(v)]
        if v.size == 0:
            return float(bounds[0]), 1.0
        lo, hi = bounds
        c = np.clip(v, lo, hi)
        n = c.size
        if epsilon <= 0:
            return float(np.mean(c)), float(max(np.std(c), 1e-6))
        eps_mean, eps_second = epsilon / 2, epsilon / 2
        sens_mean = (hi - lo) / max(n, 1)
        mean_noisy = float(np.mean(c) + self._rng.laplace(0.0, sens_mean / max(eps_mean, 1e-12)))
        m2 = np.maximum(c, 0) ** 2 if lo >= 0 else c**2
        sens_second = max(lo**2, hi**2) / max(n, 1)
        second_noisy = float(np.mean(m2) + self._rng.laplace(0.0, sens_second / max(eps_second, 1e-12)))
        var = second_noisy - (mean_noisy**2 if lo <= 0 <= hi else mean_noisy**2)
        span = max(hi - lo, 1e-6)
        std = float(np.sqrt(np.clip(var, (span * 1e-3) ** 2, (span * 2) ** 2)))
        mean_clipped = float(np.clip(mean_noisy, lo, hi))
        self.spent += epsilon
        self.details.append(
            {"stat": "mean_std", "epsilon": epsilon, "mean": mean_clipped, "std": std, "bounds": [lo, hi]}
        )
        return mean_clipped, std

    def to_dict(self) -> dict:
        return {
            "epsilon": self.spent,
            "allocated_epsilon": self.epsilon,
            "details": self.details,
            "mechanism": "Exponential-mechanism quantiles + Laplace-noised clipped moments",
        }
