"""Shared simulator utilities. All values are procedurally generated; no real records."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def synthetic_name(r: np.random.Generator) -> str:
    first = ["Avery", "Jordan", "Sam", "Riley", "Casey", "Morgan", "Taylor", "Jamie", "Drew", "Quinn", "Rowan", "Sasha"]
    last = ["Testman", "Exampleton", "Sampleworth", "Synthetic", "Placeholder", "Demofield", "Mockley", "Fictitious"]
    return f"{r.choice(first)} {r.choice(last)}"


def random_dates(
    r: np.random.Generator, n: int, start: str = "2020-01-01", end: str = "2024-12-31"
) -> pd.Series:
    lo = pd.Timestamp(start).value // 10**9
    hi = pd.Timestamp(end).value // 10**9
    secs = r.integers(lo, hi, size=n)
    return pd.to_datetime(secs, unit="s")


def correlated_categories(
    r: np.random.Generator, n: int, categories: list[str], probs: list[float]
) -> np.ndarray:
    return r.choice(categories, size=n, p=np.array(probs) / sum(probs))


def add_missing(
    series: pd.Series, r: np.random.Generator, rate: float, mechanism: str = "MCAR", driver: pd.Series | None = None
) -> pd.Series:
    """Inject missingness with documented mechanisms (MCAR, MAR, MNAR)."""
    s = series.copy()
    if rate <= 0:
        return s
    if mechanism == "MCAR":
        mask = r.random(len(s)) < rate
    elif mechanism == "MAR" and driver is not None:
        d = driver.astype(float)
        d = (d - d.min()) / max(d.max() - d.min(), 1e-9)
        mask = r.random(len(s)) < rate * (0.2 + 1.6 * d.fillna(0.5))
    elif mechanism == "MNAR":
        d = s.astype(float) if pd.api.types.is_numeric_dtype(s) else pd.Series(r.random(len(s)))
        d = (d - d.min()) / max(d.max() - d.min(), 1e-9)
        mask = r.random(len(s)) < rate * (0.2 + 1.6 * (1 - d.fillna(0.5)))
    else:
        mask = r.random(len(s)) < rate
    s = s.mask(mask)
    return s
