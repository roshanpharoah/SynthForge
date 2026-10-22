"""Enterprise extras: event/session sequences and multivariate IoT sensor series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthforge.simulator.common import rng


def simulate_sessions(n_sessions: int = 1200, seed: int = 71) -> pd.DataFrame:
    r = rng(seed)
    users = [f"U{i:05d}" for i in range(max(50, n_sessions // 8))]
    rows = []
    for i in range(n_sessions):
        user = str(r.choice(users))
        length = int(r.integers(2, 15))
        start = pd.Timestamp("2024-01-01") + pd.to_timedelta(int(r.integers(0, 365)), unit="D") + pd.to_timedelta(int(r.integers(0, 86400)), unit="s")
        events = r.choice(["view", "search", "add_to_cart", "checkout", "support", "logout"], length, p=[0.4, 0.22, 0.15, 0.1, 0.05, 0.08])
        gap = r.exponential(45, length)
        ts = start + pd.to_timedelta(np.cumsum(gap), unit="s")
        converted = int(("checkout" in events) and (r.random() < 0.55))
        for j, (e, t) in enumerate(zip(events, ts, strict=True)):
            rows.append(
                {
                    "session_id": f"S{i:06d}",
                    "user_id": user,
                    "event_index": j,
                    "event": str(e),
                    "event_ts": t,
                    "dwell_seconds": float(np.round(r.exponential(20), 2)),
                    "converted": converted,
                }
            )
    return pd.DataFrame(rows)


def simulate_iot(n: int = 6000, seed: int = 81, n_devices: int = 12) -> pd.DataFrame:
    r = rng(seed)
    devices = [f"DEV-{i:03d}" for i in range(n_devices)]
    t = pd.date_range("2024-01-01", periods=n, freq="5min")
    frames = []
    for d in devices:
        phase = r.uniform(0, 2 * np.pi)
        temp = 22 + 5 * np.sin(np.arange(n) / (24 * 12 / (2 * np.pi)) + phase) + r.normal(0, 0.35, n)
        pressure = 101 + 2.5 * np.sin(np.arange(n) / 200 + phase) + r.normal(0, 0.8, n)
        vibration = np.abs(r.lognormal(-1.2, 0.6, n))
        fault = np.zeros(n, dtype=int)
        drift = np.cumsum(r.normal(0, 0.0015, n))
        temp += drift
        # fault episodes with elevated vibration and temperature
        for _ in range(max(1, n // 2500)):
            start = int(r.integers(0, max(1, n - 200)))
            length = int(r.integers(40, 160))
            fault[start : start + length] = 1
            vibration[start : start + length] += r.uniform(1.5, 4.0)
            temp[start : start + length] += r.uniform(1.5, 5.0)
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": t,
                    "device_id": d,
                    "temperature_c": temp.round(3),
                    "pressure_kpa": pressure.round(3),
                    "vibration_mm_s": vibration.round(4),
                    "fault": fault,
                }
            )
        )
    return pd.concat(frames, ignore_index=True).sort_values(["timestamp", "device_id"]).reset_index(drop=True)


def simulate_event_log(n: int = 4000, seed: int = 91) -> pd.DataFrame:
    r = rng(seed)
    users = [f"U{i:05d}" for i in range(300)]
    actions = ["login", "export_report", "delete_record", "grant_access", "failed_login", "create_model"]
    df = pd.DataFrame(
        {
            "event_id": [f"EV{i:07d}" for i in range(n)],
            "user_id": r.choice(users, n),
            "action": r.choice(actions, n, p=[0.4, 0.12, 0.05, 0.06, 0.07, 0.3]),
            "ts": pd.Timestamp("2024-01-01") + pd.to_timedelta(r.integers(0, 31_536_000, n), unit="s"),
            "risk": np.clip(r.normal(0.2, 0.15, n), 0, 1).round(4),
        }
    )
    return df
