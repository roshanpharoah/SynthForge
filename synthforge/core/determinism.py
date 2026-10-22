"""Determinism and canonical hashing utilities.

CPU reproducibility is a first-class requirement. We seed Python, NumPy and Torch and enable
deterministic kernels where requested. GPU results may differ; manifests record the device and
tolerance is documented in docs/DECISIONS.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = True) -> None:
    """Seed all RNGs used by SynthForge."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except ImportError:  # pragma: no cover - torch is a hard dependency
        pass


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def canonical_json(obj: Any) -> str:
    """Stable JSON encoding used in hashes."""

    def _default(o: Any) -> Any:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (set, tuple)):
            return sorted(o, key=str) if isinstance(o, set) else list(o)
        if hasattr(o, "model_dump"):
            return o.model_dump(mode="json")
        return str(o)

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def stable_hash(obj: Any, *, prefix: str = "") -> str:
    h = hashlib.sha256(prefix.encode("utf-8"))
    h.update(canonical_json(obj).encode("utf-8"))
    return h.hexdigest()


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_file(path: str | os.PathLike[str], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()
