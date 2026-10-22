"""Content-addressed artifact storage on the local filesystem."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from synthforge.core.determinism import hash_bytes, hash_file
from synthforge.core.manifest import ArtifactRef
from synthforge.core.types import Dataset


@dataclass
class ArtifactStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    def run_dir(self, run_id: str) -> Path:
        p = self.root / "runs" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dataset_dir(self, dataset_id: str) -> Path:
        p = self.root / "datasets" / dataset_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def upload_dir(self, dataset_id: str) -> Path:
        p = self.root / "uploads" / dataset_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def report_dir(self, run_id: str) -> Path:
        p = self.root / "reports" / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    def write_json(self, run_id: str, name: str, payload: Any, *, kind: str = "json") -> ArtifactRef:
        path = self.run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return self.register(path, name=name, kind=kind)

    def write_text(self, run_id: str, name: str, text: str, *, kind: str = "text") -> ArtifactRef:
        path = self.run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return self.register(path, name=name, kind=kind)

    def write_bytes(self, run_id: str, name: str, data: bytes, *, kind: str = "file") -> ArtifactRef:
        path = self.run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return self.register(path, name=name, kind=kind)

    def write_frame(
        self, run_id: str, name: str, frame: pd.DataFrame, *, kind: str = "table"
    ) -> ArtifactRef:
        path = self.run_dir(run_id) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)
        return self.register(path, name=name, kind=kind)

    def save_dataset(self, dataset: Dataset, *, subdir: str, dataset_id: str | None = None) -> Path:
        target = self.dataset_dir(dataset_id or subdir) / "normalized"
        if target.exists():
            shutil.rmtree(target)
        dataset.save(target)
        return target

    def register(self, path: str | Path, *, name: str | None = None, kind: str = "file") -> ArtifactRef:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(p)
        rel = self._rel(p)
        return ArtifactRef(
            name=name or p.name,
            path=rel,
            sha256=hash_file(p),
            size=p.stat().st_size,
            kind=kind,
        )

    def copy_in(self, source: str | Path, run_id: str, name: str, *, kind: str = "source") -> ArtifactRef:
        target = self.run_dir(run_id) / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return self.register(target, name=name, kind=kind)

    def verify(self, ref: ArtifactRef) -> bool:
        p = self.abs(ref.path)
        return p.exists() and hash_file(p) == ref.sha256

    def directory_checksum(self, path: str | Path) -> str:
        """Deterministic Merkle-ish digest over sorted files (path + content)."""
        root = Path(path)
        parts: list[bytes] = []
        for p in sorted(root.rglob("*"), key=lambda x: x.as_posix()):
            if p.is_file():
                parts.append(p.relative_to(root).as_posix().encode())
                parts.append(hash_file(p).encode())
        return hash_bytes(b"|".join(parts))
