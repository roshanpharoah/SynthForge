"""Versioned on-disk model registry. The database stores the index; files are checksummed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from synthforge.core.manifest import RunManifest
from synthforge.core.protocol import BaseSynthesizer
from synthforge.core.store import ArtifactStore


@dataclass
class RegistryEntry:
    name: str
    version: int
    path: str
    checksum: str
    run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "path": self.path,
            "checksum": self.checksum,
            "run_id": self.run_id,
        }


class ModelRegistry:
    def __init__(self, store: ArtifactStore):
        self.store = store
        self.root = store.root / "registry"
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"

    def _index(self) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            return []
        return json.loads(self.index_path.read_text(encoding="utf-8"))

    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        self.index_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    def next_version(self, name: str) -> int:
        versions = [e["version"] for e in self._index() if e["name"] == name]
        return (max(versions) + 1) if versions else 1

    def save(
        self,
        model: BaseSynthesizer,
        *,
        run_id: str,
        manifest: RunManifest,
        metrics: dict[str, Any] | None = None,
        name: str | None = None,
    ) -> RegistryEntry:
        name = name or model.model_name
        version = self.next_version(name)
        target = self.root / name / f"v{version}"
        target.mkdir(parents=True, exist_ok=True)
        model.save(target / "model")
        (target / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        (target / "metrics.json").write_text(json.dumps(metrics or {}, indent=2, default=str), encoding="utf-8")
        (target / "model_card.json").write_text(
            json.dumps(model.model_card(), indent=2, default=str), encoding="utf-8"
        )
        checksum = self.store.directory_checksum(target)
        entry = RegistryEntry(
            name=name,
            version=version,
            path=target.relative_to(self.store.root).as_posix(),
            checksum=checksum,
            run_id=run_id,
        )
        entries = self._index()
        entries.append(entry.to_dict())
        self._write_index(entries)
        return entry

    def list(self, name: str | None = None) -> list[dict[str, Any]]:
        entries = self._index()
        if name:
            entries = [e for e in entries if e["name"] == name]
        return sorted(entries, key=lambda e: (e["name"], e["version"]))

    def resolve(self, name: str, version: int | str | None = None) -> Path:
        entries = [e for e in self._index() if e["name"] == name]
        if version in (None, "latest"):
            if not entries:
                raise FileNotFoundError(f"no registered model named {name!r}")
            entry = max(entries, key=lambda e: e["version"])
        else:
            match = [e for e in entries if e["version"] == int(version)]
            if not match:
                raise FileNotFoundError(f"model {name!r} version {version} not found")
            entry = match[0]
        return self.store.root / entry["path"]

    def verify(self, entry: dict[str, Any]) -> bool:
        return self.store.directory_checksum(self.store.root / entry["path"]) == entry["checksum"]


def verify_registry(registry: ModelRegistry) -> list[dict[str, Any]]:
    out = []
    for entry in registry.list():
        out.append({"name": entry["name"], "version": entry["version"], "valid": registry.verify(entry)})
    return out
