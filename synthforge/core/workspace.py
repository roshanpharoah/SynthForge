"""Local workspace layout, database bootstrap and shared services."""

from __future__ import annotations

import contextlib
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from synthforge.core.store import ArtifactStore
from synthforge.db.session import Database
from synthforge.registry import ModelRegistry


def default_root() -> Path:
    return Path(os.environ.get("SYNTHFORGE_HOME", Path.home() / ".synthforge")).expanduser()


@dataclass
class Workspace:
    root: Path = field(default_factory=default_root)

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts = ArtifactStore(self.root / "artifacts")
        self.db = Database(self.root / "metadata.db")
        self.db.init()
        self.registry = ModelRegistry(self.artifacts)
        for sub in ("uploads", "reports", "demo", "logs", "configs"):
            (self.root / sub).mkdir(exist_ok=True)
        self.secret_key = self._secret()

    def _secret(self) -> str:
        path = self.root / "secret.key"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        key = secrets.token_urlsafe(48)
        path.write_text(key, encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)
        return key

    def close(self) -> None:
        self.db.close()
