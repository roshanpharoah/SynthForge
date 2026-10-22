"""In-memory data contracts shared by ingestion, models, evaluation and storage."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from synthforge.core.determinism import canonical_json, hash_bytes, stable_hash
from synthforge.metadata.constraints import Constraint

ColumnKind = Literal["integer", "float", "boolean", "datetime", "categorical", "text", "unknown"]
ColumnRole = Literal["feature", "target", "primary_key", "foreign_key", "ignore", "label"]
Sensitivity = Literal["public", "quasi_identifier", "sensitive", "direct_identifier", "unknown"]


class ColumnSpec(BaseModel):
    name: str
    kind: ColumnKind = "unknown"
    role: ColumnRole = "feature"
    dtype: str = "object"
    nullable: bool = True
    cardinality: int | None = None
    missing_rate: float = 0.0
    sensitivity: Sensitivity = "unknown"
    semantic: str | None = None
    stats: dict[str, Any] = Field(default_factory=dict)
    dropped: bool = False


class ForeignKey(BaseModel):
    columns: list[str]
    ref_table: str
    ref_columns: list[str]


class TableSchema(BaseModel):
    name: str
    columns: list[ColumnSpec]
    primary_key: list[str] = Field(default_factory=list)
    foreign_keys: list[ForeignKey] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    row_count: int = 0
    source: str | None = None
    notes: list[str] = Field(default_factory=list)

    def column(self, name: str) -> ColumnSpec:
        for c in self.columns:
            if c.name == name:
                return c
        raise KeyError(f"column {name!r} not in table {self.name!r}")

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def feature_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.role in ("feature", "target", "label") and not c.dropped]

    @property
    def target_columns(self) -> list[str]:
        return [c.name for c in self.columns if c.role == "target"]


def infer_semantics(name: str) -> tuple[str | None, Sensitivity]:
    n = name.lower()
    direct = {
        "mrn", "ssn", "social_security", "passport", "email", "phone", "address", "name",
        "first_name", "last_name", "full_name", "patient_name", "account_number", "iban", "license",
        "device_serial", "ip_address",
    }
    quasi = {
        "age", "dob", "date_of_birth", "zip", "zipcode", "postal_code", "gender", "sex", "race",
        "ethnicity", "marital_status", "city", "state", "occupation", "education", "income",
        "admission_date", "discharge_date", "visit_date", "birth_date",
    }
    sensitive = {
        "diagnosis", "disease", "condition", "hiv", "mental_health", "salary", "default", "fraud",
        "credit_score", "medication", "lab_result", "biomarker", "pregnancy", "genetic",
    }
    if n in direct or any(t in n for t in ("_name", "email", "phone", "address", "ssn")):
        return "direct_identifier", "direct_identifier"
    if n in quasi or any(t in n for t in ("postal", "zip", "birth", "gender", "ethnicity", "age")):
        return "quasi_identifier", "quasi_identifier"
    if n in sensitive or any(t in n for t in ("diagnos", "disease", "salary", "fraud", "default")):
        return "sensitive_attribute", "sensitive"
    return None, "unknown"


def _frame_digest(frame: pd.DataFrame) -> str:
    """Stable content digest for a frame independent of dtypes where sensible."""
    if frame.empty:
        return hash_bytes(b"empty:" + canonical_json(list(frame.columns)).encode())
    cols = sorted(frame.columns, key=str)
    norm = pd.DataFrame(index=range(len(frame)))
    for c in cols:
        s = frame[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            norm[str(c)] = pd.to_datetime(s, errors="coerce").astype("int64").astype(str)
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            norm[str(c)] = s.astype("float64").round(10).astype(str)
        else:
            norm[str(c)] = s.astype("object").where(s.notna(), "<NA>").astype(str)
    hashed = pd.util.hash_pandas_object(norm, index=False).to_numpy(dtype=np.uint64)
    return hash_bytes(hashed.tobytes() + canonical_json(cols).encode())


@dataclass
class Table:
    name: str
    frame: pd.DataFrame
    schema: TableSchema

    def __post_init__(self) -> None:
        self.schema.row_count = int(len(self.frame))

    def copy(self, deep: bool = True) -> Table:
        return Table(
            name=self.name,
            frame=self.frame.copy(deep=deep),
            schema=self.schema.model_copy(deep=True),
        )


@dataclass
class SignalCohort:
    signals: np.ndarray  # (n, leads, samples), float32
    sampling_rate: int
    lead_names: list[str]
    labels: pd.DataFrame = field(default_factory=pd.DataFrame)
    quality: pd.DataFrame = field(default_factory=pd.DataFrame)
    source_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError("signals must have shape (n, leads, samples)")
        self.signals = np.asarray(self.signals, dtype=np.float32)
        if not self.lead_names:
            self.lead_names = [f"lead_{i}" for i in range(self.signals.shape[1])]
        if len(self.lead_names) != self.signals.shape[1]:
            raise ValueError("lead_names length must equal signal leads")

    def copy(self) -> SignalCohort:
        return SignalCohort(
            signals=self.signals.copy(),
            sampling_rate=self.sampling_rate,
            lead_names=list(self.lead_names),
            labels=self.labels.copy(deep=True),
            quality=self.quality.copy(deep=True),
            source_paths=list(self.source_paths),
            metadata=dict(self.metadata),
        )


@dataclass
class ImageCohort:
    images: np.ndarray  # (n, h, w) or (n, d, h, w), float32 in [0, 1]
    masks: np.ndarray | None = None
    labels: pd.DataFrame = field(default_factory=pd.DataFrame)
    spacing: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    deid_report: dict[str, Any] = field(default_factory=dict)
    source_paths: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.images = np.asarray(self.images, dtype=np.float32)
        if self.images.ndim not in (3, 4):
            raise ValueError("images must have shape (n, h, w) or (n, d, h, w)")
        if self.masks is not None:
            self.masks = np.asarray(self.masks, dtype=np.float32)
            if self.masks.shape != self.images.shape:
                raise ValueError("masks shape must equal images shape")

    @property
    def is_3d(self) -> bool:
        return self.images.ndim == 4

    def copy(self) -> ImageCohort:
        return ImageCohort(
            images=self.images.copy(),
            masks=None if self.masks is None else self.masks.copy(),
            labels=self.labels.copy(deep=True),
            spacing=tuple(self.spacing),
            metadata=dict(self.metadata),
            deid_report=dict(self.deid_report),
            source_paths=list(self.source_paths),
        )


@dataclass
class Dataset:
    name: str
    kind: Literal["tabular", "relational", "timeseries", "imaging", "multimodal"] = "tabular"
    tables: dict[str, Table] = field(default_factory=dict)
    signals: SignalCohort | None = None
    images: ImageCohort | None = None
    source: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    synthetic: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    dataset_id: str | None = None

    def table_names(self) -> list[str]:
        return list(self.tables.keys())

    @property
    def primary_table(self) -> Table | None:
        if not self.tables:
            return None
        return next(iter(self.tables.values()))

    @property
    def n_rows(self) -> int:
        if self.tables:
            return int(sum(len(t.frame) for t in self.tables.values()))
        if self.signals is not None:
            return int(self.signals.signals.shape[0])
        if self.images is not None:
            return int(self.images.images.shape[0])
        return 0

    def table_frame(self, name: str | None = None) -> pd.DataFrame:
        table = self.tables[name] if name else self.primary_table
        if table is None:
            raise KeyError("dataset has no tabular data")
        return table.frame

    def fingerprint(self) -> str:
        payload: dict[str, Any] = {
            "kind": self.kind,
            "tables": sorted(
                ({"name": t.name, "schema": t.schema.model_dump(mode="json"), "digest": _frame_digest(t.frame)} for t in self.tables.values()),
                key=lambda x: x["name"],
            ),
        }
        if self.signals is not None:
            payload["signals"] = {
                "shape": list(self.signals.signals.shape),
                "fs": self.signals.sampling_rate,
                "leads": self.signals.lead_names,
                "digest": hash_bytes(self.signals.signals.tobytes()),
                "labels": _frame_digest(self.signals.labels),
            }
        if self.images is not None:
            payload["images"] = {
                "shape": list(self.images.images.shape),
                "digest": hash_bytes(self.images.images.tobytes()),
                "mask_digest": None
                if self.images.masks is None
                else hash_bytes(self.images.masks.tobytes()),
                "labels": _frame_digest(self.images.labels),
            }
        return stable_hash(payload)

    def preview(self, n: int = 10, *, include_media: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "rows": self.n_rows,
            "tables": {},
            "warnings": self.warnings,
        }
        for name, table in self.tables.items():
            head = table.frame.head(n).copy()
            for c in head.columns:
                if pd.api.types.is_datetime64_any_dtype(head[c]):
                    head[c] = head[c].astype(str)
            out["tables"][name] = {
                "rows": int(len(table.frame)),
                "columns": list(map(str, table.frame.columns)),
                "preview": json.loads(head.to_json(orient="records", date_format="iso")),
            }
        if include_media and self.signals is not None:
            out["signals"] = {
                "n": int(self.signals.signals.shape[0]),
                "leads": self.signals.lead_names,
                "samples": int(self.signals.signals.shape[2]),
                "fs": self.signals.sampling_rate,
            }
        if include_media and self.images is not None:
            out["images"] = {
                "n": int(self.images.images.shape[0]),
                "shape": list(self.images.images.shape[1:]),
                "has_masks": self.images.masks is not None,
            }
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Persist a normalised dataset. Raw source files are kept by the caller."""
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        tables_dir = root / "tables"
        tables_dir.mkdir(exist_ok=True)
        for name, table in self.tables.items():
            table.frame.to_parquet(tables_dir / f"{name}.parquet", index=False)
        if self.signals is not None:
            np.savez_compressed(
                root / "signals.npz",
                signals=self.signals.signals,
                **{k: np.asarray(v) for k, v in self.signals.metadata.items() if isinstance(v, (str, int, float))},
            )
            self.signals.labels.to_parquet(root / "signal_labels.parquet", index=False)
            self.signals.quality.to_parquet(root / "signal_quality.parquet", index=False)
        if self.images is not None:
            payload = {"images": self.images.images}
            if self.images.masks is not None:
                payload["masks"] = self.images.masks
            np.savez_compressed(root / "images.npz", **payload)
            self.images.labels.to_parquet(root / "image_labels.parquet", index=False)
        meta = {
            "name": self.name,
            "kind": self.kind,
            "source": self.source,
            "profile": self.profile,
            "warnings": self.warnings,
            "created_at": self.created_at,
            "synthetic": self.synthetic,
            "provenance": self.provenance,
            "dataset_id": self.dataset_id,
            "schemas": {n: t.schema.model_dump(mode="json") for n, t in self.tables.items()},
            "signals": None
            if self.signals is None
            else {
                "sampling_rate": self.signals.sampling_rate,
                "lead_names": self.signals.lead_names,
                "source_paths": self.signals.source_paths,
                "metadata": _jsonable(self.signals.metadata),
            },
            "images": None
            if self.images is None
            else {
                "spacing": list(self.images.spacing),
                "metadata": _jsonable(self.images.metadata),
                "deid_report": _jsonable(self.images.deid_report),
                "source_paths": self.images.source_paths,
            },
            "fingerprint": self.fingerprint(),
        }
        (root / "dataset.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
        return root

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        root = Path(path)
        meta = json.loads((root / "dataset.json").read_text(encoding="utf-8"))
        tables: dict[str, Table] = {}
        for name, schema_dict in meta.get("schemas", {}).items():
            frame = pd.read_parquet(root / "tables" / f"{name}.parquet")
            tables[name] = Table(name=name, frame=frame, schema=TableSchema.model_validate(schema_dict))
        signals = None
        if meta.get("signals") is not None:
            sm = meta["signals"]
            npz = np.load(root / "signals.npz", allow_pickle=False)
            labels = pd.read_parquet(root / "signal_labels.parquet")
            quality = pd.read_parquet(root / "signal_quality.parquet")
            signals = SignalCohort(
                signals=npz["signals"],
                sampling_rate=int(sm["sampling_rate"]),
                lead_names=list(sm["lead_names"]),
                labels=labels,
                quality=quality,
                source_paths=list(sm.get("source_paths", [])),
                metadata=dict(sm.get("metadata", {})),
            )
        images = None
        if meta.get("images") is not None:
            im = meta["images"]
            npz = np.load(root / "images.npz", allow_pickle=False)
            images = ImageCohort(
                images=npz["images"],
                masks=npz.get("masks", None),
                labels=pd.read_parquet(root / "image_labels.parquet"),
                spacing=tuple(im.get("spacing", [])),
                metadata=dict(im.get("metadata", {})),
                deid_report=dict(im.get("deid_report", {})),
                source_paths=list(im.get("source_paths", [])),
            )
        return cls(
            name=meta["name"],
            kind=meta["kind"],
            tables=tables,
            signals=signals,
            images=images,
            source=meta.get("source", {}),
            profile=meta.get("profile", {}),
            warnings=meta.get("warnings", []),
            created_at=meta.get("created_at", ""),
            synthetic=bool(meta.get("synthetic", False)),
            provenance=meta.get("provenance", {}),
            dataset_id=meta.get("dataset_id"),
        )


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
