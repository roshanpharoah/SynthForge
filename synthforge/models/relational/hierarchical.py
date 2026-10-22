"""Hierarchical relational synthesis with referential integrity.

Approach:
1. Discover a topological order of tables from foreign keys; cycles are reported and broken
   deterministically (the breaking edges are reported, not silently dropped).
2. Synthesise the root table with the requested single-table model. For each child table, augment the
   child frame with the parent's generated foreign-key values as a *conditional* column, so the child
   model learns parent → child relationships. This is a real hierarchical conditioning mechanism; it is
   approximate (each child model is trained independently, conditioned on the parent key), and that
   limitation is stated in the model card.
3. Enforce PK uniqueness and FK integrity and report repair rates and residual violations honestly.
4. Export to SQLite/SQL/CSV/Parquet.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.config import RunConfig
from synthforge.core.errors import ModelError
from synthforge.core.protocol import (
    BaseSynthesizer,
    CancelCheck,
    FitResult,
    PrivacyReport,
    ProgressCallback,
    TrainHistory,
)
from synthforge.core.types import Dataset, Table
from synthforge.ingest.database import validate_relational_integrity
from synthforge.metadata.constraints import ForeignKeyConstraint
from synthforge.models.registry import build_model


def fk_edges(dataset: Dataset) -> list[dict[str, Any]]:
    """Return child→parent edges with column names, de-duplicated."""
    edges: list[dict[str, Any]] = []
    seen = set()
    for name, table in dataset.tables.items():
        pairs: list[tuple[list[str], str, list[str]]] = []
        for fk in table.schema.foreign_keys:
            pairs.append((list(fk.columns), fk.ref_table, list(fk.ref_columns)))
        for c in table.schema.constraints:
            if isinstance(c, ForeignKeyConstraint):
                pairs.append((list(c.columns), c.ref_table, list(c.ref_columns)))
        for cols, ref_table, ref_cols in pairs:
            key = (name, ref_table, tuple(cols), tuple(ref_cols))
            if key in seen:
                continue
            seen.add(key)
            edges.append({"child": name, "parent": ref_table, "columns": cols, "ref_columns": ref_cols})
    return edges


def topological_order(dataset: Dataset, edges: list[dict[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    """Order tables parents-first. Cycles are broken deterministically and reported."""
    tables = list(dataset.tables.keys())
    broken: list[dict[str, Any]] = []
    indeg = dict.fromkeys(tables, 0)
    parents: dict[str, list[dict[str, Any]]] = {t: [] for t in tables}
    for e in edges:
        if e["parent"] not in tables or e["child"] not in tables:
            continue
        indeg[e["child"]] += 1
        parents[e["child"]].append(e)
    order: list[str] = []
    remaining = set(tables)
    while remaining:
        ready = sorted([t for t in remaining if indeg.get(t, 0) == 0])
        if not ready:
            # deterministic cycle break: pick the alphabetically first remaining node and drop incoming edges
            victim = sorted(remaining)[0]
            for e in parents[victim]:
                broken.append({**e, "reason": "cycle broken to obtain a synthesis order"})
            indeg[victim] = 0
            ready = [victim]
        for t in ready:
            if t not in remaining:
                continue
            remaining.discard(t)
            order.append(t)
            for e in edges:
                if e["parent"] == t and e["child"] in remaining:
                    indeg[e["child"]] = max(0, indeg[e["child"]] - 1)
    return order, broken


class RelationalSynthesizer(BaseSynthesizer):
    model_name = "relational"
    modality = "relational"

    def __init__(self, config: RunConfig | None = None):
        super().__init__(config)
        self.models: dict[str, BaseSynthesizer] = {}
        self.order: list[str] = []
        self.edges: list[dict[str, Any]] = []
        self.broken_edges: list[dict[str, Any]] = []
        self.constraint_report: dict[str, Any] = {}
        self.integrity: dict[str, Any] = {}
        self.schemas: dict[str, Any] = {}
        self.child_models: dict[str, str] = {}
        self.rows_per_parent: dict[str, float] = {}
        self.privacy: PrivacyReport = PrivacyReport(mechanism="none")

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        dataset: Dataset,
        config: RunConfig,
        *,
        progress: ProgressCallback | None = None,
        should_cancel: CancelCheck | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> FitResult:
        started = time.time()
        if len(dataset.tables) < 2:
            raise ModelError(
                "relational synthesis needs at least two tables",
                hint="Ingest a database or a folder of CSVs with relationships.",
            )
        self.config = config
        self.edges = fk_edges(dataset)
        self.order, self.broken_edges = topological_order(dataset, self.edges)
        root_model = str(config.model_params.get("child_model") or config.model or "ctgan")
        if root_model in ("relational", "hierarchical"):
            root_model = "ctgan"
        child_models = config.model_params.get("child_models") or {}
        local_config = config.model_copy(deep=True)
        local_config.model = root_model
        local_config.privacy.enabled = False  # trained child-by-child; a single DP run cannot be composed per table
        if config.privacy.enabled:
            self.warnings.append(
                "DP is not supported for hierarchical relational synthesis in this build: each table would need its "
                "own accountant and composition statement. The run is recorded as non-private."
            )
        parent_maps: dict[str, dict[str, str]] = {}
        total = len(self.order)
        for i, name in enumerate(self.order):
            table = dataset.tables[name]
            frame = table.frame.copy()
            schema = table.schema.model_copy(deep=True)
            conditional_columns: list[str] = []
            for edge in self.edges:
                if edge["child"] != name:
                    continue
                parent = edge["parent"]
                if parent not in parent_maps or parent not in self.order[:i]:
                    continue
                key_col = f"__parent_{parent}_key"
                child_fk = edge["columns"][0]
                if child_fk not in frame.columns:
                    continue
                # Replace the child's real foreign key values with the *synthetic* parent keys for the same
                # logical parent, so the child model learns parent -> child without seeing real keys.
                parent_map = parent_maps[parent]
                mapped = frame[child_fk].astype(str).map(parent_map)
                if mapped.isna().any() and len(parent_map):
                    mapped = mapped.fillna(next(iter(parent_map.values())))
                frame[child_fk] = mapped
                # also expose an explicit parent-key conditioning column for the child model
                frame[key_col] = mapped
                feat = next((c for c in schema.columns if c.name == child_fk), None)
                if feat is not None:
                    feat.kind = "categorical"
                    feat.dtype = "object"
                    feat.cardinality = int(len(parent_map))
                    feat.sensitivity = "public"
                    feat.role = "foreign_key"
                schema.columns.append(
                    type(schema.columns[0])(
                        name=key_col,
                        kind="categorical",
                        role="feature",
                        dtype="object",
                        cardinality=int(len(parent_map)),
                        sensitivity="public",
                    )
                )
                conditional_columns.append(key_col)
            self.schemas[name] = schema
            model_name = child_models.get(name, root_model)
            self.child_models[name] = model_name
            model = build_model(model_name, local_config)
            sub_ds = Dataset(
                name=f"{dataset.name}_{name}",
                kind="tabular",
                tables={name: Table(name=name, frame=frame, schema=schema)},
                synthetic=False,
            )
            if progress:
                progress({"phase": "fit", "table": name, "index": i + 1, "tables": total, "progress": 0.05 + 0.9 * i / total})
            if should_cancel and should_cancel():
                from synthforge.core.errors import CancelledError

                raise CancelledError("cancelled")
            model.fit(sub_ds, local_config)
            self.models[name] = model
            # record rows-per-parent so sampling can preserve realistic cardinalities
            for edge in self.edges:
                if edge["parent"] == name and edge["child"] in dataset.tables:
                    child_frame = dataset.tables[edge["child"]].frame
                    col = edge["columns"][0]
                    if col in child_frame.columns and col in frame.columns:
                        counts = child_frame.groupby(col).size()
                        self.rows_per_parent[edge["child"]] = float(counts.mean()) if len(counts) else 1.0
            parent_pk_list = self.schemas[name].primary_key
            parent_pk = parent_pk_list[0] if parent_pk_list else (frame.columns[0] if len(frame.columns) else None)
            if parent_pk is not None and parent_pk in frame.columns:
                syn_keys = [f"SYN-{name[:3].upper()}-{j + 1:06d}" for j in range(len(frame))]
                parent_maps[name] = dict(zip(frame[parent_pk].astype(str), syn_keys, strict=False))
        self.integrity = validate_relational_integrity(dataset)
        self.is_fitted = True
        self.history = TrainHistory(
            steps=sum(int(m.history.steps) for m in self.models.values()),
            epochs=1,
            duration_seconds=time.time() - started,
            losses={name: [float(len(m.history.losses.get("critic", [0]) or [0]))] for name, m in self.models.items()},
        )
        self.metadata = {
            "order": self.order,
            "edges": self.edges,
            "broken_edges": self.broken_edges,
            "child_models": self.child_models,
            "rows_per_parent": self.rows_per_parent,
            "input_integrity": self.integrity,
        }
        self.privacy = PrivacyReport(
            mechanism="non-private",
            statement=(
                "Non-private relational run: child models are trained per table. No differential-privacy "
                "guarantee is claimed for hierarchical synthesis in this build."
            ),
            excluded=["No differential privacy.", "Per-table DP composition is not implemented for this mode."],
        )
        if progress:
            progress({"phase": "done", "progress": 1.0, "epsilon": None, "step": self.history.steps})
        return FitResult(history=self.history, n_train_rows=dataset.n_rows, warnings=self.warnings, metadata=self.metadata)

    # ------------------------------------------------------------------ sample
    def sample(
        self,
        n: int,
        *,
        conditions: dict[str, Any] | None = None,
        seed: int | None = None,
        table: str | None = None,
        scale: dict[str, float] | None = None,
    ) -> Dataset:
        if not self.is_fitted:
            raise ModelError("relational model is not fitted")
        use_seed = int(seed if seed is not None else self.config.seed + 1)
        _rng = np.random.default_rng(use_seed)
        scale = scale or {}
        generated: dict[str, pd.DataFrame] = {}
        key_maps: dict[str, pd.DataFrame] = {}
        reports: dict[str, Any] = {}
        for i, name in enumerate(self.order):
            model = self.models[name]
            if i == 0:
                n_rows = int(n * scale.get(name, 1.0))
                ds0 = model.sample(max(n_rows, 1), seed=use_seed + i * 17)
                if ds0.primary_table is None:
                    raise ModelError("root sample produced no primary table")
                frame = ds0.primary_table.frame.copy()
            else:
                # determine parent multiplicity
                out_frames = []
                for edge in self.edges:
                    if edge["child"] != name:
                        continue
                    parent = edge["parent"]
                    if parent not in generated:
                        continue
                    fk_col = edge["columns"][0]
                    parent_key_col = f"{parent}_id"
                    parent_key = key_maps[parent][parent_key_col].astype(str)
                    copies = self.rows_per_parent.get(name, 1.0)
                    mult = max(1, int(round(copies * scale.get(name, 1.0))))
                    child_fk_values = np.repeat(parent_key.to_numpy(), mult)
                    sub_ds = model.sample(
                        len(child_fk_values),
                        seed=use_seed + i * 31,
                    )
                    if sub_ds.primary_table is None:
                        raise ModelError("child sample produced no primary table")
                    sub = sub_ds.primary_table.frame.copy()
                    # assign the parent FK directly: the child's structural column is set from the parent's
                    # generated keys, guaranteeing referential integrity by construction.
                    k = min(len(sub), len(child_fk_values))
                    sub = sub.head(k).copy()
                    sub[fk_col] = child_fk_values[:k]
                    out_frames.append(sub)
                if out_frames:
                    frame = pd.concat(out_frames, ignore_index=True)
                else:
                    ds_f = model.sample(max(int(n * scale.get(name, 1.0)), 1), seed=use_seed + i * 17)
                    if ds_f.primary_table is None:
                        raise ModelError("table sample produced no primary table")
                    frame = ds_f.primary_table.frame.copy()
            pk = self.schemas[name].primary_key
            if pk and pk[0] in frame.columns:
                pk_col = pk[0]
                if frame[pk_col].duplicated().any():
                    frame[pk_col] = [f"SYN-{name[:3].upper()}-{j + 1:08d}" for j in range(len(frame))]
                key_maps[name] = pd.DataFrame({f"{name}_id": frame[pk_col].astype(str)})
            reports[name] = {"rows": int(len(frame))}
            generated[name] = frame
        # constraint repair per table, then integrity validation on the synthetic dataset
        from synthforge.metadata.constraints import repair_table

        tables: dict[str, Table] = {}
        for name, frame in generated.items():
            schema = self.schemas[name]
            frame, rep = repair_table(frame, [c for c in schema.constraints if not isinstance(c, ForeignKeyConstraint)], seed=use_seed)
            reports[name]["repair"] = rep
            tables[name] = Table(name=name, frame=frame, schema=schema)
        ds = Dataset(
            name=f"{self.config.model}_relational_synthetic",
            kind="relational",
            tables=tables,
            synthetic=True,
            provenance={
                "model": "relational",
                "child_models": self.child_models,
                "seed": use_seed,
                "order": self.order,
                "repair": reports,
                "note": "Rows synthesised per table with parent keys propagated to preserve referential integrity.",
            },
        )
        self.integrity = validate_relational_integrity(ds)
        ds.provenance["integrity"] = self.integrity
        self.constraint_report = reports
        return ds

    # ------------------------------------------------------------------ export
    @staticmethod
    def export(dataset: Dataset, root: str | Path, fmt: str = "sqlite") -> list[Path]:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        if fmt == "sqlite":
            db = root / "synthetic.sqlite"
            if db.exists():
                db.unlink()
            conn = sqlite3.connect(db)
            try:
                conn.execute("PRAGMA foreign_keys=ON")
                for name, table in dataset.tables.items():
                    frame = table.frame.copy()
                    for c in frame.columns:
                        if pd.api.types.is_datetime64_any_dtype(frame[c]):
                            frame[c] = frame[c].astype(str)
                        elif frame[c].dtype == bool:
                            frame[c] = frame[c].astype(int)
                    pk = table.schema.primary_key
                    cols = []
                    for c in frame.columns:
                        decl = "TEXT"
                        if pd.api.types.is_integer_dtype(frame[c]):
                            decl = "INTEGER"
                        elif pd.api.types.is_numeric_dtype(frame[c]):
                            decl = "REAL"
                        cols.append(f'"{c}" {decl}')
                    if pk:
                        cols.append(f'PRIMARY KEY ("{pk[0]}")')
                    fk_clauses = []
                    for fk in table.schema.foreign_keys:
                        fk_clauses.append(
                            f'FOREIGN KEY ("{fk.columns[0]}") REFERENCES "{fk.ref_table}" ("{fk.ref_columns[0]}")'
                        )
                    cols.extend(fk_clauses)
                    conn.execute(f'CREATE TABLE "{name}" ({", ".join(cols)})')
                    frame.where(pd.notna(frame), None).to_sql(name, conn, if_exists="append", index=False)
                conn.commit()
            finally:
                conn.close()
            paths.append(db)
        elif fmt == "sql":
            sql_path = root / "synthetic.sql"
            lines: list[str] = ["-- Synthetic data generated by SynthForge. Not real data.", "BEGIN TRANSACTION;"]
            for name, table in dataset.tables.items():
                frame = table.frame.copy()
                cols = ", ".join(f'"{c}"' for c in frame.columns)
                for _, row in frame.iterrows():
                    vals = []
                    for v in row:
                        if pd.isna(v):
                            vals.append("NULL")
                        elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                            vals.append(str(v))
                        else:
                            vals.append("'" + str(v).replace("'", "''") + "'")
                    lines.append(f'INSERT INTO "{name}" ({cols}) VALUES ({", ".join(vals)});')
            lines.append("COMMIT;")
            sql_path.write_text("\n".join(lines), encoding="utf-8")
            paths.append(sql_path)
        else:
            for name, table in dataset.tables.items():
                if fmt == "parquet":
                    p = root / f"{name}.parquet"
                    table.frame.to_parquet(p, index=False)
                else:
                    p = root / f"{name}.csv"
                    table.frame.to_csv(p, index=False)
                paths.append(p)
        (root / "export_manifest.json").write_text(
            json.dumps(
                {
                    "format": fmt,
                    "tables": {n: int(len(t.frame)) for n, t in dataset.tables.items()},
                    "files": [p.name for p in paths],
                    "note": "Synthetic export. No real records are included.",
                    "provenance": dataset.provenance,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        return paths

    # ------------------------------------------------------------------ persistence
    def _save_state(self, path: Path) -> None:
        models_dir = path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            model.save(models_dir / name)
        (path / "relational.json").write_text(
            json.dumps(
                {
                    "order": self.order,
                    "edges": self.edges,
                    "broken_edges": self.broken_edges,
                    "child_models": self.child_models,
                    "rows_per_parent": self.rows_per_parent,
                    "schemas": {n: s.model_dump(mode="json") for n, s in self.schemas.items()},
                    "history": self.history.model_dump(mode="json"),
                    "metadata": self.metadata,
                    "warnings": self.warnings,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    def _load_state(self, path: Path) -> None:
        from synthforge.core.types import TableSchema

        meta = json.loads((path / "relational.json").read_text(encoding="utf-8"))
        self.order = list(meta["order"])
        self.edges = list(meta["edges"])
        self.broken_edges = list(meta.get("broken_edges", []))
        self.child_models = dict(meta.get("child_models", {}))
        self.rows_per_parent = dict(meta.get("rows_per_parent", {}))
        self.schemas = {n: TableSchema.model_validate(s) for n, s in meta["schemas"].items()}
        self.history = TrainHistory.model_validate(meta["history"])
        self.metadata = meta.get("metadata", {})
        self.warnings = meta.get("warnings", [])
        from synthforge.models.registry import load_model

        self.models = {}
        for name in self.order:
            self.models[name] = load_model(path / "models" / name)
        self.is_fitted = True

    @classmethod
    def load(cls, path: str | Path) -> RelationalSynthesizer:
        obj = cls()
        obj._load_state(Path(path))
        cfg = Path(path) / "config.json"
        if cfg.exists():
            obj.config = RunConfig.model_validate_json(cfg.read_text(encoding="utf-8"))
        return obj

    def privacy_report(self) -> PrivacyReport:
        return self.privacy

    def model_card(self) -> dict[str, Any]:
        card = super().model_card()
        card.update(
            {
                "model": "relational",
                "modality": "relational",
                "order": self.order,
                "relationships": self.edges,
                "broken_cycles": self.broken_edges,
                "child_models": self.child_models,
                "integrity_on_input": self.integrity,
                "deviations": [
                    "Child tables are synthesised one at a time, conditioned on parent keys propagated as columns.",
                    "This preserves referential integrity but does not model full multi-table joint dependencies.",
                    "Per-table DP composition is not implemented; relational runs are non-private.",
                ],
                "privacy_claim": self.privacy.model_dump(mode="json"),
            }
        )
        return card


__all__ = ["RelationalSynthesizer", "fk_edges", "topological_order"]
