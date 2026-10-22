"""Relational ingestion from SQLAlchemy URLs and folders of CSVs with a relationship file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from synthforge.core.errors import IngestError
from synthforge.core.types import Dataset, ForeignKey, Table
from synthforge.ingest.profile import profile_dataset
from synthforge.ingest.tabular import SUPPORTED, read_tabular_file
from synthforge.metadata.constraints import ForeignKeyConstraint
from synthforge.metadata.schema import infer_table_schema


def _engine_from_url(url: str) -> Engine:
    if url.startswith("sqlite:///"):
        raw = url.replace("sqlite:///", "", 1)
        p = Path(raw)
        if not p.exists():
            raise IngestError(f"SQLite database not found: {p}", hint="Check the path to the .sqlite/.db file.")
    try:
        return create_engine(url, pool_pre_ping=True)
    except Exception as exc:
        raise IngestError(
            f"could not create database engine: {exc}",
            hint="For Postgres install the optional driver: pip install 'synthforge[postgres]'.",
        ) from exc


def sqlite_url(path: str | Path) -> str:
    return f"sqlite:///{Path(path).resolve().as_posix()}"


def load_database(
    url: str,
    *,
    name: str | None = None,
    tables: list[str] | None = None,
    schema: str | None = None,
    max_rows: int | None = None,
    target_columns: dict[str, str] | None = None,
    profile: bool = True,
    seed: int = 42,
) -> Dataset:
    engine = _engine_from_url(url)
    try:
        insp = inspect(engine)
        available = insp.get_table_names(schema=schema)
        if not available:
            raise IngestError("database contains no tables", hint="Create tables or select another schema.")
        chosen = tables or available
        unknown = [t for t in chosen if t not in available]
        if unknown:
            raise IngestError(
                f"tables not found: {unknown}",
                hint=f"Available tables: {', '.join(available)}",
            )
        frames: dict[str, pd.DataFrame] = {}
        pk_map: dict[str, list[str]] = {}
        fk_map: dict[str, list[ForeignKey]] = {}
        warnings: list[str] = []
        with engine.connect() as conn:
            for t in chosen:
                pk = list(insp.get_pk_constraint(t, schema=schema).get("constrained_columns") or [])
                pk_map[t] = pk
                fks = []
                for fk in insp.get_foreign_keys(t, schema=schema):
                    ref = fk.get("referred_table")
                    if ref and ref in chosen:
                        fks.append(
                            ForeignKey(
                                columns=list(fk.get("constrained_columns") or []),
                                ref_table=ref,
                                ref_columns=list(fk.get("referred_columns") or []),
                            )
                        )
                fk_map[t] = fks
                q = f'SELECT * FROM "{t}"' if not schema else f'SELECT * FROM "{schema}"."{t}"'
                if max_rows:
                    q += f" LIMIT {int(max_rows)}"
                    warnings.append(f"Table {t} limited to {max_rows} rows for this ingestion.")
                try:
                    frames[t] = pd.read_sql(text(q), conn)
                except Exception as exc:
                    raise IngestError(
                        f"could not read table {t!r}: {exc}",
                        hint="Check column types; unsupported database types should be cast in a view.",
                    ) from exc
    finally:
        engine.dispose()

    return _dataset_from_frames(
        frames,
        name=name or _db_name(url),
        primary_keys=pk_map,
        foreign_keys=fk_map,
        target_columns=target_columns,
        warnings=warnings,
        profile=profile,
        seed=seed,
        source={"type": "database", "url": _redact_url(url), "tables": chosen},
    )


def _db_name(url: str) -> str:
    tail = url.rstrip("/").split("/")[-1]
    return tail.split("?")[0] or "database"


def _redact_url(url: str) -> str:
    """Never persist credentials in source metadata or logs."""
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host = rest.split("@", 1)
    user = creds.split(":", 1)[0]
    return f"{scheme}://{user}:***@{host}"


def load_csv_folder(
    root: str | Path,
    *,
    name: str | None = None,
    relationships: str | Path | None = None,
    target_columns: dict[str, str] | None = None,
    max_rows: int | None = None,
    profile: bool = True,
    seed: int = 42,
) -> Dataset:
    root = Path(root)
    if not root.exists():
        raise IngestError(f"folder not found: {root}", hint="Check the path.")
    files = [p for p in sorted(root.iterdir()) if p.suffix.lower() in SUPPORTED and p.is_file()]
    if not files:
        raise IngestError(
            "no tabular files found in folder",
            hint=f"Supported extensions: {', '.join(sorted(SUPPORTED))}",
        )
    frames = {}
    warnings: list[str] = []
    for p in files:
        try:
            df, w, _ = read_tabular_file(p, max_rows=max_rows, seed=seed)
        except Exception as exc:
            raise IngestError(f"failed reading {p.name}: {exc}", hint="Exclude or repair this file.") from exc
        frames[p.stem] = df
        warnings.extend([f"{p.name}: {x}" for x in w])
    primary_keys: dict[str, list[str]] = {}
    foreign_keys: dict[str, list[ForeignKey]] = {}
    rel_path = Path(relationships) if relationships else root / "relationships.json"
    if rel_path.exists():
        try:
            rel = json.loads(rel_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise IngestError(f"invalid relationships file: {exc.msg}", hint="Fix the JSON or remove it.") from exc
        table_defs = rel.get("tables", rel if isinstance(rel, dict) else {})
        for t, spec in table_defs.items():
            if t not in frames:
                warnings.append(f"relationships.json references missing table {t!r}; ignored")
                continue
            if isinstance(spec, dict):
                if spec.get("primary_key"):
                    primary_keys[t] = list(spec["primary_key"])
                foreign_keys[t] = []
                for fk in spec.get("foreign_keys", []):
                    foreign_keys[t].append(ForeignKey.model_validate(fk))
    else:
        warnings.append("No relationships.json found; foreign keys inferred from *_id column names.")
        # infer from matching PK-like column names
        pk_by_name: dict[str, list[str]] = {}
        for t, df in frames.items():
            for c in df.columns:
                if c.lower().endswith("_id") and df[c].notna().all() and df[c].nunique() == len(df):
                    pk_by_name.setdefault(c.lower(), []).append(t)
                    primary_keys.setdefault(t, [c])
                    break
        for t, df in frames.items():
            fks = []
            for c in df.columns:
                lc = c.lower()
                if lc in pk_by_name and t not in pk_by_name[lc]:
                    ref = pk_by_name[lc][0]
                    ref_col = next(iter(primary_keys[ref]))
                    fks.append(ForeignKey(columns=[c], ref_table=ref, ref_columns=[ref_col]))
            foreign_keys[t] = fks
    return _dataset_from_frames(
        frames,
        name=name or root.name,
        primary_keys=primary_keys,
        foreign_keys=foreign_keys,
        target_columns=target_columns,
        warnings=warnings,
        profile=profile,
        seed=seed,
        source={"type": "csv_folder", "path": str(root.resolve())},
    )


def _dataset_from_frames(
    frames: dict[str, pd.DataFrame],
    *,
    name: str,
    primary_keys: dict[str, list[str]],
    foreign_keys: dict[str, list[ForeignKey]],
    target_columns: dict[str, str] | None,
    warnings: list[str],
    profile: bool,
    seed: int,
    source: dict[str, Any],
) -> Dataset:
    tables: dict[str, Table] = {}
    all_warnings = list(warnings)
    for tname, df in frames.items():
        try:
            schema, cleaned, report = infer_table_schema(
                df,
                tname,
                primary_key=primary_keys.get(tname),
                foreign_keys=foreign_keys.get(tname),
                target_column=(target_columns or {}).get(tname),
            )
        except ValueError as exc:
            raise IngestError(f"schema inference failed for table {tname!r}: {exc}") from exc
        tables[tname] = Table(name=tname, frame=cleaned, schema=schema)
        all_warnings.extend(report.get("warnings", []))
    dataset = Dataset(
        name=name,
        kind="relational" if len(tables) > 1 else "tabular",
        tables=tables,
        source=source,
        warnings=sorted(set(all_warnings)),
        provenance={"loader": "synthforge.ingest.database", "synthetic_source": False},
    )
    integrity = validate_relational_integrity(dataset)
    if profile:
        dataset.profile = profile_dataset(dataset, seed=seed)
        dataset.profile["relational"] = integrity
    dataset.warnings = sorted(set(dataset.warnings + integrity["warnings"]))
    return dataset


def validate_relational_integrity(dataset: Dataset) -> dict[str, Any]:
    """Report orphan FKs, missing referenced tables/columns, duplicate PKs and cycles."""
    tables = dataset.tables
    issues: list[dict[str, Any]] = []
    warnings: list[str] = []
    total_orphans = 0
    for tname, table in tables.items():
        fks = list(table.schema.foreign_keys) + [
            ForeignKey(columns=c.columns, ref_table=c.ref_table, ref_columns=c.ref_columns)
            for c in table.schema.constraints
            if isinstance(c, ForeignKeyConstraint)
        ]
        for fk in fks:
            if fk.ref_table not in tables:
                issues.append({"type": "missing_table", "table": tname, "ref_table": fk.ref_table})
                warnings.append(f"{tname} references missing table {fk.ref_table}")
                continue
            ref = tables[fk.ref_table]
            if any(c not in ref.frame.columns for c in fk.ref_columns) or any(
                c not in table.frame.columns for c in fk.columns
            ):
                issues.append({"type": "missing_column", "table": tname, "foreign_key": fk.model_dump()})
                continue
            left = table.frame[fk.columns[0]].dropna()
            right = ref.frame[fk.ref_columns[0]].dropna()
            orphan_mask = ~left.isin(set(right.tolist()))
            orphans = int(orphan_mask.sum())
            total_orphans += orphans
            if orphans:
                issues.append(
                    {
                        "type": "orphan_foreign_key",
                        "table": tname,
                        "columns": fk.columns,
                        "ref_table": fk.ref_table,
                        "orphans": orphans,
                        "rate": orphans / max(len(left), 1),
                    }
                )
                warnings.append(f"{orphans} orphan rows in {tname}.{fk.columns[0]} -> {fk.ref_table}")
        for pk in table.schema.primary_key:
            if pk in table.frame.columns:
                dup = int(table.frame[pk].duplicated().sum())
                if dup:
                    issues.append({"type": "duplicate_primary_key", "table": tname, "column": pk, "duplicates": dup})
                    warnings.append(f"{tname}.{pk} has {dup} duplicate values")
    cycles = find_fk_cycles(tables)
    for cyc in cycles:
        issues.append({"type": "cycle", "tables": cyc})
        warnings.append(f"Foreign-key cycle detected: {' -> '.join(cyc)}")
    return {
        "ok": not issues,
        "issues": issues,
        "orphan_foreign_keys": total_orphans,
        "cycles": cycles,
        "tables": list(tables),
        "warnings": warnings,
        "note": "Referential integrity is reported before training; cycles block hierarchical generation.",
    }


def find_fk_cycles(tables: dict[str, Table]) -> list[list[str]]:
    graph: dict[str, list[str]] = {t: [] for t in tables}
    for tname, table in tables.items():
        for fk in table.schema.foreign_keys:
            if fk.ref_table in graph:
                graph[tname].append(fk.ref_table)
    cycles: list[list[str]] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    stack: list[str] = []

    def dfs(node: str) -> None:
        if node in visiting:
            if node in stack:
                cycles.append(stack[stack.index(node) :] + [node])
            return
        if node in visited:
            return
        visiting.add(node)
        stack.append(node)
        for nxt in graph.get(node, []):
            dfs(nxt)
        stack.pop()
        visiting.discard(node)
        visited.add(node)

    for node in graph:
        dfs(node)
    # de-duplicate rotations
    unique: list[list[str]] = []
    seen = set()
    for cyc in cycles:
        key = tuple(sorted(cyc[:-1]))
        if key not in seen:
            seen.add(key)
            unique.append(cyc)
    return unique
