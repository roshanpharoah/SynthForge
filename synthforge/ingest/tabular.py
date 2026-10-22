"""Real tabular ingestion: CSV/TSV/Parquet/Excel/JSON-lines with actionable errors."""

from __future__ import annotations

import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from synthforge.core.determinism import hash_file
from synthforge.core.errors import IngestError, UnsupportedFormatError
from synthforge.core.types import Dataset, Table
from synthforge.ingest.profile import profile_dataset
from synthforge.metadata.schema import infer_table_schema

SUPPORTED = {".csv", ".tsv", ".txt", ".parquet", ".pq", ".xlsx", ".xlsm", ".json", ".jsonl", ".ndjson"}


def _read_head(path: Path, n: int = 65536) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(n)


def detect_encoding(path: Path) -> str:
    head = _read_head(path)
    if head.startswith(b"\xff\xfe") or head.startswith(b"\xfe\xff"):
        return "utf-16"
    if b"\x00" in head[:2048]:
        return "utf-16"
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            head.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    raise IngestError(
        "could not decode the file",
        hint="Re-export the file as UTF-8 or specify the encoding explicitly.",
        details={"path": path.name},
    )


def detect_delimiter(path: Path, encoding: str) -> str:
    sample = _read_head(path).decode(encoding, errors="replace")
    lines = [ln for ln in sample.splitlines() if ln.strip()][:30]
    if not lines:
        raise IngestError("the file is empty", hint="Provide at least a header row and one data row.")
    try:
        dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        counts = {d: sum(ln.count(d) for ln in lines) for d in [",", "\t", ";", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def count_lines(path: Path) -> int:
    count = 0
    with open(path, "rb") as fh:
        while True:
            block = fh.read(1 << 20)
            if not block:
                break
            count += block.count(b"\n")
    return count


def _make_row_skipper(step: int, offset: int) -> Callable[[int], bool]:
    """Deterministic row skipper used by pandas' skiprows callback."""

    def _skip(i: int) -> bool:
        return i > 0 and ((i - offset) % step != 0)

    return _skip


def _read_csv(path: Path, *, max_rows: int | None, seed: int) -> tuple[pd.DataFrame, list[str], dict]:
    encoding = detect_encoding(path)
    delimiter = detect_delimiter(path, encoding)
    warnings: list[str] = []
    total = count_lines(path) - 1
    skip = None
    if max_rows and total > int(max_rows * 1.2):
        step = max(1, total // max_rows)
        offset = int(np.random.default_rng(seed).integers(0, step))
        skip = _make_row_skipper(step, offset)
        warnings.append(
            f"Large file: deterministically sampled approximately {max_rows} of {total:,} rows for this dataset view. "
            "Increase max_rows or use the full profile to load everything."
        )
    try:
        df = pd.read_csv(
            path,
            sep=delimiter,
            encoding=encoding,
            skiprows=skip,
            nrows=max_rows if skip is None else None,
            low_memory=False,
            keep_default_na=True,
            na_values=["NULL", "null", "N/A", "n/a", "NA", "None", "none", "NaN", "nan", ""],
            on_bad_lines="error",
        )
    except pd.errors.ParserError as exc:
        raise IngestError(
            f"malformed delimited file: {exc}",
            hint="Check for inconsistent column counts or quoting. Fix the row or use a valid CSV.",
            details={"path": path.name, "encoding": encoding, "delimiter": delimiter},
        ) from exc
    except UnicodeDecodeError as exc:
        raise IngestError(
            f"encoding failure: {exc}",
            hint="Convert the file to UTF-8 or pass the correct encoding.",
        ) from exc
    if df.empty:
        raise IngestError("the file contained headers but no data rows", hint="Add data rows before ingesting.")
    return df, warnings, {"encoding": encoding, "delimiter": delimiter, "source_rows": total}


def _read_parquet(path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise IngestError(
            f"could not read parquet: {exc}",
            hint="Verify the file is a valid Parquet file (pyarrow) and not encrypted.",
        ) from exc
    return df, [], {"format": "parquet"}


def _read_excel(path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    try:
        df = pd.read_excel(path, sheet_name=0, engine="openpyxl")
    except Exception as exc:
        raise IngestError(
            f"could not read Excel workbook: {exc}",
            hint="Use .xlsx (openpyxl) or export the sheet to CSV. Legacy .xls requires xlrd.",
        ) from exc
    return df, ["Only the first worksheet was read."], {"format": "excel"}


def _read_jsonl(path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    records = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IngestError(
                    f"invalid JSON on line {lineno}: {exc.msg}",
                    hint="Each line must be a complete JSON object. Use a JSON array for nested documents.",
                    details={"line": lineno},
                ) from exc
            if isinstance(obj, dict):
                records.append(obj)
            else:
                records.append({"value": obj})
    if not records:
        raise IngestError("JSON-lines file contains no objects", hint="Add one JSON object per line.")
    df = pd.json_normalize(records, max_level=1)
    warnings = ["Nested JSON objects were flattened one level with dot-separated names."]
    return df, warnings, {"format": "jsonl"}


def _read_json(path: Path) -> tuple[pd.DataFrame, list[str], dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        raise IngestError(
            f"invalid JSON: {exc.msg}",
            hint="Provide a JSON array of objects or a JSON-lines file.",
            details={"line": exc.lineno},
        ) from exc
    if isinstance(data, dict):
        for key in ("data", "records", "rows", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data]
    if not isinstance(data, list):
        raise IngestError("JSON root must be an array of objects", hint="Use [ {...}, {...} ].")
    df = pd.json_normalize(data, max_level=1)
    return df, ["JSON nested objects were flattened one level."], {"format": "json"}


def read_tabular_file(
    path: str | Path, *, max_rows: int | None = None, seed: int = 42
) -> tuple[pd.DataFrame, list[str], dict]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        raise IngestError(f"file not found: {p}", hint="Check the path and upload the file again.")
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        raise UnsupportedFormatError(
            f"unsupported tabular format {suffix or '<none>'}",
            hint=f"Supported: {', '.join(sorted(SUPPORTED))}. Convert the file or use a database URL.",
        )
    if p.stat().st_size == 0:
        raise IngestError("file is empty", hint="Upload a non-empty file.")
    if suffix in (".csv", ".tsv", ".txt"):
        return _read_csv(p, max_rows=max_rows, seed=seed)
    if suffix in (".parquet", ".pq"):
        return _read_parquet(p)
    if suffix in (".xlsx", ".xlsm"):
        return _read_excel(p)
    if suffix in (".jsonl", ".ndjson"):
        return _read_jsonl(p)
    if suffix == ".json":
        return _read_json(p)
    raise UnsupportedFormatError(f"unsupported format {suffix}")


def load_tabular(
    path: str | Path,
    *,
    name: str | None = None,
    table_name: str | None = None,
    target_column: str | None = None,
    max_rows: int | None = None,
    seed: int = 42,
    max_categories: int = 50,
    treat_sentinels: bool = True,
    drop_direct_identifiers: bool = True,
    synthetic_source: bool = False,
    profile: bool = True,
) -> Dataset:
    p = Path(path).resolve()
    df, warnings, info = read_tabular_file(p, max_rows=max_rows, seed=seed)
    if len(df) < 2:
        raise IngestError(
            "at least two data rows are required to infer a schema",
            hint="Provide more rows; synthetic models need multiple observations.",
        )
    if len(df.columns) < 1:
        raise IngestError("no columns detected", hint="Ensure the file has a header row.")
    table_name = table_name or p.stem.replace(" ", "_").replace("-", "_")
    try:
        schema, cleaned, schema_report = infer_table_schema(
            df,
            table_name,
            target_column=target_column,
            max_categories=max_categories,
            treat_sentinels=treat_sentinels,
            drop_direct_identifiers=drop_direct_identifiers,
        )
    except ValueError as exc:
        raise IngestError(str(exc), hint="Rename duplicate columns and retry.") from exc
    warnings.extend(schema_report.get("warnings", []))
    if target_column and target_column not in cleaned.columns:
        raise IngestError(
            f"target column {target_column!r} not found",
            hint=f"Available columns: {', '.join(map(str, cleaned.columns))}",
        )
    table = Table(name=table_name, frame=cleaned, schema=schema)
    dataset = Dataset(
        name=name or p.stem,
        kind="tabular",
        tables={table_name: table},
        source={
            "type": "file",
            "format": info.get("format", p.suffix.lstrip(".")),
            "path": str(p),
            "sha256": hash_file(p),
            "bytes": p.stat().st_size,
            **{k: v for k, v in info.items() if k not in ("format",)},
        },
        warnings=warnings,
        synthetic=False,
        provenance={"synthetic_source": synthetic_source, "loader": "synthforge.ingest.tabular"},
    )
    dataset.profile = profile_dataset(dataset, seed=seed) if profile else {}
    return dataset


def load_csv(path: str | Path, **kwargs: Any) -> Dataset:
    return load_tabular(path, **kwargs)


def load_jsonl(path: str | Path, **kwargs: Any) -> Dataset:
    return load_tabular(path, **kwargs)


def load_parquet(path: str | Path, **kwargs: Any) -> Dataset:
    return load_tabular(path, **kwargs)


def load_excel(path: str | Path, **kwargs: Any) -> Dataset:
    return load_tabular(path, **kwargs)


def write_dataset_frames(dataset: Dataset, fmt: str, root: str | Path) -> list[Path]:
    """Write synthetic tables for export."""
    out = Path(root)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, table in dataset.tables.items():
        if fmt == "csv":
            p = out / f"{name}.csv"
            table.frame.to_csv(p, index=False)
        elif fmt in ("parquet", "pq"):
            p = out / f"{name}.parquet"
            table.frame.to_parquet(p, index=False)
        else:
            raise UnsupportedFormatError(f"export format {fmt}", hint="Use csv or parquet.")
        paths.append(p)
    return paths
