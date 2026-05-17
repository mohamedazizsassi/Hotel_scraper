"""
Output writers for the feature pipeline — Parquet snapshot + PostgreSQL.

Two destinations, two purposes:

* **Parquet snapshot** — immutable, point-in-time training input under
  ``ml/artifacts/features_{YYYY-MM-DD}.parquet``. Same calendar day
  re-runs refuse to overwrite unless ``overwrite=True`` so a snapshot
  used in a model artifact cannot be silently mutated.

* **PostgreSQL `hotel_features`** — the live serving table consumed by
  the FastAPI backend. Writes are atomic: a single transaction wraps
  ``TRUNCATE`` + chunked ``INSERT``, so a failed write leaves the prior
  contents intact. The table is a derived materialisation; partial
  updates are never valid.

Schema discovery is automatic — column SQL types are inferred from
DataFrame dtypes (pandas → Postgres mapping below). List/dict columns
are JSON-serialised into ``JSONB`` to keep the contract honest under a
relational store.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Table,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine

from .config import POSTGRES_FEATURES_TABLE

logger = logging.getLogger(__name__)


PARQUET_FILENAME_TEMPLATE: str = "features_{date}.parquet"

# PostgreSQL's extended-protocol bind-parameter cap is 65 535 (uint16).
# pandas' to_sql(method="multi") issues one INSERT with
# chunksize * ncols placeholders, so chunksize * ncols must stay < 65 535.
# The assembled feature frame is ~77 columns as of 2026-05-17. 500 leaves
# ~24x headroom (500 * 77 ~= 38 500) and survives moderate feature growth.
# COPY-via-psycopg would lift the cap entirely; deferred until profiling
# shows multi-INSERT is the actual bottleneck.
DEFAULT_INSERT_CHUNKSIZE: int = 500


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def write_parquet_snapshot(
    df: pd.DataFrame,
    artifacts_dir: Path | str,
    *,
    snapshot_date: date | None = None,
    overwrite: bool = False,
) -> Path:
    """
    Write ``df`` to ``{artifacts_dir}/features_{YYYY-MM-DD}.parquet``.

    Parameters
    ----------
    df:
        Fully-assembled, validated feature frame. Columns are sorted
        alphabetically before write so successive runs produce
        byte-comparable files when the data is unchanged.
    artifacts_dir:
        Directory to write into. Created if missing.
    snapshot_date:
        Calendar date to embed in the filename. Defaults to today (UTC
        wall-clock is fine here — the snapshot is a human-facing label,
        not a join key).
    overwrite:
        If False (default) and the target file already exists, raise
        ``FileExistsError``. Snapshots feed model artifacts; rewriting
        one in place would silently invalidate any model trained on it.

    Returns
    -------
    Path
        Absolute path of the written file.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    snap = snapshot_date or date.today()
    out = artifacts_dir / PARQUET_FILENAME_TEMPLATE.format(date=snap.isoformat())

    if out.exists() and not overwrite:
        raise FileExistsError(
            f"{out} already exists. Pass overwrite=True to replace."
        )

    sorted_cols = sorted(df.columns)
    df[sorted_cols].to_parquet(out, compression="snappy", index=False)

    logger.info(
        "wrote parquet snapshot rows=%d cols=%d path=%s",
        len(df), len(sorted_cols), out,
    )
    return out.resolve()


def open_parquet_stream(
    path: Path | str,
    schema_from_df: pd.DataFrame,
    *,
    overwrite: bool = False,
) -> "_ParquetStream":
    """
    Open a row-group-streaming Parquet writer.

    The schema is fixed from ``schema_from_df`` (typically the first
    scrape-day's feature frame). Subsequent appends must produce the
    same dtypes; otherwise ``pa.Table.from_pandas`` raises a clear
    schema error. Use the context-manager form to guarantee close:

        with open_parquet_stream(path, first_day_df, overwrite=True) as w:
            for day_df in day_frames:
                w.append(day_df)

    Memory footprint is O(one day's frame); the file grows one row
    group per ``append`` call.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass overwrite=True.")
    if path.exists():
        path.unlink()

    sorted_cols = sorted(schema_from_df.columns)
    schema_table = pa.Table.from_pandas(
        schema_from_df[sorted_cols].head(0), preserve_index=False,
    )
    writer = pq.ParquetWriter(path, schema_table.schema, compression="snappy")
    return _ParquetStream(writer, path, sorted_cols)


class _ParquetStream:
    """Context manager + append API around pyarrow.parquet.ParquetWriter."""

    def __init__(self, writer: Any, path: Path, sorted_cols: list[str]) -> None:
        self._writer = writer
        self._path = path
        self._sorted_cols = sorted_cols
        self._rows: int = 0

    def append(self, df: pd.DataFrame) -> int:
        import pyarrow as pa
        table = pa.Table.from_pandas(df[self._sorted_cols], preserve_index=False)
        self._writer.write_table(table)
        self._rows += len(df)
        return len(df)

    def close(self) -> Path:
        self._writer.close()
        logger.info("streamed parquet rows=%d path=%s", self._rows, self._path)
        return self._path.resolve()

    def __enter__(self) -> "_ParquetStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# PostgreSQL
# ---------------------------------------------------------------------------

def write_postgres(
    df: pd.DataFrame,
    connection_string: str,
    *,
    table_name: str = POSTGRES_FEATURES_TABLE,
    chunksize: int = DEFAULT_INSERT_CHUNKSIZE,
) -> int:
    """
    Atomically replace ``table_name`` with the contents of ``df``.

    Implementation: open a single transaction, ``DROP TABLE IF EXISTS``
    + ``CREATE TABLE`` (so a feature-schema change is reflected in the
    live table) + chunked ``INSERT``; commit on success, rollback on any
    exception. The prior contents are preserved on failure (Postgres
    transactional DDL covers the DROP too).

    The target table is created on first call. Column SQL types are
    inferred from DataFrame dtypes; list/dict columns become ``JSONB``
    and are JSON-serialised at write time.

    Parameters
    ----------
    df:
        Fully-assembled, validated feature frame.
    connection_string:
        SQLAlchemy-compatible Postgres URI.
    table_name:
        Target table. Defaults to ``config.POSTGRES_FEATURES_TABLE``.
    chunksize:
        Rows per ``INSERT`` batch.

    Returns
    -------
    int
        Number of rows inserted.
    """
    if df.empty:
        raise ValueError("write_postgres: refusing to truncate-and-insert an empty frame")

    engine = create_engine(connection_string)
    try:
        return _atomic_truncate_insert(df, engine, table_name, chunksize)
    finally:
        engine.dispose()


def write_postgres_append(
    df: pd.DataFrame,
    connection_string: str,
    *,
    table_name: str = POSTGRES_FEATURES_TABLE,
    chunksize: int = DEFAULT_INSERT_CHUNKSIZE,
) -> int:
    """
    Append ``df`` to ``table_name`` (INSERT only, no TRUNCATE).

    Used during chunked full-reprocess: first chunk truncates + inserts,
    subsequent chunks insert only. All rows accumulate into one table.

    Parameters
    ----------
    df:
        Feature frame chunk to append.
    connection_string:
        SQLAlchemy-compatible Postgres URI.
    table_name:
        Target table.
    chunksize:
        Rows per ``INSERT`` batch.

    Returns
    -------
    int
        Number of rows inserted.
    """
    if df.empty:
        raise ValueError("write_postgres_append: refusing to insert an empty frame")

    engine = create_engine(connection_string)
    try:
        return _atomic_insert_only(df, engine, table_name, chunksize)
    finally:
        engine.dispose()


def _atomic_truncate_insert(
    df: pd.DataFrame, engine: Engine, table_name: str, chunksize: int,
) -> int:
    metadata = MetaData()
    table = _build_table(df, table_name, metadata)

    json_cols = _jsonb_columns(df)
    payload = _prepare_payload(df, json_cols)

    with engine.begin() as conn:
        # DROP+CREATE inside the transaction (Postgres supports
        # transactional DDL) so a feature-schema change -- a new or
        # removed column on the DataFrame -- is reflected in the live
        # table. TRUNCATE alone would preserve a stale column layout
        # and the subsequent INSERT would fail with UndefinedColumn.
        # If the INSERT fails, the DROP rolls back too: atomicity
        # is preserved.
        conn.execute(text(f'DROP TABLE IF EXISTS "{table_name}"'))
        table.create(conn)
        payload.to_sql(
            table_name,
            con=conn,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            method="multi",
        )

    logger.info(
        "wrote postgres table=%s rows=%d cols=%d",
        table_name, len(df), df.shape[1],
    )
    return len(df)


def _atomic_insert_only(
    df: pd.DataFrame, engine: Engine, table_name: str, chunksize: int,
) -> int:
    """INSERT rows without truncating (append-only)."""
    metadata = MetaData()
    table = _build_table(df, table_name, metadata)
    metadata.create_all(engine, tables=[table])

    json_cols = _jsonb_columns(df)
    payload = _prepare_payload(df, json_cols)

    with engine.begin() as conn:
        payload.to_sql(
            table_name,
            con=conn,
            if_exists="append",
            index=False,
            chunksize=chunksize,
            method="multi",
        )

    logger.info(
        "appended postgres table=%s rows=%d",
        table_name, len(df),
    )
    return len(df)


def swap_table_atomic(
    connection_string: str,
    *,
    staging_table: str,
    final_table: str,
) -> None:
    """
    Atomically replace ``final_table`` with ``staging_table``.

    Postgres supports transactional DDL: DROP and ALTER ... RENAME run
    inside a single transaction, so concurrent readers see either the
    old ``final_table`` or the new one — never a missing one. If either
    statement raises, both tables remain intact and the live serving
    table keeps its prior contents.

    Used by the chunked feature-build path so a multi-day reprocess
    either fully replaces ``hotel_features`` or leaves it untouched.
    A half-written staging table from a crashed run is harmless: the
    next run's day-1 write does DROP+CREATE on the staging name.
    """
    engine = create_engine(connection_string)
    try:
        with engine.begin() as conn:
            conn.execute(text(f'DROP TABLE IF EXISTS "{final_table}"'))
            conn.execute(
                text(f'ALTER TABLE "{staging_table}" RENAME TO "{final_table}"')
            )
    finally:
        engine.dispose()
    logger.info(
        "atomic swap: %s -> %s (transaction committed)",
        staging_table, final_table,
    )


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------

def _build_table(df: pd.DataFrame, table_name: str, metadata: MetaData) -> Table:
    columns: list[Column] = []
    for name in df.columns:
        sql_type = _infer_sql_type(df[name])
        columns.append(Column(name, sql_type))
    return Table(table_name, metadata, *columns)


def _infer_sql_type(s: pd.Series) -> Any:
    """
    Map a pandas Series to a SQLAlchemy column type.

    Inference rules (most-specific-first):
        * pandas extension dtypes (Int8/Int16/Int32/Int64, Float, Bool)
          map to their narrowest Postgres equivalent.
        * numpy datetime64 -> TIMESTAMP; ``date`` objects -> DATE.
        * object columns containing list/dict -> JSONB (sniffed on the
          first non-null value; mixed types are not supported).
        * everything else -> TEXT.
    """
    dtype = s.dtype

    if pd.api.types.is_bool_dtype(dtype):
        return Boolean()
    if pd.api.types.is_integer_dtype(dtype):
        kind = str(dtype).lower()
        if "8" in kind or "16" in kind:
            return SmallInteger()
        if "32" in kind:
            return Integer()
        return BigInteger()
    if pd.api.types.is_float_dtype(dtype):
        return Float()
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return DateTime(timezone=isinstance(dtype, pd.DatetimeTZDtype))

    sample = _first_non_null(s)
    if isinstance(sample, (list, dict)):
        return JSONB()
    if isinstance(sample, date) and not isinstance(sample, datetime):
        return Date()

    return String()


def _jsonb_columns(df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for name in df.columns:
        if df[name].dtype != object:
            continue
        sample = _first_non_null(df[name])
        if isinstance(sample, (list, dict)):
            cols.append(name)
    return cols


def _prepare_payload(df: pd.DataFrame, json_cols: list[str]) -> pd.DataFrame:
    if not json_cols:
        return df
    out = df.copy()
    for col in json_cols:
        out[col] = out[col].map(_json_dump_or_none)
    return out


def _json_dump_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    return json.dumps(value, default=str, ensure_ascii=False)


def _first_non_null(s: pd.Series) -> Any:
    for v in s:
        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue
        return v
    return None
