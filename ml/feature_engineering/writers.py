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
DEFAULT_INSERT_CHUNKSIZE: int = 10_000


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

    Implementation: open a single transaction, ``TRUNCATE`` the target
    table, then chunked ``INSERT``; commit on success, rollback on any
    exception. The prior contents are preserved on failure.

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
    metadata.create_all(engine, tables=[table])

    json_cols = _jsonb_columns(df)
    payload = _prepare_payload(df, json_cols)

    with engine.begin() as conn:
        conn.execute(text(f'TRUNCATE TABLE "{table_name}"'))
        # to_sql uses the same connection — TRUNCATE and INSERTs share
        # one transaction, so a failure mid-insert rolls back the truncate.
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
