"""Tests for writers.py — Parquet snapshot + Postgres helpers."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import (
    BigInteger,
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    SmallInteger,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB

from feature_engineering import writers


# ---------------------------------------------------------------------------
# Parquet
# ---------------------------------------------------------------------------

def _toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "z_str": ["a", "b", "c"],
            "a_int": pd.array([1, 2, 3], dtype="Int32"),
            "m_float": [1.5, 2.5, 3.5],
        }
    )


def test_parquet_round_trip_equality(tmp_path: Path):
    df = _toy_frame()
    out = writers.write_parquet_snapshot(df, tmp_path, snapshot_date=date(2026, 4, 30))
    assert out.exists()
    assert out.name == "features_2026-04-30.parquet"

    read_back = pd.read_parquet(out)
    pd.testing.assert_frame_equal(
        read_back.reset_index(drop=True),
        df[sorted(df.columns)].reset_index(drop=True),
        check_dtype=True,
    )


def test_parquet_columns_written_sorted(tmp_path: Path):
    df = _toy_frame()
    out = writers.write_parquet_snapshot(df, tmp_path, snapshot_date=date(2026, 4, 30))
    read_back = pd.read_parquet(out)
    assert list(read_back.columns) == sorted(df.columns)


def test_parquet_creates_missing_artifacts_dir(tmp_path: Path):
    target = tmp_path / "nested" / "artifacts"
    out = writers.write_parquet_snapshot(
        _toy_frame(), target, snapshot_date=date(2026, 4, 30)
    )
    assert target.is_dir()
    assert out.parent == target.resolve()


def test_parquet_refuses_overwrite_by_default(tmp_path: Path):
    df = _toy_frame()
    writers.write_parquet_snapshot(df, tmp_path, snapshot_date=date(2026, 4, 30))
    with pytest.raises(FileExistsError):
        writers.write_parquet_snapshot(df, tmp_path, snapshot_date=date(2026, 4, 30))


def test_parquet_overwrite_replaces(tmp_path: Path):
    df1 = _toy_frame()
    out1 = writers.write_parquet_snapshot(df1, tmp_path, snapshot_date=date(2026, 4, 30))

    df2 = pd.DataFrame({"z_str": ["x"], "a_int": pd.array([99], dtype="Int32"), "m_float": [9.9]})
    out2 = writers.write_parquet_snapshot(
        df2, tmp_path, snapshot_date=date(2026, 4, 30), overwrite=True
    )
    assert out1 == out2
    read_back = pd.read_parquet(out2)
    assert len(read_back) == 1
    assert read_back["a_int"].iloc[0] == 99


def test_parquet_default_date_is_today(tmp_path: Path, monkeypatch):
    out = writers.write_parquet_snapshot(_toy_frame(), tmp_path)
    assert out.name == f"features_{date.today().isoformat()}.parquet"


def test_parquet_round_trip_preserves_jsonb_payload(tmp_path: Path):
    """List/dict object columns survive the parquet round-trip."""
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "tags": [["a", "b"], ["c"]],
            "meta": [{"k": 1}, {"k": 2}],
        }
    )
    out = writers.write_parquet_snapshot(df, tmp_path, snapshot_date=date(2026, 4, 30))
    read_back = pd.read_parquet(out)
    assert list(read_back["tags"].iloc[0]) == ["a", "b"]
    assert dict(read_back["meta"].iloc[0]) == {"k": 1}


# ---------------------------------------------------------------------------
# Postgres — guard + schema inference helpers
# ---------------------------------------------------------------------------

def test_write_postgres_refuses_empty_frame():
    with pytest.raises(ValueError, match="empty"):
        writers.write_postgres(pd.DataFrame(), "postgresql://unused/unused")


@pytest.mark.parametrize(
    "series, expected_type",
    [
        (pd.Series([True, False]), Boolean),
        (pd.Series(pd.array([1, 2], dtype="Int8")), SmallInteger),
        (pd.Series(pd.array([1, 2], dtype="Int16")), SmallInteger),
        (pd.Series(pd.array([1, 2], dtype="Int32")), Integer),
        (pd.Series(pd.array([1, 2], dtype="Int64")), BigInteger),
        (pd.Series([1, 2], dtype="int64"), BigInteger),
        (pd.Series([1.5, 2.5]), Float),
        (pd.Series(pd.to_datetime(["2026-01-01", "2026-01-02"])), DateTime),
        (pd.Series([date(2026, 1, 1), date(2026, 1, 2)]), Date),
        (pd.Series([{"a": 1}, {"b": 2}]), JSONB),
        (pd.Series([[1, 2], [3]]), JSONB),
        (pd.Series(["x", "y"]), String),
    ],
)
def test_infer_sql_type(series, expected_type):
    sql_type = writers._infer_sql_type(series)
    assert isinstance(sql_type, expected_type)


def test_jsonb_columns_detects_list_and_dict():
    df = pd.DataFrame(
        {
            "scalar": [1, 2],
            "tags": [["a"], ["b"]],
            "meta": [{"k": 1}, {"k": 2}],
            "text": ["x", "y"],
        }
    )
    assert set(writers._jsonb_columns(df)) == {"tags", "meta"}


def test_prepare_payload_serializes_json_cols_only():
    df = pd.DataFrame(
        {
            "scalar": [1, 2],
            "tags": [["a", "b"], ["c"]],
        }
    )
    out = writers._prepare_payload(df, ["tags"])
    assert out["scalar"].tolist() == [1, 2]
    assert out["tags"].tolist() == ['["a", "b"]', '["c"]']
    # original frame is not mutated
    assert df["tags"].iloc[0] == ["a", "b"]


def test_prepare_payload_no_json_cols_is_identity():
    df = pd.DataFrame({"a": [1, 2]})
    assert writers._prepare_payload(df, []) is df


def test_json_dump_or_none_handles_none_and_nan():
    assert writers._json_dump_or_none(None) is None
    assert writers._json_dump_or_none(float("nan")) is None
    assert writers._json_dump_or_none({"k": 1}) == '{"k": 1}'
    assert writers._json_dump_or_none(["a", "b"]) == '["a", "b"]'


def test_json_dump_or_none_serializes_dates():
    val = writers._json_dump_or_none({"d": date(2026, 4, 30)})
    assert "2026-04-30" in val


def test_first_non_null_skips_none_and_nan():
    assert writers._first_non_null(pd.Series([None, np.nan, "first", "second"])) == "first"
    assert writers._first_non_null(pd.Series([None, np.nan])) is None
    assert writers._first_non_null(pd.Series([], dtype=object)) is None


def test_infer_sql_type_object_all_null_is_string():
    assert isinstance(writers._infer_sql_type(pd.Series([None, None], dtype=object)), String)


def test_infer_sql_type_datetime_with_tz():
    s = pd.Series(pd.to_datetime(["2026-01-01"], utc=True))
    sql_type = writers._infer_sql_type(s)
    assert isinstance(sql_type, DateTime)
    assert sql_type.timezone is True


# ---------------------------------------------------------------------------
# Postgres projection — write_postgres inserts only POSTGRES_SERVING_COLUMNS
# ---------------------------------------------------------------------------

def _full_toy_frame() -> pd.DataFrame:
    """
    Minimal frame that contains all POSTGRES_SERVING_COLUMNS plus a few
    extra columns that should NOT be inserted (e.g. calendar / segment dims).
    """
    from feature_engineering.model_feature_sets import (
        POSTGRES_SERVING_COLUMNS,
        CALENDAR_FEATURES,
        SEGMENT_FEATURES,
    )
    data: dict = {}
    # Add all serving columns with dummy scalar values.
    for col in POSTGRES_SERVING_COLUMNS:
        data[col] = ["dummy"]
    # Add extra columns that must be excluded from hotel_features.
    for col in list(CALENDAR_FEATURES) + list(SEGMENT_FEATURES) + ["stars", "boarding_name", "room_name"]:
        if col not in data:
            data[col] = ["extra"]
    return pd.DataFrame(data)


def test_write_postgres_projects_to_serving_columns(monkeypatch):
    """write_postgres must pass only POSTGRES_SERVING_COLUMNS to the INSERT."""
    from feature_engineering.model_feature_sets import POSTGRES_SERVING_COLUMNS

    captured: dict = {}

    def fake_truncate_insert(df, engine, table_name, chunksize):
        captured["cols"] = list(df.columns)
        return len(df)

    class _FakeEngine:
        def dispose(self): pass

    monkeypatch.setattr(writers, "_atomic_truncate_insert", fake_truncate_insert)
    monkeypatch.setattr(writers, "create_engine", lambda uri: _FakeEngine())

    df = _full_toy_frame()
    writers.write_postgres(df, "postgresql://unused/unused")

    assert set(captured["cols"]) == set(POSTGRES_SERVING_COLUMNS), (
        f"Unexpected columns sent to Postgres: "
        f"{set(captured['cols']).symmetric_difference(set(POSTGRES_SERVING_COLUMNS))}"
    )


# ---------------------------------------------------------------------------
# Dim writers — round-trip via SQLite in-memory
# ---------------------------------------------------------------------------

def _dim_toy_frame() -> pd.DataFrame:
    """
    Toy assembled frame containing all CALENDAR_DIM_COLUMNS and
    SEGMENT_DIM_COLUMNS, with two distinct check_in dates and two
    (city_name, stars_int) pairs to verify deduplication.
    """
    from feature_engineering.model_feature_sets import (
        CALENDAR_DIM_COLUMNS,
        SEGMENT_DIM_COLUMNS,
    )
    n = 4
    data: dict = {}

    # check_in: two distinct values (repeated twice each)
    data["check_in"] = pd.to_datetime(
        ["2026-07-01", "2026-07-01", "2026-07-15", "2026-07-15"]
    )
    # stars_int: two distinct values
    data["stars_int"] = pd.array([3, 4, 3, 4], dtype="Int8")
    # city_name: two distinct values matched to stars_int
    data["city_name"] = ["hammamet", "djerba", "hammamet", "djerba"]

    # Calendar columns with dummy integer / boolean values
    for col in CALENDAR_DIM_COLUMNS:
        if col == "check_in":
            continue
        if col.startswith("is_") or col.startswith("days_to"):
            data[col] = [False] * n
        else:
            data[col] = pd.array([1] * n, dtype="Int8")

    # Segment columns with dummy string values
    for col in SEGMENT_DIM_COLUMNS:
        if col in ("check_in", "stars_int", "city_name"):
            continue
        data[col] = ["sahel_3"] * n

    return pd.DataFrame(data)


def _sqlite_engine():
    from sqlalchemy import create_engine as _ce
    return _ce("sqlite:///:memory:")


def test_write_calendar_dim_deduplicates_by_check_in():
    """After writing, calendar_dim should have one row per distinct check_in."""
    import sqlalchemy as sa

    engine = _sqlite_engine()
    df = _dim_toy_frame()
    n_written = writers.write_calendar_dim(engine, df)

    # Two distinct check_in dates → 2 rows
    assert n_written == 2
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT COUNT(*) FROM calendar_dim")).scalar()
    assert rows == 2


def test_write_calendar_dim_is_idempotent():
    """Calling write_calendar_dim twice produces the same row count."""
    import sqlalchemy as sa

    engine = _sqlite_engine()
    df = _dim_toy_frame()
    writers.write_calendar_dim(engine, df)
    writers.write_calendar_dim(engine, df)

    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT COUNT(*) FROM calendar_dim")).scalar()
    assert rows == 2


def test_write_segment_dim_deduplicates_by_city_stars():
    """After writing, segment_dim should have one row per (city_name, stars_int) pair."""
    import sqlalchemy as sa

    engine = _sqlite_engine()
    df = _dim_toy_frame()
    n_written = writers.write_segment_dim(engine, df)

    # Two distinct (city_name, stars_int) pairs → 2 rows
    assert n_written == 2
    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT COUNT(*) FROM segment_dim")).scalar()
    assert rows == 2


def test_write_segment_dim_is_idempotent():
    """Calling write_segment_dim twice produces the same row count."""
    import sqlalchemy as sa

    engine = _sqlite_engine()
    df = _dim_toy_frame()
    writers.write_segment_dim(engine, df)
    writers.write_segment_dim(engine, df)

    with engine.connect() as conn:
        rows = conn.execute(sa.text("SELECT COUNT(*) FROM segment_dim")).scalar()
    assert rows == 2
