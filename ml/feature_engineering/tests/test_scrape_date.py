"""
Unit tests for ``scrape_date.add_scrape_date``.

The derived ``scrape_date`` column is the temporal-scoping key that
prevents peer aggregates from leaking future observations into past
rows. The tests below pin its dtype, value, and contract.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.scrape_date import SCRAPE_DATE_COL, add_scrape_date


def _df(scraped_at_values: list[pd.Timestamp]) -> pd.DataFrame:
    return pd.DataFrame({
        "scraped_at": pd.to_datetime(scraped_at_values, utc=True),
        "price_per_night": [100.0] * len(scraped_at_values),
    })


def test_adds_scrape_date_column() -> None:
    df = _df([pd.Timestamp("2026-05-14T08:00:00Z")])
    out = add_scrape_date(df)
    assert SCRAPE_DATE_COL in out.columns


def test_scrape_date_is_midnight_utc_of_scraped_day() -> None:
    df = _df([
        pd.Timestamp("2026-05-14T08:00:00Z"),
        pd.Timestamp("2026-05-14T23:59:59Z"),
        pd.Timestamp("2026-05-15T00:00:01Z"),
    ])
    out = add_scrape_date(df)
    assert list(out[SCRAPE_DATE_COL]) == [
        pd.Timestamp("2026-05-14"),
        pd.Timestamp("2026-05-14"),
        pd.Timestamp("2026-05-15"),
    ]


def test_scrape_date_dtype_is_naive_datetime64() -> None:
    """
    A naive datetime64 groupby-aligns cleanly with check_in. A tz-aware
    column would silently fail to merge with check_in in the peer-key
    tuples (different dtypes). Resolution (ns/us/ms) is left to pandas.
    """
    df = _df([pd.Timestamp("2026-05-14T08:00:00Z")])
    out = add_scrape_date(df)
    assert pd.api.types.is_datetime64_any_dtype(out[SCRAPE_DATE_COL])
    # Specifically: not tz-aware.
    assert getattr(out[SCRAPE_DATE_COL].dtype, "tz", None) is None


def test_missing_scraped_at_fails_loudly() -> None:
    df = pd.DataFrame({"price_per_night": [100.0]})
    with pytest.raises(RuntimeError, match="missing 'scraped_at'"):
        add_scrape_date(df)


def test_does_not_mutate_input() -> None:
    df = _df([pd.Timestamp("2026-05-14T08:00:00Z")])
    snapshot = df.copy(deep=True)
    add_scrape_date(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_preserves_row_count() -> None:
    df = _df([
        pd.Timestamp("2026-05-01T08:00:00Z"),
        pd.Timestamp("2026-05-02T08:00:00Z"),
        pd.Timestamp("2026-05-03T08:00:00Z"),
    ])
    out = add_scrape_date(df)
    assert len(out) == len(df)


def test_distinct_days_counted_correctly(caplog: pytest.LogCaptureFixture) -> None:
    df = _df([
        pd.Timestamp("2026-05-01T08:00:00Z"),
        pd.Timestamp("2026-05-01T20:00:00Z"),  # same day
        pd.Timestamp("2026-05-02T08:00:00Z"),
    ])
    out = add_scrape_date(df)
    assert out[SCRAPE_DATE_COL].nunique() == 2
