"""Unit tests for ``calendar_features.add_calendar_features``."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.calendar_features import add_calendar_features


def _df(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"check_in": pd.to_datetime(dates)})


def test_missing_column_raises():
    with pytest.raises(RuntimeError, match="missing 'check_in'"):
        add_calendar_features(pd.DataFrame({"other": [1]}))


def test_does_not_mutate_input():
    df = _df(["2026-07-04"])
    before = df.copy()
    add_calendar_features(df)
    pd.testing.assert_frame_equal(df, before)


def test_basic_date_parts():
    # 2026-07-04 is a Saturday (dow=5), month=7, quarter=3, day=4.
    out = add_calendar_features(_df(["2026-07-04"]))
    r = out.iloc[0]
    assert r["check_in_dow"] == 5
    assert r["check_in_month"] == 7
    assert r["check_in_day_of_month"] == 4
    assert r["check_in_quarter"] == 3
    assert bool(r["is_weekend_checkin"]) is True  # Saturday
    # ISO week 27 of 2026
    assert r["check_in_week_of_year"] == 27


def test_weekend_flag_fri_sat_only():
    # Mon-Sun sweep around a known Monday (2026-07-06).
    out = add_calendar_features(_df([
        "2026-07-06",  # Mon
        "2026-07-07",  # Tue
        "2026-07-08",  # Wed
        "2026-07-09",  # Thu
        "2026-07-10",  # Fri
        "2026-07-11",  # Sat
        "2026-07-12",  # Sun
    ]))
    assert out["is_weekend_checkin"].tolist() == [
        False, False, False, False, True, True, False,
    ]


def test_ramadan_flag():
    # Ramadan 2026 window is 2026-02-17 .. 2026-03-18 inclusive.
    out = add_calendar_features(_df([
        "2026-02-16",  # day before
        "2026-02-17",  # first day
        "2026-03-01",  # middle
        "2026-03-18",  # last day
        "2026-03-19",  # day after
    ]))
    assert out["is_ramadan"].tolist() == [False, True, True, True, False]


def test_tunisia_public_holiday_fixed_and_islamic():
    # Fixed: Jul 25 (Republic Day). Islamic: 2026-05-27 (Eid al-Adha d1).
    out = add_calendar_features(_df([
        "2026-07-24",
        "2026-07-25",
        "2026-05-27",
    ]))
    assert out["is_tunisia_public_holiday"].tolist() == [False, True, True]


def test_eu_school_holiday_flags_are_independent():
    # 2026-07-04: inside FR (Été 2026-07-04..2026-08-31) and
    # inside UK (Summer 2026-07-18..2026-09-01)? No, UK summer starts 07-18.
    # So on 2026-07-04: FR=True, UK=False, DE=True (Sommer starts 2026-06-22).
    out = add_calendar_features(_df(["2026-07-04"]))
    r = out.iloc[0]
    assert bool(r["is_school_holiday_france"]) is True
    assert bool(r["is_school_holiday_germany"]) is True
    assert bool(r["is_school_holiday_uk"]) is False


def test_days_to_nearest_european_holiday_inside_window_is_zero():
    # 2026-07-04 is inside the FR Été window → 0 days.
    out = add_calendar_features(_df(["2026-07-04"]))
    assert int(out.iloc[0]["days_to_nearest_european_holiday"]) == 0


def test_days_to_nearest_european_holiday_outside_is_positive():
    # 2026-12-10: prev EU window = DE Herbst end 2026-11-07 (33d);
    # next = FR Noël start 2026-12-19 and UK Christmas start 2026-12-19
    # (9d). Min = 9.
    out = add_calendar_features(_df(["2026-12-10"]))
    d = int(out.iloc[0]["days_to_nearest_european_holiday"])
    assert d == 9, f"expected 9, got {d}"


def test_unparseable_dates_raise():
    df = pd.DataFrame({"check_in": ["not a date"]})
    with pytest.raises(RuntimeError, match="unparseable check_in"):
        add_calendar_features(df)
