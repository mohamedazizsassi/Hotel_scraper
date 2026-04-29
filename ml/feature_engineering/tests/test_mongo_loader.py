"""
Tests for ``feature_engineering.mongo_loader``.

We do not connect to a real MongoDB. The loader's testable surface is its
contract validator (``_assert_load_contract``): the actual Arrow read is
covered end-to-end in integration runs against a live collection.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.mongo_loader import (
    PROJECTED_FIELDS,
    REQUIRED_FIELDS,
    REQUIRED_NON_NULL_RATE,
    _assert_load_contract,
)


def _good_frame(n: int = 100) -> pd.DataFrame:
    """A frame that satisfies the load contract."""
    base = {f: [f"v{i}" for i in range(n)] for f in PROJECTED_FIELDS}
    # Required numeric fields need real values, not strings, but the
    # contract only checks non-null rate, so any non-null sentinel works.
    return pd.DataFrame(base)


def test_empty_frame_raises() -> None:
    with pytest.raises(RuntimeError, match="zero rows"):
        _assert_load_contract(pd.DataFrame())


def test_missing_projected_column_raises() -> None:
    df = _good_frame()
    df = df.drop(columns=["price"])
    with pytest.raises(RuntimeError, match="missing projected columns"):
        _assert_load_contract(df)


def test_required_field_below_threshold_raises() -> None:
    df = _good_frame(100)
    # Knock more than (1 - REQUIRED_NON_NULL_RATE) of price values to NaN.
    n_null = int((1 - REQUIRED_NON_NULL_RATE) * 100) + 5
    df.loc[: n_null - 1, "price"] = None
    with pytest.raises(RuntimeError, match="Required-field coverage check failed"):
        _assert_load_contract(df)


def test_required_field_at_threshold_passes() -> None:
    df = _good_frame(1000)
    # Exactly meeting REQUIRED_NON_NULL_RATE: drop a touch under (1 - rate)
    # so the check sees non_null_rate >= threshold.
    n_null = int((1 - REQUIRED_NON_NULL_RATE) * 1000) - 1
    df.loc[: n_null - 1, "price"] = None
    _assert_load_contract(df)  # should not raise


def test_required_fields_subset_of_projected() -> None:
    """Loud guard: every required field must be in the projection."""
    assert set(REQUIRED_FIELDS).issubset(set(PROJECTED_FIELDS))
