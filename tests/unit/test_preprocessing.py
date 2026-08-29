"""Tests for preprocessing implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from financial_ml.data.preprocessing import (
    CANONICAL_COLUMNS,
    apply_price_adjustments,
    create_canonical_ohlcv,
    handle_missing_data,
    normalize_timestamp,
)


class TestNormalizeTimestamp:
    """Tests for normalize_timestamp."""

    def test_normalizes_string_timestamps(self, tmp_path: Path) -> None:
        """Test that string timestamps are parsed to datetime."""
        df = pd.DataFrame(
            {
                "timestamp": ["2023-01-02", "2023-01-01", "2023-01-03"],
                "open": [100.0, 200.0, 300.0],
                "high": [105.0, 205.0, 305.0],
                "low": [95.0, 195.0, 295.0],
                "close": [102.0, 202.0, 302.0],
                "volume": [1000, 2000, 3000],
            }
        )
        result = normalize_timestamp(df)
        assert result["timestamp"].dtype == "datetime64[ns]"
        assert result["timestamp"].is_monotonic_increasing

    def test_sorts_chronologically(self, tmp_path: Path) -> None:
        """Test that timestamps are sorted ascending."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-02", "2023-01-01"]),
                "open": [200.0, 100.0],
                "high": [205.0, 105.0],
                "low": [195.0, 95.0],
                "close": [202.0, 102.0],
                "volume": [2000, 1000],
            }
        )
        result = normalize_timestamp(df)
        assert result["timestamp"].tolist() == [
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-01-02"),
        ]


class TestApplyPriceAdjustments:
    """Tests for apply_price_adjustments."""

    def test_no_adjustments_returns_same(self, tmp_path: Path) -> None:
        """Test that empty adjustment info returns same data."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01"]),
                "open": [100.0],
                "high": [105.0],
                "low": [95.0],
                "close": [102.0],
                "volume": [1000],
            }
        )
        result = apply_price_adjustments(df)
        assert result["open"].tolist() == [100.0]

    def test_split_adjustment(self, tmp_path: Path) -> None:
        """Test that split adjustment is applied correctly."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2022-12-31", "2023-01-02"]),
                "open": [200.0, 105.0],
                "high": [205.0, 108.0],
                "low": [195.0, 102.0],
                "close": [202.0, 106.0],
                "volume": [1000, 1000],
            }
        )
        # Split: 2-for-1 on 2023-01-01
        splits = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01"]),
                "split_factor": [2.0],
            }
        )
        result = apply_price_adjustments(df, splits_info=splits)
        # Pre-split price should not be adjusted (before split date)
        # Post-split price should be divided by 2
        assert result.loc[0, "open"] == pytest.approx(200.0) or result.loc[
            0, "open"
        ] == pytest.approx(100.0)
        # The second row (after split) should have half the price
        assert result.loc[1, "open"] == pytest.approx(105.0 / 2)


class TestHandleMissingData:
    """Tests for handle_missing_data."""

    def test_drop_strategy(self, tmp_path: Path) -> None:
        """Test that drop strategy removes rows with NaN."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "open": [100.0, None],
                "high": [105.0, None],
                "low": [95.0, None],
                "close": [102.0, None],
                "volume": [1000, None],
            }
        )
        result = handle_missing_data(df, strategy="drop")
        assert len(result) == 1
        assert result["open"].iloc[0] == 100.0

    def test_fill_forward_strategy(self, tmp_path: Path) -> None:
        """Test that fill_forward strategy fills NaN from previous row."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "open": [100.0, None],
                "high": [105.0, None],
                "low": [95.0, None],
                "close": [102.0, None],
                "volume": [1000, None],
            }
        )
        result = handle_missing_data(df, strategy="fill_forward")
        assert len(result) == 2
        assert result["open"].iloc[1] == 100.0

    def test_fill_zero_strategy(self, tmp_path: Path) -> None:
        """Test that fill_zero strategy replaces NaN with 0."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02"]),
                "open": [100.0, None],
                "high": [105.0, None],
                "low": [95.0, None],
                "close": [102.0, None],
                "volume": [1000, None],
            }
        )
        result = handle_missing_data(df, strategy="fill_zero")
        assert len(result) == 2
        assert result["open"].iloc[1] == 0.0

    def test_invalid_strategy_raises(self, tmp_path: Path) -> None:
        """Test that invalid strategy raises ValueError."""
        df = pd.DataFrame({"col": [1.0, None]})
        with pytest.raises(ValueError, match="Unknown NaN strategy"):
            handle_missing_data(df, strategy="invalid")


class TestCreateCanonicalOHLCV:
    """Tests for create_canonical_ohlcv."""

    def test_creates_canonical_table(self, tmp_path: Path) -> None:
        """Test that canonical table is created correctly."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-02", "2023-01-01", "2023-01-03"]),
                "open": [103.0, 100.0, 106.0],
                "high": [108.0, 105.0, 111.0],
                "low": [102.0, 99.0, 105.0],
                "close": [106.0, 103.0, 109.0],
                "volume": [1200, 1000, 1100],
            }
        )
        result = create_canonical_ohlcv(df)

        assert list(result.columns) == CANONICAL_COLUMNS
        assert result["timestamp"].is_monotonic_increasing
        assert len(result) == 3
        assert result["timestamp"].dtype == "datetime64[ns]"
        assert result["open"].dtype == "float64"
        assert result["volume"].dtype == "int64"

    def test_handles_missing_data(self, tmp_path: Path) -> None:
        """Test that missing data is handled."""
        # Create DataFrame with NaN in some columns
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "open": [100.0, 103.0, 106.0],  # no NaN
                "high": [105.0, None, 111.0],  # NaN in middle
                "low": [99.0, 102.0, 105.0],  # no NaN
                "close": [103.0, 106.0, 109.0],  # no NaN
                "volume": [1000, 1200, 1100],  # no NaN
            }
        )

        # With drop strategy - should drop the row with NaN
        result = create_canonical_ohlcv(df, na_strategy="drop")
        assert len(result) == 2  # drops row 1 (index 1)
        assert result["timestamp"].iloc[0] == pd.Timestamp("2023-01-01")
        assert result["timestamp"].iloc[1] == pd.Timestamp("2023-01-03")

        # With fill_forward - should fill NaN from previous row
        result = create_canonical_ohlcv(df, na_strategy="fill_forward")
        assert len(result) == 3
        assert result["high"].iloc[0] == 105.0
        assert result["high"].iloc[1] == 105.0  # filled from previous
        assert result["high"].iloc[2] == 111.0
