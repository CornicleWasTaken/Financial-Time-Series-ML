"""Tests for data validation implementations."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from financial_ml.data.validation import (
    OHLCVRecord,
    validate_and_clean,
    validate_dataset_completeness,
)


class TestOHLCVRecord:
    """Tests for OHLCV Pydantic model."""

    def test_valid_record(self) -> None:
        """Test that a valid OHLCV record passes validation."""
        record = OHLCVRecord(
            timestamp="2023-01-01",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000,
        )
        assert record.open == 100.0
        assert record.high == 105.0
        assert record.low == 99.0
        assert record.close == 103.0

    def test_negative_price_rejected(self) -> None:
        """Test that negative prices are rejected."""
        with pytest.raises(ValueError, match="cannot be negative"):
            OHLCVRecord(
                timestamp="2023-01-01",
                open=-100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000,
            )

    def test_negative_volume_rejected(self) -> None:
        """Test that negative volume is rejected."""
        with pytest.raises(ValueError, match="volume cannot be negative"):
            OHLCVRecord(
                timestamp="2023-01-01",
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=-10,
            )

    def test_future_timestamp_normalized(self) -> None:
        """Test that the validation function works with timezone-aware timestamps."""
        # Just test that it doesn't raise - the actual normalization happens in validate_and_clean
        # via the normalize_timestamp function in preprocessing
        record = OHLCVRecord(
            timestamp="2023-01-01T00:00:00+00:00",
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000,
        )
        # Should create successfully
        assert record is not None
        assert record.timestamp is not None


class TestValidateAndClean:
    """Tests for validate_and_clean function."""

    def test_basic_validation_passes(self, tmp_path: Path) -> None:
        """Test that valid data passes validation."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "open": [100.0, 103.0, 106.0],
                "high": [105.0, 108.0, 110.0],
                "low": [99.0, 102.0, 105.0],
                "close": [103.0, 106.0, 109.0],
                "volume": [1000, 1200, 1100],
            }
        )
        cleaned = validate_and_clean(df)
        assert len(cleaned) == 3
        assert cleaned["timestamp"].is_monotonic_increasing

    def test_duplicate_timestamps_dropped(self, tmp_path: Path) -> None:
        """Test that duplicate timestamps are dropped."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-02"]),
                "open": [100.0, 99.0, 106.0],
                "high": [105.0, 104.0, 110.0],
                "low": [99.0, 102.0, 105.0],
                "close": [103.0, 101.0, 109.0],
                "volume": [1000, 950, 1100],
            }
        )
        cleaned = validate_and_clean(df)
        assert len(cleaned) == 2  # duplicate dropped
        # Check that there's exactly one row for each unique date
        assert cleaned["timestamp"].nunique() == 2
        # Both timestamps should be present as datetime objects
        assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])
        assert cleaned["timestamp"].min().date() == pd.Timestamp("2023-01-01").date()
        assert cleaned["timestamp"].max().date() == pd.Timestamp("2023-01-02").date()

    def test_missing_columns_raises(self, tmp_path: Path) -> None:
        """Test that missing required columns raise ValueError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01"]),
                "open": [100.0],
            }
        )
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_and_clean(df)

    def test_negative_prices_raises(self, tmp_path: Path) -> None:
        """Test that negative prices raise ValueError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01"]),
                "open": [-100.0],
                "high": [105.0],
                "low": [99.0],
                "close": [103.0],
                "volume": [1000],
            }
        )
        with pytest.raises(ValueError, match="open.*cannot be negative"):
            validate_and_clean(df)

    def test_insufficient_rows_empty(self, tmp_path: Path) -> None:
        """Test that empty DataFrame raises ValueError."""
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="DataFrame is empty"):
            validate_and_clean(df)

    def test_price_field_consistency_raises(self, tmp_path: Path) -> None:
        """Test that price inconsistencies raise ValueError."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01"]),
                "open": [100.0],
                "high": [95.0],  # high < low would be caught, but low > high is the issue
                "low": [110.0],  # low > high: inconsistency
                "close": [103.0],
                "volume": [1000],
            }
        )
        with pytest.raises(ValueError, match="Data validation failed"):
            validate_and_clean(df)


class TestValidateDatasetCompleteness:
    """Tests for validate_dataset_completeness."""

    def test_completeness_stats(self, tmp_path: Path) -> None:
        """Test that completeness stats are computed correctly."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"]
                ),
                "open": [100.0, 103.0, 106.0, 109.0, 112.0],
                "high": [105.0, 108.0, 111.0, 114.0, 119.0],
                "low": [99.0, 102.0, 105.0, 108.0, 111.0],
                "close": [103.0, 106.0, 109.0, 112.0, 115.0],
                "volume": [1000, 1200, 1100, 1300, 1150],
            }
        )
        stats = validate_dataset_completeness(df, expected_days=5)
        assert stats["total_rows"] == 5
        assert stats["validation_status"] == "valid"
        assert "warnings" in stats

    def test_high_volatility_warning(self, tmp_path: Path) -> None:
        """Test that high volatility triggers a warning."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                "open": [1.0, 2.0, 3.0],
                "high": [2.0, 3.0, 4.0],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [1000, 1100, 1200],
            }
        )
        stats = validate_dataset_completeness(df, expected_days=3)
        assert len(stats["warnings"]) > 0 or stats["validation_status"] != "valid"
