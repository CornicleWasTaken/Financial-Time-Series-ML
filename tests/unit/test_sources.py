"""Tests for data source implementations."""

from __future__ import annotations

from pathlib import Path

import pytest

from financial_ml.data.sources import CSVDataSource, YFinanceSource, get_source


class TestCSVDataSource:
    """Tests for CSVDataSource."""

    def test_csv_source_loads_correctly(self, tmp_path: Path) -> None:
        """Test that CSV source loads data with correct schema."""
        # Create test CSV
        csv_data = """timestamp,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,103.0,1000
2023-01-02,103.0,108.0,102.0,106.0,1200
2023-01-03,106.0,110.0,105.0,109.0,1100"""
        csv_file = tmp_path / "spy.csv"
        csv_file.write_text(csv_data)

        # Test the source
        source = CSVDataSource(csv_dir=tmp_path)
        df = source.fetch("SPY")

        # Assertions
        assert len(df) == 3
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df["timestamp"].dtype == "datetime64[ns]"
        assert df["open"].dtype == "float64"
        assert df["volume"].dtype == "int64"
        assert df["timestamp"].is_monotonic_increasing

    def test_csv_source_file_not_found(self, tmp_path: Path) -> None:
        """Test that CSV source raises FileNotFoundError for missing file."""
        source = CSVDataSource(csv_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            source.fetch("NONEXISTENT")


class TestYFinanceSource:
    """Tests for YFinanceSource."""

    def test_yfinance_source_initialization(self) -> None:
        """Test that YFinanceSource initializes correctly."""
        pytest.importorskip("yfinance")
        source = YFinanceSource()
        assert source.max_retries == 3
        assert source.retry_delay == 2.0

    def test_yfinance_source_structure(self) -> None:
        """Test that YFinanceSource has expected methods."""
        pytest.importorskip("yfinance")
        source = YFinanceSource()
        assert hasattr(source, "fetch")
        assert callable(source.fetch)


def test_get_source_factory() -> None:
    """Test the source factory function."""
    # Test CSV source
    csv_source = get_source("csv", csv_dir=Path("."))
    assert isinstance(csv_source, CSVDataSource)

    # Test yfinance source (only if available)
    try:
        import yfinance  # noqa: F401

        yf_source = get_source("yfinance")
        assert isinstance(yf_source, YFinanceSource)
    except ImportError:
        pytest.skip("yfinance not installed")

    # Test invalid source type
    with pytest.raises(ValueError):
        get_source("invalid_source")
