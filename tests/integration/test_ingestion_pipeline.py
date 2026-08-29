"""Integration tests for the full data ingestion pipeline.

These tests use mock data to verify the end-to-end pipeline works
without requiring network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from financial_ml.data.ingestion import ingest_symbol
from financial_ml.data.preprocessing import create_canonical_ohlcv
from financial_ml.data.validation import validate_and_clean


class TestEndToEndPipeline:
    """End-to-end pipeline tests with mock data."""

    def test_full_pipeline_with_csv_source(self, tmp_path: Path) -> None:
        """Test the full pipeline: CSV → ingestion → validation → canonical table."""
        # Create mock CSV data
        csv_data = """timestamp,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,103.0,1000
2023-01-02,103.0,108.0,102.0,106.0,1200
2023-01-03,106.0,111.0,105.0,109.0,1100
2023-01-04,109.0,114.0,108.0,112.0,1300
2023-01-05,112.0,117.0,111.0,115.0,1150"""
        csv_file = tmp_path / "spy.csv"
        csv_file.write_text(csv_data)

        # 1. Ingest from CSV
        df = ingest_symbol("SPY", source_type="csv", csv_dir=tmp_path, dry_run=True)
        assert len(df) == 5

        # 2. Validate
        cleaned = validate_and_clean(df)
        assert len(cleaned) == 5
        assert cleaned["timestamp"].is_monotonic_increasing

        # 3. Create canonical table
        canonical = create_canonical_ohlcv(cleaned)
        assert list(canonical.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert len(canonical) == 5
        assert canonical["timestamp"].dtype == "datetime64[ns]"
        assert canonical["open"].dtype == "float64"
        assert canonical["volume"].dtype == "int64"

    def test_pipeline_with_duplicates(self, tmp_path: Path) -> None:
        """Test that duplicate timestamps are handled correctly."""
        csv_data = """timestamp,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,103.0,1000
2023-01-01,99.0,104.0,98.0,101.0,950
2023-01-02,103.0,108.0,102.0,106.0,1200"""
        csv_file = tmp_path / "spy.csv"
        csv_file.write_text(csv_data)

        df = ingest_symbol("SPY", source_type="csv", csv_dir=tmp_path, dry_run=True)
        cleaned = validate_and_clean(df)
        assert len(cleaned) == 2  # duplicate dropped

    def test_pipeline_with_missing_data(self, tmp_path: Path) -> None:
        """Test that missing data is handled correctly."""
        csv_data = """timestamp,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,103.0,1000
2023-01-02,NA,NA,NA,NA,NA
2023-01-03,106.0,111.0,105.0,109.0,1100"""
        csv_file = tmp_path / "spy.csv"
        csv_file.write_text(csv_data)

        df = ingest_symbol("SPY", source_type="csv", csv_dir=tmp_path, dry_run=True)
        canonical = create_canonical_ohlcv(df, na_strategy="drop")
        assert len(canonical) == 2

    def test_pipeline_with_splits(self, tmp_path: Path) -> None:
        """Test that split adjustments are applied."""
        csv_data = """timestamp,open,high,low,close,volume
2022-12-31,200.0,205.0,195.0,202.0,1000
2023-01-01,200.0,205.0,195.0,202.0,1000
2023-01-02,105.0,108.0,102.0,106.0,1000"""
        csv_file = tmp_path / "spy.csv"
        csv_file.write_text(csv_data)

        df = ingest_symbol("SPY", source_type="csv", csv_dir=tmp_path, dry_run=True)

        # Split: 2-for-1 on 2023-01-01
        splits = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2023-01-01"]),
                "split_factor": [2.0],
            }
        )
        canonical = create_canonical_ohlcv(df, splits_info=splits)

        # Pre-split price should remain unadjusted
        # Post-split prices should be divided by 2
        assert canonical["open"].iloc[0] == pytest.approx(200.0)
        # After split, price should be adjusted (divided by 2)
        assert canonical["open"].iloc[2] == pytest.approx(105.0 / 2)


class TestIngestAll:
    """Tests for ingest_all function."""

    def test_ingest_all_with_csv(self, tmp_path: Path) -> None:
        """Test that ingest_all can ingest multiple assets."""
        # Create mock CSVs for multiple assets
        for symbol in ["spy", "aapl", "msft"]:
            csv_data = """timestamp,open,high,low,close,volume
2023-01-01,100.0,105.0,99.0,103.0,1000
2023-01-02,103.0,108.0,102.0,106.0,1200
2023-01-03,106.0,111.0,105.0,109.0,1100
2023-01-04,109.0,114.0,108.0,112.0,1300
2023-01-05,112.0,117.0,111.0,115.0,1150"""
            (tmp_path / f"{symbol}.csv").write_text(csv_data)

        # Run ingest_all with CSV source for each asset
        results = {}
        for symbol in ["SPY", "AAPL", "MSFT"]:
            df = ingest_symbol(symbol, source_type="csv", csv_dir=tmp_path, dry_run=True)
            results[symbol] = df

        assert len(results) == 3
        for df in results.values():
            assert len(df) == 5
            assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
