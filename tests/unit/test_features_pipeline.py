"""Tests for the feature pipeline orchestrator."""

from __future__ import annotations

import pandas as pd
import pytest

from financial_ml.features.pipeline import build_feature_matrix


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def canonical_ohlcv() -> pd.DataFrame:
    """Small canonical OHLCV DataFrame (10 trading days)."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05",
                    "2023-01-06", "2023-01-09", "2023-01-10", "2023-01-11",
                    "2023-01-12", "2023-01-13",
                ]
            ),
            "open": [100.0, 102.0, 101.0, 105.0, 110.0, 108.0, 112.0, 115.0, 113.0, 120.0],
            "high": [103.0, 105.0, 106.0, 112.0, 115.0, 114.0, 118.0, 120.0, 116.0, 125.0],
            "low": [98.0, 100.0, 99.0, 103.0, 107.0, 106.0, 110.0, 112.0, 111.0, 117.0],
            "close": [102.0, 101.0, 105.0, 110.0, 108.0, 112.0, 115.0, 113.0, 118.0, 120.0],
            "volume": [1000, 1200, 1100, 1300, 1500, 1400, 1600, 1700, 1550, 1800],
        }
    )


@pytest.fixture
def feature_config() -> list[dict]:
    """Feature config matching configs/features.yaml structure."""
    return [
        {"name": "return_lag_1", "enabled": True, "params": {"lag": 1, "method": "simple"}},
        {"name": "log_return", "enabled": True, "params": {"method": "log"}},
        {"name": "sma_10", "enabled": True, "params": {"window": 10}},
        {"name": "ema_12", "enabled": True, "params": {"span": 12}},
        {"name": "price_sma_ratio_10", "enabled": True, "params": {"window": 10}},
        {"name": "rsi_14", "enabled": True, "params": {"period": 14}},
        {"name": "macd", "enabled": True, "params": {"fast": 12, "slow": 26, "signal": 9}},
        {"name": "volatility_10", "enabled": True, "params": {"window": 10}},
        {"name": "volatility_30", "enabled": True, "params": {"window": 30}},
        {"name": "log_volume", "enabled": True, "params": {}},
        {"name": "volume_sma_10", "enabled": True, "params": {"window": 10}},
        {"name": "day_of_week", "enabled": True, "params": {}},
        {"name": "month", "enabled": True, "params": {}},
        {"name": "quarter", "enabled": True, "params": {}},
        {"name": "is_month_start", "enabled": True, "params": {}},
        {"name": "is_month_end", "enabled": True, "params": {}},
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildFeatureMatrix:
    """Tests for build_feature_matrix."""

    def test_returns_dataframe_with_timestamp(
        self, canonical_ohlcv: pd.DataFrame, feature_config: list[dict]
    ) -> None:
        """Feature matrix must contain a timestamp column."""
        result = build_feature_matrix(canonical_ohlcv, enabled_features=feature_config)
        assert "timestamp" in result.columns

    def test_preserves_row_count(
        self, canonical_ohlcv: pd.DataFrame, feature_config: list[dict]
    ) -> None:
        """Feature matrix must have same number of rows as input."""
        result = build_feature_matrix(canonical_ohlcv, enabled_features=feature_config)
        assert len(result) == len(canonical_ohlcv)

    def test_preserves_columns_structure(
        self, canonical_ohlcv: pd.DataFrame, feature_config: list[dict]
    ) -> None:
        """Feature matrix must contain expected feature columns."""
        result = build_feature_matrix(canonical_ohlcv, enabled_features=feature_config)
        expected_cols = {
            "timestamp",
            "return_lag_1",
            "log_return",
            "sma_10",
            "ema_12",
            "price_sma_ratio_10",
            "rsi_14",
            "macd",
            "volatility_10",
            "volatility_30",
            "log_volume",
            "day_of_week",
            "month",
            "quarter",
            "is_month_start",
            "is_month_end",
        }
        actual_cols = set(result.columns)
        missing = expected_cols - actual_cols
        assert not missing, f"Missing columns: {missing}"

    def test_chronologically_sorted(
        self, canonical_ohlcv: pd.DataFrame, feature_config: list[dict]
    ) -> None:
        """Feature matrix timestamps must be sorted ascending."""
        result = build_feature_matrix(canonical_ohlcv, enabled_features=feature_config)
        ts = pd.to_datetime(result["timestamp"])
        assert ts.is_monotonic_increasing

    def test_handles_empty_config(
        self, canonical_ohlcv: pd.DataFrame
    ) -> None:
        """Empty feature config should produce just timestamp column."""
        result = build_feature_matrix(canonical_ohlcv, enabled_features=[])
        assert list(result.columns) == ["timestamp"]
        assert len(result) == len(canonical_ohlcv)

    def test_skips_disabled_features(
        self, canonical_ohlcv: pd.DataFrame, feature_config: list[dict]
    ) -> None:
        """Disabled features should not appear in output."""
        config = [
            {"name": "return_lag_1", "enabled": True, "params": {"lag": 1, "method": "simple"}},
            {"name": "sma_10", "enabled": False, "params": {"window": 10}},
            {"name": "log_return", "enabled": True, "params": {"method": "log"}},
        ]
        result = build_feature_matrix(canonical_ohlcv, enabled_features=config)
        assert "return_lag_1" in result.columns
        assert "sma_10" not in result.columns
        assert "log_return" in result.columns

    def test_handles_unknown_feature_name(
        self, canonical_ohlcv: pd.DataFrame
    ) -> None:
        """Unknown feature names should be skipped with warning, not crash."""
        config = [
            {"name": "nonexistent_feature", "enabled": True, "params": {}},
            {"name": "return_lag_1", "enabled": True, "params": {"lag": 1, "method": "simple"}},
        ]
        # Should not raise
        result = build_feature_matrix(canonical_ohlcv, enabled_features=config)
        assert "return_lag_1" in result.columns
        assert "nonexistent_feature" not in result.columns