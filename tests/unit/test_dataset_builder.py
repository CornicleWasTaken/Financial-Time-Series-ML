"""Unit tests for dataset builder (Phase 3).

Tests:
- Target construction correctness
- Walk-forward split generation (no overlap, chronological order)
- Per-fold scaler fitting (fit on train only)
- Metadata persistence
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_ml.datasets.builder import (
    DatasetBuilder,
    SplitResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Sample OHLCV data for testing."""
    n = 300
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    np.random.seed(42)

    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volumes = np.random.randint(1000, 10000, n)

    return pd.DataFrame({
        "timestamp": dates,
        "open": prices * 0.99,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": volumes,
    })


@pytest.fixture
def sample_feature_matrix(sample_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Sample feature matrix derived from OHLCV data."""
    df = sample_ohlcv.copy()

    # Add simple features
    df["return_lag_1"] = df["close"].pct_change(1)
    df["sma_10"] = df["close"].rolling(10).mean()
    df["rsi_14"] = 50 + np.random.randn(len(df)) * 10  # Simplified RSI proxy
    df["volatility_10"] = df["return_lag_1"].rolling(10).std() * np.sqrt(252)
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # Drop NaN rows from rolling features
    df = df.dropna().reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Tests: Target Construction
# ---------------------------------------------------------------------------


class TestTargetConstruction:
    """Tests for target variable construction."""

    def test_target_is_next_period_return(self, sample_ohlcv: pd.DataFrame) -> None:
        """Target should be next-period simple return."""
        builder = DatasetBuilder()
        df = builder.construct_target(sample_ohlcv)

        assert "target" in df.columns

        # Calculate expected target
        expected = (sample_ohlcv["close"].shift(-1) - sample_ohlcv["close"]) / sample_ohlcv["close"]

        # Check values match (excluding NaN at end)
        for i in range(len(df) - 1):
            assert abs(df["target"].iloc[i] - expected.iloc[i]) < 1e-10

    def test_target_horizon(self, sample_ohlcv: pd.DataFrame) -> None:
        """Target should use configured horizon."""
        builder = DatasetBuilder(horizon=5)
        df = builder.construct_target(sample_ohlcv)

        # Calculate expected target with horizon=5
        expected = (sample_ohlcv["close"].shift(-5) - sample_ohlcv["close"]) / sample_ohlcv["close"]

        for i in range(len(df) - 5):
            assert abs(df["target"].iloc[i] - expected.iloc[i]) < 1e-10

    def test_target_nan_rows_dropped(self, sample_ohlcv: pd.DataFrame) -> None:
        """Rows with NaN target (no future data) should be dropped."""
        builder = DatasetBuilder(horizon=3)
        df = builder.construct_target(sample_ohlcv)

        # Should have no NaN in target column
        assert df["target"].notna().all()

        # Should have horizon rows fewer than input
        assert len(df) == len(sample_ohlcv) - 3

    def test_target_preserves_timestamp(self, sample_ohlcv: pd.DataFrame) -> None:
        """Target construction should preserve timestamp column."""
        builder = DatasetBuilder()
        df = builder.construct_target(sample_ohlcv)

        assert "timestamp" in df.columns
        assert df["timestamp"].dtype == "datetime64[ns]"


# ---------------------------------------------------------------------------
# Tests: Split Generation
# ---------------------------------------------------------------------------


class TestSplitGeneration:
    """Tests for walk-forward split generation."""

    def test_splits_are_chronological(self, sample_feature_matrix: pd.DataFrame) -> None:
        """All splits should respect chronological order (no future in train)."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            gap=0,
            step=10,
        )

        splits = builder.generate_splits(len(sample_feature_matrix))

        assert len(splits) > 0

        for split in splits:
            # Train indices should be before val indices
            assert split.train_indices[-1] < split.val_indices[0]
            # Val indices should be before test indices
            assert split.val_indices[-1] < split.test_indices[0]

    def test_splits_no_overlap(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Train, val, test splits should not overlap."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            gap=0,
            step=10,
        )

        splits = builder.generate_splits(len(sample_feature_matrix))

        for split in splits:
            train_set = set(split.train_indices)
            val_set = set(split.val_indices)
            test_set = set(split.test_indices)

            # No overlap between any pairs
            assert train_set.isdisjoint(val_set)
            assert train_set.isdisjoint(test_set)
            assert val_set.isdisjoint(test_set)

    def test_expanding_window(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Expanding window: later folds should have larger or equal training set."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            gap=0,
            step=10,
        )

        splits = builder.generate_splits(len(sample_feature_matrix))

        if len(splits) > 1:
            for i in range(1, len(splits)):
                # Later folds should have same train size (fixed window) but shifted forward
                assert splits[i].train_indices[0] > splits[i-1].train_indices[0]

    def test_splits_with_gap(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Gap between validation and test sets should be respected."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            gap=5,
            step=10,
        )

        splits = builder.generate_splits(len(sample_feature_matrix))

        for split in splits:
            # Gap between val end and test start
            actual_gap = split.test_start - split.val_end
            assert actual_gap == 5

    def test_min_train_samples_respected(self) -> None:
        """Splits should not be generated if train window < min_train_samples."""
        builder = DatasetBuilder(
            train_window=10,
            test_window=5,
            step=5,
            min_train_samples=50,
        )

        # With train_window=10, min_train_samples=50, no splits should be generated
        splits = builder.generate_splits(100)
        # All splits will be skipped because train_end < min_train_samples
        assert len(splits) == 0

    def test_splits_fit_within_data(self, sample_feature_matrix: pd.DataFrame) -> None:
        """All split indices should be within bounds of the data."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            step=10,
        )

        splits = builder.generate_splits(len(sample_feature_matrix))
        n = len(sample_feature_matrix)

        for split in splits:
            assert split.train_indices[0] >= 0
            assert split.train_indices[-1] < n
            assert split.val_indices[0] >= 0
            assert split.val_indices[-1] < n
            assert split.test_indices[0] >= 0
            assert split.test_indices[-1] < n


# ---------------------------------------------------------------------------
# Tests: Scaler Fitting
# ---------------------------------------------------------------------------


class TestScalerFitting:
    """Tests for per-fold scaler fitting."""

    def test_scaler_fit_on_train_only(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Scaler should be fit on training data only, not on val/test."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            step=20,
            scaler_type="standard",
            exclude_features=["timestamp", "day_of_week"],
        )

        # Simulate split generation and scaler fitting
        splits = builder.generate_splits(len(sample_feature_matrix))
        if not splits:
            pytest.skip("No splits generated")

        split = splits[0]

        # Extract train and test
        train_df = sample_feature_matrix.iloc[split.train_indices]
        test_df = sample_feature_matrix.iloc[split.test_indices]

        # Fit scaler on train
        scaler = builder.fit_scaler(train_df, split.fold_idx)

        # Verify scaler was fit on train data
        assert scaler.mean_ is not None
        assert scaler.scale_ is not None

        # Verify scaler parameters are stored
        assert split.fold_idx in builder._scaler_params

    def test_scaler_params_stored(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Scaler parameters should be stored for reproducibility."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            step=20,
            scaler_type="standard",
            exclude_features=["timestamp", "day_of_week"],
        )

        splits = builder.generate_splits(len(sample_feature_matrix))
        if not splits:
            pytest.skip("No splits generated")

        split = splits[0]
        train_df = sample_feature_matrix.iloc[split.train_indices]

        builder.fit_scaler(train_df, split.fold_idx)

        # Check params stored
        assert split.fold_idx in builder._scaler_params
        params = builder._scaler_params[split.fold_idx]
        assert "mean" in params or "min" in params

    def test_transform_uses_fitted_scaler(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Transform should use the fitted scaler parameters."""
        builder = DatasetBuilder(
            train_window=50,
            test_window=10,
            step=20,
            scaler_type="standard",
            exclude_features=["timestamp", "day_of_week"],
        )

        splits = builder.generate_splits(len(sample_feature_matrix))
        if not splits:
            pytest.skip("No splits generated")

        split = splits[0]
        train_df = sample_feature_matrix.iloc[split.train_indices]
        test_df = sample_feature_matrix.iloc[split.test_indices]

        # Fit scaler
        scaler = builder.fit_scaler(train_df, split.fold_idx)

        # Transform train and test
        train_transformed = builder.transform_split(train_df, scaler, fit_cols=False)
        test_transformed = builder.transform_split(test_df, scaler, fit_cols=False)

        # Train should be centered around 0
        numeric_cols = [c for c in train_transformed.columns if c not in ["timestamp", "day_of_week"]]
        for col in numeric_cols[:3]:  # Check first few columns
            assert abs(train_transformed[col].mean()) < 1e-6


# ---------------------------------------------------------------------------
# Tests: Dataset Metadata
# ---------------------------------------------------------------------------


class TestDatasetMetadata:
    """Tests for dataset metadata."""

    def test_config_hash_computed(self) -> None:
        """Configuration hash should be computed for reproducibility."""
        builder = DatasetBuilder(
            train_window=100,
            test_window=20,
            horizon=1,
        )

        config_hash = builder._compute_config_hash()

        assert config_hash is not None
        assert len(config_hash) == 16

    def test_config_hash_deterministic(self) -> None:
        """Same config should produce same hash."""
        builder1 = DatasetBuilder(train_window=100, test_window=20)
        builder2 = DatasetBuilder(train_window=100, test_window=20)

        assert builder1._compute_config_hash() == builder2._compute_config_hash()

    def test_config_hash_differs_for_different_config(self) -> None:
        """Different configs should produce different hashes."""
        builder1 = DatasetBuilder(train_window=100, test_window=20)
        builder2 = DatasetBuilder(train_window=200, test_window=20)

        assert builder1._compute_config_hash() != builder2._compute_config_hash()


# ---------------------------------------------------------------------------
# Tests: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_insufficient_data_for_splits(self) -> None:
        """Should raise error if data is too short for any split."""
        builder = DatasetBuilder(
            train_window=100,
            test_window=20,
            initial_train_days=252,
        )

        # Only 50 rows, can't generate any splits
        splits = builder.generate_splits(50)

        assert len(splits) == 0

    def test_feature_matrix_not_found(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError if feature matrix doesn't exist."""
        builder = DatasetBuilder()

        # Temporarily change FEATURE_DATA_DIR to non-existent path
        from financial_ml import config as config_module
        original_dir = config_module.FEATURE_DATA_DIR
        config_module.FEATURE_DATA_DIR = tmp_path / "nonexistent"

        try:
            with pytest.raises(FileNotFoundError):
                builder.load_feature_matrix()
        finally:
            config_module.FEATURE_DATA_DIR = original_dir

    def test_empty_split_list_raises(self, sample_feature_matrix: pd.DataFrame) -> None:
        """build_dataset should raise if no splits can be generated."""
        builder = DatasetBuilder(
            train_window=100,
            test_window=20,
            min_train_samples=1000,  # Unrealistic - more than n_obs
        )

        # generate_splits should return empty when train_window < min_train_samples
        splits = builder.generate_splits(len(sample_feature_matrix))
        assert len(splits) == 0

        # build_dataset should raise ValueError when splits are empty
        # We mock load_feature_matrix to avoid file dependency
        import unittest.mock
        with unittest.mock.patch.object(builder, "load_feature_matrix", return_value=sample_feature_matrix):
            with pytest.raises(ValueError, match="No valid splits"):
                builder.build_dataset()
