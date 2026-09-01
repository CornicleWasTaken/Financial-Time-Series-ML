"""Integration tests for dataset building pipeline (Phase 3).

End-to-end tests that:
1. Write a feature matrix to a temp directory
2. Monkeypatch FEATURE_DATA_DIR to point at the temp file
3. Build the dataset
4. Verify all artifacts (parquet files + metadata) and leakage safety
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from financial_ml import config as config_module
from financial_ml.datasets.builder import DatasetBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_sample_feature_matrix(path: Path, n: int = 600) -> pd.DataFrame:
    """Write a small but realistic feature matrix to ``path`` and return it."""
    np.random.seed(7)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = np.random.randint(1000, 10_000, n)

    df = pd.DataFrame({
        "timestamp": dates,
        "close": close,
        "volume": volume,
        "return_lag_1": np.concatenate([[0.0], np.diff(close) / close[:-1]]),
        "return_lag_5": pd.Series(close).pct_change(5).fillna(0).to_numpy(),
        "log_return": np.log(close / np.concatenate([[close[0]], close[:-1]])),
        "sma_10": pd.Series(close).rolling(10).mean().bfill().to_numpy(),
        "sma_20": pd.Series(close).rolling(20).mean().bfill().to_numpy(),
        "sma_50": pd.Series(close).rolling(50).mean().bfill().to_numpy(),
        "ema_12": pd.Series(close).ewm(span=12, adjust=False).mean().to_numpy(),
        "ema_26": pd.Series(close).ewm(span=26, adjust=False).mean().to_numpy(),
        "price_sma_ratio_10": close / pd.Series(close).rolling(10).mean().bfill().to_numpy(),
        "rsi_14": 50 + np.random.randn(n) * 5,
        "macd": np.random.randn(n) * 0.1,
        "volatility_10": np.abs(np.random.randn(n)) * 0.01,
        "volatility_30": np.abs(np.random.randn(n)) * 0.015,
        "log_volume": np.log(volume),
        "volume_sma_10": pd.Series(volume).rolling(10).mean().bfill().to_numpy(),
        "volume_price_trend": np.random.randn(n) * 100,
        "day_of_week": dates.dayofweek,
        "month": dates.month,
        "quarter": dates.quarter,
        "is_month_start": dates.is_month_start.astype(int),
        "is_month_end": dates.is_month_end.astype(int),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


@pytest.fixture
def feature_matrix_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write a feature matrix and point FEATURE_DATA_DIR at it."""
    feature_path = tmp_path / "features" / "features.parquet"
    df = _write_sample_feature_matrix(feature_path)
    monkeypatch.setattr(config_module, "FEATURE_DATA_DIR", feature_path.parent)
    return {"df": df, "path": feature_path, "tmp": tmp_path}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDatasetPipelineEndToEnd:
    """End-to-end: feature matrix → dataset → splits → metadata."""

    def test_full_pipeline_writes_files(self, feature_matrix_dir) -> None:
        """The full pipeline writes per-fold parquets and a metadata file."""
        builder = DatasetBuilder(
            train_window=120,
            test_window=20,
            gap=0,
            step=20,
            initial_train_days=120,
            scaler_type="standard",
            exclude_features=["timestamp", "day_of_week", "month", "quarter", "is_month_start", "is_month_end"],
            output_dir=feature_matrix_dir["tmp"] / "dataset",
        )

        result = builder.build_dataset()

        # Per-fold artifacts
        for fold_name, fold_data in result["folds"].items():
            for split_name in ("train", "val", "test"):
                p = feature_matrix_dir["tmp"] / "dataset" / fold_name / f"{split_name}.parquet"
                assert p.exists(), f"Missing {p}"

        # Metadata
        metadata_path = feature_matrix_dir["tmp"] / "dataset" / "dataset_metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert "feature_columns" in metadata
        assert "target_column" in metadata
        assert "splits" in metadata
        assert "scaler_params" in metadata
        assert metadata["config_hash"]

    def test_no_future_data_in_training(self, feature_matrix_dir) -> None:
        """For every fold, training indices must precede val and test indices."""
        builder = DatasetBuilder(
            train_window=120,
            test_window=20,
            step=20,
            initial_train_days=120,
            exclude_features=["timestamp", "day_of_week"],
            output_dir=feature_matrix_dir["tmp"] / "dataset",
        )

        result = builder.build_dataset()
        metadata = result["metadata"]

        for split in metadata.splits:
            train_end = split["train_indices"]["end"]
            val_start = split["val_indices"]["start"]
            val_end = split["val_indices"]["end"]
            test_start = split["test_indices"]["start"]

            assert train_end <= val_start, "Training contains future data"
            assert val_end <= test_start, "Validation contains future data"

    def test_scaler_fit_per_fold(self, feature_matrix_dir) -> None:
        """Each fold has its own scaler parameters stored in metadata."""
        builder = DatasetBuilder(
            train_window=120,
            test_window=20,
            step=40,
            initial_train_days=120,
            scaler_type="standard",
            exclude_features=["timestamp", "day_of_week", "month", "quarter"],
            output_dir=feature_matrix_dir["tmp"] / "dataset",
        )

        result = builder.build_dataset()
        scaler_params = result["metadata"].scaler_params

        # At least one fold should have params
        assert len(scaler_params) > 0
        # Each fold should have either "mean"/"scale" (standard) or "min"/"scale" (minmax)
        for fold_idx, params in scaler_params.items():
            assert "mean" in params or "min" in params
            assert "scale" in params

    def test_target_column_aligned(self, feature_matrix_dir) -> None:
        """The target column should be present in train/val/test DataFrames."""
        builder = DatasetBuilder(
            train_window=120,
            test_window=20,
            step=20,
            initial_train_days=120,
            scaler_type="standard",
            exclude_features=["day_of_week", "month", "quarter", "is_month_start", "is_month_end"],
            output_dir=feature_matrix_dir["tmp"] / "dataset",
        )

        result = builder.build_dataset()
        first_fold = result["folds"]["fold_0"]

        for split_name in ("train", "val", "test"):
            df = first_fold[split_name]
            assert "target" in df.columns
            assert df["target"].notna().all()
            assert "timestamp" in df.columns


class TestDatasetConfigFromYaml:
    """Verify the config system exposes the new dataset section."""

    def test_dataset_section_loads(self) -> None:
        cfg = config_module.load_config()
        assert hasattr(cfg, "dataset")
        assert cfg.dataset.dataset.target.name == "next_return"
        assert cfg.dataset.dataset.splits.train_window == 100
        assert cfg.dataset.dataset.scaling.type in ("standard", "minmax")
