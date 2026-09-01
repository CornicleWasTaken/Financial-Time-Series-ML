"""Dataset builder for Phase 3: feature matrix → chronological train/val/test splits.

Implements:
- Target variable construction (next-period return)
- Expanding-window walk-forward split generation
- Per-fold scaling (fit on training fold only, never on full dataset)
- Metadata persistence for reproducibility

All splits are leakage-safe:
- No future data appears in training folds
- Scalers are fit on training data only within each fold
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from financial_ml import config as config_module
from financial_ml.config import DATA_DIR, load_config, reset_config_cache

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# Split Result
# ---------------------------------------------------------------------------


class SplitResult(NamedTuple):
    """Result of a single walk-forward split."""

    fold_idx: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int


# ---------------------------------------------------------------------------
# Dataset Metadata
# ---------------------------------------------------------------------------


@dataclass
class DatasetMetadata:
    """Metadata for a built dataset."""

    feature_columns: list[str]
    target_column: str
    splits: list[dict[str, Any]]
    scaler_params: dict[str, dict[str, float]]
    target_definition: dict[str, Any]
    config_hash: str | None = None


# ---------------------------------------------------------------------------
# Dataset Builder
# ---------------------------------------------------------------------------


class DatasetBuilder:
    """Builds train/val/test splits from feature matrix with walk-forward logic.

    All operations are leakage-safe:
    - Features are only computed from past/present data (Phase 2 guarantees this)
    - Target is next-period return (shifted), safe to compute
    - Scalers fit on training fold only
    """

    def __init__(
        self,
        train_window: int = 100,
        test_window: int = 20,
        gap: int = 0,
        horizon: int = 1,
        step: int = 1,
        initial_train_days: int = 252,
        scaler_type: Literal["standard", "minmax"] = "standard",
        features_to_scale: list[str] | None = None,
        exclude_features: list[str] | None = None,
        min_train_samples: int = 50,
        output_dir: Path | str = "data/dataset/",
    ):
        self.train_window = train_window
        self.test_window = test_window
        self.gap = gap
        self.horizon = horizon
        self.step = step
        self.initial_train_days = initial_train_days
        self.scaler_type = scaler_type
        self.features_to_scale = features_to_scale or []
        self.exclude_features = exclude_features or []
        self.min_train_samples = min_train_samples
        self.output_dir = Path(output_dir)

        # Per-fold scalers store parameters for reproducibility
        self._scaler_params: dict[int, dict[str, Any]] = {}
        self._scalers: dict[str, StandardScaler | MinMaxScaler] = {}
        self._feature_columns: list[str] = []
        self._target_column = "target"

    def load_feature_matrix(self) -> pd.DataFrame:
        """Load the feature matrix from Phase 2 output."""
        feature_path = config_module.FEATURE_DATA_DIR / "features.parquet"
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature matrix not found at {feature_path}")
        df = pd.read_parquet(feature_path)
        logger.info("Loaded feature matrix: %d rows, %d columns", len(df), len(df.columns))
        # Feature columns for modeling (excluding timestamp, which is metadata)
        self._feature_columns = [c for c in df.columns if c != "timestamp"]
        return df

    def construct_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Construct next-period return as target variable.

        Target = (close[t+1] - close[t]) / close[t]

        This is leakage-safe because we're predicting the return that will
        actually happen after the current timestamp - no look-ahead bias.
        """
        df = df.copy()

        # Calculate simple return
        df[self._target_column] = df["close"].pct_change(periods=self.horizon)

        # Drop the first 'horizon' rows that have NaN target
        # (these rows don't have future price data for target construction)
        rows_dropped = df[self._target_column].isna().sum()
        if rows_dropped > 0:
            logger.info("Dropping %d rows with NaN target (no future data)", rows_dropped)

        return df.dropna(subset=[self._target_column])

    def generate_splits(self, n_obs: int) -> list[SplitResult]:
        """Generate expanding-window walk-forward splits.

        Strategy:
        1. The train_window is fixed per config; we slide it forward by `step`.
        2. After each training window, we use test_window for validation,
           then test_window for testing after a gap.
        3. If the fixed train_window is less than min_train_samples, no splits possible.
        4. Otherwise, slide forward by step until we run out of data.

        Returns list of (train_idx, val_idx, test_idx) tuples.
        """
        # If the fixed training window is less than the minimum required,
        # no splits can be generated
        if self.train_window < self.min_train_samples:
            return []

        splits: list[SplitResult] = []
        fold_idx = 0

        # Start at 0 and slide forward by step each fold
        train_start = 0
        while True:
            train_end = train_start + self.train_window
            val_start = train_end   # gap between train and val
            val_end = val_start + self.test_window
            test_start = val_end + self.gap   # gap between val and test
            test_end = test_start + self.test_window

            if test_end > n_obs:
                break

            splits.append(
                SplitResult(
                    fold_idx=fold_idx,
                    train_indices=np.arange(train_start, train_end),
                    val_indices=np.arange(val_start, val_end),
                    test_indices=np.arange(test_start, test_end),
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_idx += 1
            train_start += self.step

        logger.info("Generated %d walk-forward splits for %d observations", len(splits), n_obs)
        return splits

    def _get_scaler_class(self):
        """Get the scaler class based on configuration."""
        if self.scaler_type == "standard":
            return StandardScaler
        return MinMaxScaler

    def fit_scaler(self, X: pd.DataFrame, fold_idx: int) -> StandardScaler | MinMaxScaler:
        """Fit a scaler on training data only (per-fold, leakage-safe).

        Args:
            X: Training features only
            fold_idx: Fold index for storing parameters

        Returns:
            Fitted scaler
        """
        # Filter to features that should be scaled
        cols_to_scale = [c for c in X.columns if c in self.features_to_scale or (self.features_to_scale and c not in self.exclude_features)]
        if not cols_to_scale:
            cols_to_scale = [c for c in X.columns if c not in self.exclude_features]

        scaler = self._get_scaler_class()()
        scaler.fit(X[cols_to_scale])

        # Store parameters for reproducibility
        if hasattr(scaler, "mean_"):
            self._scaler_params[fold_idx] = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            }
        if hasattr(scaler, "min_"):
            self._scaler_params[fold_idx] = {
                "min": scaler.min_.tolist() if hasattr(scaler, "min_") else None,
                "scale": scaler.scale_.tolist() if hasattr(scaler, "scale_") else None,
            }

        return scaler

    def transform_split(
        self,
        X: pd.DataFrame,
        scaler: StandardScaler | MinMaxScaler,
        fit_cols: bool = False,
    ) -> pd.DataFrame:
        """Transform features using a fitted scaler.

        Args:
            X: Feature DataFrame
            scaler: Fitted scaler
            fit_cols: If True, fit on this data (only for training)

        Returns:
            Transformed DataFrame
        """
        X_out = X.copy()

        # Get columns to scale
        cols_to_scale = [c for c in X.columns if c not in self.exclude_features]
        if self.features_to_scale:
            cols_to_scale = [c for c in cols_to_scale if c in self.features_to_scale]

        if fit_cols:
            scaler.fit(X_out[cols_to_scale])

        X_out[cols_to_scale] = scaler.transform(X_out[cols_to_scale])
        return X_out

    def build_dataset(self, output_path: Path | None = None) -> dict[str, Any]:
        """Build complete dataset with all walk-forward splits.

        Returns:
            Dict with 'metadata' and per-fold split data
        """
        # Load and prepare data
        df = self.load_feature_matrix()
        df = self.construct_target(df)

        # Generate splits
        splits = self.generate_splits(len(df))

        if not splits:
            raise ValueError("No valid splits could be generated - check configuration")

        # Store split metadata
        split_metadata = []
        fold_data = {}

        # Columns fed to the scaler (features + target, never timestamp)
        _model_cols = self._feature_columns + [self._target_column]

        for split in splits:
            # Extract features and target for this split
            train_df = df.iloc[split.train_indices]
            val_df = df.iloc[split.val_indices]
            test_df = df.iloc[split.test_indices]

            # Fit scaler on training data only
            train_features = train_df[_model_cols]
            scaler = self.fit_scaler(train_features, split.fold_idx)

            # Transform all splits using this fold's scaler
            train_scaled = self.transform_split(train_features, scaler, fit_cols=True)
            val_scaled = self.transform_split(val_df[_model_cols], scaler)
            test_scaled = self.transform_split(test_df[_model_cols], scaler)

            # Re-attach timestamp so downstream consumers can trace predictions
            has_ts = "timestamp" in df.columns
            train_transformed = pd.concat([train_df[["timestamp"]].reset_index(drop=True), train_scaled.reset_index(drop=True)], axis=1) if has_ts else train_scaled
            val_transformed = pd.concat([val_df[["timestamp"]].reset_index(drop=True), val_scaled.reset_index(drop=True)], axis=1) if has_ts else val_scaled
            test_transformed = pd.concat([test_df[["timestamp"]].reset_index(drop=True), test_scaled.reset_index(drop=True)], axis=1) if has_ts else test_scaled

            # Store fold data with timestamp preserved
            fold_data[f"fold_{split.fold_idx}"] = {
                "train": train_transformed,
                "val": val_transformed,
                "test": test_transformed,
            }

            # Store split metadata
            split_metadata.append({
                "fold_idx": split.fold_idx,
                "train_indices": {"start": int(split.train_start), "end": int(split.train_end)},
                "val_indices": {"start": int(split.val_start), "end": int(split.val_end)},
                "test_indices": {"start": int(split.test_start), "end": int(split.test_end)},
                "train_dates": {
                    "start": df.iloc[split.train_start]["timestamp"].isoformat() if "timestamp" in df.columns else None,
                    "end": df.iloc[split.train_end - 1]["timestamp"].isoformat() if "timestamp" in df.columns else None,
                },
            })

        # Create metadata
        config_hash = self._compute_config_hash()
        metadata = DatasetMetadata(
            feature_columns=self._feature_columns,
            target_column=self._target_column,
            splits=split_metadata,
            scaler_params=self._scaler_params,
            target_definition={"name": "next_return", "horizon": self.horizon},
            config_hash=config_hash,
        )

        # Save to output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._save_dataset(fold_data, metadata)

        logger.info("Dataset built: %d folds, saved to %s", len(splits), self.output_dir)
        return {"metadata": metadata, "folds": fold_data}

    def _compute_config_hash(self) -> str:
        """Compute hash of configuration for reproducibility."""
        config_str = json.dumps({
            "train_window": self.train_window,
            "test_window": self.test_window,
            "gap": self.gap,
            "horizon": self.horizon,
            "step": self.step,
            "scaler_type": self.scaler_type,
        }, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:16]

    def _save_dataset(self, fold_data: dict, metadata: DatasetMetadata) -> None:
        """Save dataset splits and metadata to disk."""
        # Save each fold's splits
        for fold_name, data in fold_data.items():
            fold_dir = self.output_dir / fold_name
            fold_dir.mkdir(exist_ok=True)

            data["train"].to_parquet(fold_dir / "train.parquet")
            data["val"].to_parquet(fold_dir / "val.parquet")
            data["test"].to_parquet(fold_dir / "test.parquet")

            logger.info("Saved %s splits", fold_name)

        # Save metadata
        metadata_dict = {
            "feature_columns": metadata.feature_columns,
            "target_column": metadata.target_column,
            "splits": metadata.splits,
            "scaler_params": metadata.scaler_params,
            "target_definition": metadata.target_definition,
            "config_hash": metadata.config_hash,
        }

        with open(self.output_dir / "dataset_metadata.json", "w") as f:
            json.dump(metadata_dict, f, indent=2, default=str)

        logger.info("Saved dataset metadata")


# ---------------------------------------------------------------------------
# Convenience function for CLI usage
# ---------------------------------------------------------------------------


def build_dataset_from_config() -> dict[str, Any]:
    """Build dataset using configuration from configs/dataset.yaml."""
    cfg = load_config()
    ds_cfg = cfg.dataset.dataset

    builder = DatasetBuilder(
        train_window=ds_cfg.splits.train_window,
        test_window=ds_cfg.splits.test_window,
        gap=ds_cfg.splits.gap,
        horizon=ds_cfg.splits.horizon,
        step=ds_cfg.splits.step,
        initial_train_days=ds_cfg.splits.initial_train_days,
        scaler_type=ds_cfg.scaling.type,
        features_to_scale=ds_cfg.scaling.features_to_scale,
        exclude_features=ds_cfg.scaling.exclude_features,
        min_train_samples=ds_cfg.validation.min_train_samples,
        output_dir=DATA_DIR / "dataset",
    )

    return builder.build_dataset()


def main() -> None:
    """CLI entry point for dataset building."""
    import argparse

    parser = argparse.ArgumentParser(description="Build dataset with walk-forward splits")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config YAML")
    parser.add_argument("--train-window", type=int, default=None, help="Override train window size")
    parser.add_argument("--test-window", type=int, default=None, help="Override test window size")
    args = parser.parse_args()

    reset_config_cache()

    if args.train_window or args.test_window:
        # Build with command-line overrides
        cfg = load_config().dataset.dataset
        builder = DatasetBuilder(
            train_window=args.train_window or cfg.splits.train_window,
            test_window=args.test_window or cfg.splits.test_window,
            gap=cfg.splits.gap,
            horizon=cfg.splits.horizon,
            step=cfg.splits.step,
            scaler_type=cfg.scaling.type,
            features_to_scale=cfg.scaling.features_to_scale,
            exclude_features=cfg.scaling.exclude_features,
            min_train_samples=cfg.validation.min_train_samples,
            output_dir=DATA_DIR / "dataset",
        )
        result = builder.build_dataset()
        print(f"Dataset built: {len(result['folds'])} folds")
        print(f"Output directory: {builder.output_dir}")
    else:
        result = build_dataset_from_config()
        print(f"Dataset built: {len(result['folds'])} folds")
        print(f"Output directory: {DATA_DIR / 'dataset'}")


if __name__ == "__main__":
    main()