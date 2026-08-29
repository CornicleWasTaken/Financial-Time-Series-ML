"""Data preprocessing: convert raw data to canonical OHLCV format.

Implements timestamp normalization, corporate action adjustments, and
creation of the canonical OHLCV table used downstream in feature engineering.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Timestamp normalization
# ---------------------------------------------------------------------------


def normalize_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Normalize timestamp column to timezone-naive datetime.

    Ensures all timestamps are timezone-naive and sorted chronologically ascending.

    Args:
        df: DataFrame with timestamp column
        column: Name of the timestamp column

    Returns:
        DataFrame with normalized timestamps
    """
    df = df.copy()

    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    df[column] = pd.to_datetime(df[column])

    # Convert timezone-aware timestamps to naive
    if df[column].dt.tz is not None:
        df[column] = df[column].dt.tz_convert(None)

    # Sort chronologically
    df = df.sort_values(column).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Corporate action adjustments
# ---------------------------------------------------------------------------


def apply_price_adjustments(
    df: pd.DataFrame,
    splits_info: pd.DataFrame | None = None,
    dividends_info: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply split and dividend adjustments to OHLCV data.

    Applies historical split and dividend factors to all OHLC price columns.
    This is a simplified implementation — production code would use the
    cumulative-adjustment-factor approach with a single multiplicative factor
    per row.

    Args:
        df: Input OHLCV DataFrame
        splits_info: DataFrame with 'timestamp' and 'split_factor' columns
            (e.g., 2.0 for 2-for-1 split)
        dividends_info: DataFrame with 'timestamp' and 'dividend_amount' columns

    Returns:
        DataFrame with adjusted prices
    """
    df = df.copy()

    if splits_info is None:
        splits_info = pd.DataFrame(columns=["timestamp", "split_factor"])
    if dividends_info is None:
        dividends_info = pd.DataFrame(columns=["timestamp", "dividend_amount"])

    if splits_info.empty and dividends_info.empty:
        return df  # No adjustments to apply

    # Compute cumulative split factor per row
    if not splits_info.empty:
        splits_info = splits_info.sort_values("timestamp")
        cumulative_split = []
        current_factor = 1.0
        split_iter = iter(zip(splits_info["timestamp"], splits_info["split_factor"], strict=False))
        next_split = next(split_iter, None)
        for ts in df["timestamp"]:
            while next_split is not None and ts >= next_split[0]:
                current_factor *= next_split[1]
                next_split = next(split_iter, None)
            cumulative_split.append(current_factor)
        df["split_factor"] = cumulative_split
    else:
        df["split_factor"] = 1.0

    if not dividends_info.empty:
        dividends_info = dividends_info.sort_values("timestamp")
        cumulative_dividend = []
        cum_div = 0.0
        div_iter = iter(
            zip(dividends_info["timestamp"], dividends_info["dividend_amount"], strict=False)
        )
        next_div = next(div_iter, None)
        for ts in df["timestamp"]:
            while next_div is not None and ts >= next_div[0]:
                cum_div += next_div[1]
                next_div = next(div_iter, None)
            cumulative_dividend.append(cum_div)
        df["cum_dividend"] = cumulative_dividend
    else:
        df["cum_dividend"] = 0.0

    # Apply adjustments: divide OHLC by split factor, subtract cum-dividend
    df["open"] = df["open"] / df["split_factor"] - df["cum_dividend"]
    df["high"] = df["high"] / df["split_factor"] - df["cum_dividend"]
    df["low"] = df["low"] / df["split_factor"] - df["cum_dividend"]
    df["close"] = df["close"] / df["split_factor"] - df["cum_dividend"]

    df = df.drop(columns=["split_factor", "cum_dividend"], errors="ignore")

    return df


# ---------------------------------------------------------------------------
# Missing-data handling
# ---------------------------------------------------------------------------


def handle_missing_data(
    df: pd.DataFrame,
    strategy: Literal["drop", "fill_forward", "fill_zero"] = "drop",
) -> pd.DataFrame:
    """Handle missing values in OHLCV data.

    Args:
        df: OHLCV DataFrame
        strategy: Strategy for handling missing values

    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()

    if strategy == "drop":
        df = df.dropna()
    elif strategy == "fill_forward":
        df = df.ffill()
    elif strategy == "fill_zero":
        df = df.fillna(0)
    else:
        raise ValueError(f"Unknown NaN strategy: {strategy}")

    return df


# ---------------------------------------------------------------------------
# Canonical OHLCV construction
# ---------------------------------------------------------------------------


CANONICAL_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def create_canonical_ohlcv(
    df: pd.DataFrame,
    splits_info: pd.DataFrame | None = None,
    dividends_info: pd.DataFrame | None = None,
    na_strategy: Literal["drop", "fill_forward", "fill_zero"] = "drop",
) -> pd.DataFrame:
    """Build the canonical OHLCV table from a raw DataFrame.

    The canonical table has columns ``['timestamp', 'open', 'high', 'low',
    'close', 'volume']``, no NaN values, sorted chronologically ascending,
    with consistent dtypes. This is the table downstream phases consume.

    Args:
        df: Raw OHLCV DataFrame.
        splits_info: Optional corporate-action data for split adjustment.
        dividends_info: Optional corporate-action data for dividend adjustment.
        na_strategy: How to handle missing observations.

    Returns:
        Canonical OHLCV DataFrame.
    """
    # 1. Normalize timestamps
    df = normalize_timestamp(df)

    # 2. Handle missing data
    df = handle_missing_data(df, strategy=na_strategy)

    # 3. Apply price adjustments
    df = apply_price_adjustments(df, splits_info=splits_info, dividends_info=dividends_info)

    # 4. Select canonical columns and enforce dtypes
    df = df[CANONICAL_COLUMNS]
    df = df.astype(
        {
            "timestamp": "datetime64[ns]",
            "open": "float64",
            "high": "float64",
            "low": "float64",
            "close": "float64",
            "volume": "int64",
        }
    )

    # 5. Final sort
    df = df.sort_values("timestamp").reset_index(drop=True)

    return df
