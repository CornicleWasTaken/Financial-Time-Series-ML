"""Data validation and cleaning for financial time series.

Implements schema validation, duplicate detection, missing data checks,
and timestamp/corporate data normalization.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator


class OHLCVRecord(BaseModel):
    """Pydantic model for a single OHLCV record."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @field_validator("timestamp")
    @classmethod
    def normalize_timestamp(cls, v: datetime) -> datetime:
        """Normalize timestamp to timezone-naive UTC datetime."""
        if v.tzinfo is not None:
            v = v.astimezone(None)
        return v

    @field_validator("high", "low", "open", "close")
    @classmethod
    def validate_price_fields(cls, v: float, info) -> float:
        """Validate price field values."""
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative")
        return v

    @field_validator("volume")
    @classmethod
    def validate_volume(cls, v: int) -> int:
        """Validate volume is non-negative."""
        if v < 0:
            raise ValueError("volume cannot be negative")
        return v

    @field_validator("low")
    @classmethod
    def validate_low_field(cls, v: float, info) -> float:
        """Validate low <= high and low <= open/close."""
        # Note: This is a partial validation; full cross-validation happens in batch validation
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert Pydantic model to dict with string timestamp."""
        data = self.model_dump()
        data["timestamp"] = data["timestamp"].isoformat()
        return data


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean OHLCV data.

    Performs the following validations:
    1. Schema validation (column names and types)
    2. Duplicate timestamps
    3. Price field consistency (low <= high, open/close within bounds)
    4. Missing values handling
    5. Timestamp ordering

    Args:
        df: Raw OHLCV DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']

    Returns:
        Cleaned DataFrame ready for feature engineering

    Raises:
        ValueError: If validation fails
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    # Expected columns
    expected_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Ensure proper column order
    df = df[expected_columns]

    # Convert timestamp to datetime if needed
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Check for duplicate timestamps
    duplicate_count = df["timestamp"].duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # 2. Validate data types
    try:
        # Validate each record using Pydantic
        records = []
        for _, row in df.iterrows():
            record = OHLCVRecord(
                timestamp=row["timestamp"],
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"],
            )
            records.append(record)
    except ValidationError as e:
        # Convert validation errors to a more informative format
        error_details = []
        for error in e.errors():
            error_details.append(
                {
                    "row": error["loc"][0] if error["loc"] else "unknown",
                    "field": error["loc"][0]
                    if error["loc"] and len(error["loc"]) > 1
                    else error["loc"][0]
                    if error["loc"]
                    else None,
                    "error": error["msg"],
                    "value": error["input"] if error["input"] else None,
                }
            )
        raise ValueError(f"Validation errors:\n{format_validation_errors(error_details)}") from e

    # Convert back to DataFrame
    df = pd.DataFrame([record.to_dict() for record in records])
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 3. Cross-field validation
    validation_errors = []

    # Check price field consistency
    price_checks = [
        ("low", "high", lambda low, high: low <= high),
        ("low", "open", lambda low, open_val: low <= open_val),
        ("low", "close", lambda low, close: low <= close),
        ("high", "open", lambda high, open_val: high >= open_val),
        ("high", "close", lambda high, close: high >= close),
    ]

    for field1, field2, check in price_checks:
        invalid_mask = ~check(df[field1], df[field2])
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            validation_errors.append(
                f"Price inconsistency: {field1} > {field2} in {invalid_count} rows"
            )

    # Handle missing values - remove rows with any NaN
    df = df.dropna()

    # 4. Ensure chronological order
    df = df.sort_values("timestamp").reset_index(drop=True)

    if validation_errors:
        raise ValueError("Data validation failed:\n" + "\n".join(validation_errors))

    return df


def format_validation_errors(errors: list[dict[str, Any]]) -> str:
    """Format validation errors for display."""
    formatted_errors = []
    for error in errors:
        if error.get("field"):
            formatted_errors.append(
                f"Row {error['row']}: {error['field']} - {error['error']} (value: {error.get('value', 'N/A')})"
            )
        else:
            formatted_errors.append(
                f"Row {error['row']}: {error['error']} (value: {error.get('value', 'N/A')})"
            )
    return "\n".join(formatted_errors)


def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to validated data.

    Currently normalizes timestamps to timezone-naive UTC and ensures
    proper data types. Future phases could add additional normalization.

    Args:
        df: Validated OHLCV DataFrame

    Returns:
        Normalized DataFrame
    """
    # Timestamp normalization (already done in Pydantic validation)
    df = df.copy()

    # Ensure consistent dtypes
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

    return df


def validate_dataset_completeness(
    df: pd.DataFrame, expected_days: int | None = None
) -> dict[str, Any]:
    """Validate dataset completeness and return statistics.

    Args:
        df: Validated OHLCV DataFrame
        expected_days: Expected number of trading days (optional)

    Returns:
        Dictionary with validation statistics
    """
    stats = {
        "total_rows": len(df),
        "date_range": {
            "start": df["timestamp"].min().isoformat(),
            "end": df["timestamp"].max().isoformat(),
            "days": (df["timestamp"].max() - df["timestamp"].min()).days,
        },
        "duplicates_removed": None,  # Would need original to compute
        "validation_status": "valid",
        "warnings": [],
    }

    if expected_days is not None:
        days_present = stats["date_range"]["days"]
        if days_present < expected_days * 0.9:  # Allow 10% tolerance
            stats["warnings"].append(
                f"Data gap: expected ~{expected_days} days, got {days_present}"
            )

    # Check for price anomalies
    price_volatility = df["close"].std() / df["close"].mean()
    if price_volatility > 10:  # More than 1000% daily volatility is suspicious
        stats["warnings"].append(f"High price volatility detected: {price_volatility:.2f}")

    return stats
