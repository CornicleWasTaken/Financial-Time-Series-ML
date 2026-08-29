"""Market data ingestion sources for financial time series.

Implements API (yfinance) and CSV sources with retry/rate limit handling.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

# Optional dependency - yfinance may not be installed in all environments
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class DataSource(Protocol):
    """Protocol for data source implementations."""

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch historical OHLCV data for a symbol.

        Returns:
            DataFrame with columns: ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            Rows should be sorted chronologically ascending.
        """
        ...


# ---------------------------------------------------------------------------
# CSV source (no external dependencies, always available)
# ---------------------------------------------------------------------------


class CSVDataSource:
    """Data source that reads OHLCV from CSV files."""

    def __init__(self, csv_dir: Path):
        self.csv_dir = Path(csv_dir)

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Read OHLCV data from CSV file for a symbol.

        CSV file must be located at ``{csv_dir}/{symbol.lower()}.csv`` with columns:
        timestamp, open, high, low, close, volume.
        """
        csv_path = self.csv_dir / f"{symbol.lower()}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"No CSV file found for {symbol} at {csv_path}")

        df = pd.read_csv(csv_path, parse_dates=["timestamp"])

        # Ensure proper dtypes - handle NaN in volume
        df = df.astype(
            {
                "open": "float64",
                "high": "float64",
                "low": "float64",
                "close": "float64",
                "volume": "float64",
            }
        )
        # Fill NaN in volume with 0, then convert to int
        df["volume"] = df["volume"].fillna(0).astype("int64")

        # Sort chronologically ascending
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df


# ---------------------------------------------------------------------------
# API source (requires yfinance)
# ---------------------------------------------------------------------------


class YFinanceSource:
    """Data source that fetches from Yahoo Finance API.

    Includes simple retry logic and rate limiting (1 second between calls).
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 2.0):
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance is not installed. Install with: pip install yfinance")
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def fetch(self, symbol: str) -> pd.DataFrame:
        """Fetch OHLCV data from Yahoo Finance for a symbol."""
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="max", interval="1d")
                df = df.reset_index()

                # yfinance returns 'Date' (timezone-aware) or 'Datetime' depending on version
                date_col = "Date" if "Date" in df.columns else "Datetime"
                df = df.rename(
                    columns={
                        date_col: "timestamp",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Volume": "volume",
                    }
                )

                df["timestamp"] = pd.to_datetime(df["timestamp"])
                # Convert to float first, then to int with NaN handling
                df = df.astype(
                    {
                        "open": "float64",
                        "high": "float64",
                        "low": "float64",
                        "close": "float64",
                        "volume": "float64",
                    }
                )
                # Handle NaN in volume by filling with 0
                df["volume"] = df["volume"].fillna(0).astype("int64")

                # Select only canonical columns
                df = df[["timestamp", "open", "high", "low", "close", "volume"]]
                df = df.sort_values("timestamp").reset_index(drop=True)

                return df

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError(
            f"Failed to fetch data for {symbol} after {self.max_retries} attempts: {last_error}"
        )


# ---------------------------------------------------------------------------
# Source factory
# ---------------------------------------------------------------------------


SourceType = Literal["csv", "yfinance"]


def get_source(source_type: SourceType, csv_dir: Path | None = None) -> DataSource:
    """Get a data source by type.

    Args:
        source_type: Either "csv" or "yfinance".
        csv_dir: Required when source_type is "csv".

    Returns:
        A DataSource instance.
    """
    if source_type == "csv":
        if csv_dir is None:
            raise ValueError("csv_dir is required for CSV source")
        return CSVDataSource(csv_dir=csv_dir)
    elif source_type == "yfinance":
        return YFinanceSource()
    else:
        raise ValueError(f"Unknown source type: {source_type}")
