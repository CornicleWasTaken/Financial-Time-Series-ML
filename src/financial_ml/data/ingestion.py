"""Ingestion script: pull market data from source and store as immutable raw data.

Usage:
    python -m financial_ml.data.ingestion --asset SPY
    python -m financial_ml.data.ingestion --all
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from financial_ml.config import RAW_DATA_DIR, load_config, reset_config_cache
from financial_ml.data.sources import CSVDataSource, YFinanceSource

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------


def raw_parquet_path(symbol: str) -> Path:
    """Path to the immutable Parquet file for a symbol."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_DATA_DIR / f"{symbol.lower()}.parquet"


# ---------------------------------------------------------------------------
# Ingestion logic
# ---------------------------------------------------------------------------


def ingest_symbol(
    symbol: str,
    source_type: str = "yfinance",
    csv_dir: Path | None = None,
    dry_run: bool = False,
) -> pd.DataFrame:
    """Ingest historical OHLCV data for a single symbol.

    Args:
        symbol: Ticker symbol.
        source_type: "yfinance" or "csv".
        csv_dir: Directory with CSV files (required for csv source).
        dry_run: If True, fetch but do not write to disk.

    Returns:
        Raw DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    """
    logger.info("Ingesting %s from %s", symbol, source_type)

    if source_type == "yfinance":
        source = YFinanceSource()
    elif source_type == "csv":
        if csv_dir is None:
            raise ValueError("csv_dir is required for CSV source")
        source = CSVDataSource(csv_dir=csv_dir)
    else:
        raise ValueError(f"Unknown source type: {source_type}")

    df = source.fetch(symbol)

    logger.info(
        "Fetched %d rows for %s (%s → %s)",
        len(df),
        symbol,
        df["timestamp"].min().date(),
        df["timestamp"].max().date(),
    )

    if not dry_run:
        out_path = raw_parquet_path(symbol)
        df.to_parquet(out_path, index=False)
        logger.info("Stored raw data to %s", out_path)

    return df


def ingest_all(
    assets_config: list | None = None,
    dry_run: bool = False,
) -> dict[str, pd.DataFrame]:
    """Ingest data for all enabled assets in the configuration.

    Applies rate-limiting between symbols (1 second delay).

    Args:
        assets_config: List of asset config dicts. If None, loads from configs/assets.yaml.
        dry_run: If True, fetch but do not write to disk.

    Returns:
        Dict mapping symbol → DataFrame.
    """
    if assets_config is None:
        reset_config_cache()
        cfg = load_config()
        assets_config = [
            {"symbol": a.symbol, "source": a.source} for a in cfg.assets.assets if a.enabled
        ]

    results: dict[str, pd.DataFrame] = {}
    for asset in assets_config:
        symbol = asset["symbol"]
        source = asset.get("source", "yfinance")

        try:
            df = ingest_symbol(symbol, source_type=source, dry_run=dry_run)
            results[symbol] = df
        except Exception as e:
            logger.error("Failed to ingest %s: %s", symbol, e)
            results[symbol] = pd.DataFrame()  # empty on failure

        # Rate limit: 1 second between symbols
        if not dry_run:
            time.sleep(1)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest market data for financial time-series ML.")
    parser.add_argument(
        "--asset",
        type=str,
        default=None,
        help="Single asset symbol to ingest (e.g. SPY).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Ingest all enabled assets from configs/assets.yaml.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="yfinance",
        choices=["yfinance", "csv"],
        help="Data source to use (default: yfinance).",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="CSV directory (required when --source csv).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch data but do not write to disk.",
    )
    args = parser.parse_args()

    if args.all:
        results = ingest_all(dry_run=args.dry_run)
        for symbol, df in results.items():
            if df.empty:
                logger.warning("%s: ingestion failed (empty result)", symbol)
            else:
                logger.info("%s: %d rows stored", symbol, len(df))
    elif args.asset:
        csv_dir = Path(args.csv_dir) if args.csv_dir else None
        df = ingest_symbol(
            args.asset, source_type=args.source, csv_dir=csv_dir, dry_run=args.dry_run
        )
        logger.info("%s: %d rows %s", args.asset, len(df), "fetched" if args.dry_run else "stored")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
