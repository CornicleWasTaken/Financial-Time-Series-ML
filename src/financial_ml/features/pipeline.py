"""Feature engineering pipeline.

Reads the canonical OHLCV table and applies user-configurable technical indicators
from ``configs/features.yaml``. Outputs a feature matrix stored in
``data/features/features.parquet``. Every feature is computed using only past and
present information — no look-ahead bias is possible by construction.

Phase 2 feature groups supported:
  - Returns & lags
  - Simple & exponential moving averages
  - Price-to-MA ratios
  - RSI
  - MACD
  - Volatility
  - Volume features
  - Calendar features
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from financial_ml.config import FEATURE_DATA_DIR, load_config, reset_config_cache
from financial_ml.features.technical import (
    calculate_returns,
    ema,
    log_volume,
    macd,
    price_ema_ratio,
    price_sma_ratio,
    quarter,
    rsi,
    sma,
    day_of_week,
    is_month_end,
    is_month_start,
    month,
    lagged_return,
    rolling_volatility,
    volume_sma,
    volume_price_trend,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# ---------------------------------------------------------------------------
# Mapping from config feature names to technical-function calls
# ---------------------------------------------------------------------------

# Each entry maps a config feature name to a tuple of:
# (function, args_kwargs_dict, output_series_name)
FEATURE_FUNCS: dict[str, tuple] = {
    # ---- Returns & lags ----
    "return_lag_1": (
        lagged_return,
        {"lag": 1, "method": "simple"},
        "return_lag_1",
    ),
    "return_lag_5": (
        lagged_return,
        {"lag": 5, "method": "simple"},
        "return_lag_5",
    ),
    "log_return": (
        calculate_returns,
        {"method": "log"},
        "log_return",
    ),

    # ---- Simple moving averages ----
    "sma_10": (sma, {"window": 10}, "sma_10"),
    "sma_50": (sma, {"window": 50}, "sma_50"),
    "sma_20": (sma, {"window": 20}, "sma_20"),

    # ---- Exponential moving averages ----
    "ema_12": (ema, {"span": 12}, "ema_12"),
    "ema_26": (ema, {"span": 26}, "ema_26"),

    # ---- Price-to-MA ratios ----
    "price_sma_ratio_10": (
        price_sma_ratio,
        {"window": 10},
        "price_sma_ratio_10",
    ),
    "price_ema_ratio_12": (
        price_ema_ratio,
        {"span": 12},
        "price_ema_ratio_12",
    ),

    # ---- RSI ----
    "rsi_14": (rsi, {"period": 14}, "rsi_14"),

    # ---- MACD ----
    "macd": (macd, {"fast": 12, "slow": 26, "signal": 9}, "macd"),

    # ---- Volatility ----
    "volatility_10": (
        rolling_volatility,
        {"window": 10},
        "volatility_10",
    ),
    "volatility_30": (
        rolling_volatility,
        {"window": 30},
        "volatility_30",
    ),

    # ---- Volume features ----
    "log_volume": (log_volume, {}, "log_volume"),
    "volume_sma_10": (volume_sma, {"window": 10}, "volume_sma_10"),
    "volume_price_trend": (volume_price_trend, {}, "volume_price_trend"),

    # ---- Calendar features ----
    "day_of_week": (day_of_week, {}, "day_of_week"),
    "month": (month, {}, "month"),
    "quarter": (quarter, {}, "quarter"),
    "is_month_start": (is_month_start, {}, "is_month_start"),
    "is_month_end": (is_month_end, {}, "is_month_end"),
}


def _infer_price_series(ohlcv: pd.DataFrame) -> pd.Series:
    """Return the close price series from a canonical OHLCV table."""
    if "close" in ohlcv.columns:
        return ohlcv["close"]
    msg = "OHLCV DataFrame must contain a 'close' column"
    raise ValueError(msg)


def _infer_volume_series(ohlcv: pd.DataFrame) -> pd.Series | None:
    """Return the volume series from a canonical OHLCV table, if present."""
    if "volume" in ohlcv.columns:
        return ohlcv["volume"]
    return None


def _apply_feature(
    func,
    args: dict,
    series: pd.Series,
    series_name: str,
) -> pd.Series:
    """Apply a technical function to a series, handling edge cases.

    Most functions expect a Series; some may need to compute on ``series``
    directly or on derivative series (returns). We dispatch accordingly.
    """
    # Special dispatch for functions that need returns series
    if func == calculate_returns:
        return func(series, method=args["method"])  # type: ignore[arg-type]
    if func == rolling_volatility:
        # volatility takes a *returns* series
        returns = calculate_returns(series, method="simple")
        return func(returns, window=args["window"])  # type: ignore[arg-type]
    if func == macd:
        # macd takes price series directly and returns dict
        return macd(series, **args)["macd"]  # type: ignore[arg-type]
    if func == lagged_return:
        return func(series, lag=args["lag"], method=args["method"])  # type: ignore[arg-type]
    if func == volume_price_trend:
        # needs both series — caller handles
        raise NotImplementedError("volume_price_trend handled in pipeline assembly")
    # Default: pure-series functions
    return func(series, **args)  # type: ignore[arg-type]


def _apply_calendar_feature(
    func,
    series: pd.Series,
    series_name: str,
) -> pd.Series:
    """Apply calendar-feature functions (take a datetime Series)."""
    return func(series)


def build_feature_matrix(
    ohlcv: pd.DataFrame,
    enabled_features: list[dict] | None = None,
) -> pd.DataFrame:
    """Build the feature matrix from a canonical OHLCV table.

    Args:
        ohlcv: Canonical OHLCV DataFrame (output of
            :func:`financial_ml.data.preprocessing.create_canonical_ohlcv`).
        enabled_features: List of feature dicts from ``configs/features.yaml``.
            If ``None``, loads from config and enables everything listed.

    Returns:
        DataFrame with original timestamps + all enabled features.
        Columns: ``timestamp`` + feature-name columns.
    """
    # Load configuration
    if enabled_features is None:
        reset_config_cache()
        cfg = load_config()
        enabled_features = cfg.features.features  # type: ignore[attr-defined]

    # Start with timestamp column
    result: pd.DataFrame = ohlcv[["timestamp"]].copy()

    # Helper to extract series safely
    price = _infer_price_series(ohlcv)
    volume = _infer_volume_series(ohlcv)

    for feat_cfg in enabled_features:
        if not feat_cfg.get("enabled", True):
            logger.info("Skipping disabled feature: %s", feat_cfg["name"])
            continue

        name = feat_cfg["name"]
        params = feat_cfg.get("params", {})

        if name not in FEATURE_FUNCS:
            logger.warning(
                "Unknown feature %s — skipping. Known: %s",
                name,
                ", ".join(FEATURE_FUNCS.keys()),
            )
            continue

        func, func_args, out_name = FEATURE_FUNCS[name]

        # Collect series arguments depending on feature type
        if name in {
            "return_lag_1",
            "return_lag_5",
            "log_return",
            "sma_10",
            "sma_50",
            "sma_20",
            "ema_12",
            "ema_26",
            "price_sma_ratio_10",
            "price_ema_ratio_12",
            "rsi_14",
            "volatility_10",
            "volatility_30",
        }:
            # Functions that take price series
            series_arg = price
            out = _apply_feature(func, func_args, series_arg, out_name)

        elif name in {"log_volume", "volume_sma_10", "volume_price_trend"}:
            # Functions that take volume series
            if volume is None:
                logger.warning(
                    "Feature %s enabled but OHLCV has no 'volume' column — skipping",
                    name,
                )
                continue
            series_arg = volume
            if name == "volume_price_trend":
                # Need price series too for sign
                price_series = price
                out = func(series_arg, price_series)  # type: ignore[arg-type]
            else:
                out = _apply_feature(func, func_args, series_arg, out_name)

        elif name in {"macd"}:
            out = _apply_feature(func, func_args, price, out_name)

        elif name in {"day_of_week", "month", "quarter", "is_month_start", "is_month_end"}:
            out = _apply_calendar_feature(func, ohlcv["timestamp"], out_name)

        else:
            logger.warning("Unhandled feature %s — skipping", name)
            continue

        # Align on timestamp index
        result[out_name] = out

        logger.info(
            "Applied feature %s -> column %s (%d rows)",
            name,
            out_name,
            len(out.dropna()),
        )

    # Final chronological sort (already should be, but safety)
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result


def save_feature_matrix(
    ohlcv: pd.DataFrame,
    output_path: Path | None = None,
    enabled_features: list[dict] | None = None,
) -> pd.DataFrame:
    """Build and persist the feature matrix to parquet.

    Args:
        ohlcv: Canonical OHLCV DataFrame.
        output_path: Where to write ``features.parquet``.
            Defaults to ``FEATURE_DATA_DIR / "features.parquet"``.
        enabled_features: Feature config list (see :func:`build_feature_matrix`).

    Returns:
        The feature matrix DataFrame.
    """
    if output_path is None:
        FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        output_path = FEATURE_DATA_DIR / "features.parquet"

    matrix = build_feature_matrix(ohlcv, enabled_features=enabled_features)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matrix.to_parquet(output_path, index=False)
    logger.info("Feature matrix written to %s (%d rows, %d columns)",
                output_path, len(matrix), len(matrix.columns))

    return matrix


def load_feature_matrix(
    path: Path | None = None,
) -> pd.DataFrame:
    """Load a previously saved feature matrix.

    Args:
        path: Path to parquet file. Defaults to
            ``FEATURE_DATA_DIR / "features.parquet"``.

    Returns:
        DataFrame with timestamp + feature columns.
    """
    if path is None:
        FEATURE_DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = FEATURE_DATA_DIR / "features.parquet"

    if not path.exists():
        msg = f"Feature matrix not found at {path}"
        raise FileNotFoundError(msg)

    return pd.read_parquet(path)