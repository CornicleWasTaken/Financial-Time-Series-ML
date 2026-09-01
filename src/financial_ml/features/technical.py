"""Technical indicators for financial time-series features.

All functions are pure: they accept a pandas Series (or DataFrame for calendar)
and return a Series with the same index. Every function is leakage-safe by
design — it uses only past and present data (never look-ahead).

Phase 2 implements the core indicator library. The pipeline module
(``pipeline.py``) orchestrates these functions against the canonical OHLCV table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Return calculations
# ---------------------------------------------------------------------------


def calculate_returns(
    series: pd.Series,
    method: str = "simple",
) -> pd.Series:
    """Compute period-over-period returns.

    Args:
        series: Price series.
        method: "simple" (P_t / P_{t-1} - 1) or "log" (ln(P_t / P_{t-1})).

    Returns:
        Returns series with same index as input. First element is always NaN.
    """
    if method == "simple":
        return series.pct_change()
    if method == "log":
        return np.log(series / series.shift(1))
    msg = f"Unknown method: {method!r}. Use 'simple' or 'log'."
    raise ValueError(msg)


def lagged_return(
    series: pd.Series,
    lag: int = 1,
    method: str = "simple",
) -> pd.Series:
    """Shifted return (return from lag periods ago).

    Args:
        series: Price series.
        lag: Number of periods to shift (1 = yesterday's return).
        method: "simple" or "log".

    Returns:
        Return series shifted by ``lag`` periods. The most recent ``lag`` values
        will be NaN.
    """
    returns = calculate_returns(series, method=method)
    return returns.shift(lag)


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


def sma(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Simple (unweighted) moving average.

    Args:
        series: Input series.
        window: Lookback window in periods.

    Returns:
        SMA series aligned to the input index. The first ``window`` values
        are NaN.
    """
    return series.rolling(window=window, min_periods=window).mean()


def ema(
    series: pd.Series,
    span: int,
) -> pd.Series:
    """Exponential moving average (span-based).

    Args:
        series: Input series.
        span: Span in periods (equivalent to ``alpha = 2 / (span + 1)``).

    Returns:
        EMA series aligned to the input index. The first ``span`` values
        may be NaN or statistically unreliable.
    """
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


# ---------------------------------------------------------------------------
# Price-to-MA ratios
# ---------------------------------------------------------------------------


def price_sma_ratio(
    series: pd.Series,
    window: int,
) -> pd.Series:
    """Price divided by its simple moving average.

    Args:
        series: Price series.
        window: SMA lookback window.

    Returns:
        Ratio series. Values > 1 indicate price above its SMA.
    """
    return series / sma(series, window=window)


def price_ema_ratio(
    series: pd.Series,
    span: int,
) -> pd.Series:
    """Price divided by its exponential moving average.

    Args:
        series: Price series.
        span: EMA span.

    Returns:
        Ratio series.
    """
    return series / ema(series, span=span)


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------


def rsi(
    series: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Relative Strength Index (Wilder's formulation).

    Args:
        series: Price series (typically close).
        period: RSI lookback period (default 14).

    Returns:
        RSI series bounded in [0, 100]. The first ``period`` values are NaN.
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder's smoothing: initial SMA using only valid (non-NaN) periods
    # First compute SMA over the first `period` valid gain/loss values
    # Since diff() produces NaN at index 0, we have `len(series)-1` valid returns
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Find the first index where both avg_gain and avg_loss are non-NaN
    first_valid = None
    for i in range(len(series)):
        if not (pd.isna(avg_gain.iloc[i]) or pd.isna(avg_loss.iloc[i])):
            first_valid = i
            break

    if first_valid is None:
        return pd.Series(np.nan, index=series.index, name="rsi")

    # Apply Wilder smoothing starting from the index AFTER the first valid SMA
    # The first_valid index already contains the SMA of the first `period` valid values
    for i in range(first_valid + 1, len(series)):
        prev_gain = avg_gain.iloc[i - 1]
        prev_loss = avg_loss.iloc[i - 1]
        curr_gain = gain.iloc[i]
        curr_loss = loss.iloc[i]

        # Wilder's smoothing formula: prev_avg * (period-1)/period + curr/period
        avg_gain.iloc[i] = (prev_gain * (period - 1) + curr_gain) / period
        avg_loss.iloc[i] = (prev_loss * (period - 1) + curr_loss) / period

    # Calculate RS and RSI
    # When avg_loss is 0 (all gains, uptrend), RS = inf, RSI = 100
    # When avg_gain is 0 (all losses, downtrend), RS = 0, RSI = 0
    # Use np.errstate to allow inf, then handle explicitly
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / avg_loss.abs()
    rsi_series = 100 - (100 / (1 + rs))
    # Where RS is inf (avg_loss=0), RSI = 100 (uptrend boundary)
    rsi_series = rsi_series.where(~np.isinf(rs), 100.0)
    # Where RS is 0 (avg_gain=0, avg_loss>0), RSI = 0 (downtrend boundary)
    rsi_series = rsi_series.where(~(rs == 0), 0.0)

    # Clamp boundary values only for non-NaN entries
    # uptrend (no losses): avg_loss = 0 -> RSI = 100
    # downtrend (no gains): avg_gain = 0 -> RSI = 0
    valid_mask = ~rsi_series.isna()
    rsi_series = rsi_series.where(~(valid_mask & (avg_loss == 0)), 100.0)
    rsi_series = rsi_series.where(~(valid_mask & (avg_gain == 0)), 0.0)
    rsi_series.index = series.index
    return rsi_series


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, pd.Series]:
    """Moving Average Convergence Divergence.

    Args:
        series: Price series (typically close).
        fast: Fast EMA span.
        slow: Slow EMA span.
        signal: Signal line EMA span.

    Returns:
        Dict with keys "macd", "signal", "histogram".
    """
    ema_fast = ema(series, span=fast)
    ema_slow = ema(series, span=slow)

    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line

    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    }


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def rolling_volatility(
    returns: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling standard deviation of returns (annualised).

    Args:
        returns: Return series (e.g. from :func:`calculate_returns`).
        window: Lookback window in periods.

    Returns:
        Annualised volatility series. The first ``window`` values are NaN.
    """
    return returns.rolling(window=window, min_periods=window).std() * np.sqrt(252)


# ---------------------------------------------------------------------------
# Volume features
# ---------------------------------------------------------------------------


def log_volume(volume: pd.Series) -> pd.Series:
    """Natural log of volume.

    Args:
        volume: Volume series.

    Returns:
        Log-volume series. NaN for zero-volume rows.
    """
    return np.log(volume.replace(0, np.nan))


def volume_sma(
    volume: pd.Series,
    window: int,
) -> pd.Series:
    """Simple moving average of volume.

    Args:
        volume: Volume series.
        window: Lookback window.

    Returns:
        Volume SMA series. The first ``window`` values are NaN.
    """
    return volume.rolling(window=window, min_periods=window).mean()


def volume_price_trend(volume: pd.Series, returns: pd.Series) -> pd.Series:
    """Volume-price trend (volume times return sign).

    A simple proxy for on-balance volume direction.

    Args:
        volume: Volume series.
        returns: Return series.

    Returns:
        Signed volume series.
    """
    # Treat NaN returns as 0 (no change) for VPT calculation
    return volume * np.sign(returns.fillna(0))


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def day_of_week(timestamp: pd.Series) -> pd.Series:
    """Day of week as integer (Monday=0, Sunday=6).

    Args:
        timestamp: Datetime series.

    Returns:
        Integer series.
    """
    return timestamp.dt.dayofweek


def month(timestamp: pd.Series) -> pd.Series:
    """Month as integer (January=1, December=12).

    Args:
        timestamp: Datetime series.

    Returns:
        Integer series.
    """
    return timestamp.dt.month


def quarter(timestamp: pd.Series) -> pd.Series:
    """Quarter as integer (Q1=1, Q4=4).

    Args:
        timestamp: Datetime series.

    Returns:
        Integer series.
    """
    return timestamp.dt.quarter


def is_month_start(timestamp: pd.Series) -> pd.Series:
    """Boolean: True if the date is the first trading day of the month.

    Args:
        timestamp: Datetime series.

    Returns:
        Boolean series.
    """
    return timestamp.dt.is_month_start


def is_month_end(timestamp: pd.Series) -> pd.Series:
    """Boolean: True if the date is the last trading day of the month.

    Args:
        timestamp: Datetime series.

    Returns:
        Boolean series.
    """
    return timestamp.dt.is_month_end
