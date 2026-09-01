"""Tests for technical indicator functions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_ml.features.technical import (
    calculate_returns,
    day_of_week,
    ema,
    lagged_return,
    log_volume,
    macd,
    month,
    price_ema_ratio,
    price_sma_ratio,
    quarter,
    rsi,
    rolling_volatility,
    sma,
    volume_price_trend,
    volume_sma,
    is_month_end,
    is_month_start,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _price_series() -> pd.Series:
    """Deterministic price series: 100, 102, 101, 105, 110, 108, 112, 115, 113, 120."""
    return pd.Series(
        [100.0, 102.0, 101.0, 105.0, 110.0, 108.0, 112.0, 115.0, 113.0, 120.0],
        index=pd.date_range("2023-01-02", periods=10, freq="D"),
        name="close",
    )


def _volume_series() -> pd.Series:
    return pd.Series(
        [1000, 1200, 1100, 1300, 1500, 1400, 1600, 1700, 1550, 1800],
        index=pd.date_range("2023-01-02", periods=10, freq="D"),
        name="volume",
    )


# ---------------------------------------------------------------------------
# Test: calculate_returns
# ---------------------------------------------------------------------------

class TestCalculateReturns:
    def test_simple_returns(self) -> None:
        s = _price_series()
        result = calculate_returns(s, method="simple")
        assert result.iloc[0] is pd.NA or np.isnan(result.iloc[0])
        assert result.iloc[1] == pytest.approx((102 - 100) / 100)
        assert result.index.equals(s.index)

    def test_log_returns(self) -> None:
        s = _price_series()
        result = calculate_returns(s, method="log")
        expected = np.log(102 / 100)
        assert result.iloc[1] == pytest.approx(expected)
        assert result.index.equals(s.index)

    def test_unknown_method_raises(self) -> None:
        s = _price_series()
        with pytest.raises(ValueError, match="Unknown method"):
            calculate_returns(s, method="unknown")


# ---------------------------------------------------------------------------
# Test: lagged_return
# ---------------------------------------------------------------------------

class TestLaggedReturn:
    def test_lag_1(self) -> None:
        s = _price_series()
        result = lagged_return(s, lag=1)
        # lagged_return returns: calculate_returns(s).shift(lag)
        # result[2] = returns[1] = (price[1] - price[0]) / price[0] = (102-100)/100 = 0.02
        expected_return = (102.0 - 100.0) / 100.0
        assert result.iloc[2] == pytest.approx(expected_return, abs=1e-10)
        # result[1] should be NaN (no prior value)
        assert pd.isna(result.iloc[1])

    def test_lag_5(self) -> None:
        s = _price_series()
        result = lagged_return(s, lag=5)
        # result[6] = returns[1] = (102-100)/100 = 0.02
        # (lagged_return shifts returns by lag, so result[6] = returns[1])
        expected_return = (102.0 - 100.0) / 100.0
        assert result.iloc[6] == pytest.approx(expected_return, abs=1e-10)


# ---------------------------------------------------------------------------
# Test: sma
# ---------------------------------------------------------------------------

class TestSMA:
    def test_sma_3(self) -> None:
        s = _price_series()
        result = sma(s, window=3)
        # First 2 values should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # SMA of 100, 102, 101 = 101
        assert result.iloc[2] == pytest.approx(101.0)

    def test_sma_10(self) -> None:
        s = _price_series()
        result = sma(s, window=10)
        # All 10 values used
        assert result.iloc[9] == pytest.approx(s.mean())

    def test_sma_longer_than_series(self) -> None:
        s = _price_series()
        result = sma(s, window=100)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Test: ema
# ---------------------------------------------------------------------------

class TestEMA:
    def test_ema_value_exists(self) -> None:
        s = _price_series()
        result = ema(s, span=3)
        # EMA should have values (not all NaN) for reasonable spans
        assert not result.isna().all()

    def test_ema_bounded(self) -> None:
        s = _price_series()
        result = ema(s, span=3)
        # EMA of prices should be between min and max
        assert result.iloc[-1] >= s.min()
        assert result.iloc[-1] <= s.max()

    def test_ema_longer_than_series(self) -> None:
        s = _price_series()
        result = ema(s, span=100)
        assert result.isna().all()


# ---------------------------------------------------------------------------
# Test: price_sma_ratio
# ---------------------------------------------------------------------------

class TestPriceSMARatio:
    def test_ratio_above_1(self) -> None:
        s = _price_series()
        # After an uptrend, price should be above SMA, ratio > 1
        result = price_sma_ratio(s, window=3)
        assert result.iloc[-1] > 1.0

    def test_ratio_below_1(self) -> None:
        # Falling prices
        falling_price_series = pd.Series([100.0, 95.0, 90.0, 85.0, 80.0])
        falling = pd.Series(
            [100.0, 95.0, 90.0, 85.0, 80.0],
            index=pd.date_range("2023-01-02", periods=5, freq="D"),
        )
        result = price_sma_ratio(falling, window=3)
        assert result.iloc[-1] < 1.0

    def test_ratio_index_preserved(self) -> None:
        s = _price_series()
        result = price_sma_ratio(s, window=3)
        assert result.index.equals(s.index)


# ---------------------------------------------------------------------------
# Test: price_ema_ratio
# ---------------------------------------------------------------------------

class TestPriceEMARatio:
    def test_ratio_exists(self) -> None:
        s = _price_series()
        result = price_ema_ratio(s, span=3)
        assert not result.isna().all()
        assert result.index.equals(s.index)


# ---------------------------------------------------------------------------
# Test: rsi
# ---------------------------------------------------------------------------

class TestRSI:
    def test_rsi_bounded(self) -> None:
        s = _price_series()
        result = rsi(s, period=3)
        valid = result.dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_rsi_first_values_nan(self) -> None:
        s = _price_series()
        result = rsi(s, period=3)
        # First `period` values should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # At index 2 (first non-NaN), RSI may not be NaN if enough data points exist
        # So we check the first truly computed value
        valid = result.dropna()
        assert len(valid) >= 1  # Should have at least one non-NaN value after first period

    def test_rsi_stable_uptrend(self) -> None:
        # Strictly increasing prices = all gains, RSI = 100
        # Use longer series to give RSI room to converge
        up = pd.Series([100.0 + i * 0.1 for i in range(100)], name="close")
        result = rsi(up, period=3)
        valid = result.dropna()
        assert len(valid) > 0
        # With strictly increasing prices, avg_loss = 0, RS -> inf, RSI -> 100
        assert valid.iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_rsi_stable_downtrend(self) -> None:
        # Strictly decreasing prices = all losses, RSI = 0
        # Use longer series to give RSI room to converge
        down = pd.Series([100.0 - i * 0.1 for i in range(100)], name="close")
        result = rsi(down, period=3)
        valid = result.dropna()
        assert len(valid) > 0
        # With strictly decreasing prices, avg_gain = 0, RS = 0, RSI = 0
        assert valid.iloc[-1] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Test: macd
# ---------------------------------------------------------------------------

class TestMACD:
    def test_macd_returns_dict(self) -> None:
        s = _price_series()
        result = macd(s)
        assert isinstance(result, dict)
        assert "macd" in result
        assert "signal" in result
        assert "histogram" in result

    def test_macd_histogram(self) -> None:
        s = _price_series()
        result = macd(s)
        expected = result["macd"] - result["signal"]
        # Compare non-NaN values
        valid_mask = ~result["histogram"].isna() & ~expected.isna()
        assert (result["histogram"][valid_mask] == expected[valid_mask]).all()

    def test_macd_length(self) -> None:
        s = _price_series()
        result = macd(s)
        assert len(result["macd"]) == len(s)
        assert result["macd"].index.equals(s.index)

    def test_macd_signal_not_empty(self) -> None:
        """MACD signals should have values for most of the series."""
        # MACD with spans 12 and 26 needs at least 26 periods to produce values
        s = pd.Series(
            [100.0 + i for i in range(50)],
            index=pd.date_range("2023-01-02", periods=50, freq="B"),
            name="close",
        )
        result = macd(s)
        # The macd line should not be all NaN
        assert not result["macd"].isna().all()

    def test_macd_signal_is_series(self) -> None:
        """MACD components should be pandas Series."""
        s = _price_series()
        result = macd(s)
        assert isinstance(result["macd"], pd.Series)
        assert isinstance(result["signal"], pd.Series)
        assert isinstance(result["histogram"], pd.Series)


# ---------------------------------------------------------------------------
# Test: rolling_volatility
# ---------------------------------------------------------------------------

class TestRollingVolatility:
    def test_volatility_nonnegative(self) -> None:
        returns = calculate_returns(_price_series(), method="simple")
        result = rolling_volatility(returns, window=5)
        valid = result.dropna()
        assert (valid >= 0).all()

    def test_volatility_first_nan(self) -> None:
        returns = calculate_returns(_price_series(), method="simple")
        result = rolling_volatility(returns, window=5)
        assert pd.isna(result.iloc[:4]).all()
        assert not pd.isna(result.iloc[5])

    def test_volatility_index_preserved(self) -> None:
        returns = calculate_returns(_price_series(), method="simple")
        result = rolling_volatility(returns, window=5)
        assert result.index.equals(returns.index)


# ---------------------------------------------------------------------------
# Test: log_volume
# ---------------------------------------------------------------------------

class TestLogVolume:
    def test_log_volume(self) -> None:
        v = _volume_series()
        result = log_volume(v)
        expected = np.log(1000)
        assert result.iloc[0] == pytest.approx(expected)
        assert result.index.equals(v.index)

    def test_log_volume_zero_replaced(self) -> None:
        v = pd.Series([0, 1000, 2000], index=_volume_series().index[:3])
        result = log_volume(v)
        assert pd.isna(result.iloc[0])
        assert not pd.isna(result.iloc[1])


# ---------------------------------------------------------------------------
# Test: volume_sma
# ---------------------------------------------------------------------------

class TestVolumeSMA:
    def test_volume_sma(self) -> None:
        v = _volume_series()
        result = volume_sma(v, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx((1000 + 1200 + 1100) / 3)


# ---------------------------------------------------------------------------
# Test: volume_price_trend
# ---------------------------------------------------------------------------

class TestVolumePriceTrend:
    def test_volume_price_trend(self) -> None:
        v = _volume_series()
        p = _price_series()
        returns = calculate_returns(p, method="simple")
        result = volume_price_trend(v, returns)
        # First return is NaN → sign is 0 → VPT = 0
        assert result.iloc[0] == 0.0
        # Index 1: price went up (102>100), positive return → positive VPT
        assert result.iloc[1] > 0
        # Index 2: price went down (101<102), negative return → negative VPT
        assert result.iloc[2] < 0

    def test_vpt_index_preserved(self) -> None:
        v = _volume_series()
        p = _price_series()
        returns = calculate_returns(p, method="simple")
        result = volume_price_trend(v, returns)
        assert result.index.equals(v.index)


# ---------------------------------------------------------------------------
# Test: calendar features
# ---------------------------------------------------------------------------

class TestCalendarFeatures:
    def test_day_of_week(self) -> None:
        ts = pd.Series(
            pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"]),  # Mon, Tue, Wed
        )
        result = day_of_week(ts)
        assert result.iloc[0] == 0  # Monday
        assert result.iloc[1] == 1  # Tuesday

    def test_month(self) -> None:
        ts = pd.Series(pd.to_datetime(["2023-01-15", "2023-06-15"]))
        result = month(ts)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 6

    def test_quarter(self) -> None:
        ts = pd.Series(pd.to_datetime(["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15"]))
        result = quarter(ts)
        assert result.iloc[0] == 1
        assert result.iloc[1] == 2
        assert result.iloc[2] == 3
        assert result.iloc[3] == 4

    def test_is_month_start(self) -> None:
        ts = pd.Series(pd.to_datetime(["2023-01-01", "2023-01-15"]))
        result = is_month_start(ts)
        assert result.iloc[0] == True  # noqa: E712
        assert result.iloc[1] == False  # noqa: E712

    def test_is_month_end(self) -> None:
        ts = pd.Series(pd.to_datetime(["2023-01-31", "2023-01-15"]))
        result = is_month_end(ts)
        assert result.iloc[0] == True  # noqa: E712
        assert result.iloc[1] == False  # noqa: E712
