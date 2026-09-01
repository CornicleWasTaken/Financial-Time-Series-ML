"""Leakage tests for Phase 2 feature engineering.

These tests verify that every feature function uses only past/present data
relative to the prediction timestamp. The test methodology:
- Take a feature and a time point t.
- Record the feature value at t.
- Modify data at times > t.
- Re-compute the feature at t.
- Verify the value did not change (i.e., no look-ahead bias).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from financial_ml.features.technical import (
    calculate_returns,
    ema,
    lagged_return,
    macd,
    price_ema_ratio,
    price_sma_ratio,
    rsi,
    rolling_volatility,
    sma,
)


# ---------------------------------------------------------------------------
# Helper: create price series with known structure
# ---------------------------------------------------------------------------

def _price_series() -> pd.Series:
    """Price series: 100, 102, 101, 105, 110, 108, 112, 115, 113, 120."""
    return pd.Series(
        [100.0, 102.0, 101.0, 105.0, 110.0, 108.0, 112.0, 115.0, 113.0, 120.0],
        index=pd.date_range("2023-01-02", periods=10, freq="D"),
        name="close",
    )


def _returns_series() -> pd.Series:
    """Simple returns from _price_series()."""
    price = _price_series()
    return price.pct_change()


# ---------------------------------------------------------------------------
# Leakage test base class
# ---------------------------------------------------------------------------


class TestLeakageSafety:
    """Mixin: provides a method to verify a feature is leakage-free.

    Subclasses must implement:
    - `make_feature`: return a feature Series given a price Series.
    - `test_time_idx`: the index (0-based) at which to check leakage safety.
    """

    @pytest.fixture(autouse=True)
    def setup_price(self) -> pd.Series:
        self.price = _price_series()
        return self.price

    def _verify_no_leakage(
        self,
        make_feature,
        test_time_idx: int,
        future_idx: int,
        future_value: float,
    ) -> None:
        """Assert that feature value at test_time_idx is unchanged after
        modifying data at future_idx.

        Args:
            make_feature: callable(price_series) -> result_series
            test_time_idx: index in result_series where we check the value
            future_idx: index > test_time_idx whose data we will modify
            future_value: new value at future_idx
        """
        # 1. Compute with original data
        original_result = make_feature(self.price)
        original_value = original_result.iloc[test_time_idx]

        # 2. Modify future data point
        modified_price = self.price.copy()
        modified_price.iloc[future_idx] = future_value

        # 3. Re-compute
        modified_result = make_feature(modified_price)
        modified_value = modified_result.iloc[test_time_idx]

        # 4. Assert they match (no leakage) - handle NaN values
        import math
        if math.isnan(original_value) and math.isnan(modified_value):
            # Both NaN — considered matching (no leakage)
            pass
        elif original_value != modified_value:
            raise AssertionError(
                f"LEAK DETECTED: feature at t={test_time_idx} changed from "
                f"{original_value} to {modified_value} when future data "
                f"at t={future_idx} was modified from {self.price.iloc[future_idx]} "
                f"to {future_value}"
            )

    # ------------------------------------------------------------------
    # Concrete leakage test methods per feature
    # ------------------------------------------------------------------

    def test_sma_no_leakage(self) -> None:
        """SMA at t=3 should not be affected by price at t=8."""
        make_feature = lambda s: sma(s, window=3)
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=3,
            future_idx=8,
            future_value=999.0,  # extreme future value
        )

    def test_ema_no_leakage(self) -> None:
        """EMA at t=5 should not be affected by price at t=9."""
        make_feature = lambda s: ema(s, span=3)
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=5,
            future_idx=9,
            future_value=999.0,
        )

    def test_price_sma_ratio_no_leakage(self) -> None:
        """Price/SMA at t=4 should not use price at t=8."""
        make_feature = lambda s: price_sma_ratio(s, window=3)
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=4,
            future_idx=8,
            future_value=999.0,
        )

    def test_rsi_no_leakage(self) -> None:
        """RSI at t=5 should not use price moves at t=9."""
        make_feature = lambda s: rsi(s, period=3)
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=5,
            future_idx=9,
            future_value=999.0,
        )

    def test_returns_no_leakage(self) -> None:
        """Simple return at t=3 should not use price at t=8."""
        make_feature = lambda s: calculate_returns(s, method="simple")
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=3,
            future_idx=8,
            future_value=999.0,
        )

    def test_lagged_return_no_leakage(self) -> None:
        """Lagged return at t=5 should not use price at t=9."""
        make_feature = lambda s: lagged_return(s, lag=1)
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=5,
            future_idx=9,
            future_value=999.0,
        )

    def test_rolling_volatility_no_leakage(self) -> None:
        """Volatility at t=5 (using returns up to t=5) should not use return at t=9."""
        make_feature = lambda s: rolling_volatility(
            calculate_returns(s, method="simple"), window=3
        )
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=5,
            future_idx=9,
            future_value=999.0,
        )

    def test_macd_no_leakage(self) -> None:
        """MACD at t=5 should not use price at t=9."""
        make_feature = lambda s: macd(s)["macd"]
        self._verify_no_leakage(
            make_feature=make_feature,
            test_time_idx=5,
            future_idx=9,
            future_value=999.0,
        )

    def test_ema_constant_future_no_leakage(self) -> None:
        """Set future prices all to the same extreme value; EMA at t=3 should not move."""
        make_feature = lambda s: ema(s, span=3)
        # Set ALL future points to 999.0 (indices 4..9)
        modified_price = self.price.copy()
        modified_price.iloc[4:] = 999.0
        original_result = make_feature(self.price)
        modified_result = make_feature(modified_price)
        # Values at t=3 should be identical regardless of future 999s
        assert original_result.iloc[3] == modified_result.iloc[3], (
            "LEAK DETECTED: EMA changed when future prices were set to extreme values"
        )


# ---------------------------------------------------------------------------
# Specific leakage test classes
# ---------------------------------------------------------------------------


class TestPriceSeriesLeakage(TestLeakageSafety):
    """Leakage tests that need price series."""

    @pytest.fixture(autouse=True)
    def setup_price(self) -> pd.Series:
        self.price = _price_series()


class TestReturnsLeakage(TestLeakageSafety):
    """Leakage tests for return-based features."""

    @pytest.fixture(autouse=True)
    def setup_returns(self) -> pd.Series:
        self.price = _returns_series()


# ---------------------------------------------------------------------------
# Additional fine-grained leakage checks
# ---------------------------------------------------------------------------


class TestFineGrainedLeakage:
    """Fine-grained checks: verify exact index positions and NaN handling."""

    def test_sma_nan_boundary(self) -> None:
        """SMA(3): indices 0,1 should be NaN; index 2 should have value."""
        s = pd.Series([100.0, 102.0, 101.0, 105.0, 110.0])
        result = sma(s, window=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(101.0)
        assert result.iloc[3] == pytest.approx((102 + 101 + 105) / 3)  # window includes 102,101,105
        assert result.iloc[4] == pytest.approx((101 + 105 + 110) / 3)

    def test_rsi_nan_first_periods(self) -> None:
        """RSI first `period` values should be NaN."""
        period = 5
        s = pd.Series([100.0 + i for i in range(15)], name="close")
        result = rsi(s, period=period)
        # First `period` values should be NaN
        assert pd.isna(result.iloc[:period]).all()
        # After that, NaN values should be within [0, 100] range
        valid = result.iloc[period:].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_volatility_nan_boundary(self) -> None:
        """Rolling volatility: first `window-1` values should be NaN."""
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02, -0.015], name="return")
        result = rolling_volatility(returns, window=3)
        assert pd.isna(result.iloc[:2]).all()  # window=3 → first 2 are NaN
        assert not pd.isna(result.iloc[2])

    def test_macd_length_equals_price_length(self) -> None:
        """MACD output lengths should match input price length."""
        s = pd.Series([100.0 + i for i in range(20)], name="close")
        result = macd(s)
        for key in ("macd", "signal", "histogram"):
            assert len(result[key]) == len(s), (
                f"MACD {key} length {len(result[key])} != price length {len(s)}"
            )
            assert result[key].index.equals(s.index)