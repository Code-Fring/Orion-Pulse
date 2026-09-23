"""Unit tests for core models."""

from datetime import date

import pytest

from orion_pulse.core.models import (
    OHLCV,
    DailyReturn,
    MovingAverage,
    Trend,
    TrendAnalysis,
    Volatility,
    VolumeRatio,
)


class TestOHLCV:
    def test_creates_valid_ohlcv(self):
        ohlcv = OHLCV(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )

        assert ohlcv.symbol == "TEST"
        assert ohlcv.trading_date == date(2024, 1, 1)
        assert ohlcv.open == 100.0
        assert ohlcv.high == 105.0
        assert ohlcv.low == 99.0
        assert ohlcv.close == 103.0
        assert ohlcv.volume == 1000000

    def test_creates_with_alias(self):
        """Test creating with 'date' alias."""
        ohlcv = OHLCV(
            symbol="TEST",
            date=date(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        assert ohlcv.trading_date == date(2024, 1, 1)

    def test_validates_positive_prices(self):
        with pytest.raises(Exception):
            OHLCV(
                symbol="TEST",
                trading_date=date(2024, 1, 1),
                open=-100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=1000000,
            )

    def test_validates_non_negative_volume(self):
        with pytest.raises(Exception):
            OHLCV(
                symbol="TEST",
                trading_date=date(2024, 1, 1),
                open=100.0,
                high=105.0,
                low=99.0,
                close=103.0,
                volume=-1,
            )

    def test_typical_price(self):
        ohlcv = OHLCV(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        # (105 + 99 + 103) / 3 = 102.333...
        assert abs(ohlcv.typical_price - 102.333) < 0.01

    def test_price_change(self):
        ohlcv = OHLCV(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        assert ohlcv.price_change == 3.0

    def test_price_change_pct(self):
        ohlcv = OHLCV(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        assert abs(ohlcv.price_change_pct - 3.0) < 0.01

    def test_immutable(self):
        ohlcv = OHLCV(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            open=100.0,
            high=105.0,
            low=99.0,
            close=103.0,
            volume=1000000,
        )
        with pytest.raises(Exception):
            ohlcv.close = 200.0


class TestTrend:
    def test_trend_values(self):
        assert Trend.BULLISH == "bullish"
        assert Trend.BEARISH == "bearish"
        assert Trend.SIDEWAYS == "sideways"


class TestDailyReturn:
    def test_creates_daily_return(self):
        dr = DailyReturn(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            return_pct=0.02,
            log_return=0.0198,
        )
        assert dr.symbol == "TEST"
        assert dr.return_pct == 0.02

    def test_creates_with_alias(self):
        dr = DailyReturn(
            symbol="TEST",
            date=date(2024, 1, 1),
            return_pct=0.02,
            log_return=0.0198,
        )
        assert dr.trading_date == date(2024, 1, 1)


class TestMovingAverage:
    def test_creates_moving_average(self):
        ma = MovingAverage(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            period=20,
            value=100.0,
            ma_type="SMA",
        )
        assert ma.period == 20
        assert ma.ma_type == "SMA"


class TestVolatility:
    def test_creates_volatility(self):
        vol = Volatility(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            window=20,
            value=0.15,
            annualized=True,
        )
        assert vol.window == 20
        assert vol.annualized is True


class TestVolumeRatio:
    def test_creates_volume_ratio(self):
        vr = VolumeRatio(
            symbol="TEST",
            trading_date=date(2024, 1, 1),
            ratio=1.5,
            avg_volume=1000000,
            current_volume=1500000,
        )
        assert vr.ratio == 1.5


class TestTrendAnalysis:
    def test_creates_trend_analysis(self):
        ta = TrendAnalysis(
            symbol="TEST",
            as_of=date(2024, 1, 1),
            trend=Trend.BULLISH,
            last_price=105.0,
            ma_20=100.0,
            ma_50=95.0,
            ma_200=90.0,
        )
        assert ta.trend == Trend.BULLISH
        assert ta.ma_20 == 100.0

    def test_trend_signals_computed_manually(self):
        """Test that signals can be set manually."""
        ta = TrendAnalysis(
            symbol="TEST",
            as_of=date(2024, 1, 1),
            trend=Trend.BULLISH,
            last_price=105.0,
            ma_20=100.0,
            ma_50=95.0,
            ma_200=90.0,
            price_above_ma20=True,
            price_above_ma50=True,
            price_above_ma200=True,
            ma20_above_ma50=True,
            ma50_above_ma200=True,
        )
        assert ta.price_above_ma20 is True
        assert ta.price_above_ma50 is True
        assert ta.price_above_ma200 is True
        assert ta.ma20_above_ma50 is True
        assert ta.ma50_above_ma200 is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
