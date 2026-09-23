"""Unit tests for technical indicators."""

import polars as pl
import pytest

from orion_pulse.analysis.indicators import (
    analyze_trend,
    calculate_daily_returns,
    calculate_moving_average,
    calculate_rolling_volatility,
    calculate_volume_ratio,
    classify_trend,
    prepare_analysis_data,
)
from orion_pulse.core.models import Trend


def create_sample_data() -> pl.DataFrame:
    """Create sample OHLCV data for testing."""
    dates = [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
    ]
    return pl.DataFrame(
        {
            "date": dates,
            "open": [
                100.0,
                101.0,
                102.0,
                101.5,
                103.0,
                104.0,
                105.0,
                104.5,
                106.0,
                107.0,
            ],
            "high": [
                102.0,
                103.0,
                104.0,
                103.5,
                105.0,
                106.0,
                107.0,
                106.5,
                108.0,
                109.0,
            ],
            "low": [
                99.0,
                100.0,
                101.0,
                100.5,
                102.0,
                103.0,
                104.0,
                103.5,
                105.0,
                106.0,
            ],
            "close": [
                101.0,
                102.0,
                103.0,
                102.5,
                104.0,
                105.0,
                106.0,
                105.5,
                107.0,
                108.0,
            ],
            "volume": [
                1000000,
                1100000,
                1050000,
                1150000,
                1200000,
                1100000,
                1250000,
                1300000,
                1200000,
                1350000,
            ],
            "symbol": ["TEST"] * 10,
        }
    )


class TestCalculateDailyReturns:
    def test_calculates_returns(self):
        df = create_sample_data()
        result = calculate_daily_returns(df)

        assert "return_pct" in result.columns
        assert "log_return" in result.columns
        assert len(result) == len(df)

        # First row should be null (no previous day)
        assert result[0, "return_pct"] is None

    def test_return_values_are_correct(self):
        df = create_sample_data()
        result = calculate_daily_returns(df)

        # Second day: (102 - 101) / 101 = 0.0099...
        second_return = result[1, "return_pct"]
        assert second_return is not None
        assert abs(second_return - 0.0099) < 0.001


class TestCalculateMovingAverage:
    def test_calculates_ma(self):
        df = create_sample_data()
        result = calculate_moving_average(df, period=3)

        assert "ma_3" in result.columns
        assert len(result) == len(df)

        # First 2 rows should be null (not enough data)
        assert result[0, "ma_3"] is None
        assert result[1, "ma_3"] is None

        # Third row: (101 + 102 + 103) / 3 = 102
        third_ma = result[2, "ma_3"]
        assert third_ma is not None
        assert abs(third_ma - 102.0) < 0.01

    def test_custom_column(self):
        df = create_sample_data()
        result = calculate_moving_average(df, period=3, column="volume")

        assert "ma_3" in result.columns
        third_ma = result[2, "ma_3"]
        expected = (1000000 + 1100000 + 1050000) / 3
        assert abs(third_ma - expected) < 1


class TestCalculateRollingVolatility:
    def test_calculates_volatility(self):
        df = create_sample_data()
        df = calculate_daily_returns(df)
        result = calculate_rolling_volatility(df, window=3)

        assert "volatility_3" in result.columns
        assert len(result) == len(df)

    def test_annualization_factor(self):
        df = create_sample_data()
        df = calculate_daily_returns(df)

        result_annual = calculate_rolling_volatility(df, window=3, annualize=True)
        result_raw = calculate_rolling_volatility(df, window=3, annualize=False)

        # Annualized should be larger by sqrt(252)
        assert result_annual[3, "volatility_3"] > result_raw[3, "volatility_3"]


class TestCalculateVolumeRatio:
    def test_calculates_volume_ratio(self):
        df = create_sample_data()
        result = calculate_volume_ratio(df, window=3)

        assert "avg_volume_3" in result.columns
        assert "volume_ratio_3" in result.columns

        # Check ratio calculation
        third_ratio = result[2, "volume_ratio_3"]
        third_volume = df[2, "volume"]
        avg_volume = (1000000 + 1100000 + 1050000) / 3
        expected = third_volume / avg_volume
        assert abs(third_ratio - expected) < 0.01


class TestClassifyTrend:
    def test_bullish_trend(self):
        # Price above all MAs
        trend = classify_trend(110.0, 105.0, 100.0, 95.0)
        assert trend == Trend.BULLISH

    def test_bearish_trend(self):
        # Price below all MAs
        trend = classify_trend(90.0, 95.0, 100.0, 105.0)
        assert trend == Trend.BEARISH

    def test_sideways_trend(self):
        # Price mixed vs MAs
        trend = classify_trend(102.0, 100.0, 105.0, 95.0)
        assert trend == Trend.SIDEWAYS

    def test_no_mas_returns_sideways(self):
        trend = classify_trend(100.0, None, None, None)
        assert trend == Trend.SIDEWAYS

    def test_partial_mas(self):
        # Only MA20 available
        trend = classify_trend(105.0, 100.0, None, None)
        assert trend == Trend.BULLISH

        trend = classify_trend(95.0, 100.0, None, None)
        assert trend == Trend.BEARISH


class TestAnalyzeTrend:
    def test_analyzes_complete_trend(self):
        df = create_sample_data()
        # Add MAs manually for testing
        df = calculate_moving_average(df, 3)
        df = calculate_moving_average(df, 5)
        df = calculate_daily_returns(df)
        df = calculate_rolling_volatility(df, window=3)
        df = calculate_volume_ratio(df, window=3)

        result = analyze_trend("TEST", df, ma_periods=[3, 5])

        assert result.symbol == "TEST"
        assert result.trend in [Trend.BULLISH, Trend.BEARISH, Trend.SIDEWAYS]
        assert result.last_price > 0
        assert result.ma_20 is None  # Not in ma_periods
        assert result.ma_50 is None
        assert result.ma_200 is None

    def test_raises_on_empty_data(self):
        df = pl.DataFrame({"date": [], "close": [], "volume": []})
        with pytest.raises(ValueError, match="No data provided"):
            analyze_trend("TEST", df)


class TestPrepareAnalysisData:
    def test_prepares_all_indicators(self):
        df = create_sample_data()
        result = prepare_analysis_data(
            df, ma_periods=[3, 5], volatility_window=3, volume_window=3
        )

        # Check all columns exist
        assert "return_pct" in result.columns
        assert "log_return" in result.columns
        assert "ma_3" in result.columns
        assert "ma_5" in result.columns
        assert "volatility_3" in result.columns
        assert "avg_volume_3" in result.columns
        assert "volume_ratio_3" in result.columns

        assert len(result) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
