"""Unit tests for forecasting module."""

from datetime import date

import numpy as np
import polars as pl
import pytest

from orion_pulse.forecasting import (
    EnsembleForecaster,
    ForecastingError,
    ForecastingService,
    ForecastModel,
    ForecastResult,
    HistoricalMeanForecaster,
    RandomWalkForecaster,
    TrendMomentumForecaster,
    VolatilityBasedForecaster,
)


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    np.random.seed(42)
    n = 100
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    # Generate realistic price series with slight upward drift
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.002, n)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
            "symbol": ["TEST"] * n,
        }
    )


from datetime import timedelta


class TestHistoricalMeanForecaster:
    def test_forecast_basic(self, sample_market_data):
        forecaster = HistoricalMeanForecaster()
        result = forecaster.forecast("TEST", sample_market_data, horizon_days=5)

        assert isinstance(result, ForecastResult)
        assert result.symbol == "TEST"
        assert result.model_name == "historical_mean"
        assert result.horizon_days == 5
        assert result.forecast_date == date.today()
        assert result.target_date == date.today()
        assert isinstance(result.expected_return, float)
        assert 0 <= result.prob_positive <= 1
        assert result.p10 <= result.p25 <= result.p50 <= result.p75 <= result.p90

    def test_forecast_with_lookback(self, sample_market_data):
        forecaster = HistoricalMeanForecaster()
        result = forecaster.forecast(
            "TEST", sample_market_data, horizon_days=10, lookback_days=50
        )

        assert result.horizon_days == 10
        assert "lookback_days" in result.factors
        assert result.factors["lookback_days"] == 50


class TestVolatilityBasedForecaster:
    def test_forecast_basic(self, sample_market_data):
        forecaster = VolatilityBasedForecaster()
        result = forecaster.forecast("TEST", sample_market_data, horizon_days=5)

        assert result.model_name == "volatility_based"
        assert "current_annual_vol" in result.factors
        assert result.factors["current_annual_vol"] > 0


class TestTrendMomentumForecaster:
    def test_forecast_basic(self, sample_market_data):
        forecaster = TrendMomentumForecaster()
        result = forecaster.forecast("TEST", sample_market_data, horizon_days=5)

        assert result.model_name == "trend_momentum"
        assert "trend_signals" in result.factors
        assert "momentum" in result.factors


class TestRandomWalkForecaster:
    def test_forecast_basic(self, sample_market_data):
        forecaster = RandomWalkForecaster()
        result = forecaster.forecast("TEST", sample_market_data, horizon_days=5)

        assert result.model_name == "random_walk"
        assert result.expected_return == 0.0
        assert result.prob_positive == 0.5
        assert result.confidence == 0.3


class TestEnsembleForecaster:
    def test_forecast_basic(self, sample_market_data):
        forecaster = EnsembleForecaster()
        result = forecaster.forecast("TEST", sample_market_data, horizon_days=5)

        assert result.model_name == "ensemble"
        assert "models" in result.factors
        assert "weights" in result.factors
        assert len(result.factors["models"]) == 4  # All 4 sub-forecasters


class TestForecastingService:
    def test_generate_forecast(self, sample_market_data):
        service = ForecastingService()
        result = service.generate_forecast(
            "TEST",
            sample_market_data,
            model=ForecastModel.HISTORICAL_MEAN,
            horizon_days=5,
        )

        assert result.model_name == "historical_mean"

    def test_generate_all_forecasts(self, sample_market_data):
        service = ForecastingService()
        results = service.generate_all_forecasts(
            "TEST", sample_market_data, horizon_days=5
        )

        assert len(results) == 5  # All 5 models
        for model_name in ForecastModel:
            assert model_name.value in results
            # Some models might fail, but at least one should work
        assert any(r is not None for r in results.values())

    def test_unknown_model_raises(self, sample_market_data):
        service = ForecastingService()

        # Create a fake model enum value
        from enum import Enum

        class FakeModel(Enum):
            FAKE = "fake"

        with pytest.raises(ForecastingError, match="Unknown model"):
            service.generate_forecast("TEST", sample_market_data, model=FakeModel.FAKE)

    def test_empty_data_raises(self):
        service = ForecastingService()
        empty_df = pl.DataFrame({"date": [], "close": [], "volume": []})

        with pytest.raises(ForecastingError):
            service.generate_forecast("TEST", empty_df)


class TestForecastResult:
    def test_to_dict(self):
        result = ForecastResult(
            symbol="TEST",
            model_name="test",
            horizon_days=5,
            forecast_date=date(2024, 1, 15),
            target_date=date(2024, 1, 22),
            expected_return=0.02,
            prob_positive=0.6,
            p10=-0.05,
            p25=-0.01,
            p50=0.02,
            p75=0.05,
            p90=0.09,
            volatility=0.15,
            confidence=0.7,
            factors={"test": "value"},
            input_features={"mean": 0.001},
        )

        d = result.to_dict()
        assert d["symbol"] == "TEST"
        assert d["expected_return"] == 0.02
        assert d["factors"]["test"] == "value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
