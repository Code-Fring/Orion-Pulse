"""Unit tests for backtesting module."""

from datetime import date, timedelta
from unittest.mock import Mock, patch

import polars as pl
import pytest

from orion_pulse.backtesting import (
    BacktestConfig,
    BacktestEngine,
    BacktestError,
    BacktestPrediction,
    BacktestResult,
)
from orion_pulse.forecasting import ForecastModel, ForecastResult


@pytest.fixture
def sample_market_data():
    """Create sample market data for backtesting."""
    import numpy as np

    np.random.seed(42)
    n = 500
    dates = [date(2023, 1, 1) + timedelta(days=i) for i in range(n)]
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame(
        {
            "date": dates,
            "open": prices * 0.999,
            "high": prices * 1.005,
            "low": prices * 0.995,
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
            "symbol": ["TEST"] * n,
        }
    )


class TestBacktestConfig:
    def test_create_config(self):
        config = BacktestConfig(
            symbol="TEST",
            model=ForecastModel.HISTORICAL_MEAN,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            horizon_days=5,
            train_window=252,
            step_size=5,
        )

        assert config.symbol == "TEST"
        assert config.model == ForecastModel.HISTORICAL_MEAN
        assert config.horizon_days == 5


class TestBacktestPrediction:
    def test_create_prediction(self):
        pred = BacktestPrediction(
            forecast_date=date(2024, 1, 15),
            target_date=date(2024, 1, 22),
            expected_return=0.02,
            prob_positive=0.6,
            actual_return=0.015,
            p10=-0.05,
            p25=-0.01,
            p50=0.02,
            p75=0.05,
            p90=0.09,
            volatility=0.15,
        )

        assert pred.expected_return == 0.02
        assert pred.actual_return == 0.015


class TestBacktestEngine:
    def test_run_backtest_basic(self, sample_market_data):
        # Mock the forecasting service to return predictable results
        with patch("orion_pulse.backtesting.ForecastingService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service

            # Mock forecast generation
            mock_forecast = ForecastResult(
                symbol="TEST",
                model_name="historical_mean",
                horizon_days=5,
                forecast_date=date(2024, 1, 15),
                target_date=date(2024, 1, 22),
                expected_return=0.01,
                prob_positive=0.55,
                p10=-0.03,
                p25=-0.01,
                p50=0.01,
                p75=0.03,
                p90=0.05,
                volatility=0.12,
                confidence=0.6,
                factors={},
                input_features={},
            )
            mock_service.generate_forecast.return_value = mock_forecast

            # Mock market data repository
            with patch(
                "orion_pulse.backtesting.MarketDataRepository"
            ) as mock_repo_class:
                mock_repo = Mock()
                mock_repo_class.return_value = mock_repo
                mock_repo.get_market_data.return_value = sample_market_data

                engine = BacktestEngine()
                config = BacktestConfig(
                    symbol="TEST",
                    model=ForecastModel.HISTORICAL_MEAN,
                    start_date=date(2024, 3, 1),
                    end_date=date(2024, 3, 31),
                    horizon_days=5,
                    train_window=100,
                    step_size=5,
                    min_train_samples=50,
                )

                result = engine.run_backtest(config, save_to_db=False)

                assert isinstance(result, BacktestResult)
                assert result.symbol == "TEST"
                assert result.model_name == "historical_mean"
                assert result.total_predictions > 0
                assert len(result.predictions) > 0

    def test_calculate_metrics(self, sample_market_data):
        engine = BacktestEngine()

        predictions = [
            BacktestPrediction(
                forecast_date=date(2024, 1, 15),
                target_date=date(2024, 1, 22),
                expected_return=0.02,
                prob_positive=0.6,
                actual_return=0.015,
            ),
            BacktestPrediction(
                forecast_date=date(2024, 1, 22),
                target_date=date(2024, 1, 29),
                expected_return=-0.01,
                prob_positive=0.4,
                actual_return=-0.005,
            ),
            BacktestPrediction(
                forecast_date=date(2024, 1, 29),
                target_date=date(2024, 2, 5),
                expected_return=0.03,
                prob_positive=0.7,
                actual_return=0.025,
            ),
        ]

        config = BacktestConfig(
            symbol="TEST",
            model=ForecastModel.HISTORICAL_MEAN,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 28),
            horizon_days=5,
        )

        result = engine._calculate_metrics(config, predictions)

        assert result.total_predictions == 3
        assert result.directional_accuracy == 1.0  # All correct direction
        assert result.mae is not None
        assert result.rmse is not None
        assert result.cumulative_return is not None

    def test_calibration_error(self):
        engine = BacktestEngine()

        # Perfect calibration
        predictions = [
            BacktestPrediction(
                forecast_date=date(2024, 1, 15),
                target_date=date(2024, 1, 22),
                expected_return=0.02,
                prob_positive=0.8,
                actual_return=0.01,
            ),
            BacktestPrediction(
                forecast_date=date(2024, 1, 22),
                target_date=date(2024, 1, 29),
                expected_return=0.01,
                prob_positive=0.8,
                actual_return=0.02,
            ),
        ]

        # Both have high prob_positive and both actual returns are positive
        error = engine._calculate_calibration_error(predictions)
        # With only 2 predictions in same bin, calibration error should be low
        assert error >= 0

    def test_run_backtest_insufficient_data(self, sample_market_data):
        with patch("orion_pulse.backtesting.MarketDataRepository") as mock_repo_class:
            mock_repo = Mock()
            mock_repo_class.return_value = mock_repo
            # Return empty data
            mock_repo.get_market_data.return_value = pl.DataFrame(
                {
                    "date": [],
                    "open": [],
                    "high": [],
                    "low": [],
                    "close": [],
                    "volume": [],
                    "symbol": [],
                }
            )

            engine = BacktestEngine()
            config = BacktestConfig(
                symbol="TEST",
                model=ForecastModel.HISTORICAL_MEAN,
                start_date=date(2024, 1, 1),
                end_date=date(2024, 6, 30),
            )

            with pytest.raises(BacktestError, match="No market data"):
                engine.run_backtest(config)

    def test_run_backtest_invalid_dates(self):
        engine = BacktestEngine()
        config = BacktestConfig(
            symbol="TEST",
            model=ForecastModel.HISTORICAL_MEAN,
            start_date=date(2024, 6, 30),
            end_date=date(2024, 1, 1),  # End before start
        )

        with pytest.raises(BacktestError):
            engine.run_backtest(config)


class TestBacktestResult:
    def test_to_dict(self):
        result = BacktestResult(
            symbol="TEST",
            model_name="historical_mean",
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            horizon_days=5,
            total_predictions=50,
            directional_accuracy=0.55,
            mae=0.02,
            rmse=0.03,
            calibration_error=0.05,
            cumulative_return=0.12,
            annualized_return=0.24,
            volatility=0.18,
            max_drawdown=-0.08,
            sharpe_ratio=1.33,
            sortino_ratio=1.8,
            parameters={"train_window": 252},
            detailed_results={},
            predictions=[],
        )

        d = result.to_dict()
        assert d["symbol"] == "TEST"
        assert d["directional_accuracy"] == 0.55
        assert len(d["predictions"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
