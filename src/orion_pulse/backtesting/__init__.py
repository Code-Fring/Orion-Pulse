"""Backtesting framework for Orion Pulse."""

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
import polars as pl

from orion_pulse.forecasting import (
    BaseForecaster,
    ForecastingService,
    ForecastModel,
    ForecastResult,
)
from orion_pulse.storage.repositories import BacktestRepository, MarketDataRepository


class BacktestError(Exception):
    """Exception raised during backtesting."""

    pass


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for backtest."""

    symbol: str
    model: ForecastModel
    start_date: date
    end_date: date
    horizon_days: int = 5
    train_window: int = 252  # Days of training data
    test_window: int = 20  # Days between re-training
    step_size: int = 1  # Step forward each iteration
    min_train_samples: int = 60


@dataclass(frozen=True)
class BacktestPrediction:
    """Single prediction in backtest."""

    forecast_date: date
    target_date: date
    expected_return: float
    prob_positive: float
    actual_return: float | None = None
    p10: float = 0.0
    p25: float = 0.0
    p50: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    volatility: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    """Complete backtest result."""

    symbol: str
    model_name: str
    start_date: date
    end_date: date
    horizon_days: int
    total_predictions: int
    directional_accuracy: float | None = None
    mae: float | None = None
    rmse: float | None = None
    calibration_error: float | None = None
    cumulative_return: float | None = None
    annualized_return: float | None = None
    volatility: float | None = None
    max_drawdown: float | None = None
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    detailed_results: dict[str, Any] = field(default_factory=dict)
    predictions: list[BacktestPrediction] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "horizon_days": self.horizon_days,
            "total_predictions": self.total_predictions,
            "directional_accuracy": self.directional_accuracy,
            "mae": self.mae,
            "rmse": self.rmse,
            "calibration_error": self.calibration_error,
            "cumulative_return": self.cumulative_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "parameters": self.parameters,
            "detailed_results": self.detailed_results,
            "predictions": [
                {
                    "forecast_date": p.forecast_date.isoformat(),
                    "target_date": p.target_date.isoformat(),
                    "expected_return": p.expected_return,
                    "prob_positive": p.prob_positive,
                    "actual_return": p.actual_return,
                    "p10": p.p10,
                    "p25": p.p25,
                    "p50": p.p50,
                    "p75": p.p75,
                    "p90": p.p90,
                    "volatility": p.volatility,
                }
                for p in self.predictions
            ],
        }


class BacktestEngine:
    """Walk-forward backtesting engine."""

    def __init__(
        self,
        forecasting_service: ForecastingService | None = None,
        market_repo: MarketDataRepository | None = None,
        backtest_repo: BacktestRepository | None = None,
    ):
        self.forecasting_service = forecasting_service or ForecastingService()
        self.market_repo = market_repo or MarketDataRepository()
        self.backtest_repo = backtest_repo or BacktestRepository()

    def run_backtest(
        self,
        config: BacktestConfig,
        save_to_db: bool = True,
    ) -> BacktestResult:
        """Run walk-forward backtest."""
        # Load all historical data needed
        # We need data from (start_date - train_window) to end_date
        data_start = config.start_date - timedelta(days=config.train_window + 100)

        df = self.market_repo.get_market_data(
            config.symbol,
            start_date=data_start,
            end_date=config.end_date,
        )

        if df.is_empty():
            raise BacktestError(f"No market data for {config.symbol}")

        # Ensure data is sorted by date
        df = df.sort("date")

        # Convert date strings to date objects
        from datetime import date as date_type

        dates = [
            date_type.fromisoformat(str(d)) if isinstance(d, str) else d
            for d in df["date"].to_list()
        ]
        closes = df["close"].to_list()

        # Find indices for backtest period
        start_idx = None
        end_idx = None
        for i, d in enumerate(dates):
            if d >= config.start_date and start_idx is None:
                start_idx = i
            if d <= config.end_date:
                end_idx = i

        if start_idx is None or end_idx is None or start_idx >= end_idx:
            raise BacktestError("Invalid date range for backtest")

        predictions = []
        current_idx = start_idx

        while current_idx + config.horizon_days <= end_idx:
            # Training data: from (current_idx - train_window) to current_idx
            train_start = max(0, current_idx - config.train_window)
            train_data = df.slice(train_start, current_idx - train_start + 1)

            if len(train_data) < config.min_train_samples:
                current_idx += config.step_size
                continue

            # Target date is current_idx + horizon_days
            target_idx = current_idx + config.horizon_days
            if target_idx >= len(dates):
                break

            forecast_date = dates[current_idx]
            target_date = dates[target_idx]

            # Generate forecast using training data only (no look-ahead!)
            try:
                forecast = self.forecasting_service.generate_forecast(
                    symbol=config.symbol,
                    df=train_data,
                    model=config.model,
                    horizon_days=config.horizon_days,
                    save_to_db=False,  # Don't save individual forecasts
                )
            except Exception:
                # Skip this iteration
                current_idx += config.step_size
                continue

            # Calculate actual return
            actual_return = (closes[target_idx] - closes[current_idx]) / closes[
                current_idx
            ]

            prediction = BacktestPrediction(
                forecast_date=forecast_date,
                target_date=target_date,
                expected_return=forecast.expected_return,
                prob_positive=forecast.prob_positive,
                actual_return=actual_return,
                p10=forecast.p10,
                p25=forecast.p25,
                p50=forecast.p50,
                p75=forecast.p75,
                p90=forecast.p90,
                volatility=forecast.volatility,
            )

            predictions.append(prediction)
            current_idx += config.step_size

        if not predictions:
            raise BacktestError("No valid predictions generated")

        # Calculate metrics
        result = self._calculate_metrics(config, predictions)

        if save_to_db:
            self.backtest_repo.save_backtest(result.to_dict())

        return result

    def _calculate_metrics(
        self,
        config: BacktestConfig,
        predictions: list[BacktestPrediction],
    ) -> BacktestResult:
        """Calculate backtest performance metrics."""
        # Filter predictions with actual returns
        valid_preds = [p for p in predictions if p.actual_return is not None]

        if not valid_preds:
            return BacktestResult(
                symbol=config.symbol,
                model_name=config.model.value,
                start_date=config.start_date,
                end_date=config.end_date,
                horizon_days=config.horizon_days,
                total_predictions=len(predictions),
                predictions=predictions,
            )

        n = len(valid_preds)

        # Directional accuracy
        correct_direction = sum(
            1
            for p in valid_preds
            if (p.expected_return > 0 and p.actual_return > 0)
            or (p.expected_return < 0 and p.actual_return < 0)
            or (p.expected_return == 0 and p.actual_return == 0)
        )
        directional_accuracy = correct_direction / n

        # MAE and RMSE
        errors = [abs(p.expected_return - p.actual_return) for p in valid_preds]
        squared_errors = [
            (p.expected_return - p.actual_return) ** 2 for p in valid_preds
        ]
        mae = np.mean(errors)
        rmse = np.sqrt(np.mean(squared_errors))

        # Calibration error (for probabilistic forecasts)
        # Bin predictions by prob_positive and check actual frequency
        calibration_error = self._calculate_calibration_error(valid_preds)

        # Returns-based metrics
        actual_returns = [p.actual_return for p in valid_preds]
        predicted_returns = [p.expected_return for p in valid_preds]

        cumulative_return = np.prod([1 + r for r in actual_returns]) - 1

        # Annualized return (assuming 252 trading days)
        avg_period_days = (
            (config.end_date - config.start_date).days / n if n > 0 else 20
        )
        periods_per_year = 252 / avg_period_days if avg_period_days > 0 else 12
        annualized_return = (
            (1 + cumulative_return) ** periods_per_year - 1
            if cumulative_return > -1
            else -1
        )

        # Volatility of actual returns
        volatility = (
            np.std(actual_returns) * np.sqrt(252) if len(actual_returns) > 1 else 0
        )

        # Max drawdown
        cumulative = np.cumprod([1 + r for r in actual_returns])
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0

        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = (
            (np.mean(actual_returns) * 252) / (volatility + 1e-8)
            if volatility > 0
            else 0
        )

        # Sortino ratio (downside deviation)
        negative_returns = [r for r in actual_returns if r < 0]
        downside_std = (
            np.std(negative_returns) * np.sqrt(252)
            if len(negative_returns) > 1
            else volatility
        )
        sortino_ratio = (
            (np.mean(actual_returns) * 252) / (downside_std + 1e-8)
            if downside_std > 0
            else 0
        )

        # Detailed results
        detailed = {
            "directional_accuracy_breakdown": {
                "correct_up": sum(
                    1
                    for p in valid_preds
                    if p.expected_return > 0 and p.actual_return > 0
                ),
                "correct_down": sum(
                    1
                    for p in valid_preds
                    if p.expected_return < 0 and p.actual_return < 0
                ),
                "wrong_up": sum(
                    1
                    for p in valid_preds
                    if p.expected_return > 0 and p.actual_return < 0
                ),
                "wrong_down": sum(
                    1
                    for p in valid_preds
                    if p.expected_return < 0 and p.actual_return > 0
                ),
            },
            "error_distribution": {
                "mean_error": float(
                    np.mean([p.expected_return - p.actual_return for p in valid_preds])
                ),
                "median_error": float(
                    np.median(
                        [p.expected_return - p.actual_return for p in valid_preds]
                    )
                ),
                "std_error": float(
                    np.std([p.expected_return - p.actual_return for p in valid_preds])
                ),
            },
        }

        return BacktestResult(
            symbol=config.symbol,
            model_name=config.model.value,
            start_date=config.start_date,
            end_date=config.end_date,
            horizon_days=config.horizon_days,
            total_predictions=n,
            directional_accuracy=directional_accuracy,
            mae=float(mae),
            rmse=float(rmse),
            calibration_error=calibration_error,
            cumulative_return=float(cumulative_return),
            annualized_return=float(annualized_return),
            volatility=float(volatility),
            max_drawdown=max_drawdown,
            sharpe_ratio=float(sharpe_ratio),
            sortino_ratio=float(sortino_ratio),
            parameters={
                "train_window": config.train_window,
                "test_window": config.test_window,
                "step_size": config.step_size,
                "min_train_samples": config.min_train_samples,
            },
            detailed_results=detailed,
            predictions=predictions,
        )

    def _calculate_calibration_error(
        self,
        predictions: list[BacktestPrediction],
    ) -> float:
        """Calculate calibration error for probabilistic forecasts."""
        # Bin predictions by prob_positive
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_errors = []

        for i in range(len(bins) - 1):
            bin_preds = [
                p for p in predictions if bins[i] <= p.prob_positive < bins[i + 1]
            ]

            if not bin_preds:
                continue

            # Actual frequency of positive returns
            actual_positive = sum(1 for p in bin_preds if p.actual_return > 0) / len(
                bin_preds
            )
            predicted_positive = np.mean([p.prob_positive for p in bin_preds])

            bin_errors.append(abs(actual_positive - predicted_positive))

        return float(np.mean(bin_errors)) if bin_errors else 0.0

    def run_comparative_backtest(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        horizon_days: int = 5,
        models: list[ForecastModel] | None = None,
    ) -> dict[str, BacktestResult]:
        """Run backtest for multiple models and compare."""
        if models is None:
            models = list(ForecastModel)

        results = {}
        for model in models:
            config = BacktestConfig(
                symbol=symbol,
                model=model,
                start_date=start_date,
                end_date=end_date,
                horizon_days=horizon_days,
            )
            try:
                results[model.value] = self.run_backtest(config, save_to_db=True)
            except BacktestError:
                results[model.value] = None  # Failed

        return results
