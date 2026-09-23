"""Forecasting engine for Orion Pulse."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Any

import numpy as np
import polars as pl

from orion_pulse.storage.repositories import ForecastRepository


def _as_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, handling None and Polars types."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_float_series(series: pl.Series) -> float:
    """Extract single float from a Polars series, handling None."""
    if series.is_empty():
        return 0.0
    val = series[0]
    if val is None:
        return 0.0
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


class ForecastModel(str, Enum):
    """Available forecast models."""

    HISTORICAL_MEAN = "historical_mean"
    VOLATILITY_BASED = "volatility_based"
    TREND_MOMENTUM = "trend_momentum"
    RANDOM_WALK = "random_walk"
    ENSEMBLE = "ensemble"


@dataclass(frozen=True)
class ForecastResult:
    """Forecast result with probabilistic output."""

    symbol: str
    model_name: str
    horizon_days: int
    forecast_date: date
    target_date: date
    expected_return: float
    prob_positive: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    volatility: float
    confidence: float
    factors: dict[str, Any]
    input_features: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "model_name": self.model_name,
            "horizon_days": self.horizon_days,
            "forecast_date": self.forecast_date.isoformat(),
            "target_date": self.target_date.isoformat(),
            "expected_return": self.expected_return,
            "prob_positive": self.prob_positive,
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "volatility": self.volatility,
            "confidence": self.confidence,
            "factors": self.factors,
            "input_features": self.input_features,
        }


class ForecastingError(Exception):
    """Exception raised during forecasting."""

    pass


class BaseForecaster(ABC):
    """Abstract base class for forecasters."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return model name."""
        ...

    @abstractmethod
    def forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate forecast for a symbol."""
        ...

    def _calculate_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """Calculate daily returns."""
        return df.with_columns(
            pl.col("close").pct_change().alias("return_pct"),
        )

    def _get_return_stats(self, returns: pl.Series) -> dict[str, float]:
        """Get statistics from return series."""
        clean_returns = returns.drop_nulls()
        if len(clean_returns) == 0:
            return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}

        return {
            "mean": _as_float(clean_returns.mean()),
            "std": _as_float(clean_returns.std()),
            "skew": _as_float(clean_returns.skew()),
            "kurtosis": _as_float(clean_returns.kurtosis()),
        }

    def _percentile_forecast(
        self,
        mean: float,
        std: float,
        horizon_days: int,
        prob_positive_prior: float = 0.5,
    ) -> tuple[float, float, dict[str, float]]:
        """Calculate percentile forecasts assuming normal distribution."""
        # Scale for horizon
        horizon_mean = mean * horizon_days
        horizon_std = std * np.sqrt(horizon_days)

        # Probability of positive return
        from scipy import stats

        prob_positive = (
            float(stats.norm.cdf(horizon_mean / horizon_std))
            if horizon_std > 0
            else prob_positive_prior
        )

        # Percentiles
        percentiles = {}
        for p in [10, 25, 50, 75, 90]:
            z = stats.norm.ppf(p / 100)
            percentiles[f"p{p}"] = float(horizon_mean + z * horizon_std)

        return prob_positive, float(horizon_std), percentiles


class HistoricalMeanForecaster(BaseForecaster):
    """Historical mean return forecaster."""

    @property
    def model_name(self) -> str:
        return ForecastModel.HISTORICAL_MEAN.value

    def forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        **kwargs: Any,
    ) -> ForecastResult:
        """Forecast using historical mean returns."""
        lookback_days = kwargs.get("lookback_days", 252)
        if df.is_empty():
            raise ForecastingError("No data provided")

        # Use recent data for mean calculation
        recent_df = df.tail(min(lookback_days, len(df)))
        df_with_returns = self._calculate_returns(recent_df)
        returns = df_with_returns["return_pct"].drop_nulls()

        stats = self._get_return_stats(returns)
        mean_return = stats["mean"]
        std_return = stats["std"]

        prob_positive, horizon_vol, percentiles = self._percentile_forecast(
            mean_return, std_return, horizon_days
        )

        forecast_date = date.today()
        target_date = forecast_date  # Simplified - would add business days

        return ForecastResult(
            symbol=symbol,
            model_name=self.model_name,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            target_date=target_date,
            expected_return=float(mean_return * horizon_days),
            prob_positive=prob_positive,
            p10=percentiles["p10"],
            p25=percentiles["p25"],
            p50=percentiles["p50"],
            p75=percentiles["p75"],
            p90=percentiles["p90"],
            volatility=float(horizon_vol),
            confidence=0.5,  # Low confidence for simple model
            factors={"method": "historical_mean", "lookback_days": lookback_days},
            input_features={
                "mean_daily_return": mean_return,
                "daily_volatility": std_return,
                "sample_size": len(returns),
            },
        )


class VolatilityBasedForecaster(BaseForecaster):
    """Volatility-based forecaster using GARCH-like scaling."""

    @property
    def model_name(self) -> str:
        return ForecastModel.VOLATILITY_BASED.value

    def forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        **kwargs: Any,
    ) -> ForecastResult:
        """Forecast using current volatility regime."""
        vol_window = kwargs.get("vol_window", 20)
        if df.is_empty():
            raise ForecastingError("No data provided")

        df_with_returns = self._calculate_returns(df)

        # Calculate rolling volatility
        df_with_vol = df_with_returns.with_columns(
            (
                pl.col("return_pct").rolling_std(window_size=vol_window) * np.sqrt(252)
            ).alias("annual_vol")
        )

        # Get current volatility
        current_vol = _as_float_series(df_with_vol.tail(1)["annual_vol"])
        if current_vol == 0.0:
            # Fallback to overall volatility
            returns = df_with_returns["return_pct"].drop_nulls()
            current_vol = float(returns.std() * np.sqrt(252))

        # Get recent mean return
        recent_returns = df_with_returns.tail(20)["return_pct"].drop_nulls()
        mean_return = (
            _as_float(recent_returns.mean()) if len(recent_returns) > 0 else 0.0
        )

        # Convert annual vol to daily
        daily_vol = current_vol / np.sqrt(252)

        prob_positive, horizon_vol, percentiles = self._percentile_forecast(
            mean_return, daily_vol, horizon_days
        )

        forecast_date = date.today()
        target_date = forecast_date

        return ForecastResult(
            symbol=symbol,
            model_name=self.model_name,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            target_date=target_date,
            expected_return=float(mean_return * horizon_days),
            prob_positive=prob_positive,
            p10=percentiles["p10"],
            p25=percentiles["p25"],
            p50=percentiles["p50"],
            p75=percentiles["p75"],
            p90=percentiles["p90"],
            volatility=float(horizon_vol),
            confidence=0.6,
            factors={
                "method": "volatility_based",
                "current_annual_vol": float(current_vol),
                "vol_window": vol_window,
            },
            input_features={
                "mean_daily_return": mean_return,
                "daily_volatility": daily_vol,
                "current_annual_vol": float(current_vol),
            },
        )


class TrendMomentumForecaster(BaseForecaster):
    """Trend and momentum based forecaster."""

    @property
    def model_name(self) -> str:
        return ForecastModel.TREND_MOMENTUM.value

    def forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        **kwargs: Any,
    ) -> ForecastResult:
        """Forecast using trend and momentum signals."""
        ma_periods = kwargs.get("ma_periods", [20, 50, 200])
        if df.is_empty():
            raise ForecastingError("No data provided")

        if ma_periods is None:
            ma_periods = [20, 50, 200]

        df_with_returns = self._calculate_returns(df)

        # Calculate moving averages
        for period in ma_periods:
            df_with_returns = df_with_returns.with_columns(
                pl.col("close").rolling_mean(window_size=period).alias(f"ma_{period}")
            )

        # Get latest values
        latest = df_with_returns.tail(1).to_dicts()[0]
        close = latest["close"]

        # Trend signals
        trend_signals = []
        for period in ma_periods:
            ma_val = latest.get(f"ma_{period}")
            if ma_val is not None:
                trend_signals.append(1.0 if close > ma_val else -1.0)

        # Momentum (recent return)
        recent_returns = df_with_returns.tail(20)["return_pct"].drop_nulls()
        momentum = _as_float(recent_returns.mean()) if len(recent_returns) > 0 else 0.0

        # Combine signals
        trend_score = np.mean(trend_signals) if trend_signals else 0.0

        # Expected return based on trend + momentum
        base_return = momentum
        trend_adjustment = trend_score * 0.001  # Small adjustment per signal
        expected_daily = base_return + trend_adjustment

        # Volatility
        daily_vol = _as_float(recent_returns.std()) if len(recent_returns) > 1 else 0.02

        prob_positive, horizon_vol, percentiles = self._percentile_forecast(
            expected_daily, daily_vol, horizon_days
        )

        forecast_date = date.today()
        target_date = forecast_date

        # Confidence based on trend consistency
        confidence = 0.5 + abs(trend_score) * 0.3

        return ForecastResult(
            symbol=symbol,
            model_name=self.model_name,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            target_date=target_date,
            expected_return=float(expected_daily * horizon_days),
            prob_positive=prob_positive,
            p10=percentiles["p10"],
            p25=percentiles["p25"],
            p50=percentiles["p50"],
            p75=percentiles["p75"],
            p90=percentiles["p90"],
            volatility=float(horizon_vol),
            confidence=float(min(confidence, 0.85)),
            factors={
                "method": "trend_momentum",
                "trend_signals": {
                    f"ma_{p}": s
                    for p, s in zip(ma_periods, trend_signals, strict=False)
                },
                "momentum": momentum,
            },
            input_features={
                "mean_daily_return": expected_daily,
                "daily_volatility": daily_vol,
                "trend_score": trend_score,
                "momentum": momentum,
            },
        )


class RandomWalkForecaster(BaseForecaster):
    """Random walk (driftless) forecaster - baseline."""

    @property
    def model_name(self) -> str:
        return ForecastModel.RANDOM_WALK.value

    def forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        **kwargs: Any,
    ) -> ForecastResult:
        """Forecast using random walk with drift."""
        if df.is_empty():
            raise ForecastingError("No data provided")

        df_with_returns = self._calculate_returns(df)
        returns = df_with_returns["return_pct"].drop_nulls()

        # Random walk assumes zero drift, only volatility
        stats = self._get_return_stats(returns)
        daily_vol = stats["std"]

        prob_positive, horizon_vol, percentiles = self._percentile_forecast(
            0.0, daily_vol, horizon_days, prob_positive_prior=0.5
        )

        forecast_date = date.today()
        target_date = forecast_date

        return ForecastResult(
            symbol=symbol,
            model_name=self.model_name,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            target_date=target_date,
            expected_return=0.0,
            prob_positive=0.5,
            p10=percentiles["p10"],
            p25=percentiles["p25"],
            p50=percentiles["p50"],
            p75=percentiles["p75"],
            p90=percentiles["p90"],
            volatility=float(horizon_vol),
            confidence=0.3,  # Lowest confidence - pure uncertainty
            factors={"method": "random_walk"},
            input_features={
                "daily_volatility": daily_vol,
            },
        )


class EnsembleForecaster(BaseForecaster):
    """Ensemble forecaster combining multiple models."""

    def __init__(self) -> None:
        self.forecasters = [
            HistoricalMeanForecaster(),
            VolatilityBasedForecaster(),
            TrendMomentumForecaster(),
            RandomWalkForecaster(),
        ]

    @property
    def model_name(self) -> str:
        return ForecastModel.ENSEMBLE.value

    def forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        **kwargs: Any,
    ) -> ForecastResult:
        """Forecast using weighted ensemble of models."""
        weights = kwargs.get("weights")
        if df.is_empty():
            raise ForecastingError("No data provided")

        if weights is None:
            # Equal weights
            weights = [1.0 / len(self.forecasters)] * len(self.forecasters)

        forecasts = []
        for forecaster, weight in zip(self.forecasters, weights, strict=True):
            try:
                fc = forecaster.forecast(symbol, df, horizon_days)
                forecasts.append((fc, weight))
            except Exception as e:
                # Skip failed forecasters but log the error
                logging.warning(
                    "Forecaster %s failed, skipping: %s", forecaster.model_name, e
                )

        if not forecasts:
            raise ForecastingError("All ensemble forecasters failed")

        # Weighted average of expected returns
        total_weight = sum(w for _, w in forecasts)
        expected_return = (
            sum(fc.expected_return * w for fc, w in forecasts) / total_weight
        )
        prob_positive = sum(fc.prob_positive * w for fc, w in forecasts) / total_weight
        volatility = sum(fc.volatility * w for fc, w in forecasts) / total_weight
        confidence = sum(fc.confidence * w for fc, w in forecasts) / total_weight

        # Percentiles - use weighted average
        percentiles = {}
        for p in ["p10", "p25", "p50", "p75", "p90"]:
            percentiles[p] = (
                sum(getattr(fc, p) * w for fc, w in forecasts) / total_weight
            )

        forecast_date = date.today()
        target_date = forecast_date

        return ForecastResult(
            symbol=symbol,
            model_name=self.model_name,
            horizon_days=horizon_days,
            forecast_date=forecast_date,
            target_date=target_date,
            expected_return=expected_return,
            prob_positive=prob_positive,
            p10=percentiles["p10"],
            p25=percentiles["p25"],
            p50=percentiles["p50"],
            p75=percentiles["p75"],
            p90=percentiles["p90"],
            volatility=volatility,
            confidence=float(min(confidence, 0.8)),
            factors={
                "method": "ensemble",
                "models": [fc.model_name for fc, _ in forecasts],
                "weights": weights[: len(forecasts)],
            },
            input_features={},
        )


class ForecastingService:
    """Service for generating forecasts."""

    def __init__(self, forecast_repo: ForecastRepository | None = None):
        self.forecasters = {
            ForecastModel.HISTORICAL_MEAN: HistoricalMeanForecaster(),
            ForecastModel.VOLATILITY_BASED: VolatilityBasedForecaster(),
            ForecastModel.TREND_MOMENTUM: TrendMomentumForecaster(),
            ForecastModel.RANDOM_WALK: RandomWalkForecaster(),
            ForecastModel.ENSEMBLE: EnsembleForecaster(),
        }
        self.forecast_repo = forecast_repo or ForecastRepository()

    def generate_forecast(
        self,
        symbol: str,
        df: pl.DataFrame,
        model: ForecastModel = ForecastModel.ENSEMBLE,
        horizon_days: int = 5,
        save_to_db: bool = True,
        **kwargs: Any,
    ) -> ForecastResult:
        """Generate forecast using specified model."""
        forecaster = self.forecasters.get(model)
        if not forecaster:
            raise ForecastingError(f"Unknown model: {model}")

        try:
            result = forecaster.forecast(symbol, df, horizon_days, **kwargs)
        except Exception as e:
            raise ForecastingError(f"Forecast failed: {e}") from e

        if save_to_db:
            self.forecast_repo.save_forecast(result.to_dict())

        return result

    def generate_all_forecasts(
        self,
        symbol: str,
        df: pl.DataFrame,
        horizon_days: int = 5,
        save_to_db: bool = True,
    ) -> dict[str, ForecastResult | None]:
        """Generate forecasts from all models."""
        results: dict[str, ForecastResult | None] = {}
        for model_name, forecaster in self.forecasters.items():
            try:
                results[model_name.value] = forecaster.forecast(
                    symbol, df, horizon_days
                )
            except Exception:
                results[model_name.value] = None  # Failed

        if save_to_db:
            for result in results.values():
                if result:
                    self.forecast_repo.save_forecast(result.to_dict())

        return results

    def get_forecaster(self, model: ForecastModel) -> BaseForecaster:
        """Get a specific forecaster."""
        forecaster = self.forecasters.get(model)
        if forecaster is None:
            raise ForecastingError(f"Unknown model: {model}")
        return forecaster
