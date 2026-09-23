"""Analysis service for market data."""

from datetime import date

import polars as pl

from orion_pulse.analysis.indicators import (
    analyze_trend,
    prepare_analysis_data,
)
from orion_pulse.config.settings import settings
from orion_pulse.core.models import AnalysisReport
from orion_pulse.data.providers.factory import ProviderFactory


class AnalysisError(Exception):
    """Exception raised during analysis."""

    pass


class AnalysisService:
    """Service for performing market analysis."""

    def __init__(self, provider_name: str = "yfinance"):
        self.provider = ProviderFactory.get_provider(provider_name)

    def analyze(
        self,
        symbol: str,
        lookback_days: int | None = None,
        ma_periods: list[int] | None = None,
        volatility_window: int | None = None,
    ) -> AnalysisReport:
        """Perform complete analysis for a symbol."""
        if lookback_days is None:
            lookback_days = settings.default_lookback_days

        if ma_periods is None:
            ma_periods = settings.default_ma_periods

        if volatility_window is None:
            volatility_window = settings.volatility_window

        # Fetch data
        try:
            raw_data = self.provider.get_latest_data(symbol, lookback_days)
        except Exception as e:
            raise AnalysisError(f"Failed to fetch data for {symbol}: {e}") from e

        if raw_data.is_empty():
            raise AnalysisError(f"No data available for symbol {symbol}")

        # Prepare data with indicators
        enriched_data = prepare_analysis_data(
            raw_data,
            ma_periods=ma_periods,
            volatility_window=volatility_window,
            volume_window=20,
        )

        # Perform trend analysis
        trend_analysis = analyze_trend(
            symbol,
            enriched_data,
            ma_periods=ma_periods,
            volatility_window=volatility_window,
        )

        # Build report
        report = AnalysisReport(
            symbol=symbol.upper(),
            generated_at=date.today(),
            trend_analysis=trend_analysis,
            lookback_days=lookback_days,
            data_points=len(enriched_data),
        )

        return report

    def get_raw_data(
        self,
        symbol: str,
        lookback_days: int | None = None,
    ) -> pl.DataFrame:
        """Get raw market data without analysis."""
        if lookback_days is None:
            lookback_days = settings.default_lookback_days

        return self.provider.get_latest_data(symbol, lookback_days)
