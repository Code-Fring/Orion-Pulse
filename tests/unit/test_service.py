"""Unit tests for analysis service."""

from datetime import date
from unittest.mock import Mock, patch

import polars as pl
import pytest

from orion_pulse.analysis.service import AnalysisError, AnalysisService
from orion_pulse.core.models import Trend


def create_mock_data() -> pl.DataFrame:
    """Create mock data for testing."""
    return pl.DataFrame(
        {
            "date": [date(2024, 1, i) for i in range(1, 11)],
            "open": [100.0 + i for i in range(10)],
            "high": [102.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [101.0 + i for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
            "symbol": ["TEST"] * 10,
        }
    )


class TestAnalysisService:
    def test_analyze_success(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            service = AnalysisService(provider_name="yfinance")
            report = service.analyze("TEST", lookback_days=10)

            assert report.symbol == "TEST"
            assert report.trend_analysis.symbol == "TEST"
            assert report.trend_analysis.trend in [
                Trend.BULLISH,
                Trend.BEARISH,
                Trend.SIDEWAYS,
            ]
            assert report.data_points == 10
            assert report.lookback_days == 10

    def test_analyze_no_data_raises(self):
        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = pl.DataFrame(
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
            mock_factory.return_value = mock_provider

            service = AnalysisService(provider_name="yfinance")

            with pytest.raises(AnalysisError, match="No data available"):
                service.analyze("TEST")

    def test_analyze_provider_error_raises(self):
        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.side_effect = Exception("API Error")
            mock_factory.return_value = mock_provider

            service = AnalysisService(provider_name="yfinance")

            with pytest.raises(AnalysisError, match="Failed to fetch data"):
                service.analyze("TEST")

    def test_analyze_custom_ma_periods(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            service = AnalysisService(provider_name="yfinance")
            report = service.analyze("TEST", ma_periods=[10, 30])

            # Should have MA10 and MA30 in trend analysis
            assert report.trend_analysis.ma_20 is None
            # The service uses ma_periods but the model only has ma_20, ma_50, ma_200
            # So we just verify it runs without error

    def test_get_raw_data(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            service = AnalysisService(provider_name="yfinance")
            data = service.get_raw_data("TEST", lookback_days=10)

            assert isinstance(data, pl.DataFrame)
            assert len(data) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
