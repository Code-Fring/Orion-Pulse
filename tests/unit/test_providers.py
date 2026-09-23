"""Unit tests for data providers."""

from datetime import date
from unittest.mock import Mock, patch

import polars as pl
import pytest

from orion_pulse.data.providers.base import (
    DataNotFoundError,
    DataProviderError,
    ProviderUnavailableError,
)
from orion_pulse.data.providers.factory import ProviderFactory
from orion_pulse.data.providers.yfinance_provider import YFinanceProvider


class TestYFinanceProvider:
    def test_get_provider_name(self):
        provider = YFinanceProvider()
        assert provider.get_provider_name() == "yfinance"

    def test_is_available_enabled(self):
        with patch(
            "orion_pulse.data.providers.yfinance_provider.settings"
        ) as mock_settings:
            mock_settings.yfinance_enabled = True
            provider = YFinanceProvider()
            assert provider.is_available() is True

    def test_is_available_disabled(self):
        with patch(
            "orion_pulse.data.providers.yfinance_provider.settings"
        ) as mock_settings:
            mock_settings.yfinance_enabled = False
            provider = YFinanceProvider()
            assert provider.is_available() is False

    def test_get_historical_data_provider_unavailable(self):
        with patch(
            "orion_pulse.data.providers.yfinance_provider.settings"
        ) as mock_settings:
            mock_settings.yfinance_enabled = False
            provider = YFinanceProvider()

            with pytest.raises(ProviderUnavailableError):
                provider.get_historical_data(
                    "TEST", date(2024, 1, 1), date(2024, 1, 31)
                )

    @patch("orion_pulse.data.providers.yfinance_provider.yf.Ticker")
    def test_get_historical_data_success(self, mock_ticker):
        # Setup mock
        mock_hist = Mock()
        mock_hist.empty = False
        mock_hist.index.date = [date(2024, 1, 1), date(2024, 1, 2)]

        # Create mock series with .values attribute
        def create_mock_series(values):
            mock_series = Mock()
            mock_series.values = values
            return mock_series

        mock_hist.__getitem__ = Mock(
            side_effect=lambda key: create_mock_series(
                {
                    "Open": [100.0, 101.0],
                    "High": [102.0, 103.0],
                    "Low": [99.0, 100.0],
                    "Close": [101.0, 102.0],
                    "Volume": [1000000, 1100000],
                    "Adj Close": [100.5, 101.5],
                }[key]
            )
        )
        mock_hist.columns = ["Open", "High", "Low", "Close", "Volume", "Adj Close"]
        mock_ticker.return_value.history.return_value = mock_hist

        with patch(
            "orion_pulse.data.providers.yfinance_provider.settings"
        ) as mock_settings:
            mock_settings.yfinance_enabled = True
            mock_settings.yfinance_timeout = 30
            provider = YFinanceProvider()

            result = provider.get_historical_data(
                "TEST", date(2024, 1, 1), date(2024, 1, 31)
            )

        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2
        assert "symbol" in result.columns
        assert result[0, "symbol"] == "TEST"

    @patch("orion_pulse.data.providers.yfinance_provider.yf.Ticker")
    def test_get_historical_data_not_found(self, mock_ticker):
        mock_hist = Mock()
        mock_hist.empty = True
        mock_ticker.return_value.history.return_value = mock_hist

        with patch(
            "orion_pulse.data.providers.yfinance_provider.settings"
        ) as mock_settings:
            mock_settings.yfinance_enabled = True
            mock_settings.yfinance_timeout = 30
            provider = YFinanceProvider()

            with pytest.raises(DataNotFoundError):
                provider.get_historical_data(
                    "INVALID", date(2024, 1, 1), date(2024, 1, 31)
                )

    @patch("orion_pulse.data.providers.yfinance_provider.yf.Ticker")
    def test_get_historical_data_exception(self, mock_ticker):
        mock_ticker.return_value.history.side_effect = Exception("Network error")

        with patch(
            "orion_pulse.data.providers.yfinance_provider.settings"
        ) as mock_settings:
            mock_settings.yfinance_enabled = True
            mock_settings.yfinance_timeout = 30
            provider = YFinanceProvider()

            with pytest.raises(DataProviderError):
                provider.get_historical_data(
                    "TEST", date(2024, 1, 1), date(2024, 1, 31)
                )


class TestProviderFactory:
    def test_get_yfinance_provider(self):
        ProviderFactory.clear_cache()
        provider = ProviderFactory.get_provider("yfinance")
        assert isinstance(provider, YFinanceProvider)

    def test_get_provider_caches(self):
        ProviderFactory.clear_cache()
        provider1 = ProviderFactory.get_provider("yfinance")
        provider2 = ProviderFactory.get_provider("yfinance")
        assert provider1 is provider2

    def test_get_unknown_provider_raises(self):
        ProviderFactory.clear_cache()
        with pytest.raises(ValueError, match="Unknown provider"):
            ProviderFactory.get_provider("unknown")

    def test_get_available_providers(self):
        ProviderFactory.clear_cache()
        with patch(
            "orion_pulse.data.providers.factory.ProviderFactory.get_provider"
        ) as mock_get:
            mock_provider = Mock()
            mock_provider.is_available.return_value = True
            mock_get.return_value = mock_provider

            available = ProviderFactory.get_available_providers()
            assert "yfinance" in available

    def test_clear_cache(self):
        ProviderFactory.get_provider("yfinance")
        ProviderFactory.clear_cache()
        # Should not raise
        ProviderFactory.get_provider("yfinance")


class TestExceptions:
    def test_data_provider_error(self):
        err = DataProviderError("Test error", provider="test", symbol="TEST")
        assert err.provider == "test"
        assert err.symbol == "TEST"
        assert str(err) == "Test error"

    def test_data_not_found_error(self):
        err = DataNotFoundError("Not found", provider="test", symbol="TEST")
        assert isinstance(err, DataProviderError)

    def test_provider_unavailable_error(self):
        err = ProviderUnavailableError("Unavailable", provider="test")
        assert isinstance(err, DataProviderError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
