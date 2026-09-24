"""Integration tests for CLI commands."""

from unittest.mock import Mock, patch

import polars as pl
import pytest
from typer.testing import CliRunner

from orion_pulse.cli.main import app

runner = CliRunner()


def create_mock_data() -> pl.DataFrame:
    """Create mock data for testing."""
    return pl.DataFrame(
        {
            "date": [f"2024-01-{i:02d}" for i in range(1, 11)],
            "open": [100.0 + i for i in range(10)],
            "high": [102.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [101.0 + i for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
            "symbol": ["TEST"] * 10,
        }
    )


class TestAnalyzeCommand:
    def test_analyze_success(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            result = runner.invoke(app, ["analyze", "TEST"])

            assert result.exit_code == 0
            assert "TEST" in result.output

    def test_analyze_with_json_output(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            result = runner.invoke(app, ["analyze", "TEST", "--json"])

            assert result.exit_code == 0
            assert '"symbol": "TEST"' in result.output
            assert '"trend"' in result.output

    def test_analyze_with_custom_lookback(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            result = runner.invoke(app, ["analyze", "TEST", "--lookback", "90"])

            assert result.exit_code == 0

    def test_analyze_with_custom_ma_periods(self):
        mock_data = create_mock_data()

        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.return_value = mock_data
            mock_factory.return_value = mock_provider

            result = runner.invoke(app, ["analyze", "TEST", "--ma", "10,20,50"])

            assert result.exit_code == 0

    def test_analyze_invalid_ma_periods(self):
        result = runner.invoke(app, ["analyze", "TEST", "--ma", "invalid"])
        assert result.exit_code == 1
        assert "Invalid MA periods format" in result.output

    def test_analyze_no_data(self):
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

            result = runner.invoke(app, ["analyze", "TEST"])
            assert result.exit_code == 1
            assert "No data available" in result.output

    def test_analyze_provider_error(self):
        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_provider"
        ) as mock_factory:
            mock_provider = Mock()
            mock_provider.get_latest_data.side_effect = Exception("API Error")
            mock_factory.return_value = mock_provider

            result = runner.invoke(app, ["analyze", "TEST"])
            assert result.exit_code == 1
            assert "Failed to fetch data" in result.output


class TestProvidersCommand:
    def test_providers_list(self):
        with patch(
            "orion_pulse.analysis.service.ProviderFactory.get_available_providers"
        ) as mock_get:
            mock_get.return_value = ["yfinance"]

            result = runner.invoke(app, ["providers"])

            assert result.exit_code == 0
            assert "yfinance" in result.output


class TestVersionCallback:
    def test_version_flag(self):
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert "Orion Pulse v0.2.0" in result.output


class TestHelp:
    def test_main_help(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Orion Pulse" in result.output
        assert "analyze" in result.output

    def test_analyze_help(self):
        result = runner.invoke(app, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "lookback" in result.output
        assert "ma" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
