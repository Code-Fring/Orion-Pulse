"""Unit tests for CLI output formatting."""

from datetime import date, datetime
from unittest.mock import patch

from orion_pulse.cli.output import (
    format_pct,
    format_price,
    format_ratio,
    format_trend,
    format_volatility,
    render_analysis_report,
)
from orion_pulse.config.settings import settings
from orion_pulse.core.models import AnalysisReport, Trend, TrendAnalysis


def create_test_report() -> AnalysisReport:
    """Create a test analysis report."""
    ta = TrendAnalysis(
        symbol="TEST",
        as_of=date(2024, 1, 15),
        trend=Trend.BULLISH,
        last_price=150.0,
        ma_20=145.0,
        ma_50=140.0,
        ma_200=130.0,
        volatility=0.25,
        volume_ratio=1.2,
        price_above_ma20=True,
        price_above_ma50=True,
        price_above_ma200=True,
        ma20_above_ma50=True,
        ma50_above_ma200=True,
    )
    return AnalysisReport(
        symbol="TEST",
        generated_at=datetime(2024, 1, 15, 10, 30, 0),
        trend_analysis=ta,
        lookback_days=365,
        data_points=252,
    )


class TestFormatters:
    def test_format_trend_bullish(self):
        with patch.object(settings, "no_color", False):
            result = format_trend(Trend.BULLISH)
            assert "BULLISH" in result.plain
            assert result.style == "bold green"

    def test_format_trend_bearish(self):
        with patch.object(settings, "no_color", False):
            result = format_trend(Trend.BEARISH)
            assert "BEARISH" in result.plain
            assert result.style == "bold red"

    def test_format_trend_sideways(self):
        with patch.object(settings, "no_color", False):
            result = format_trend(Trend.SIDEWAYS)
            assert "SIDEWAYS" in result.plain
            assert result.style == "bold yellow"

    def test_format_trend_no_color(self):
        with patch.object(settings, "no_color", True):
            result = format_trend(Trend.BULLISH)
            assert result.plain == "BULLISH"

    def test_format_price(self):
        assert format_price(1500.0) == "$1,500.00"
        assert format_price(150.0) == "$150.00"
        assert format_price(1.5) == "$1.50"
        assert format_price(0.0015) == "$0.0015"

    def test_format_pct(self):
        assert format_pct(5.0) == "+5.00%"
        assert format_pct(-3.0) == "-3.00%"
        assert format_pct(None) == "N/A"

    def test_format_ratio(self):
        assert format_ratio(1.5) == "1.50x"
        assert format_ratio(0.8) == "0.80x"
        assert format_ratio(None) == "N/A"

    def test_format_volatility(self):
        assert format_volatility(0.25) == "25.00%"
        assert format_volatility(0.15) == "15.00%"
        assert format_volatility(None) == "N/A"


class TestRenderAnalysisReport:
    def test_renders_text_output(self, capsys):
        report = create_test_report()
        with (
            patch.object(settings, "json_output", False),
            patch.object(settings, "no_color", True),
        ):
            render_analysis_report(report, json_output=False)

        captured = capsys.readouterr()
        assert "ORION PULSE" in captured.out
        assert "TEST" in captured.out
        assert "BULLISH" in captured.out
        assert "$150.00" in captured.out
        assert "145.00" in captured.out
        assert "25.00%" in captured.out
        assert "1.20x" in captured.out

    def test_renders_json_output(self, capsys):
        report = create_test_report()
        with patch.object(settings, "json_output", True):
            render_analysis_report(report, json_output=True)

        captured = capsys.readouterr()
        # Should contain JSON
        assert '"symbol": "TEST"' in captured.out
        assert '"trend": "bullish"' in captured.out


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
