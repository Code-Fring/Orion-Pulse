"""Unit tests for reporting module."""

from datetime import date, datetime
from unittest.mock import Mock

import pytest

from orion_pulse.backtesting import BacktestResult
from orion_pulse.core.models import AnalysisReport, Trend, TrendAnalysis
from orion_pulse.forecasting import ForecastResult
from orion_pulse.llm import MockLLMProvider
from orion_pulse.news.analysis import DirectionalBias, EventAnalysis, EventCategory
from orion_pulse.reporting import (
    ComprehensiveReport,
    ReportGenerator,
    TerminalReportRenderer,
)


@pytest.fixture
def sample_analysis_report():
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
        generated_at=datetime(2024, 1, 15, 10, 0),
        trend_analysis=ta,
        lookback_days=365,
        data_points=252,
    )


@pytest.fixture
def sample_forecast_result():
    return ForecastResult(
        symbol="TEST",
        model_name="ensemble",
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
        factors={"method": "ensemble"},
        input_features={},
    )


@pytest.fixture
def sample_backtest_result():
    return BacktestResult(
        symbol="TEST",
        model_name="ensemble",
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


@pytest.fixture
def sample_events():
    return [
        EventAnalysis(
            article_id="1",
            symbol="TEST",
            event_category=EventCategory.EARNINGS,
            headline="Earnings beat",
            summary="Beat estimates",
            source="Test",
            published_at=datetime(2024, 1, 15),
            directional_bias=DirectionalBias.POSITIVE,
            confidence=0.8,
            relevance_score=0.9,
            key_factors=["earnings"],
            raw_article=Mock(),
        ),
    ]


class TestComprehensiveReport:
    def test_create_report(self, sample_analysis_report, sample_forecast_result):
        report = ComprehensiveReport(
            symbol="TEST",
            generated_at=datetime(2024, 1, 15, 10, 0),
            analysis_report=sample_analysis_report,
            forecast_results={"ensemble": sample_forecast_result},
        )

        assert report.symbol == "TEST"
        assert report.analysis_report is not None
        assert "ensemble" in report.forecast_results

    def test_to_dict(self, sample_analysis_report, sample_forecast_result):
        report = ComprehensiveReport(
            symbol="TEST",
            generated_at=datetime(2024, 1, 15, 10, 0),
            analysis_report=sample_analysis_report,
            forecast_results={"ensemble": sample_forecast_result},
        )

        d = report.to_dict()
        assert d["symbol"] == "TEST"
        assert d["analysis_report"] is not None
        assert d["forecast_results"]["ensemble"]["model_name"] == "ensemble"


class TestReportGenerator:
    def test_generate_report_basic(
        self, sample_analysis_report, sample_forecast_result, sample_backtest_result
    ):
        llm = MockLLMProvider()
        generator = ReportGenerator(llm_provider_name="mock")

        report = generator.generate_report(
            symbol="TEST",
            analysis_report=sample_analysis_report,
            forecast_results={"ensemble": sample_forecast_result},
            backtest_result=sample_backtest_result,
            include_news=False,
            use_llm=False,
        )

        assert isinstance(report, ComprehensiveReport)
        assert report.symbol == "TEST"
        assert report.analysis_report == sample_analysis_report
        assert report.backtest_result == sample_backtest_result
        assert report.llm_summary is None

    def test_generate_report_with_llm(
        self, sample_analysis_report, sample_forecast_result
    ):
        llm = MockLLMProvider()
        generator = ReportGenerator(llm_provider_name="mock")

        report = generator.generate_report(
            symbol="TEST",
            analysis_report=sample_analysis_report,
            forecast_results={"ensemble": sample_forecast_result},
            include_news=False,
            use_llm=True,
        )

        assert report.llm_summary is not None
        assert "MOCK" in report.llm_summary

    def test_generate_report_with_news(
        self, sample_analysis_report, sample_forecast_result, sample_events
    ):
        llm = MockLLMProvider()
        news_service = Mock()
        news_service.fetch_and_analyze.return_value = sample_events

        generator = ReportGenerator(llm_provider_name="mock", news_service=news_service)

        report = generator.generate_report(
            symbol="TEST",
            analysis_report=sample_analysis_report,
            forecast_results={"ensemble": sample_forecast_result},
            include_news=True,
            news_days=7,
            use_llm=False,
        )

        assert len(report.news_events) == 1
        assert report.news_events[0].event_category == EventCategory.EARNINGS
        news_service.fetch_and_analyze.assert_called_once()


class TestTerminalReportRenderer:
    def test_render_analysis(self, sample_analysis_report, capsys):
        renderer = TerminalReportRenderer(no_color=True)
        renderer._render_analysis(sample_analysis_report)

        captured = capsys.readouterr()
        assert "QUANTITATIVE ANALYSIS" in captured.out
        assert "BULLISH" in captured.out
        assert "150.00" in captured.out

    def test_render_forecasts(self, sample_forecast_result, capsys):
        renderer = TerminalReportRenderer(no_color=True)
        renderer._render_forecasts({"ensemble": sample_forecast_result})

        captured = capsys.readouterr()
        assert "PROBABILISTIC FORECASTS" in captured.out
        assert "ensemble" in captured.out
        assert "2.00%" in captured.out  # expected_return

    def test_render_backtest(self, sample_backtest_result, capsys):
        renderer = TerminalReportRenderer(no_color=True)
        renderer._render_backtest(sample_backtest_result)

        captured = capsys.readouterr()
        assert "BACKTEST PERFORMANCE" in captured.out
        assert "55.0%" in captured.out  # directional_accuracy
        assert "1.33" in captured.out  # sharpe_ratio

    def test_render_news_events(self, sample_events, capsys):
        renderer = TerminalReportRenderer(no_color=True)
        renderer._render_news_events(sample_events)

        captured = capsys.readouterr()
        assert "RECENT EVENTS" in captured.out
        assert "Earnings beat" in captured.out
        assert "earnings" in captured.out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
