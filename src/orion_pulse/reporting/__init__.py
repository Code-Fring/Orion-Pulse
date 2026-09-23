"""Reporting module for Orion Pulse."""

import contextlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from orion_pulse.backtesting import BacktestResult
from orion_pulse.core.models import AnalysisReport
from orion_pulse.forecasting import ForecastResult
from orion_pulse.llm import LLMMessage, LLMProviderFactory
from orion_pulse.news.analysis import EventAnalysis, NewsAnalysisService


@dataclass(frozen=True)
class ComprehensiveReport:
    """Complete comprehensive report."""

    symbol: str
    generated_at: datetime
    analysis_report: AnalysisReport | None = None
    forecast_results: dict[str, ForecastResult] = field(default_factory=dict)
    backtest_result: BacktestResult | None = None
    news_events: list[EventAnalysis] = field(default_factory=list)
    llm_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "generated_at": self.generated_at.isoformat(),
            "analysis_report": self.analysis_report.model_dump(mode="json")
            if self.analysis_report
            else None,
            "forecast_results": {
                k: v.to_dict() for k, v in self.forecast_results.items()
            },
            "backtest_result": self.backtest_result.to_dict()
            if self.backtest_result
            else None,
            "news_events": [e.to_dict() for e in self.news_events],
            "llm_summary": self.llm_summary,
        }


class ReportingError(Exception):
    """Exception raised during report generation."""

    pass


class ReportGenerator:
    """Generate comprehensive reports combining analysis, forecasts, and news."""

    def __init__(
        self,
        llm_provider_name: str = "mock",
        news_service: NewsAnalysisService | None = None,
    ):
        self.llm = LLMProviderFactory.get_provider(llm_provider_name)
        self.news_service = news_service

    def generate_report(
        self,
        symbol: str,
        analysis_report: AnalysisReport | None = None,
        forecast_results: dict[str, ForecastResult] | None = None,
        backtest_result: BacktestResult | None = None,
        include_news: bool = True,
        news_days: int = 7,
        use_llm: bool = True,
    ) -> ComprehensiveReport:
        """Generate comprehensive report."""
        news_events = []

        if include_news and self.news_service:
            with contextlib.suppress(Exception):
                news_events = self.news_service.fetch_and_analyze(
                    symbol=symbol,
                    days_back=news_days,
                    limit=20,
                    save_to_db=True,
                )

        report = ComprehensiveReport(
            symbol=symbol,
            generated_at=datetime.now(),
            analysis_report=analysis_report,
            forecast_results=forecast_results or {},
            backtest_result=backtest_result,
            news_events=news_events,
        )

        if use_llm and self.llm.is_available():
            with contextlib.suppress(Exception):
                report = self._add_llm_summary(report)

        return report

    def _add_llm_summary(self, report: ComprehensiveReport) -> ComprehensiveReport:
        """Add LLM-generated summary to report."""
        # Build context for LLM
        context = self._build_context(report)

        messages = [
            LLMMessage(
                role="system",
                content="""You are a financial analyst writing a concise market intelligence report.
Focus on:
1. Key quantitative findings (trend, volatility, momentum)
2. Forecast probabilities and uncertainty
3. Recent news/events and their potential impact
4. Backtest performance of the forecasting model
5. Clear distinction between observed data and model-generated interpretations

Use professional, objective tone. Do not give financial advice. Always emphasize uncertainty and probability.""",
            ),
            LLMMessage(
                role="user",
                content=context,
            ),
        ]

        response = self.llm.complete(messages, temperature=0.3, max_tokens=1500)

        return ComprehensiveReport(
            symbol=report.symbol,
            generated_at=report.generated_at,
            analysis_report=report.analysis_report,
            forecast_results=report.forecast_results,
            backtest_result=report.backtest_result,
            news_events=report.news_events,
            llm_summary=response.content,
        )

    def _build_context(self, report: ComprehensiveReport) -> str:
        """Build context string for LLM."""
        parts = [f"ORION PULSE REPORT FOR {report.symbol}"]
        parts.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
        parts.append("")

        # Analysis section
        if report.analysis_report:
            ta = report.analysis_report.trend_analysis
            parts.append("=== QUANTITATIVE ANALYSIS ===")
            parts.append(f"Trend: {ta.trend.value.upper()}")
            parts.append(f"Last Price: ${ta.last_price:.2f}")
            parts.append(f"Data Points: {report.analysis_report.data_points}")
            parts.append(f"Lookback: {report.analysis_report.lookback_days} days")
            parts.append("")

            if ta.ma_20:
                parts.append(
                    f"MA20: ${ta.ma_20:.2f} (Price {'above' if ta.price_above_ma20 else 'below'})"
                )
            if ta.ma_50:
                parts.append(
                    f"MA50: ${ta.ma_50:.2f} (Price {'above' if ta.price_above_ma50 else 'below'})"
                )
            if ta.ma_200:
                parts.append(
                    f"MA200: ${ta.ma_200:.2f} (Price {'above' if ta.price_above_ma200 else 'below'})"
                )

            if ta.volatility:
                parts.append(f"Volatility (20D): {ta.volatility:.2%}")
            if ta.volume_ratio:
                parts.append(f"Volume Ratio: {ta.volume_ratio:.2f}x")

            parts.append("")

        # Forecast section
        if report.forecast_results:
            parts.append("=== FORECASTS ===")
            for model_name, forecast in report.forecast_results.items():
                parts.append(f"Model: {model_name}")
                parts.append(f"  Horizon: {forecast.horizon_days} days")
                parts.append(f"  Expected Return: {forecast.expected_return:.2%}")
                parts.append(f"  Prob Positive: {forecast.prob_positive:.1%}")
                parts.append(
                    f"  Range (P10-P90): {forecast.p10:.2%} to {forecast.p90:.2%}"
                )
                parts.append(f"  Confidence: {forecast.confidence:.1%}")
                parts.append("")

        # Backtest section
        if report.backtest_result:
            bt = report.backtest_result
            parts.append("=== BACKTEST PERFORMANCE ===")
            parts.append(f"Model: {bt.model_name}")
            parts.append(f"Period: {bt.start_date} to {bt.end_date}")
            parts.append(f"Predictions: {bt.total_predictions}")
            if bt.directional_accuracy is not None:
                parts.append(f"Directional Accuracy: {bt.directional_accuracy:.1%}")
            if bt.mae is not None:
                parts.append(f"MAE: {bt.mae:.4f}")
            if bt.rmse is not None:
                parts.append(f"RMSE: {bt.rmse:.4f}")
            if bt.calibration_error is not None:
                parts.append(f"Calibration Error: {bt.calibration_error:.4f}")
            if bt.cumulative_return is not None:
                parts.append(f"Cumulative Return: {bt.cumulative_return:.2%}")
            if bt.sharpe_ratio is not None:
                parts.append(f"Sharpe Ratio: {bt.sharpe_ratio:.2f}")
            if bt.max_drawdown is not None:
                parts.append(f"Max Drawdown: {bt.max_drawdown:.2%}")
            parts.append("")

        # News section
        if report.news_events:
            parts.append("=== RECENT EVENTS ===")
            for event in report.news_events[:5]:
                parts.append(
                    f"- {event.published_at.strftime('%Y-%m-%d')}: {event.headline}"
                )
                parts.append(
                    f"  Category: {event.event_category.value}, Bias: {event.directional_bias.value}"
                )
                parts.append(
                    f"  Confidence: {event.confidence:.1%}, Relevance: {event.relevance_score:.1%}"
                )
                parts.append("")

        parts.append("=== INSTRUCTIONS ===")
        parts.append(
            "Generate a concise, professional report synthesizing the above information."
        )
        parts.append("Clearly separate observed facts from model interpretations.")
        parts.append("Emphasize uncertainty and probabilistic nature of forecasts.")

        return "\n".join(parts)


class TerminalReportRenderer:
    """Render reports to terminal with Rich formatting."""

    def __init__(self, no_color: bool = False):
        self.no_color = no_color
        from rich.console import Console

        self.console = Console()

    def render(self, report: ComprehensiveReport) -> None:
        """Render comprehensive report to terminal."""
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        # Header
        header = Text.assemble(
            ("ORION PULSE", "bold cyan"),
            ("  ", ""),
            ("COMPREHENSIVE REPORT", "bold white"),
        )
        self.console.print(Panel(header, box=box.DOUBLE, border_style="cyan"))
        self.console.print()

        # Symbol info
        info_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
        info_table.add_column("Label", style="dim", width=20)
        info_table.add_column("Value", style="white")

        info_table.add_row("Symbol:", report.symbol)
        info_table.add_row(
            "Generated:", report.generated_at.strftime("%Y-%m-%d %H:%M:%S")
        )

        self.console.print(info_table)
        self.console.print()

        # Analysis
        if report.analysis_report:
            self._render_analysis(report.analysis_report)

        # Forecasts
        if report.forecast_results:
            self._render_forecasts(report.forecast_results)

        # Backtest
        if report.backtest_result:
            self._render_backtest(report.backtest_result)

        # News events
        if report.news_events:
            self._render_news_events(report.news_events)

        # LLM Summary
        if report.llm_summary:
            self._render_llm_summary(report.llm_summary)

    def _render_analysis(self, report: AnalysisReport) -> None:
        from rich import box
        from rich.table import Table

        ta = report.trend_analysis

        self.console.print("[bold cyan]QUANTITATIVE ANALYSIS[/bold cyan]")
        self.console.print()

        # Trend
        trend_style = {"bullish": "green", "bearish": "red", "sideways": "yellow"}.get(
            ta.trend.value, "white"
        )
        self.console.print(
            f"Trend: [{trend_style}]{ta.trend.value.upper()}[/{trend_style}]"
        )
        self.console.print(f"Last Price: ${ta.last_price:,.2f}")
        self.console.print(f"As of: {ta.as_of}")
        self.console.print()

        # Moving averages
        ma_table = Table(title="Moving Averages", box=box.SIMPLE)
        ma_table.add_column("Period", style="cyan")
        ma_table.add_column("Value", style="white", justify="right")
        ma_table.add_column("Price vs MA", justify="center")

        for period, value, signal in [
            (20, ta.ma_20, ta.price_above_ma20),
            (50, ta.ma_50, ta.price_above_ma50),
            (200, ta.ma_200, ta.price_above_ma200),
        ]:
            if value is not None:
                sig = "▲" if signal else "▼" if signal is not None else "–"
                ma_table.add_row(f"{period}-Day", f"${value:,.2f}", sig)

        self.console.print(ma_table)
        self.console.print()

        # Indicators
        ind_table = Table(title="Indicators", box=box.SIMPLE)
        ind_table.add_column("Indicator", style="cyan")
        ind_table.add_column("Value", style="white", justify="right")

        if ta.volatility is not None:
            ind_table.add_row("Volatility (20D)", f"{ta.volatility:.2%}")
        if ta.volume_ratio is not None:
            ind_table.add_row("Volume Ratio (20D)", f"{ta.volume_ratio:.2f}x")

        self.console.print(ind_table)
        self.console.print()

    def _render_forecasts(self, forecasts: dict[str, ForecastResult]) -> None:
        from rich import box
        from rich.table import Table

        self.console.print("[bold cyan]PROBABILISTIC FORECASTS[/bold cyan]")
        self.console.print()

        for model_name, fc in forecasts.items():
            self.console.print(f"[bold]Model: {model_name}[/bold]")

            fc_table = Table(box=box.SIMPLE)
            fc_table.add_column("Metric", style="cyan")
            fc_table.add_column("Value", style="white", justify="right")

            fc_table.add_row("Horizon", f"{fc.horizon_days} days")
            fc_table.add_row("Expected Return", f"{fc.expected_return:.2%}")
            fc_table.add_row("Prob Positive", f"{fc.prob_positive:.1%}")
            fc_table.add_row("P10 (Worst)", f"{fc.p10:.2%}")
            fc_table.add_row("P25", f"{fc.p25:.2%}")
            fc_table.add_row("P50 (Median)", f"{fc.p50:.2%}")
            fc_table.add_row("P75", f"{fc.p75:.2%}")
            fc_table.add_row("P90 (Best)", f"{fc.p90:.2%}")
            fc_table.add_row("Volatility", f"{fc.volatility:.2%}")
            fc_table.add_row("Confidence", f"{fc.confidence:.1%}")

            self.console.print(fc_table)
            self.console.print()

    def _render_backtest(self, bt: BacktestResult) -> None:
        from rich import box
        from rich.table import Table

        self.console.print("[bold cyan]BACKTEST PERFORMANCE[/bold cyan]")
        self.console.print()

        bt_table = Table(title=f"Backtest: {bt.model_name}", box=box.SIMPLE)
        bt_table.add_column("Metric", style="cyan")
        bt_table.add_column("Value", style="white", justify="right")

        bt_table.add_row("Period", f"{bt.start_date} to {bt.end_date}")
        bt_table.add_row("Total Predictions", str(bt.total_predictions))

        if bt.directional_accuracy is not None:
            bt_table.add_row("Directional Accuracy", f"{bt.directional_accuracy:.1%}")
        if bt.mae is not None:
            bt_table.add_row("MAE", f"{bt.mae:.4f}")
        if bt.rmse is not None:
            bt_table.add_row("RMSE", f"{bt.rmse:.4f}")
        if bt.calibration_error is not None:
            bt_table.add_row("Calibration Error", f"{bt.calibration_error:.4f}")
        if bt.cumulative_return is not None:
            bt_table.add_row("Cumulative Return", f"{bt.cumulative_return:.2%}")
        if bt.annualized_return is not None:
            bt_table.add_row("Annualized Return", f"{bt.annualized_return:.2%}")
        if bt.volatility is not None:
            bt_table.add_row("Volatility", f"{bt.volatility:.2%}")
        if bt.max_drawdown is not None:
            bt_table.add_row("Max Drawdown", f"{bt.max_drawdown:.2%}")
        if bt.sharpe_ratio is not None:
            bt_table.add_row("Sharpe Ratio", f"{bt.sharpe_ratio:.2f}")
        if bt.sortino_ratio is not None:
            bt_table.add_row("Sortino Ratio", f"{bt.sortino_ratio:.2f}")

        self.console.print(bt_table)
        self.console.print()

    def _render_news_events(self, events: list) -> None:
        from rich import box
        from rich.table import Table

        self.console.print("[bold cyan]RECENT EVENTS & NEWS[/bold cyan]")
        self.console.print()

        news_table = Table(title="Events", box=box.SIMPLE)
        news_table.add_column("Date", style="cyan", width=12)
        news_table.add_column("Headline", style="white", width=60)
        news_table.add_column("Category", style="yellow", width=18)
        news_table.add_column("Bias", style="white", width=10, justify="center")
        news_table.add_column("Conf", style="white", width=6, justify="right")

        for event in events[:10]:
            bias_color = {
                "positive": "green",
                "negative": "red",
                "neutral": "yellow",
                "unknown": "white",
            }.get(event.directional_bias.value, "white")

            news_table.add_row(
                event.published_at.strftime("%Y-%m-%d"),
                event.headline[:58] + "..."
                if len(event.headline) > 58
                else event.headline,
                event.event_category.value,
                f"[{bias_color}]{event.directional_bias.value[0].upper()}[/{bias_color}]",
                f"{event.confidence:.0%}",
            )

        self.console.print(news_table)
        self.console.print()

    def _render_llm_summary(self, summary: str) -> None:
        from rich import box
        from rich.panel import Panel

        self.console.print("[bold cyan]AI-GENERATED SYNTHESIS[/bold cyan]")
        self.console.print()
        self.console.print(Panel(summary, box=box.ROUNDED, border_style="dim cyan"))
        self.console.print()
