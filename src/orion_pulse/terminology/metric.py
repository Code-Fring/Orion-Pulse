"""Terminology system for Orion Pulse providing dual-label metric definitions."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    """Definition of a metric with technical and human-friendly names."""

    code_name: str
    technical_name: str
    common_name: str
    explanation: str
    unit: str = ""
    source_type: str = "calculated"  # market_data, calculated, model, news, config
    category: str = "general"
    aliases: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code_name": self.code_name,
            "technical_name": self.technical_name,
            "common_name": self.common_name,
            "explanation": self.explanation,
            "unit": self.unit,
            "source_type": self.source_type,
            "category": self.category,
            "aliases": self.aliases,
        }


class MetricRegistry:
    """Centralized registry of all metric definitions."""

    def __init__(self) -> None:
        self._metrics: dict[str, MetricDefinition] = {}
        self._aliases: dict[str, str] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register all default metric definitions."""
        definitions = [
            # Moving Averages
            MetricDefinition(
                code_name="ma_20",
                technical_name="MA20",
                common_name="20-Day Moving Average",
                explanation="Average closing price over the last 20 trading days",
                unit="USD",
                source_type="calculated",
                category="moving_average",
                aliases=["ma20", "20d_ma", "20_day_ma"],
            ),
            MetricDefinition(
                code_name="ma_50",
                technical_name="MA50",
                common_name="50-Day Moving Average",
                explanation="Average closing price over the last 50 trading days",
                unit="USD",
                source_type="calculated",
                category="moving_average",
                aliases=["ma50", "50d_ma", "50_day_ma"],
            ),
            MetricDefinition(
                code_name="ma_200",
                technical_name="MA200",
                common_name="200-Day Moving Average",
                explanation="Long-term average price over the last 200 trading days",
                unit="USD",
                source_type="calculated",
                category="moving_average",
                aliases=["ma200", "200d_ma", "200_day_ma"],
            ),
            # Price vs MA signals
            MetricDefinition(
                code_name="price_above_ma20",
                technical_name="Price > MA20",
                common_name="Price Above 20-Day Average",
                explanation="Whether the current price is above the 20-day moving average",
                unit="",
                source_type="calculated",
                category="trend_signal",
                aliases=["price_vs_ma20"],
            ),
            MetricDefinition(
                code_name="price_above_ma50",
                technical_name="Price > MA50",
                common_name="Price Above 50-Day Average",
                explanation="Whether the current price is above the 50-day moving average",
                unit="",
                source_type="calculated",
                category="trend_signal",
                aliases=["price_vs_ma50"],
            ),
            MetricDefinition(
                code_name="price_above_ma200",
                technical_name="Price > MA200",
                common_name="Price Above 200-Day Average",
                explanation="Whether the current price is above the 200-day moving average",
                unit="",
                source_type="calculated",
                category="trend_signal",
                aliases=["price_vs_ma200"],
            ),
            MetricDefinition(
                code_name="ma20_above_ma50",
                technical_name="MA20 > MA50",
                common_name="Short-Term Average Above Medium-Term",
                explanation="Whether the 20-day average is above the 50-day average (short-term momentum)",
                unit="",
                source_type="calculated",
                category="trend_signal",
                aliases=["ma20_vs_ma50"],
            ),
            MetricDefinition(
                code_name="ma50_above_ma200",
                technical_name="MA50 > MA200",
                common_name="Medium-Term Average Above Long-Term",
                explanation="Whether the 50-day average is above the 200-day average (long-term trend)",
                unit="",
                source_type="calculated",
                category="trend_signal",
                aliases=["ma50_vs_ma200"],
            ),
            # Volatility
            MetricDefinition(
                code_name="volatility_20",
                technical_name="Volatility (20D)",
                common_name="Price Volatility",
                explanation="How much the price has been moving around (annualized standard deviation of daily returns over 20 days)",
                unit="%",
                source_type="calculated",
                category="volatility",
                aliases=["volatility", "vol_20", "annualized_volatility"],
            ),
            # Volume
            MetricDefinition(
                code_name="volume_ratio_20",
                technical_name="Volume Ratio (20D)",
                common_name="Trading Activity",
                explanation="Today's trading volume compared with its recent 20-day average",
                unit="x",
                source_type="calculated",
                category="volume",
                aliases=["volume_ratio", "vol_ratio", "relative_volume"],
            ),
            # Trend
            MetricDefinition(
                code_name="trend",
                technical_name="Trend",
                common_name="Market Trend",
                explanation="Overall trend classification based on price vs moving averages",
                unit="",
                source_type="calculated",
                category="trend",
                aliases=["market_trend", "trend_classification"],
            ),
            # Forecast metrics
            MetricDefinition(
                code_name="expected_return",
                technical_name="Expected Return",
                common_name="Model-Expected Change",
                explanation="The model's estimated average price change over the forecast horizon",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["expected_change", "mean_forecast"],
            ),
            MetricDefinition(
                code_name="prob_positive",
                technical_name="Probability Positive",
                common_name="Chance of Positive Return",
                explanation="Estimated probability that the return will be above zero over the forecast horizon",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["prob_up", "positive_probability"],
            ),
            MetricDefinition(
                code_name="p10",
                technical_name="P10",
                common_name="10th Percentile Outcome",
                explanation="A lower-end scenario: 10% chance the return will be below this level",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["percentile_10", "worst_case"],
            ),
            MetricDefinition(
                code_name="p25",
                technical_name="P25",
                common_name="25th Percentile Outcome",
                explanation="A below-average scenario: 25% chance the return will be below this level",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["percentile_25", "below_average"],
            ),
            MetricDefinition(
                code_name="p50",
                technical_name="P50 / Median",
                common_name="Median Outcome",
                explanation="The middle scenario: 50% chance the return will be below, 50% above this level",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["percentile_50", "median", "middle_scenario"],
            ),
            MetricDefinition(
                code_name="p75",
                technical_name="P75",
                common_name="75th Percentile Outcome",
                explanation="An above-average scenario: 75% chance the return will be below this level",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["percentile_75", "above_average"],
            ),
            MetricDefinition(
                code_name="p90",
                technical_name="P90",
                common_name="90th Percentile Outcome",
                explanation="A higher-end scenario: 90% chance the return will be below this level",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["percentile_90", "best_case"],
            ),
            MetricDefinition(
                code_name="forecast_volatility",
                technical_name="Forecast Volatility",
                common_name="Forecast Uncertainty",
                explanation="Model's estimated volatility of returns over the forecast horizon",
                unit="%",
                source_type="model",
                category="forecast",
                aliases=["horizon_volatility", "forecast_uncertainty"],
            ),
            MetricDefinition(
                code_name="confidence",
                technical_name="Confidence",
                common_name="Model Confidence",
                explanation="How confident the model is in its forecast (0-1 scale)",
                unit="",
                source_type="model",
                category="forecast",
                aliases=["model_confidence", "forecast_confidence"],
            ),
            # Backtest metrics
            MetricDefinition(
                code_name="directional_accuracy",
                technical_name="Directional Accuracy",
                common_name="Direction Accuracy",
                explanation="Percentage of forecasts that correctly predicted the direction (up/down) of price movement",
                unit="%",
                source_type="calculated",
                category="backtest",
                aliases=["direction_accuracy", "hit_rate"],
            ),
            MetricDefinition(
                code_name="mae",
                technical_name="MAE",
                common_name="Mean Absolute Error",
                explanation="Average absolute difference between predicted and actual returns",
                unit="",
                source_type="calculated",
                category="backtest",
                aliases=["mean_absolute_error"],
            ),
            MetricDefinition(
                code_name="rmse",
                technical_name="RMSE",
                common_name="Root Mean Square Error",
                explanation="Square root of average squared errors - penalizes larger errors more",
                unit="",
                source_type="calculated",
                category="backtest",
                aliases=["root_mean_square_error"],
            ),
            MetricDefinition(
                code_name="calibration_error",
                technical_name="Calibration Error",
                common_name="Probability Calibration Error",
                explanation="How well forecast probabilities match actual outcomes (lower is better)",
                unit="",
                source_type="calculated",
                category="backtest",
                aliases=["calibration"],
            ),
            MetricDefinition(
                code_name="cumulative_return",
                technical_name="Cumulative Return",
                common_name="Total Return",
                explanation="Total compounded return over the backtest period",
                unit="%",
                source_type="calculated",
                category="backtest",
                aliases=["total_return"],
            ),
            MetricDefinition(
                code_name="annualized_return",
                technical_name="Annualized Return",
                common_name="Annualized Growth Rate",
                explanation="Average annualized growth rate (Compound Annual Growth Rate equivalent)",
                unit="%",
                source_type="calculated",
                category="backtest",
                aliases=["cagr", "annual_return"],
            ),
            MetricDefinition(
                code_name="max_drawdown",
                technical_name="Max Drawdown",
                common_name="Maximum Decline",
                explanation="Largest peak-to-trough decline during the backtest period",
                unit="%",
                source_type="calculated",
                category="backtest",
                aliases=["maximum_drawdown", "worst_decline"],
            ),
            MetricDefinition(
                code_name="sharpe_ratio",
                technical_name="Sharpe Ratio",
                common_name="Risk-Adjusted Return",
                explanation="Return per unit of risk (higher is better, >1 is good)",
                unit="",
                source_type="calculated",
                category="backtest",
                aliases=["sharpe"],
            ),
            MetricDefinition(
                code_name="sortino_ratio",
                technical_name="Sortino Ratio",
                common_name="Downside Risk-Adjusted Return",
                explanation="Return per unit of downside risk (only penalizes negative volatility)",
                unit="",
                source_type="calculated",
                category="backtest",
                aliases=["sortino"],
            ),
            # Market data
            MetricDefinition(
                code_name="last_price",
                technical_name="Last Price",
                common_name="Current Market Price",
                explanation="Most recent available closing price from the data provider",
                unit="USD",
                source_type="market_data",
                category="price",
                aliases=["close", "current_price", "latest_price"],
            ),
            MetricDefinition(
                code_name="data_points",
                technical_name="Data Points",
                common_name="Historical Data Points",
                explanation="Number of trading days of historical data used in analysis",
                unit="days",
                source_type="market_data",
                category="data_quality",
                aliases=["sample_size", "observations"],
            ),
            MetricDefinition(
                code_name="lookback_days",
                technical_name="Lookback Period",
                common_name="Analysis Window",
                explanation="Number of days of historical data included in the analysis",
                unit="days",
                source_type="config",
                category="config",
                aliases=["lookback", "window"],
            ),
        ]

        for metric in definitions:
            self.register(metric)

    def register(self, metric: MetricDefinition) -> None:
        """Register a metric definition."""
        self._metrics[metric.code_name] = metric
        # Register aliases
        for alias in metric.aliases:
            self._aliases[alias.lower()] = metric.code_name
        # Also register technical name and common name as aliases
        self._aliases[metric.technical_name.lower()] = metric.code_name
        self._aliases[metric.common_name.lower()] = metric.code_name

    def get(self, key: str) -> MetricDefinition | None:
        """Get metric by code name, alias, technical name, or common name."""
        key_lower = key.lower()
        # Direct lookup
        if key in self._metrics:
            return self._metrics[key]
        # Alias lookup
        if key_lower in self._aliases:
            return self._metrics[self._aliases[key_lower]]
        return None

    def get_by_category(self, category: str) -> list[MetricDefinition]:
        """Get all metrics in a category."""
        return [m for m in self._metrics.values() if m.category == category]

    def get_by_source_type(self, source_type: str) -> list[MetricDefinition]:
        """Get all metrics by source type."""
        return [m for m in self._metrics.values() if m.source_type == source_type]

    def all(self) -> list[MetricDefinition]:
        """Get all registered metrics."""
        return list(self._metrics.values())

    def format_metric(
        self,
        code_name: str,
        value: Any,
        show_explanation: bool = True,
        no_color: bool = False,
    ) -> str:
        """Format a metric value with dual labels."""
        metric = self.get(code_name)
        if not metric:
            return f"{code_name}: {value}"

        lines = []
        # Technical name
        lines.append(f"{metric.technical_name}")
        # Common name
        lines.append(f"  {metric.common_name}")
        if show_explanation:
            lines.append(f"  {metric.explanation}")
        # Value with unit
        if metric.unit:
            if metric.unit == "%":
                if isinstance(value, (int, float)):
                    lines.append(f"  Value: {value:+.2f}%")
                else:
                    lines.append(f"  Value: {value}%")
            elif metric.unit == "x":
                lines.append(f"  Value: {value:.2f}x")
            elif metric.unit == "USD":
                lines.append(f"  Value: ${value:,.2f}")
            else:
                lines.append(f"  Value: {value} {metric.unit}")
        else:
            lines.append(f"  Value: {value}")
        # Source type badge
        source_badge = {
            "market_data": "[MARKET DATA]",
            "calculated": "[CALCULATED]",
            "model": "[MODEL]",
            "news": "[NEWS]",
            "config": "[CONFIG]",
        }.get(metric.source_type, f"[{metric.source_type.upper()}]")
        lines.append(f"  Source: {source_badge}")

        return "\n".join(lines)


# Global registry instance
_registry: MetricRegistry | None = None


def get_registry() -> MetricRegistry:
    """Get global metric registry instance."""
    global _registry
    if _registry is None:
        _registry = MetricRegistry()
    return _registry


def get_metric(code_name: str) -> MetricDefinition | None:
    """Convenience function to get a metric definition."""
    return get_registry().get(code_name)


def format_metric(
    code_name: str,
    value: Any,
    show_explanation: bool = True,
    no_color: bool = False,
) -> str:
    """Convenience function to format a metric."""
    return get_registry().format_metric(code_name, value, show_explanation, no_color)
