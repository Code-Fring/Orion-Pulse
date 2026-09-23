"""CLI output formatting for Orion Pulse."""

from datetime import datetime

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from orion_pulse.core.models import AnalysisReport, Trend

console = Console()


def format_trend(trend: Trend, no_color: bool = False) -> Text:
    """Format trend with appropriate color."""
    if no_color:
        return Text(trend.value.upper())

    colors = {
        Trend.BULLISH: "green",
        Trend.BEARISH: "red",
        Trend.SIDEWAYS: "yellow",
    }
    return Text(trend.value.upper(), style=f"bold {colors.get(trend, 'white')}")


def format_price(price: float) -> str:
    """Format price with appropriate precision."""
    if price >= 1000 or price >= 100:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"


def format_pct(value: float | None) -> str:
    """Format percentage value."""
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def format_ratio(value: float | None) -> str:
    """Format ratio value."""
    if value is None:
        return "N/A"
    return f"{value:.2f}x"


def format_volatility(value: float | None) -> str:
    """Format volatility value."""
    if value is None:
        return "N/A"
    return f"{value:.2%}"


def render_analysis_report(
    report: AnalysisReport, json_output: bool = False, no_color: bool = False
) -> None:
    """Render analysis report to console."""
    if json_output:
        import json

        console.print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        return

    ta = report.trend_analysis

    # Header
    header = Text.assemble(
        ("ORION PULSE", "bold cyan"),
        ("  ", ""),
        (f"v{__import__('orion_pulse').__version__}", "dim"),
    )
    console.print(Panel(header, box=box.DOUBLE, border_style="cyan"))
    console.print()

    # Symbol info
    info_table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    info_table.add_column("Label", style="dim", width=16)
    info_table.add_column("Value", style="white")

    info_table.add_row("Symbol:", ta.symbol)
    info_table.add_row("Last Price:", format_price(ta.last_price))
    info_table.add_row("Trend:", format_trend(ta.trend, no_color))
    info_table.add_row("As of:", str(ta.as_of))
    info_table.add_row("Data Points:", str(report.data_points))
    info_table.add_row("Lookback:", f"{report.lookback_days} days")

    console.print(info_table)
    console.print()

    # Moving Averages
    ma_table = Table(title="Moving Averages", box=box.SIMPLE, show_header=True)
    ma_table.add_column("Period", style="cyan", justify="right")
    ma_table.add_column("Value", style="white", justify="right")
    ma_table.add_column("Price vs MA", style="white", justify="center")

    if ta.ma_20 is not None:
        signal = (
            "▲"
            if ta.price_above_ma20
            else "▼"
            if ta.price_above_ma20 is not None
            else "–"
        )
        ma_table.add_row("20-Day", format_price(ta.ma_20), signal)
    if ta.ma_50 is not None:
        signal = (
            "▲"
            if ta.price_above_ma50
            else "▼"
            if ta.price_above_ma50 is not None
            else "–"
        )
        ma_table.add_row("50-Day", format_price(ta.ma_50), signal)
    if ta.ma_200 is not None:
        signal = (
            "▲"
            if ta.price_above_ma200
            else "▼"
            if ta.price_above_ma200 is not None
            else "–"
        )
        ma_table.add_row("200-Day", format_price(ta.ma_200), signal)

    console.print(ma_table)
    console.print()

    # Indicators
    ind_table = Table(title="Indicators", box=box.SIMPLE, show_header=True)
    ind_table.add_column("Indicator", style="cyan")
    ind_table.add_column("Value", style="white", justify="right")

    ind_table.add_row("Volatility (20D)", format_volatility(ta.volatility))
    ind_table.add_row("Volume Ratio (20D)", format_ratio(ta.volume_ratio))

    console.print(ind_table)
    console.print()

    # Trend Signals
    sig_table = Table(title="Trend Signals", box=box.SIMPLE, show_header=True)
    sig_table.add_column("Signal", style="cyan")
    sig_table.add_column("Status", style="white", justify="center")

    signals = [
        ("Price > MA20", ta.price_above_ma20),
        ("Price > MA50", ta.price_above_ma50),
        ("Price > MA200", ta.price_above_ma200),
        ("MA20 > MA50", ta.ma20_above_ma50),
        ("MA50 > MA200", ta.ma50_above_ma200),
    ]

    for label, value in signals:
        if value is None:
            status = "–"
        elif value:
            status = "[green]✓[/green]" if not no_color else "YES"
        else:
            status = "[red]✗[/red]" if not no_color else "NO"
        sig_table.add_row(label, status)

    console.print(sig_table)
    console.print()

    # Footer
    footer = Text.assemble(
        ("Generated: ", "dim"),
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "dim"),
    )
    console.print(footer)
