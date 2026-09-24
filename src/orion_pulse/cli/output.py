"""CLI output formatting for Orion Pulse."""

from datetime import datetime

from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from orion_pulse.core.models import AnalysisReport, Trend

console = Console()


ORION_BANNER = r"""
    ██████╗ ██████╗ ██████╗ ██████╗ ███████╗███████╗████████╗
    ██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝╚══██╔══╝
    ██████╔╝██████╔╝██████╔╝██████╔╝█████╗  ███████╗   ██║
    ██╔═══╝ ██╔══██╗██╔══██╗██╔══██╗██╔══╝  ╚════██║   ██║
    ██║     ██║  ██║██║  ██║██║  ██║███████╗███████║   ██║
    ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚══════╝   ╚═╝
"""

ORION_COMPACT = "    ██████╗ ██████╗ ██████╗ ██████╗ ███████╗███████╗████████╗"


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


def print_banner(compact: bool = False, version: str = "0.1.0") -> None:
    """Print the Orion Pulse banner."""
    if compact:
        text = Text(ORION_COMPACT, style="bold cyan")
        text.append(f"  v{version}", style="dim")
        console.print(Align.center(text))
    else:
        text = Text(ORION_BANNER, style="bold cyan")
        text.append(f"    v{version}", style="dim")
        console.print(Align.center(text))


def print_section(title: str, icon: str = "◆") -> None:
    """Print a section header."""
    console.print()
    console.print(
        Text.assemble(
            (f"  {icon} ", "bold cyan"),
            (title, "bold white"),
        )
    )
    console.print()


def render_analysis_report(
    report: AnalysisReport, json_output: bool = False, no_color: bool = False
) -> None:
    """Render analysis report to console."""
    if json_output:
        import json

        console.print(json.dumps(report.model_dump(mode="json"), indent=2, default=str))
        return

    ta = report.trend_analysis

    # Compact banner for analysis
    print_banner(compact=True)

    # Symbol header panel
    header_text = Text.assemble(
        (ta.symbol, "bold white"),
        ("  ", ""),
        (format_price(ta.last_price), "bold cyan"),
        ("  ", ""),
        format_trend(ta.trend, no_color),
    )
    console.print(
        Panel(header_text, box=box.ROUNDED, border_style="cyan", padding=(0, 1))
    )
    console.print()

    # Symbol info
    info_table = Table(
        box=box.SIMPLE, show_header=False, pad_edge=False, collapse_padding=True
    )
    info_table.add_column("Label", style="dim", width=14)
    info_table.add_column("Value", style="white")

    info_table.add_row("As of:", str(ta.as_of))
    info_table.add_row("Data Points:", str(report.data_points))
    info_table.add_row("Lookback:", f"{report.lookback_days} days")

    console.print(info_table)
    console.print()

    # Moving Averages
    print_section("Moving Averages", "📈")
    ma_table = Table(box=box.SIMPLE, show_header=True, collapse_padding=True)
    ma_table.add_column("Period", style="cyan", justify="right")
    ma_table.add_column("Value", style="white", justify="right")
    ma_table.add_column("vs Price", style="white", justify="center")

    for period, value, above in [
        (20, ta.ma_20, ta.price_above_ma20),
        (50, ta.ma_50, ta.price_above_ma50),
        (200, ta.ma_200, ta.price_above_ma200),
    ]:
        if value is not None:
            signal = "▲" if above else "▼" if above is not None else "–"
            signal_style = "green" if above else "red" if above is not None else "dim"
            ma_table.add_row(
                f"{period}-Day",
                format_price(value),
                Text(signal, style=signal_style),
            )

    console.print(ma_table)
    console.print()

    # Indicators
    print_section("Indicators", "📊")
    ind_table = Table(box=box.SIMPLE, show_header=True, collapse_padding=True)
    ind_table.add_column("Indicator", style="cyan")
    ind_table.add_column("Value", style="white", justify="right")

    ind_table.add_row("Volatility (20D)", format_volatility(ta.volatility))
    ind_table.add_row("Volume Ratio (20D)", format_ratio(ta.volume_ratio))

    console.print(ind_table)
    console.print()

    # Trend Signals
    print_section("Trend Signals", "🎯")
    sig_table = Table(box=box.SIMPLE, show_header=True, collapse_padding=True)
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
            status = Text("–", style="dim")
        elif value:
            status = Text("✓", style="green")
        else:
            status = Text("✗", style="red")
        sig_table.add_row(label, status)

    console.print(sig_table)
    console.print()

    # Footer
    footer = Text.assemble(
        ("Generated: ", "dim"),
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "dim"),
    )
    console.print(Align.right(footer))


def print_error(message: str) -> None:
    """Print error message."""
    console.print(Text.assemble(("✗ ", "bold red"), (message, "red")))


def print_success(message: str) -> None:
    """Print success message."""
    console.print(Text.assemble(("✓ ", "bold green"), (message, "green")))


def print_warning(message: str) -> None:
    """Print warning message."""
    console.print(Text.assemble(("⚠ ", "bold yellow"), (message, "yellow")))


def print_info(message: str) -> None:
    """Print info message."""
    console.print(Text.assemble(("ℹ ", "bold cyan"), (message, "cyan")))
