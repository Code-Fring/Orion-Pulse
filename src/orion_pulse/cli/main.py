"""Main CLI entry point for Orion Pulse."""

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich import box
from rich.console import Console
from rich.table import Table

from orion_pulse.analysis.service import AnalysisError, AnalysisService
from orion_pulse.backtesting import BacktestConfig, BacktestEngine
from orion_pulse.cli.output import render_analysis_report
from orion_pulse.config.settings import settings
from orion_pulse.data.providers.factory import ProviderFactory
from orion_pulse.forecasting import ForecastingService, ForecastModel
from orion_pulse.llm import LLMProviderFactory
from orion_pulse.news import NewsAnalysisService, NewsProviderFactory
from orion_pulse.reporting import ReportGenerator, TerminalReportRenderer

app = typer.Typer(
    name="orion-pulse",
    help="Orion Pulse - Market Intelligence & Probabilistic Forecasting Platform",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        from orion_pulse import __version__

        console.print(f"Orion Pulse v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit",
        ),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Orion Pulse - CLI-first market intelligence platform."""
    settings.json_output = json_output
    settings.no_color = no_color


@app.command()
def analyze(
    symbol: Annotated[
        str, typer.Argument(help="Market symbol to analyze (e.g., NVDA, AAPL)")
    ],
    lookback: Annotated[
        int,
        typer.Option(
            "--lookback", "-d", help="Lookback period in days", min=1, max=3650
        ),
    ] = 365,
    ma_periods: Annotated[
        str | None,
        typer.Option(
            "--ma", help="Comma-separated moving average periods (e.g., 20,50,200)"
        ),
    ] = None,
    volatility_window: Annotated[
        int,
        typer.Option(
            "--vol-window", help="Volatility calculation window", min=5, max=252
        ),
    ] = 20,
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="Data provider to use"),
    ] = "yfinance",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Analyze a market symbol and display trend analysis."""
    # Parse MA periods
    parsed_ma_periods = None
    if ma_periods:
        try:
            parsed_ma_periods = [int(p.strip()) for p in ma_periods.split(",")]
        except ValueError:
            console.print(
                "[red]Error:[/red] Invalid MA periods format. Use comma-separated integers."
            )
            raise typer.Exit(code=1) from None

    # Create analysis service
    try:
        service = AnalysisService(provider_name=provider)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    # Run analysis
    try:
        with console.status(f"[cyan]Analyzing {symbol.upper()}...[/cyan]"):
            report = service.analyze(
                symbol=symbol,
                lookback_days=lookback,
                ma_periods=parsed_ma_periods,
                volatility_window=volatility_window,
            )
    except AnalysisError as exc:
        console.print(f"[red]Analysis Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        console.print(f"[red]Unexpected Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Render output
    render_analysis_report(report, json_output=json_output, no_color=no_color)


@app.command()
def providers(
    provider_type: Annotated[
        str | None,
        typer.Option("--type", "-t", help="Provider type: market, news, llm"),
    ] = None,
) -> None:
    """List available data providers."""
    if provider_type is None or provider_type == "market":
        console.print("[bold]Market Data Providers:[/bold]")
        available = ProviderFactory.get_available_providers()
        if not available:
            console.print("  [yellow]No providers available[/yellow]")
        else:
            table = Table(box=box.SIMPLE)
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")
            for name in available:
                table.add_row(name, "Available")
            console.print(table)

    if provider_type is None or provider_type == "news":
        console.print("\n[bold]News Providers:[/bold]")
        available = NewsProviderFactory.get_available_providers()
        if not available:
            console.print("  [yellow]No providers available[/yellow]")
        else:
            table = Table(box=box.SIMPLE)
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")
            for name in available:
                table.add_row(name, "Available")
            console.print(table)

    if provider_type is None or provider_type == "llm":
        console.print("\n[bold]LLM Providers:[/bold]")
        available = LLMProviderFactory.get_available_providers()
        if not available:
            console.print("  [yellow]No providers available[/yellow]")
        else:
            table = Table(box=box.SIMPLE)
            table.add_column("Provider", style="cyan")
            table.add_column("Status", style="green")
            for name in available:
                table.add_row(name, "Available")
            console.print(table)


@app.command()
def news(
    symbol: Annotated[str, typer.Argument(help="Market symbol (e.g., NVDA, AAPL)")],
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Days back to fetch news", min=1, max=90),
    ] = 7,
    limit: Annotated[
        int,
        typer.Option("--limit", "-l", help="Maximum articles to fetch", min=1, max=100),
    ] = 20,
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="News provider to use"),
    ] = "newsapi",
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Fetch and analyze news for a symbol."""
    try:
        news_service = NewsAnalysisService(provider_name=provider)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    try:
        with console.status(f"[cyan]Fetching news for {symbol.upper()}...[/cyan]"):
            events = news_service.fetch_and_analyze(
                symbol=symbol,
                days_back=days,
                limit=limit,
                save_to_db=True,
            )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        import json

        console.print(json.dumps([e.to_dict() for e in events], indent=2, default=str))
        return

    if not events:
        console.print(f"[yellow]No news events found for {symbol.upper()}[/yellow]")
        return

    # Summary
    summary = news_service.summarize_events(events)
    console.print(f"\n[bold]News Summary for {symbol.upper()}[/bold]")
    console.print(f"  Total Events: {summary['total_events']}")
    console.print(f"  Avg Confidence: {summary['avg_confidence']:.1%}")
    console.print(f"  Avg Relevance: {summary['avg_relevance']:.1%}")
    console.print(f"  By Category: {summary['by_category']}")
    console.print(f"  By Bias: {summary['by_bias']}")
    console.print()

    # Events table
    from rich.table import Table

    events_table = Table(title=f"Events for {symbol.upper()}", box=box.SIMPLE)
    events_table.add_column("Date", style="cyan", width=12)
    events_table.add_column("Headline", style="white", width=60)
    events_table.add_column("Category", style="yellow", width=18)
    events_table.add_column("Bias", style="white", width=10, justify="center")
    events_table.add_column("Conf", style="white", width=6, justify="right")

    for event in events:
        bias_color = {
            "positive": "green",
            "negative": "red",
            "neutral": "yellow",
            "unknown": "white",
        }.get(event.directional_bias.value, "white")

        events_table.add_row(
            event.published_at.strftime("%Y-%m-%d"),
            event.headline[:58] + "..." if len(event.headline) > 58 else event.headline,
            event.event_category.value,
            f"[{bias_color}]{event.directional_bias.value[0].upper()}[/{bias_color}]",
            f"{event.confidence:.0%}",
        )

    console.print(events_table)


@app.command()
def forecast(
    symbol: Annotated[
        str, typer.Argument(help="Market symbol to forecast (e.g., NVDA, AAPL)")
    ],
    horizon: Annotated[
        int,
        typer.Option(
            "--horizon", "-h", help="Forecast horizon in trading days", min=1, max=30
        ),
    ] = 5,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Forecast model to use"),
    ] = "ensemble",
    lookback: Annotated[
        int,
        typer.Option(
            "--lookback", "-d", help="Lookback period for training", min=30, max=3650
        ),
    ] = 365,
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="Market data provider"),
    ] = "yfinance",
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Generate forecasts from all models"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Generate probabilistic forecast for a symbol."""
    # Validate model
    try:
        model_enum = ForecastModel(model)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown model: {model}")
        console.print(f"Available models: {[m.value for m in ForecastModel]}")
        raise typer.Exit(code=1) from None

    # Create services
    try:
        analysis_service = AnalysisService(provider_name=provider)
        forecast_service = ForecastingService()
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    # Get market data
    try:
        with console.status(f"[cyan]Fetching data for {symbol.upper()}...[/cyan]"):
            raw_data = analysis_service.get_raw_data(symbol, lookback_days=lookback)
    except Exception as exc:
        console.print(f"[red]Error fetching data:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if raw_data.is_empty():
        console.print(f"[red]Error:[/red] No data available for {symbol}")
        raise typer.Exit(code=1) from None

    # Generate forecasts
    try:
        with console.status("[cyan]Generating forecast...[/cyan]"):
            if all_models:
                results = forecast_service.generate_all_forecasts(
                    symbol, raw_data, horizon_days=horizon
                )
            else:
                results = {
                    model_enum.value: forecast_service.generate_forecast(
                        symbol, raw_data, model=model_enum, horizon_days=horizon
                    )
                }
    except Exception as exc:
        console.print(f"[red]Forecast Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        import json

        output = {k: v.to_dict() if v else None for k, v in results.items()}
        console.print(json.dumps(output, indent=2, default=str))
        return

    # Render forecasts
    from orion_pulse.reporting import TerminalReportRenderer

    renderer = TerminalReportRenderer(no_color=no_color)
    renderer._render_forecasts({k: v for k, v in results.items() if v})


@app.command()
def backtest(
    symbol: Annotated[
        str, typer.Argument(help="Market symbol to backtest (e.g., NVDA, AAPL)")
    ],
    start_date: Annotated[
        str,
        typer.Option("--start", help="Start date (YYYY-MM-DD)"),
    ],
    end_date: Annotated[
        str,
        typer.Option("--end", help="End date (YYYY-MM-DD)"),
    ],
    horizon: Annotated[
        int,
        typer.Option("--horizon", "-h", help="Forecast horizon in days", min=1, max=30),
    ] = 5,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Forecast model to backtest"),
    ] = "ensemble",
    train_window: Annotated[
        int,
        typer.Option(
            "--train-window", help="Training window in days", min=60, max=1000
        ),
    ] = 252,
    step_size: Annotated[
        int,
        typer.Option("--step", help="Step size in days", min=1, max=20),
    ] = 5,
    all_models: Annotated[
        bool,
        typer.Option("--all", help="Backtest all models"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Run walk-forward backtest for a symbol."""
    # Parse dates
    try:
        start = date.fromisoformat(start_date)
        end = date.fromisoformat(end_date)
    except ValueError:
        console.print("[red]Error:[/red] Invalid date format. Use YYYY-MM-DD.")
        raise typer.Exit(code=1) from None

    if start >= end:
        console.print("[red]Error:[/red] Start date must be before end date.")
        raise typer.Exit(code=1) from None

    # Validate model
    if not all_models:
        try:
            model_enum = ForecastModel(model)
        except ValueError:
            console.print(f"[red]Error:[/red] Unknown model: {model}")
            console.print(f"Available models: {[m.value for m in ForecastModel]}")
            raise typer.Exit(code=1) from None

    # Create engine
    engine = BacktestEngine()

    try:
        with console.status(f"[cyan]Running backtest for {symbol.upper()}...[/cyan]"):
            if all_models:
                results = engine.run_comparative_backtest(
                    symbol=symbol,
                    start_date=start,
                    end_date=end,
                    horizon_days=horizon,
                )
            else:
                config = BacktestConfig(
                    symbol=symbol,
                    model=model_enum,
                    start_date=start,
                    end_date=end,
                    horizon_days=horizon,
                    train_window=train_window,
                    step_size=step_size,
                )
                results = {model_enum.value: engine.run_backtest(config)}
    except Exception as exc:
        console.print(f"[red]Backtest Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        import json

        output = {k: v.to_dict() if v else None for k, v in results.items()}
        console.print(json.dumps(output, indent=2, default=str))
        return

    # Render results
    from orion_pulse.reporting import TerminalReportRenderer

    renderer = TerminalReportRenderer(no_color=no_color)

    for model_name, result in results.items():
        if result:
            renderer._render_backtest(result)
        else:
            console.print(f"[red]Backtest failed for {model_name}[/red]")


@app.command()
def report(
    symbol: Annotated[
        str, typer.Argument(help="Market symbol for comprehensive report")
    ],
    lookback: Annotated[
        int,
        typer.Option(
            "--lookback", "-d", help="Lookback period for analysis", min=30, max=3650
        ),
    ] = 365,
    horizon: Annotated[
        int,
        typer.Option("--horizon", "-h", help="Forecast horizon in days", min=1, max=30),
    ] = 5,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Forecast model to use"),
    ] = "ensemble",
    news_days: Annotated[
        int,
        typer.Option("--news-days", help="Days of news to include", min=0, max=90),
    ] = 7,
    provider: Annotated[
        str,
        typer.Option("--provider", "-p", help="Market data provider"),
    ] = "yfinance",
    llm_provider: Annotated[
        str,
        typer.Option("--llm", help="LLM provider for synthesis"),
    ] = "mock",
    no_llm: Annotated[
        bool,
        typer.Option("--no-llm", help="Disable LLM synthesis"),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Output as JSON"),
    ] = False,
    no_color: Annotated[
        bool,
        typer.Option("--no-color", help="Disable colored output"),
    ] = False,
) -> None:
    """Generate comprehensive report combining analysis, forecast, backtest, and news."""
    # Validate model
    try:
        model_enum = ForecastModel(model)
    except ValueError:
        console.print(f"[red]Error:[/red] Unknown model: {model}")
        raise typer.Exit(code=1) from None

    # Create services
    try:
        analysis_service = AnalysisService(provider_name=provider)
        forecast_service = ForecastingService()
        backtest_engine = BacktestEngine()
        news_service = NewsAnalysisService()
        report_generator = ReportGenerator(
            llm_provider_name=llm_provider, news_service=news_service
        )
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from None

    # Get market data
    try:
        with console.status(f"[cyan]Fetching data for {symbol.upper()}...[/cyan]"):
            raw_data = analysis_service.get_raw_data(symbol, lookback_days=lookback)
    except Exception as exc:
        console.print(f"[red]Error fetching data:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if raw_data.is_empty():
        console.print(f"[red]Error:[/red] No data available for {symbol}")
        raise typer.Exit(code=1) from None

    # Run analysis
    try:
        with console.status("[cyan]Running analysis...[/cyan]"):
            analysis_report = analysis_service.analyze(
                symbol=symbol,
                lookback_days=lookback,
            )
    except Exception as exc:
        console.print(f"[red]Analysis Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Generate forecasts
    try:
        with console.status("[cyan]Generating forecast...[/cyan]"):
            forecast_results = forecast_service.generate_all_forecasts(
                symbol, raw_data, horizon_days=horizon
            )
    except Exception as exc:
        console.print(f"[red]Forecast Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Run quick backtest (last 6 months)
    backtest_result = None
    try:
        with console.status("[cyan]Running backtest...[/cyan]"):
            bt_end = date.today() - timedelta(days=1)
            bt_start = bt_end - timedelta(days=180)
            config = BacktestConfig(
                symbol=symbol,
                model=model_enum,
                start_date=bt_start,
                end_date=bt_end,
                horizon_days=horizon,
                train_window=120,
                step_size=5,
            )
            backtest_result = backtest_engine.run_backtest(config)
    except Exception as e:
        # Backtest is optional, log and continue
        logging.warning("Backtest failed, continuing without it: %s", e)

    # Generate comprehensive report
    try:
        with console.status("[cyan]Generating report...[/cyan]"):
            # Filter out None forecast results
            valid_forecasts = {
                k: v for k, v in forecast_results.items() if v is not None
            }
            report = report_generator.generate_report(
                symbol=symbol,
                analysis_report=analysis_report,
                forecast_results=valid_forecasts,
                backtest_result=backtest_result,
                include_news=news_days > 0,
                news_days=news_days,
                use_llm=not no_llm,
            )
    except Exception as exc:
        console.print(f"[red]Report Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if json_output:
        import json

        console.print(json.dumps(report.to_dict(), indent=2, default=str))
        return

    # Render report
    renderer = TerminalReportRenderer(no_color=no_color)
    renderer.render(report)


@app.command()
def config(
    show: Annotated[
        bool,
        typer.Option("--show", help="Show current configuration"),
    ] = False,
    init: Annotated[
        bool,
        typer.Option("--init", help="Create example .env file"),
    ] = False,
) -> None:
    """Manage configuration."""
    if init:
        env_content = """# Orion Pulse Configuration
# Copy this to .env and fill in your API keys

# Data storage
ORION_PULSE_DATA_DIR=~/.orion-pulse/data
ORION_PULSE_CACHE_DIR=~/.orion-pulse/cache

# Market data providers
ORION_PULSE_YFINANCE_ENABLED=true
ORION_PULSE_YFINANCE_TIMEOUT=30

# Analysis defaults
ORION_PULSE_DEFAULT_LOOKBACK_DAYS=365
ORION_PULSE_DEFAULT_MA_PERIODS=20,50,200
ORION_PULSE_VOLATILITY_WINDOW=20

# Output
ORION_PULSE_JSON_OUTPUT=false
ORION_PULSE_NO_COLOR=false

# API Keys (required for respective providers)
ORION_PULSE_NEWSAPI_KEY=your_newsapi_key_here
ORION_PULSE_NVIDIA_API_KEY=your_nvidia_api_key_here
ORION_PULSE_ALPHA_VANTAGE_KEY=your_alpha_vantage_key_here
ORION_PULSE_POLYGON_KEY=your_polygon_key_here
"""
        env_path = Path(".env")
        if env_path.exists() and not typer.confirm(".env already exists. Overwrite?"):
            console.print("Aborted.")
            return

        env_path.write_text(env_content)
        console.print(f"[green]Created {env_path.absolute()}[/green]")
        console.print("Edit this file to add your API keys.")
        return

    if show:
        console.print("[bold]Current Configuration:[/bold]")
        console.print()

        table = Table(box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        # Show non-sensitive settings
        settings_dict = {
            "data_dir": str(settings.data_dir),
            "cache_dir": str(settings.cache_dir),
            "yfinance_enabled": settings.yfinance_enabled,
            "yfinance_timeout": settings.yfinance_timeout,
            "default_lookback_days": settings.default_lookback_days,
            "default_ma_periods": settings.default_ma_periods,
            "volatility_window": settings.volatility_window,
            "json_output": settings.json_output,
            "no_color": settings.no_color,
            "newsapi_configured": bool(settings.newsapi_key),
            "nvidia_api_configured": bool(settings.nvidia_api_key),
        }

        for key, value in settings_dict.items():
            table.add_row(key, str(value))

        console.print(table)
        return

    console.print("Use --show to view config or --init to create example .env file")


if __name__ == "__main__":
    app()
