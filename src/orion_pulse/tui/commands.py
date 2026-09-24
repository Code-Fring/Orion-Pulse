"""Commands for Orion Pulse TUI."""

from orion_pulse.analysis.service import AnalysisService
from orion_pulse.backtesting import (
    BacktestConfig,
    BacktestEngine,
)
from orion_pulse.backtesting import (
    ForecastModel as BacktestForecastModel,
)
from orion_pulse.forecasting import (
    ForecastingService,
    ForecastModel,
)
from orion_pulse.news.analysis import NewsAnalysisService
from orion_pulse.tui.command_engine import Command, ParsedCommand
from orion_pulse.tui.provider_manager import ConnectScreen
from orion_pulse.llm import (
    LLMMessage,
    LLMProviderError,
    LLMProviderFactory,
    NVIDIAProvider,
    LLMResponse,
)


class AnalyzeCommand(Command):
    """Analyze a market symbol."""

    name = "analyze"
    aliases = ["a"]
    description = "Analyze a market symbol"
    usage = "analyze SYMBOL [--lookback DAYS] [--ma PERIODS] [--provider NAME]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the analyze command."""
        if not parsed.args:
            self.app.write_output("[red]Error: Symbol is required[/red]")
            return

        symbol = parsed.args[0]

        # Parse options
        lookback_days = int(
            parsed.kwargs.get("lookback", self.app.config.default_lookback_days)
        )
        ma_periods_str = parsed.kwargs.get("ma", self.app.config.default_ma_periods)
        ma_periods = (
            [int(p) for p in ma_periods_str.split(",")]
            if ma_periods_str
            else [20, 50, 200]
        )
        provider_name = parsed.kwargs.get("provider", "yfinance")

        # Get provider
        provider = self.app.provider_manager.get_active_provider()
        if not provider:
            self.app.write_output(
                "[red]Error: No active provider. Use 'connect' to configure one.[/red]"
            )
            return

        try:
            # Run analysis
            service = AnalysisService(provider_name)
            report = service.analyze(
                symbol=symbol,
                lookback_days=lookback_days,
                ma_periods=ma_periods,
                volatility_window=self.app.config.volatility_window,
            )

            # Render report using existing CLI output formatter
            self.app.write_output(f"[green]Analysis complete for {symbol}[/green]")
            self.app.write_output(f"Trend: {report.trend_analysis.trend.value}")
            self.app.write_output(
                f"Last Price: ${report.trend_analysis.last_price:.2f}"
            )

        except Exception as e:
            self.app.write_output(f"[red]Error analyzing {symbol}: {str(e)}[/red]")


class ProvidersCommand(Command):
    """List available data providers."""

    name = "providers"
    aliases = ["p"]
    description = "List available data providers"
    usage = "providers"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the providers command."""
        available = self.app.provider_manager.get_available_providers()
        active = self.app.provider_manager.active_provider

        self.app.write_output("[bold cyan]Available Providers:[/bold cyan]")
        for provider in available:
            status = " [green](active)[/green]" if provider == active else ""
            self.app.write_output(f"  {provider}{status}")
        self.app.write_output("")


class ConnectCommand(Command):
    """Connect to a data provider."""

    name = "connect"
    aliases = ["c"]
    description = "Connect to a data provider"
    usage = "connect PROVIDER_NAME"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the connect command."""
        if not parsed.args:
            # Show available providers
            available = self.app.provider_manager.get_available_providers()
            self.app.write_output("[bold cyan]Available Providers:[/bold cyan]")
            for provider in available:
                self.app.write_output(f"  {provider}")
            self.app.write_output("Use: connect PROVIDER_NAME")
            return

        provider_name = parsed.args[0]

        # Check if provider is available
        available = self.app.provider_manager.get_available_providers()
        if provider_name not in available:
            self.app.write_output(
                f"[red]Error: Provider '{provider_name}' not available[/red]"
            )
            return

        # Show connection screen
        screen = ConnectScreen(provider_name, self.app.provider_manager)
        result = await self.app.push_screen_wait(screen)

        if result is not None:
            self.app.write_output(
                f"[green]Successfully connected to {provider_name}[/green]"
            )
        else:
            self.app.write_output("[yellow]Connection cancelled[/yellow]")


class HelpCommand(Command):
    """Show help information."""

    name = "help"
    aliases = ["h", "?"]
    description = "Show help information"
    usage = "help [COMMAND]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the help command."""
        if parsed.args:
            # Show specific command help
            command_name = parsed.args[0]
            command = self.app.command_registry.get(command_name)
            if command:
                self.app.write_output(command.format_help())
            else:
                self.app.write_output(f"[red]Unknown command: {command_name}[/red]")
        else:
            # Show general help
            self.app.write_output("[bold cyan]Orion Pulse Commands:[/bold cyan]")
            for command in self.app.command_registry.all_commands():
                self.app.write_output(command.format_help())
            self.app.write_output(
                "[dim]Use 'help COMMAND' for detailed information about a specific command.[/dim]"
            )
            self.app.write_output("")


class ExitCommand(Command):
    """Exit the application."""

    name = "exit"
    aliases = ["quit", "q"]
    description = "Exit the application"
    usage = "exit"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the exit command."""
        raise SystemExit()


class ForecastCommand(Command):
    """Generate probabilistic forecasts for a symbol."""

    name = "forecast"
    aliases = ["f"]
    description = "Generate probabilistic forecasts for a symbol"
    usage = "forecast SYMBOL [--model MODEL] [--horizon DAYS] [--lookback DAYS] [--provider NAME]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the forecast command."""
        if not parsed.args:
            self.app.write_output("[red]Error: Symbol is required[/red]")
            return

        symbol = parsed.args[0]

        # Parse options
        model_name = parsed.kwargs.get("model", "ensemble").upper()
        try:
            model = ForecastModel[model_name]
        except KeyError:
            self.app.write_output(
                f"[red]Error: Unknown model '{model_name}'. Available: {[m.name for m in ForecastModel]}[/red]"
            )
            return

        horizon_days = int(parsed.kwargs.get("horizon", "5"))
        lookback_days = int(
            parsed.kwargs.get("lookback", str(self.app.config.default_lookback_days))
        )
        provider_name = parsed.kwargs.get("provider", "yfinance")

        # Get provider
        provider = self.app.provider_manager.get_active_provider()
        if not provider:
            self.app.write_output(
                "[red]Error: No active provider. Use 'connect' to configure one.[/red]"
            )
            return

        try:
            # Get market data
            from orion_pulse.data.providers.factory import DataProviderFactory

            provider_instance = DataProviderFactory.get_provider(provider_name)
            df = provider_instance.get_data(symbol, lookback_days=lookback_days)

            if df.is_empty():
                self.app.write_output(f"[red]Error: No data found for {symbol}[/red]")
                return

            # Generate forecast
            service = ForecastingService()
            result = service.generate_forecast(
                symbol=symbol,
                df=df,
                model=model,
                horizon_days=horizon_days,
                save_to_db=True,
                lookback_days=lookback_days,
            )

            # Display forecast
            self.app.write_output(f"[green]Forecast complete for {symbol}[/green]")
            self.app.write_output(f"Model: {result.model_name}")
            self.app.write_output(f"Horizon: {result.horizon_days} days")
            self.app.write_output(f"Expected Return: {result.expected_return:.2%}")
            self.app.write_output(f"Prob Positive: {result.prob_positive:.1%}")
            self.app.write_output(f"Confidence: {result.confidence:.0%}")
            self.app.write_output("")
            self.app.write_output("[bold]Percentile Forecasts:[/bold]")
            self.app.write_output(f"  P10: {result.p10:.2%}")
            self.app.write_output(f"  P25: {result.p25:.2%}")
            self.app.write_output(f"  P50 (Median): {result.p50:.2%}")
            self.app.write_output(f"  P75: {result.p75:.2%}")
            self.app.write_output(f"  P90: {result.p90:.2%}")
            self.app.write_output("")
            self.app.write_output("[bold]Key Factors:[/bold]")
            for key, value in result.factors.items():
                self.app.write_output(f"  {key}: {value}")

        except Exception as e:
            self.app.write_output(f"[red]Error forecasting {symbol}: {str(e)}[/red]")


class BacktestCommand(Command):
    """Run walk-forward backtesting for a symbol."""

    name = "backtest"
    aliases = ["bt"]
    description = "Run walk-forward backtesting for a symbol"
    usage = "backtest SYMBOL [--model MODEL] [--start DATE] [--end DATE] [--horizon DAYS] [--train-window DAYS] [--step DAYS]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the backtest command."""
        if not parsed.args:
            self.app.write_output("[red]Error: Symbol is required[/red]")
            return

        symbol = parsed.args[0]

        # Parse options
        model_name = parsed.kwargs.get("model", "ensemble").upper()
        try:
            model = BacktestForecastModel[model_name]
        except KeyError:
            self.app.write_output(
                f"[red]Error: Unknown model '{model_name}'. Available: {[m.name for m in BacktestForecastModel]}[/red]"
            )
            return

        from datetime import date, timedelta

        end_date_str = parsed.kwargs.get("end", date.today().isoformat())
        start_date_str = parsed.kwargs.get(
            "start", (date.today() - timedelta(days=252)).isoformat()
        )

        try:
            start_date = date.fromisoformat(start_date_str)
            end_date = date.fromisoformat(end_date_str)
        except ValueError:
            self.app.write_output(
                "[red]Error: Invalid date format. Use YYYY-MM-DD[/red]"
            )
            return

        horizon_days = int(parsed.kwargs.get("horizon", "5"))
        train_window = int(parsed.kwargs.get("train-window", "252"))
        step_size = int(parsed.kwargs.get("step", "1"))

        self.app.write_output(f"[yellow]Running backtest for {symbol}...[/yellow]")
        self.app.write_output(
            f"Model: {model.value}, Period: {start_date} to {end_date}"
        )

        try:
            engine = BacktestEngine()
            config = BacktestConfig(
                symbol=symbol,
                model=model,
                start_date=start_date,
                end_date=end_date,
                horizon_days=horizon_days,
                train_window=train_window,
                step_size=step_size,
            )
            result = engine.run_backtest(config, save_to_db=True)

            # Display results
            self.app.write_output(f"[green]Backtest complete for {symbol}[/green]")
            self.app.write_output(f"Total Predictions: {result.total_predictions}")
            self.app.write_output(
                f"Directional Accuracy: {result.directional_accuracy:.1%}"
                if result.directional_accuracy
                else "Directional Accuracy: N/A"
            )
            self.app.write_output(
                f"MAE: {result.mae:.4f}" if result.mae else "MAE: N/A"
            )
            self.app.write_output(
                f"RMSE: {result.rmse:.4f}" if result.rmse else "RMSE: N/A"
            )
            self.app.write_output(
                f"Calibration Error: {result.calibration_error:.4f}"
                if result.calibration_error
                else "Calibration Error: N/A"
            )
            self.app.write_output(
                f"Cumulative Return: {result.cumulative_return:.2%}"
                if result.cumulative_return
                else "Cumulative Return: N/A"
            )
            self.app.write_output(
                f"Annualized Return: {result.annualized_return:.2%}"
                if result.annualized_return
                else "Annualized Return: N/A"
            )
            self.app.write_output(
                f"Volatility: {result.volatility:.2%}"
                if result.volatility
                else "Volatility: N/A"
            )
            self.app.write_output(
                f"Max Drawdown: {result.max_drawdown:.2%}"
                if result.max_drawdown
                else "Max Drawdown: N/A"
            )
            self.app.write_output(
                f"Sharpe Ratio: {result.sharpe_ratio:.2f}"
                if result.sharpe_ratio
                else "Sharpe Ratio: N/A"
            )
            self.app.write_output(
                f"Sortino Ratio: {result.sortino_ratio:.2f}"
                if result.sortino_ratio
                else "Sortino Ratio: N/A"
            )

        except Exception as e:
            self.app.write_output(f"[red]Error backtesting {symbol}: {str(e)}[/red]")


class NewsCommand(Command):
    """Browse and analyze financial news."""

    name = "news"
    aliases = ["n"]
    description = "Browse and analyze financial news for a symbol"
    usage = "news SYMBOL [--days DAYS] [--limit COUNT] [--provider NAME]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the news command."""
        if not parsed.args:
            self.app.write_output("[red]Error: Symbol is required[/red]")
            return

        symbol = parsed.args[0]
        days_back = int(parsed.kwargs.get("days", "7"))
        limit = int(parsed.kwargs.get("limit", "20"))
        provider_name = parsed.kwargs.get("provider", "newsapi")

        self.app.write_output(f"[yellow]Fetching news for {symbol}...[/yellow]")

        try:
            service = NewsAnalysisService(provider_name=provider_name)
            events = service.fetch_and_analyze(
                symbol=symbol,
                days_back=days_back,
                limit=limit,
                save_to_db=True,
            )

            if not events:
                self.app.write_output(
                    f"[yellow]No news found for {symbol} in the last {days_back} days[/yellow]"
                )
                return

            self.app.write_output(
                f"[green]Found {len(events)} news events for {symbol}[/green]"
            )
            self.app.write_output("")

            for i, event in enumerate(events, 1):
                bias_color = {
                    "positive": "green",
                    "negative": "red",
                    "neutral": "yellow",
                    "unknown": "dim",
                }.get(event.directional_bias.value, "white")

                self.app.write_output(f"[bold]{i}. {event.headline}[/bold]")
                self.app.write_output(
                    f"  Source: {event.source} | {event.published_at.strftime('%Y-%m-%d %H:%M')}"
                )
                self.app.write_output(
                    f"  Category: {event.event_category.value} | Bias: [{bias_color}]{event.directional_bias.value}[/{bias_color}] (conf: {event.confidence:.0%})"
                )
                self.app.write_output(
                    f"  Relevance: {event.relevance_score:.0%} | Factors: {', '.join(event.key_factors) if event.key_factors else 'N/A'}"
                )
                if event.summary:
                    self.app.write_output(
                        f"  {event.summary[:200]}{'...' if len(event.summary) > 200 else ''}"
                    )
                self.app.write_output("")

            # Summary
            summary = service.summarize_events(events)
            self.app.write_output("[bold]Summary:[/bold]")
            self.app.write_output(f"  Total Events: {summary['total_events']}")
            self.app.write_output(f"  By Category: {summary['by_category']}")
            self.app.write_output(f"  By Bias: {summary['by_bias']}")
            self.app.write_output(f"  Avg Confidence: {summary['avg_confidence']:.0%}")
            self.app.write_output(f"  Avg Relevance: {summary['avg_relevance']:.0%}")

        except Exception as e:
            self.app.write_output(
                f"[red]Error fetching news for {symbol}: {str(e)}[/red]"
            )


class ChatCommand(Command):
    """Chat with the AI model."""

    name = "chat"
    aliases = ["ch"]
    description = "Chat with the AI model"
    usage = "chat [PROMPT]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the chat command."""
        # Get the prompt from args or ask user
        if parsed.args:
            prompt = " ".join(parsed.args)
        else:
            self.app.write_output(
                "[yellow]Enter your prompt (or 'clear' to clear, 'help' for options):[/yellow]"
            )
            # We'll handle this differently - just show help for now
            return

        if not prompt.strip():
            self.app.write_output("[red]Error: Prompt is required[/red]")
            return

        # Get the LLM provider - try NVIDIA first, fall back to mock
        LLMProviderFactory.clear_cache()
        from orion_pulse.config.settings import settings

        try:
            provider = LLMProviderFactory.get_provider("nvidia")
            if not provider.is_available():
                self.app.write_output(
                    "[red]Error: NVIDIA LLM provider not available. Check API key configuration.[/red]"
                )
                # Fall back to mock provider
                provider = LLMProviderFactory.get_provider("mock")

            # Complete the prompt
            messages = [LLMMessage(role="user", content=prompt)]
            response: LLMResponse = provider.complete(messages)

            # Display the response
            self.app.write_output("")
            self.app.write_output("[bold cyan]AI Response:[/bold cyan]")
            self.app.write_output(response.content)
            self.app.write_output("")

        except LLMProviderError as e:
            self.app.write_output(f"[red]LLM Provider Error: {e}[/red]")
            # Fall back to mock provider on error
            try:
                fallback_provider = LLMProviderFactory.get_provider("mock")
                messages = [LLMMessage(role="user", content=prompt)]
                response: LLMResponse = fallback_provider.complete(messages)
                self.app.write_output("")
                self.app.write_output("[bold cyan]AI Response (mock):[/bold cyan]")
                self.app.write_output(response.content)
                self.app.write_output("")
            except Exception:
                self.app.write_output("[red]Failed to get AI response[/red]")
        except Exception as e:
            self.app.write_output(f"[red]Error chatting with AI: {str(e)}[/red]")
            # Fall back to mock provider on unexpected error
            try:
                fallback_provider = LLMProviderFactory.get_provider("mock")
                messages = [LLMMessage(role="user", content=prompt)]
                response: LLMResponse = fallback_provider.complete(messages)
                self.app.write_output("")
                self.app.write_output("[bold cyan]AI Response (mock):[/bold cyan]")
                self.app.write_output(response.content)
                self.app.write_output("")
            except Exception:
                self.app.write_output("[red]Failed to get AI response[/red]")


class SettingsCommand(Command):
    """View and manage application settings."""

    name = "settings"
    aliases = ["cfg", "config"]
    description = "View and manage application settings"
    usage = "settings [--show] [--set KEY=VALUE] [--reset]"

    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the settings command."""
        if parsed.kwargs.get("show") or "show" in parsed.kwargs:
            self._show_settings()
            return

        if parsed.kwargs.get("reset") or "reset" in parsed.kwargs:
            self._reset_settings()
            return

        # Handle --set KEY=VALUE
        set_args = [k for k in parsed.kwargs if k.startswith("set")]
        if set_args or parsed.args:
            # Check for key=value in args
            for arg in parsed.args:
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    self._set_setting(key, value)
                    return
            # Check kwargs
            for _key in set_args:
                # This handles --set=key=value format
                pass
            self.app.write_output("[yellow]Usage: settings --set KEY=VALUE[/yellow]")
            return

        # Default: show settings
        self._show_settings()

    def _show_settings(self) -> None:
        """Display current settings."""
        self.app.write_output("[bold cyan]Orion Pulse Settings:[/bold cyan]")
        self.app.write_output("")

        config = self.app.config
        settings = [
            ("Data Directory", str(config.data_dir)),
            ("Cache Directory", str(config.cache_dir)),
            ("YFinance Enabled", str(config.yfinance_enabled)),
            ("YFinance Timeout", f"{config.yfinance_timeout}s"),
            ("Default Lookback", f"{config.default_lookback_days} days"),
            ("Default MA Periods", config.default_ma_periods),
            ("Volatility Window", f"{config.volatility_window} days"),
            ("JSON Output", str(config.json_output)),
            ("No Color", str(config.no_color)),
            ("Default Provider", config.default_provider),
            ("Currency", config.currency),
            ("Timezone", config.timezone),
            ("LLM Model", config.llm_model),
        ]

        for key, value in settings:
            self.app.write_output(f"  [bold]{key}:[/bold] {value}")

        self.app.write_output("")
        self.app.write_output("[dim]API Keys (masked):[/dim]")
        api_keys = [
            ("Alpha Vantage", config.alpha_vantage_key),
            ("Polygon", config.polygon_key),
            ("NewsAPI", config.newsapi_key),
        ]
        for name, key in api_keys:
            masked = f"{key[:4]}...{key[-4:]}" if key and len(key) > 8 else "Not set"
            self.app.write_output(f"  {name}: {masked}")

    def _set_setting(self, key: str, value: str) -> None:
        """Set a setting value."""
        # This would update the .env file
        # For now, just show what would be set
        self.app.write_output(
            f"[yellow]Setting {key} = {value} (requires .env update)[/yellow]"
        )

    def _reset_settings(self) -> None:
        """Reset settings to defaults."""
        self.app.write_output(
            "[yellow]Settings reset not yet implemented. Edit .env file manually.[/yellow]"
        )
