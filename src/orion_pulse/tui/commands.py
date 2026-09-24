"""Commands for Orion Pulse TUI."""

from __future__ import annotations

from orion_pulse.analysis.service import AnalysisService
from orion_pulse.tui.command_engine import Command, ParsedCommand
from orion_pulse.tui.provider_manager import ConnectScreen


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
