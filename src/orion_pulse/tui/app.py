"""Main TUI application for Orion Pulse."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal
from textual.widgets import Footer, Header, Input, RichLog

if TYPE_CHECKING:
    from rich.text import Text

from orion_pulse.tui.command_engine import CommandEngine, CommandRegistry
from orion_pulse.tui.commands import (
    AnalyzeCommand,
    ConnectCommand,
    ExitCommand,
    HelpCommand,
    ProvidersCommand,
)
from orion_pulse.tui.config_manager import initialize_config
from orion_pulse.tui.provider_manager import ProviderManager


class OrionPulseApp(App[None]):
    """Main Orion Pulse TUI application."""

    ENABLE_COMMAND_PALETTE = False
    CSS_PATH = "styles.tcss"

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("ctrl+c", "quit", "Quit", priority=True),
        Binding("ctrl+d", "quit", "Quit", priority=True),
        Binding("ctrl+l", "clear_output", "Clear", show=False),
        Binding("up", "history_previous", "Previous Command", show=False),
        Binding("down", "history_next", "Next Command", show=False),
    ]

    def __init__(self) -> None:
        super().__init__()
        self.config = initialize_config()
        self.provider_manager = ProviderManager(self)
        self.command_registry = CommandRegistry()
        self.command_engine = CommandEngine(self.command_registry)
        self._setup_commands()

    def _setup_commands(self) -> None:
        """Set up available commands."""
        commands = [
            AnalyzeCommand(self),
            ProvidersCommand(self),
            ConnectCommand(self),
            HelpCommand(self),
            ExitCommand(self),
        ]

        for command in commands:
            self.command_registry.register(command)

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Container(
            RichLog(id="output", wrap=True, highlight=True, markup=True),
            id="main-content",
        )
        yield Container(
            Horizontal(
                Input(placeholder="Enter command...", id="command-input"),
                id="input-container",
            ),
            id="input-area",
        )
        yield Footer()

    def on_mount(self) -> None:
        """Initialize the app when mounted."""
        self.title = "Orion Pulse"
        self.sub_title = "Market Intelligence Platform"

        # Focus the command input
        input_widget = self.query_one("#command-input", Input)
        input_widget.focus()

        # Show welcome message
        self.show_welcome()

    def show_welcome(self) -> None:
        """Show welcome message."""
        output = self.query_one("#output", RichLog)
        output.write("[bold cyan]Welcome to Orion Pulse![/bold cyan]")
        output.write(
            "Type [bold]'help'[/bold] for available commands or [bold]'exit'[/bold] to quit."
        )
        output.write("")

    @on(Input.Submitted, "#command-input")
    async def handle_command_submission(self, event: Input.Submitted) -> None:
        """Handle command submission."""
        command_input = event.input
        raw_command = command_input.value.strip()

        if not raw_command:
            return

        # Clear the input
        command_input.value = ""

        # Echo the command
        output = self.query_one("#output", RichLog)
        output.write(f"[dim]> {raw_command}[/dim]")

        # Execute the command
        should_continue, error = await self.command_engine.execute(raw_command)

        if error:
            output.write(f"[bold red]Error:[/bold red] {error}")
            output.write("")

        if not should_continue:
            self.exit()

    def action_quit(self) -> None:
        """Quit the application."""
        self.exit()

    def action_clear_output(self) -> None:
        """Clear the output log."""
        output = self.query_one("#output", RichLog)
        output.clear()

    def action_history_previous(self) -> None:
        """Navigate to previous command in history."""
        command_input = self.query_one("#command-input", Input)
        previous = self.command_engine.get_history_up()
        if previous is not None:
            command_input.value = previous
            # Move cursor to end
            command_input.cursor_position = len(previous)

    def action_history_next(self) -> None:
        """Navigate to next command in history."""
        command_input = self.query_one("#command-input", Input)
        next_cmd = self.command_engine.get_history_down()
        if next_cmd is not None:
            command_input.value = next_cmd
            # Move cursor to end
            command_input.cursor_position = len(next_cmd)

    def write_output(self, content: str | Text) -> None:
        """Write content to the output log."""
        output = self.query_one("#output", RichLog)
        output.write(content)

    def get_provider_status(self) -> str:
        """Get provider status for display."""
        return self.provider_manager.get_provider_status()


def main() -> None:
    """Main entry point for the TUI application."""
    app = OrionPulseApp()
    app.run()


if __name__ == "__main__":
    main()
