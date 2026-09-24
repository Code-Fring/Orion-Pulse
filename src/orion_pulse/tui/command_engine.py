"""Command engine for Orion Pulse TUI."""

from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.text import Text

if TYPE_CHECKING:
    from orion_pulse.tui.app import OrionPulseApp


@dataclass(frozen=True)
class ParsedCommand:
    """Parsed command with name, args, and kwargs."""

    name: str
    args: list[str] = field(default_factory=list)
    kwargs: dict[str, str] = field(default_factory=dict)
    raw: str = ""


class Command(ABC):
    """Base class for all commands."""

    name: str
    aliases: list[str] = field(default_factory=list)
    description: str = ""
    usage: str = ""

    def __init__(self, app: OrionPulseApp) -> None:
        self.app = app

    @abstractmethod
    async def execute(self, parsed: ParsedCommand) -> None:
        """Execute the command."""
        ...

    def format_help(self) -> Text:
        """Format help text for this command."""
        text = Text()
        text.append(f"  {self.name}", style="bold cyan")
        if self.aliases:
            text.append(f" ({', '.join(self.aliases)})", style="dim")
        text.append(f"\n    {self.description}\n")
        if self.usage:
            text.append(f"    Usage: {self.usage}\n", style="dim")
        return text


class CommandRegistry:
    """Registry for all available commands."""

    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}
        self._aliases: dict[str, str] = {}

    def register(self, command: Command) -> None:
        """Register a command."""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get(self, name: str) -> Command | None:
        """Get command by name or alias."""
        if name in self._commands:
            return self._commands[name]
        if name in self._aliases:
            return self._commands[self._aliases[name]]
        return None

    def all_commands(self) -> list[Command]:
        """Get all registered commands."""
        return list(self._commands.values())

    def all_names(self) -> list[str]:
        """Get all command names and aliases."""
        names = list(self._commands.keys())
        names.extend(self._aliases.keys())
        return sorted(set(names))


class CommandEngine:
    """Engine for parsing and executing commands."""

    def __init__(self, registry: CommandRegistry) -> None:
        self.registry = registry
        self.history: list[str] = []
        self.history_index: int = -1

    def parse(self, raw_input: str) -> ParsedCommand:
        """Parse raw input into a ParsedCommand."""
        raw_input = raw_input.strip()
        if not raw_input:
            return ParsedCommand(name="", raw=raw_input)

        try:
            parts = shlex.split(raw_input)
        except ValueError:
            parts = raw_input.split()

        if not parts:
            return ParsedCommand(name="", raw=raw_input)

        name = parts[0].lower()
        args = []
        kwargs = {}

        for part in parts[1:]:
            if part.startswith("--"):
                key = part[2:]
                if "=" in key:
                    k, v = key.split("=", 1)
                    kwargs[k] = v
                else:
                    kwargs[key] = "true"
            elif part.startswith("-") and len(part) > 1:
                for ch in part[1:]:
                    kwargs[ch] = "true"
            else:
                args.append(part)

        return ParsedCommand(name=name, args=args, kwargs=kwargs, raw=raw_input)

    async def execute(self, raw_input: str) -> tuple[bool, str | None]:
        """Execute a command. Returns (should_continue, error_message)."""
        raw_input = raw_input.strip()
        if not raw_input:
            return True, None

        self.history.append(raw_input)
        self.history_index = len(self.history)

        parsed = self.parse(raw_input)

        if not parsed.name:
            return True, None

        command = self.registry.get(parsed.name)
        if not command:
            return (
                True,
                f"Unknown command: {parsed.name}. Type 'help' for available commands.",
            )

        try:
            await command.execute(parsed)
        except SystemExit:
            return False, None
        except Exception as e:
            return True, f"Command error: {e}"

        return True, None

    def get_history_up(self) -> str | None:
        """Get previous history entry."""
        if not self.history:
            return None
        if self.history_index > 0:
            self.history_index -= 1
        return (
            self.history[self.history_index]
            if self.history_index < len(self.history)
            else None
        )

    def get_history_down(self) -> str | None:
        """Get next history entry."""
        if not self.history:
            return None
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            return self.history[self.history_index]
        self.history_index = len(self.history)
        return ""

    def reset_history_index(self) -> None:
        """Reset history index to end."""
        self.history_index = len(self.history)
