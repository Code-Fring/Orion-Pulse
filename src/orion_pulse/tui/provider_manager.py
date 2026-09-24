"""Provider management for Orion Pulse TUI."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from textual import on
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Input, Label, Static

from orion_pulse.data.providers.factory import ProviderFactory

if TYPE_CHECKING:
    from textual.app import ComposeResult

    from orion_pulse.data.providers.base import MarketDataProvider
    from orion_pulse.tui.app import OrionPulseApp


class ProviderManager:
    """Manage data providers for the TUI."""

    def __init__(self, app: OrionPulseApp) -> None:
        self.app = app
        self.providers: dict[str, MarketDataProvider] = {}
        self.active_provider: str | None = None

    def get_available_providers(self) -> list[str]:
        """Get list of available provider names."""
        return ProviderFactory.get_available_providers()

    def get_active_provider(self) -> MarketDataProvider | None:
        """Get the currently active provider."""
        if self.active_provider and self.active_provider in self.providers:
            return self.providers[self.active_provider]
        return None

    def set_active_provider(self, name: str) -> bool:
        """Set the active provider."""
        if name in self.providers:
            self.active_provider = name
            return True
        return False

    def add_provider(self, name: str, provider: MarketDataProvider) -> None:
        """Add a provider to the manager."""
        self.providers[name] = provider

    def remove_provider(self, name: str) -> bool:
        """Remove a provider from the manager."""
        if name in self.providers:
            del self.providers[name]
            if self.active_provider == name:
                self.active_provider = None
            return True
        return False

    def get_provider_status(self) -> str:
        """Get status string for the active provider."""
        if not self.active_provider:
            return "No provider configured"

        provider = self.get_active_provider()
        if not provider:
            return f"Provider '{self.active_provider}' not initialized"

        # Try to get a quick status check
        try:
            # This is a placeholder - actual implementation would depend on the provider
            return f"Connected to {self.active_provider}"
        except Exception:
            return f"Provider '{self.active_provider}' configured but not connected"


class ConnectScreen(ModalScreen[dict[str, Any] | None]):
    """Screen for connecting to a data provider."""

    BINDINGS = [
        ("escape", "dismiss", "Cancel"),
    ]

    def __init__(self, provider_name: str, provider_manager: ProviderManager) -> None:
        super().__init__()
        self.provider_name = provider_name
        self.provider_manager = provider_manager
        self.credentials: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield Header()
        yield Vertical(
            Label(f"Configure {self.provider_name}", id="title"),
            Static("", id="error-message"),
            id="connect-container",
        )
        yield Horizontal(
            Button("Connect", variant="primary", id="connect-button"),
            Button("Cancel", variant="error", id="cancel-button"),
            id="buttons",
        )

    def on_mount(self) -> None:
        """Initialize the screen when mounted."""
        self.populate_fields()

    def populate_fields(self) -> None:
        """Populate credential fields based on provider."""
        container = self.query_one("#connect-container", Vertical)

        # Clear existing fields
        for child in container.children[2:]:  # Skip title and error message
            child.remove()

        # Add fields based on provider
        if self.provider_name == "yfinance":
            # Yahoo Finance doesn't require credentials
            container.mount(
                Label("Yahoo Finance doesn't require credentials", id="no-creds")
            )
        else:
            # Add generic credential fields
            container.mount(Label("API Key:", id="api-key-label"))
            container.mount(Input(placeholder="Enter API key", id="api-key-input"))

    @on(Button.Pressed, "#connect-button")
    def connect_pressed(self) -> None:
        """Handle connect button press."""
        error_label = self.query_one("#error-message", Static)
        error_label.update("")

        try:
            # Validate and save credentials
            if self.provider_name != "yfinance":
                api_key_input = self.query_one("#api-key-input", Input)
                api_key = api_key_input.value.strip()
                if not api_key:
                    error_label.update("API key is required")
                    return
                self.credentials["api_key"] = api_key

            # Create provider instance
            provider = ProviderFactory.get_provider(self.provider_name)

            # Store credentials securely
            if self.credentials:
                # In a real implementation, we'd use keyring or similar
                # For now, we'll just store in memory
                for key, value in self.credentials.items():
                    os.environ[
                        f"ORION_PULSE_{self.provider_name.upper()}_{key.upper()}"
                    ] = value

            # Add to provider manager
            self.provider_manager.add_provider(self.provider_name, provider)
            self.provider_manager.set_active_provider(self.provider_name)

            # Dismiss with success
            self.dismiss(self.credentials)
        except Exception as e:
            error_label.update(f"Connection failed: {str(e)}")

    @on(Button.Pressed, "#cancel-button")
    def cancel_pressed(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)

    def action_dismiss(self, result: dict[str, str] | None = None) -> None:
        """Dismiss the screen."""
        self.dismiss(result)
