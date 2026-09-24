"""Provider management for Orion Pulse TUI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual import on
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Header, Input, Label, Static

from orion_pulse.data.providers.factory import ProviderFactory
from orion_pulse.tui.config_manager import get_config

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
        self._load_configured_providers()

    def _load_configured_providers(self) -> None:
        """Load providers from config with saved credentials."""
        config = get_config()
        for provider_name, provider_config in config.providers.items():
            if provider_config.enabled:
                self._initialize_provider(provider_name, provider_config.credentials)

    def _initialize_provider(
        self, provider_name: str, credentials: dict[str, str]
    ) -> bool:
        """Initialize a provider with credentials."""
        try:
            if provider_name == "yfinance":
                provider = ProviderFactory.create_provider(provider_name)
                self.providers[provider_name] = provider
                self.active_provider = provider_name
                return True
            # Future providers (alpha_vantage, polygon, etc.) would go here
            return False
        except Exception:
            return False

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

        return f"Connected to {self.active_provider}"

    def connect_provider(self, provider_name: str, credentials: dict[str, str]) -> bool:
        """Connect to a provider with credentials and save to config."""
        try:
            # Save credentials securely
            config = get_config()
            if provider_name not in config.providers:
                config.providers[provider_name] = {
                    "name": provider_name,
                    "enabled": True,
                    "credentials": {},
                }
            for key, value in credentials.items():
                config.set_secure_credential(provider_name, key, value)

            # Initialize provider
            success = self._initialize_provider(provider_name, credentials)
            if success:
                # Save config to file
                config.save_to_file(Path.home() / ".orion-pulse" / "config.json")
            return success
        except Exception:
            return False


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
        for child in list(container.children)[2:]:  # Skip title and error message
            child.remove()

        # Add fields based on provider
        if self.provider_name == "yfinance":
            # Yahoo Finance doesn't require credentials
            container.mount(
                Label("Yahoo Finance doesn't require credentials", id="no-creds")
            )
            container.mount(
                Label("Press Connect to enable Yahoo Finance provider", id="yf-hint")
            )
        elif self.provider_name == "newsapi":
            # Check for existing credentials
            config = get_config()
            existing_key = config.get_secure_credential("newsapi", "api_key")

            container.mount(Label("NewsAPI.org API Key:", id="api-key-label"))
            api_input = Input(
                placeholder="Enter NewsAPI.org API key",
                id="api-key-input",
            )
            if existing_key:
                api_input.value = "•" * len(existing_key)
                api_input.placeholder = (
                    "API key already configured (leave blank to keep)"
                )
            container.mount(api_input)
            container.mount(
                Label(
                    "Get a free API key at https://newsapi.org/register",
                    id="newsapi-hint",
                )
            )
        else:
            # Generic credential fields
            container.mount(Label("API Key:", id="api-key-label"))
            container.mount(Input(placeholder="Enter API key", id="api-key-input"))

    @on(Button.Pressed, "#connect-button")
    def connect_pressed(self) -> None:
        """Handle connect button press."""
        error_label = self.query_one("#error-message", Static)
        error_label.update("")

        try:
            if self.provider_name == "yfinance":
                # Yahoo Finance just needs to be enabled
                self.credentials = {}
            elif self.provider_name == "newsapi":
                api_key_input = self.query_one("#api-key-input", Input)
                api_key = api_key_input.value.strip()
                if not api_key:
                    error_label.update("API key is required")
                    return
                self.credentials = {"api_key": api_key}
            else:
                api_key_input = self.query_one("#api-key-input", Input)
                api_key = api_key_input.value.strip()
                if not api_key:
                    error_label.update("API key is required")
                    return
                self.credentials = {"api_key": api_key}

            # Connect and save
            success = self.provider_manager.connect_provider(
                self.provider_name, self.credentials
            )
            if success:
                self.dismiss(self.credentials)
            else:
                error_label.update("Connection failed: unable to initialize provider")

        except Exception as e:
            error_label.update(f"Connection failed: {str(e)}")

    @on(Button.Pressed, "#cancel-button")
    def cancel_pressed(self) -> None:
        """Handle cancel button press."""
        self.dismiss(None)

    def action_dismiss(self, result: dict[str, str] | None = None) -> None:
        """Dismiss the screen."""
        self.dismiss(result)
