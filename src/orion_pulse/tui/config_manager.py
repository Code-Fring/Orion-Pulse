"""Configuration management for Orion Pulse TUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import keyring

    HAS_KEYRING = True
except ImportError:
    HAS_KEYRING = False
    keyring = None

from cryptography.fernet import Fernet
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderConfig(BaseModel):
    """Configuration for a data provider."""

    name: str
    enabled: bool = True
    credentials: dict[str, str] = Field(default_factory=dict)


class TUIConfig(BaseModel):
    """TUI-specific configuration."""

    theme: str = "dark"
    max_history: int = 1000
    refresh_interval: int = 30  # seconds


class AppConfig(BaseSettings):
    """Main application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Data directories
    data_dir: Path = Field(default=Path.home() / ".orion-pulse" / "data")
    cache_dir: Path = Field(default=Path.home() / ".orion-pulse" / "cache")

    # Analysis defaults
    default_lookback_days: int = 365
    default_ma_periods: str = "20,50,200"
    volatility_window: int = 20

    # Output settings
    json_output: bool = False
    no_color: bool = False

    # Provider configurations
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    # TUI configuration
    tui: TUIConfig = Field(default_factory=TUIConfig)

    # Encryption key for secure storage
    encryption_key: str = ""

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize encryption
        if not self.encryption_key:
            self.encryption_key = Fernet.generate_key().decode()

    def get_encryption_cipher(self) -> Fernet:
        """Get cipher for encrypting/decrypting sensitive data."""
        return Fernet(self.encryption_key.encode())

    def save_to_file(self, filepath: Path | str) -> None:
        """Save configuration to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Encrypt sensitive data before saving
        config_dict = self.model_dump()
        if "encryption_key" in config_dict:
            del config_dict["encryption_key"]

        # Encrypt provider credentials
        for _provider_name, provider_config in config_dict.get("providers", {}).items():
            if isinstance(provider_config, dict) and "credentials" in provider_config:
                encrypted_creds = {}
                cipher = self.get_encryption_cipher()
                for key, value in provider_config["credentials"].items():
                    encrypted_creds[key] = cipher.encrypt(value.encode()).decode()
                provider_config["credentials"] = encrypted_creds

        with open(filepath, "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def load_from_file(cls, filepath: Path | str) -> AppConfig:
        """Load configuration from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            return cls()

        with open(filepath) as f:
            data = json.load(f)

        # Decrypt provider credentials
        if "providers" in data:
            for _provider_name, provider_config in data["providers"].items():
                if (
                    isinstance(provider_config, dict)
                    and "credentials" in provider_config
                ):
                    decrypted_creds = {}
                    # We need the encryption key to decrypt, which should be loaded separately
                    # In a real implementation, we'd handle this more carefully
                    for key, value in provider_config["credentials"].items():
                        decrypted_creds[key] = value  # Placeholder
                    provider_config["credentials"] = decrypted_creds

        return cls(**data)

    def get_secure_credential(self, provider: str, key: str) -> str | None:
        """Get a secure credential for a provider."""
        # Try keyring first if available
        if HAS_KEYRING:
            try:
                cred = keyring.get_password(f"orion-pulse-{provider}", key)
                if cred:
                    return cred
            except Exception:
                pass

        # Fall back to config-stored credentials
        if provider in self.providers:
            provider_config = self.providers[provider]
            if key in provider_config.credentials:
                # Decrypt the credential
                try:
                    cipher = self.get_encryption_cipher()
                    encrypted_value = provider_config.credentials[key]
                    return cipher.decrypt(encrypted_value.encode()).decode()
                except Exception:
                    return provider_config.credentials[
                        key
                    ]  # Return as-is if decryption fails

        return None

    def set_secure_credential(self, provider: str, key: str, value: str) -> None:
        """Set a secure credential for a provider."""
        # Try keyring first if available
        if HAS_KEYRING:
            try:
                keyring.set_password(f"orion-pulse-{provider}", key, value)
                return
            except Exception:
                pass

        # Fall back to config storage
        if provider not in self.providers:
            self.providers[provider] = ProviderConfig(name=provider)

        # Encrypt the credential before storing
        try:
            cipher = self.get_encryption_cipher()
            encrypted_value = cipher.encrypt(value.encode()).decode()
            self.providers[provider].credentials[key] = encrypted_value
        except Exception:
            # If encryption fails, store as plain text (not ideal but better than nothing)
            self.providers[provider].credentials[key] = value


# Global config instance
config = AppConfig()


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    return config


def initialize_config(config_path: Path | str | None = None) -> AppConfig:
    """Initialize configuration from file or create default."""
    global config

    if config_path is None:
        config_path = Path.home() / ".orion-pulse" / "config.json"
    else:
        config_path = Path(config_path)

    if config_path.exists():
        config = AppConfig.load_from_file(config_path)
    else:
        config = AppConfig()
        config.save_to_file(config_path)

    return config
