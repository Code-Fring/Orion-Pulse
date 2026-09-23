"""Configuration settings for Orion Pulse."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Data storage
    data_dir: Path = Field(
        default=Path.home() / ".orion-pulse" / "data",
        description="Directory for cached market data",
    )
    cache_dir: Path = Field(
        default=Path.home() / ".orion-pulse" / "cache",
        description="Directory for temporary cache files",
    )

    # Market data providers
    yfinance_enabled: bool = Field(
        default=True,
        description="Enable Yahoo Finance provider",
    )
    yfinance_timeout: int = Field(
        default=30,
        description="Timeout for Yahoo Finance requests in seconds",
    )

    # Analysis defaults
    default_lookback_days: int = Field(
        default=365,
        description="Default lookback period for analysis in days",
    )
    default_ma_periods: list[int] = Field(
        default_factory=lambda: [20, 50, 200],
        description="Default moving average periods",
    )
    volatility_window: int = Field(
        default=20,
        description="Window for rolling volatility calculation",
    )

    # Output
    json_output: bool = Field(
        default=False,
        description="Enable JSON output mode",
    )
    no_color: bool = Field(
        default=False,
        description="Disable colored output",
    )

    # API keys
    alpha_vantage_key: str | None = Field(
        default=None,
        description="Alpha Vantage API key",
    )
    polygon_key: str | None = Field(
        default=None,
        description="Polygon.io API key",
    )
    newsapi_key: str | None = Field(
        default=None,
        description="NewsAPI.org API key",
    )
    nvidia_api_key: str | None = Field(
        default=None,
        description="NVIDIA API key for LLM",
    )


settings = Settings()
