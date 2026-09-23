"""Market data provider abstraction."""

from abc import ABC, abstractmethod
from datetime import date

import polars as pl


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data for a symbol."""
        ...

    @abstractmethod
    def get_latest_data(
        self,
        symbol: str,
        lookback_days: int = 365,
    ) -> pl.DataFrame:
        """Fetch latest available data for a symbol."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available/configured."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        ...


class DataProviderError(Exception):
    """Exception raised when data provider encounters an error."""

    def __init__(self, message: str, provider: str, symbol: str | None = None):
        self.provider = provider
        self.symbol = symbol
        super().__init__(message)


class DataNotFoundError(DataProviderError):
    """Exception raised when data is not found for a symbol."""

    pass


class ProviderUnavailableError(DataProviderError):
    """Exception raised when a provider is not available."""

    pass
