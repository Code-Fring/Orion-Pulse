"""Provider factory for market data."""

from orion_pulse.data.providers.base import MarketDataProvider
from orion_pulse.data.providers.yfinance_provider import YFinanceProvider


class ProviderFactory:
    """Factory for creating market data providers."""

    _providers: dict[str, MarketDataProvider] = {}

    @classmethod
    def get_provider(cls, name: str = "yfinance", **kwargs) -> MarketDataProvider:
        """Get or create a provider instance."""
        if name not in cls._providers:
            if name == "yfinance":
                cls._providers[name] = YFinanceProvider(**kwargs)
            else:
                raise ValueError(f"Unknown provider: {name}")

        return cls._providers[name]

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        providers = []
        for name in ["yfinance"]:
            provider = cls.get_provider(name)
            if provider.is_available():
                providers.append(name)
        return providers

    @classmethod
    def clear_cache(cls) -> None:
        """Clear provider cache (useful for testing)."""
        cls._providers.clear()

    @classmethod
    def create_provider(cls, name: str, **kwargs) -> MarketDataProvider:
        """Create a new provider instance (not cached)."""
        if name == "yfinance":
            return YFinanceProvider(**kwargs)
        raise ValueError(f"Unknown provider: {name}")
