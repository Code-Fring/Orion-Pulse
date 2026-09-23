"""Factory for news providers."""

from orion_pulse.news.base import NewsProvider
from orion_pulse.news.newsapi_provider import MockNewsProvider, NewsAPIProvider


class NewsProviderFactory:
    """Factory for creating news providers."""

    _providers: dict[str, NewsProvider] = {}

    @classmethod
    def get_provider(cls, name: str = "newsapi") -> NewsProvider:
        """Get or create a provider instance."""
        if name not in cls._providers:
            if name == "newsapi":
                cls._providers[name] = NewsAPIProvider()
            elif name == "mock":
                cls._providers[name] = MockNewsProvider()
            else:
                raise ValueError(f"Unknown news provider: {name}")

        return cls._providers[name]

    @classmethod
    def get_available_providers(cls) -> list[str]:
        """Get list of available provider names."""
        providers = []
        for name in ["newsapi", "mock"]:
            provider = cls.get_provider(name)
            if provider.is_available():
                providers.append(name)
        return providers

    @classmethod
    def clear_cache(cls) -> None:
        """Clear provider cache (useful for testing)."""
        cls._providers.clear()
