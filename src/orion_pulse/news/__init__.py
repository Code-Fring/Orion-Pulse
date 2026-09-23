"""News module for Orion Pulse."""

from orion_pulse.news.analysis import (
    DirectionalBias,
    EventAnalysis,
    EventCategory,
    NewsAnalysisService,
)
from orion_pulse.news.base import (
    NewsArticle,
    NewsCategory,
    NewsProvider,
    NewsProviderError,
)
from orion_pulse.news.factory import NewsProviderFactory
from orion_pulse.news.newsapi_provider import MockNewsProvider, NewsAPIProvider

__all__ = [
    "NewsProvider",
    "NewsProviderError",
    "NewsArticle",
    "NewsCategory",
    "NewsProviderFactory",
    "NewsAPIProvider",
    "MockNewsProvider",
    "NewsAnalysisService",
    "EventAnalysis",
    "EventCategory",
    "DirectionalBias",
]
