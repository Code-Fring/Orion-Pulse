"""News provider abstraction for Orion Pulse."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class NewsCategory(str, Enum):
    """News category classification."""

    EARNINGS = "earnings"
    MERGERS_ACQUISITIONS = "mergers_acquisitions"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    MACROECONOMIC = "macroeconomic"
    ANALYST_RATING = "analyst_rating"
    GENERAL = "general"


@dataclass(frozen=True)
class NewsArticle:
    """Normalized news article."""

    id: str
    headline: str
    summary: str | None
    url: str
    source: str
    published_at: datetime
    symbol: str | None = None
    category: NewsCategory | None = None
    relevance_score: float | None = None
    sentiment_score: float | None = None
    entities: list[str] = field(default_factory=list)
    raw_data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "id": self.id,
            "symbol": self.symbol,
            "headline": self.headline,
            "summary": self.summary,
            "url": self.url,
            "source": self.source,
            "published_at": self.published_at,
            "relevance_score": self.relevance_score,
            "category": self.category.value if self.category else None,
            "sentiment_score": self.sentiment_score,
            "entities": self.entities,
            "raw_data": self.raw_data,
        }


class NewsProviderError(Exception):
    """Exception raised when news provider encounters an error."""

    def __init__(self, message: str, provider: str, symbol: str | None = None):
        self.provider = provider
        self.symbol = symbol
        super().__init__(message)


class NewsProvider(ABC):
    """Abstract base class for news providers."""

    @abstractmethod
    def get_news(
        self,
        symbol: str | None = None,
        keywords: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Fetch news articles."""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available/configured."""
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name."""
        ...
