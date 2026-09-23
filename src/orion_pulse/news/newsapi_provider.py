"""NewsAPI provider implementation."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

import requests

from orion_pulse.config.settings import settings
from orion_pulse.news.base import (
    NewsArticle,
    NewsCategory,
    NewsProvider,
    NewsProviderError,
)

logger = logging.getLogger(__name__)


class NewsAPIProvider(NewsProvider):
    """NewsAPI.org provider implementation."""

    BASE_URL = "https://newsapi.org/v2"

    def __init__(
        self,
        api_key: str | None = None,
        timeout: int = 30,
    ):
        self.api_key = api_key or settings.newsapi_key
        self.timeout = timeout
        self._enabled = bool(self.api_key)

    def get_provider_name(self) -> str:
        return "newsapi"

    def is_available(self) -> bool:
        return self._enabled

    def get_news(
        self,
        symbol: str | None = None,
        keywords: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Fetch news articles from NewsAPI."""
        if not self._enabled:
            raise NewsProviderError(
                "NewsAPI provider is not configured (missing API key)",
                provider=self.get_provider_name(),
            )

        # Build query
        query_parts = []
        if symbol:
            # Try to get company name for better search
            query_parts.append(symbol)
        if keywords:
            query_parts.extend(keywords)

        query = " OR ".join(query_parts) if query_parts else "stocks"

        # Default to last 7 days if no dates specified
        if end_date is None:
            end_date = datetime.now()
        if start_date is None:
            start_date = end_date - timedelta(days=7)

        params: dict[str, Any] = {
            "q": query,
            "from": start_date.strftime("%Y-%m-%d"),
            "to": end_date.strftime("%Y-%m-%d"),
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": min(limit, 100),  # NewsAPI max is 100
            "apiKey": self.api_key,
        }

        try:
            response = requests.get(
                f"{self.BASE_URL}/everything",
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("status") != "ok":
                raise NewsProviderError(
                    f"NewsAPI error: {data.get('message', 'Unknown error')}",
                    provider=self.get_provider_name(),
                )

            articles = []
            for item in data.get("articles", [])[:limit]:
                article = self._parse_article(item, symbol)
                if article:
                    articles.append(article)

            return articles

        except requests.exceptions.Timeout as e:
            raise NewsProviderError(
                "NewsAPI request timed out",
                provider=self.get_provider_name(),
            ) from e
        except requests.exceptions.RequestException as e:
            logger.error(f"NewsAPI request error: {e}")
            raise NewsProviderError(
                f"Failed to fetch news: {e}",
                provider=self.get_provider_name(),
            ) from e

    def _parse_article(
        self, item: dict[str, Any], symbol: str | None
    ) -> NewsArticle | None:
        """Parse NewsAPI article into NewsArticle."""
        try:
            published_at = datetime.fromisoformat(
                item["publishedAt"].replace("Z", "+00:00")
            )
        except (ValueError, KeyError):
            published_at = datetime.now()

        # Simple relevance scoring based on symbol mention
        relevance = self._calculate_relevance(item, symbol)

        # Simple category detection
        category = self._detect_category(item)

        return NewsArticle(
            id=str(uuid.uuid4()),
            headline=item.get("title", ""),
            summary=item.get("description"),
            url=item.get("url", ""),
            source=item.get("source", {}).get("name", "Unknown"),
            published_at=published_at,
            symbol=symbol.upper() if symbol else None,
            category=category,
            relevance_score=relevance,
            sentiment_score=None,  # Would need sentiment analysis
            entities=self._extract_entities(item),
            raw_data=item,
        )

    def _calculate_relevance(self, item: dict[str, Any], symbol: str | None) -> float:
        """Calculate relevance score for an article."""
        if not symbol:
            return 0.5

        text = f"{item.get('title', '')} {item.get('description', '')}".lower()
        symbol_lower = symbol.lower()

        # Check for symbol mentions
        if symbol_lower in text:
            return 0.9
        if f"${symbol_lower}" in text:
            return 0.95
        if symbol_lower.upper() in text:
            return 0.85

        return 0.3

    def _detect_category(self, item: dict[str, Any]) -> NewsCategory | None:
        """Detect news category from article content."""
        text = f"{item.get('title', '')} {item.get('description', '')}".lower()

        categories = {
            NewsCategory.EARNINGS: [
                "earnings",
                "quarterly results",
                "eps",
                "revenue beat",
                "revenue miss",
            ],
            NewsCategory.MERGERS_ACQUISITIONS: [
                "merger",
                "acquisition",
                "buyout",
                "takeover",
                "acquire",
            ],
            NewsCategory.PRODUCT_LAUNCH: [
                "launch",
                "unveil",
                "announce",
                "release",
                "new product",
            ],
            NewsCategory.REGULATORY: [
                "sec",
                "regulation",
                "fda",
                "approval",
                "investigation",
                "lawsuit",
            ],
            NewsCategory.MACROECONOMIC: [
                "fed",
                "interest rate",
                "inflation",
                "gdp",
                "unemployment",
                "cpi",
            ],
            NewsCategory.ANALYST_RATING: [
                "upgrade",
                "downgrade",
                "price target",
                "rating",
                "analyst",
            ],
        }

        for category, keywords in categories.items():
            if any(kw in text for kw in keywords):
                return category

        return NewsCategory.GENERAL

    def _extract_entities(self, item: dict[str, Any]) -> list[str]:
        """Extract named entities (simplified)."""
        # In a real implementation, this would use NER
        # For now, return empty list
        return []


class MockNewsProvider(NewsProvider):
    """Mock news provider for testing and development."""

    def __init__(self, articles: list[NewsArticle] | None = None):
        self._articles = articles or []
        self._enabled = True

    def get_provider_name(self) -> str:
        return "mock"

    def is_available(self) -> bool:
        return self._enabled

    def get_news(
        self,
        symbol: str | None = None,
        keywords: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 50,
    ) -> list[NewsArticle]:
        """Return mock articles filtered by symbol if provided."""
        filtered = self._articles

        if symbol:
            filtered = [a for a in filtered if a.symbol == symbol.upper()]

        if start_date:
            filtered = [a for a in filtered if a.published_at >= start_date]

        if end_date:
            filtered = [a for a in filtered if a.published_at <= end_date]

        return filtered[:limit]

    def set_articles(self, articles: list[NewsArticle]) -> None:
        """Set mock articles."""
        self._articles = articles
