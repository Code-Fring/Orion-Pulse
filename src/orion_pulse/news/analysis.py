"""News analysis service for Orion Pulse."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from orion_pulse.news.base import (
    NewsArticle,
    NewsCategory,
)
from orion_pulse.news.factory import NewsProviderFactory
from orion_pulse.storage.repositories import NewsRepository


class EventCategory(str, Enum):
    """Event category for structured analysis."""

    EARNINGS = "earnings"
    MERGERS_ACQUISITIONS = "mergers_acquisitions"
    PRODUCT_LAUNCH = "product_launch"
    REGULATORY = "regulatory"
    MACROECONOMIC = "macroeconomic"
    ANALYST_RATING = "analyst_rating"
    GENERAL = "general"


class DirectionalBias(str, Enum):
    """Directional interpretation of event."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EventAnalysis:
    """Structured event analysis result."""

    article_id: str
    symbol: str
    event_category: EventCategory
    headline: str
    summary: str
    source: str
    published_at: datetime
    directional_bias: DirectionalBias
    confidence: float
    relevance_score: float
    key_factors: list[str]
    raw_article: NewsArticle

    def to_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "symbol": self.symbol,
            "event_category": self.event_category.value,
            "headline": self.headline,
            "summary": self.summary,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "directional_bias": self.directional_bias.value,
            "confidence": self.confidence,
            "relevance_score": self.relevance_score,
            "key_factors": self.key_factors,
        }


class NewsAnalysisError(Exception):
    """Exception raised during news analysis."""

    pass


class NewsAnalysisService:
    """Service for analyzing news and extracting structured events."""

    # Keywords for directional bias detection
    POSITIVE_KEYWORDS = [
        "beat",
        "exceed",
        "surpass",
        "strong",
        "growth",
        "increase",
        "rise",
        "gain",
        "profit",
        "record",
        "high",
        "bullish",
        "upgrade",
        "outperform",
        "buy",
        "partnership",
        "deal",
        "contract",
        "approval",
        "launch",
        "expand",
    ]

    NEGATIVE_KEYWORDS = [
        "miss",
        "fall",
        "drop",
        "decline",
        "decrease",
        "loss",
        "weak",
        "low",
        "bearish",
        "downgrade",
        "underperform",
        "sell",
        "lawsuit",
        "investigation",
        "recall",
        "delay",
        "cancel",
        "cut",
        "layoff",
        "bankruptcy",
        "default",
    ]

    def __init__(
        self,
        provider_name: str = "newsapi",
        news_repo: NewsRepository | None = None,
    ):
        self.provider = NewsProviderFactory.get_provider(provider_name)
        self.news_repo = news_repo or NewsRepository()

    def fetch_and_analyze(
        self,
        symbol: str,
        days_back: int = 7,
        limit: int = 50,
        save_to_db: bool = True,
    ) -> list[EventAnalysis]:
        """Fetch news and analyze events for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)

        # Fetch news
        try:
            articles = self.provider.get_news(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                limit=limit,
            )
        except Exception as e:
            raise NewsAnalysisError(f"Failed to fetch news: {e}") from e

        # Analyze each article
        events = []
        for article in articles:
            event = self.analyze_article(article)
            if event:
                events.append(event)

        # Save to database if requested
        if save_to_db and events:
            for event in events:
                self.news_repo.save_article(
                    {
                        **event.raw_article.to_dict(),
                        "event_analysis": event.to_dict(),
                    }
                )

        return events

    def analyze_article(self, article: NewsArticle) -> EventAnalysis | None:
        """Analyze a single article and extract structured event."""
        if not article.symbol:
            return None

        # Determine event category
        category = self._map_category(article.category or NewsCategory.GENERAL)

        # Determine directional bias
        bias, confidence = self._determine_bias(article)

        # Extract key factors
        key_factors = self._extract_key_factors(article)

        # Calculate relevance
        relevance = article.relevance_score or 0.5

        return EventAnalysis(
            article_id=article.id,
            symbol=article.symbol,
            event_category=category,
            headline=article.headline,
            summary=article.summary or "",
            source=article.source,
            published_at=article.published_at,
            directional_bias=bias,
            confidence=confidence,
            relevance_score=relevance,
            key_factors=key_factors,
            raw_article=article,
        )

    def _map_category(self, news_category: NewsCategory) -> EventCategory:
        """Map news category to event category."""
        mapping = {
            NewsCategory.EARNINGS: EventCategory.EARNINGS,
            NewsCategory.MERGERS_ACQUISITIONS: EventCategory.MERGERS_ACQUISITIONS,
            NewsCategory.PRODUCT_LAUNCH: EventCategory.PRODUCT_LAUNCH,
            NewsCategory.REGULATORY: EventCategory.REGULATORY,
            NewsCategory.MACROECONOMIC: EventCategory.MACROECONOMIC,
            NewsCategory.ANALYST_RATING: EventCategory.ANALYST_RATING,
            NewsCategory.GENERAL: EventCategory.GENERAL,
        }
        return mapping.get(news_category, EventCategory.GENERAL)

    def _determine_bias(
        self,
        article: NewsArticle,
    ) -> tuple[DirectionalBias, float]:
        """Determine directional bias from article content."""
        text = f"{article.headline} {article.summary or ''}".lower()

        positive_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text)
        negative_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text)

        total = positive_count + negative_count

        if total == 0:
            return DirectionalBias.UNKNOWN, 0.1

        if positive_count > negative_count:
            confidence = min(0.5 + (positive_count - negative_count) * 0.1, 0.9)
            return DirectionalBias.POSITIVE, confidence
        elif negative_count > positive_count:
            confidence = min(0.5 + (negative_count - positive_count) * 0.1, 0.9)
            return DirectionalBias.NEGATIVE, confidence
        else:
            return DirectionalBias.NEUTRAL, 0.3

    def _extract_key_factors(self, article: NewsArticle) -> list[str]:
        """Extract key factors mentioned in article."""
        text = f"{article.headline} {article.summary or ''}".lower()
        factors = []

        factor_keywords = {
            "earnings": ["earnings", "eps", "revenue", "quarterly"],
            "guidance": ["guidance", "outlook", "forecast", "expect"],
            "product": ["launch", "product", "service", "feature"],
            "partnership": ["partnership", "deal", "agreement", "collaboration"],
            "regulation": ["sec", "fda", "regulation", "approval", "compliance"],
            "analyst": ["upgrade", "downgrade", "price target", "rating", "analyst"],
            "macro": ["fed", "interest rate", "inflation", "gdp", "economy"],
            "legal": ["lawsuit", "investigation", "court", "legal", "settlement"],
        }

        for factor, keywords in factor_keywords.items():
            if any(kw in text for kw in keywords):
                factors.append(factor)

        return factors

    def get_recent_events(
        self,
        symbol: str,
        days_back: int = 30,
        limit: int = 20,
    ) -> list[EventAnalysis]:
        """Get recent analyzed events from database."""
        articles = self.news_repo.get_latest_articles(symbol, limit=limit * 2)

        events = []
        for article_data in articles:
            # Reconstruct article
            article = NewsArticle(
                id=article_data["id"],
                headline=article_data["headline"],
                summary=article_data["summary"],
                url=article_data["url"],
                source=article_data["source"],
                published_at=article_data["published_at"],
                symbol=article_data["symbol"],
                category=NewsCategory(article_data["category"])
                if article_data.get("category")
                else None,
                relevance_score=article_data.get("relevance_score"),
                sentiment_score=article_data.get("sentiment_score"),
                entities=article_data.get("entities", []),
                raw_data=article_data.get("raw_data", {}),
            )

            event = self.analyze_article(article)
            if event:
                events.append(event)

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days_back)
        events = [e for e in events if e.published_at >= cutoff]

        return events[:limit]

    def summarize_events(self, events: list[EventAnalysis]) -> dict[str, Any]:
        """Summarize a list of events."""
        if not events:
            return {
                "total_events": 0,
                "by_category": {},
                "by_bias": {},
                "avg_confidence": 0.0,
                "avg_relevance": 0.0,
            }

        by_category: dict[str, int] = {}
        by_bias: dict[str, int] = {}

        for event in events:
            cat = event.event_category.value
            bias = event.directional_bias.value
            by_category[cat] = by_category.get(cat, 0) + 1
            by_bias[bias] = by_bias.get(bias, 0) + 1

        return {
            "total_events": len(events),
            "by_category": by_category,
            "by_bias": by_bias,
            "avg_confidence": sum(e.confidence for e in events) / len(events),
            "avg_relevance": sum(e.relevance_score for e in events) / len(events),
            "latest_event": events[0].to_dict() if events else None,
        }
