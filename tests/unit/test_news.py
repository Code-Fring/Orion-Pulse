"""Unit tests for news module."""

from datetime import datetime
from unittest.mock import Mock, patch

import pytest

from orion_pulse.news.analysis import (
    DirectionalBias,
    EventAnalysis,
    EventCategory,
    NewsAnalysisService,
)
from orion_pulse.news.base import (
    NewsArticle,
    NewsCategory,
    NewsProviderError,
)
from orion_pulse.news.factory import NewsProviderFactory
from orion_pulse.news.newsapi_provider import MockNewsProvider, NewsAPIProvider


class TestNewsArticle:
    def test_create_news_article(self):
        article = NewsArticle(
            id="test-1",
            headline="Test Headline",
            summary="Test Summary",
            url="https://example.com",
            source="Test Source",
            published_at=datetime(2024, 1, 15, 10, 0),
            symbol="TEST",
            category=NewsCategory.EARNINGS,
            relevance_score=0.9,
            sentiment_score=0.5,
            entities=["TEST"],
        )

        assert article.symbol == "TEST"
        assert article.category == NewsCategory.EARNINGS

    def test_to_dict(self):
        article = NewsArticle(
            id="test-1",
            headline="Test Headline",
            summary="Test Summary",
            url="https://example.com",
            source="Test Source",
            published_at=datetime(2024, 1, 15, 10, 0),
            symbol="TEST",
            category=NewsCategory.EARNINGS,
        )

        d = article.to_dict()
        assert d["symbol"] == "TEST"
        assert d["category"] == "earnings"


class TestMockNewsProvider:
    def test_get_news(self):
        articles = [
            NewsArticle(
                id="1",
                headline="Test 1",
                summary="Summary 1",
                url="https://example.com/1",
                source="Source 1",
                published_at=datetime(2024, 1, 15),
                symbol="TEST",
            ),
            NewsArticle(
                id="2",
                headline="Test 2",
                summary="Summary 2",
                url="https://example.com/2",
                source="Source 2",
                published_at=datetime(2024, 1, 16),
                symbol="AAPL",
            ),
        ]

        provider = MockNewsProvider(articles=articles)

        # Test filtering by symbol
        result = provider.get_news(symbol="TEST")
        assert len(result) == 1
        assert result[0].symbol == "TEST"

        # Test no filter
        result = provider.get_news()
        assert len(result) == 2

    def test_is_available(self):
        provider = MockNewsProvider()
        assert provider.is_available() is True


class TestNewsAPIProvider:
    def test_get_provider_name(self):
        with patch("orion_pulse.news.newsapi_provider.settings") as mock_settings:
            mock_settings.newsapi_key = "test-key"
            provider = NewsAPIProvider()
            assert provider.get_provider_name() == "newsapi"

    def test_is_available_with_key(self):
        with patch("orion_pulse.news.newsapi_provider.settings") as mock_settings:
            mock_settings.newsapi_key = "test-key"
            provider = NewsAPIProvider()
            assert provider.is_available() is True

    def test_is_available_without_key(self):
        with patch("orion_pulse.news.newsapi_provider.settings") as mock_settings:
            mock_settings.newsapi_key = None
            provider = NewsAPIProvider()
            assert provider.is_available() is False

    def test_get_news_without_key_raises(self):
        with patch("orion_pulse.news.newsapi_provider.settings") as mock_settings:
            mock_settings.newsapi_key = None
            provider = NewsAPIProvider()

            with pytest.raises(NewsProviderError, match="not configured"):
                provider.get_news(symbol="TEST")


class TestNewsProviderFactory:
    def test_get_newsapi_provider(self):
        NewsProviderFactory.clear_cache()
        with patch("orion_pulse.news.newsapi_provider.settings") as mock_settings:
            mock_settings.newsapi_key = "test-key"
            provider = NewsProviderFactory.get_provider("newsapi")
            assert isinstance(provider, NewsAPIProvider)

    def test_get_mock_provider(self):
        NewsProviderFactory.clear_cache()
        provider = NewsProviderFactory.get_provider("mock")
        assert isinstance(provider, MockNewsProvider)

    def test_get_unknown_provider_raises(self):
        NewsProviderFactory.clear_cache()
        with pytest.raises(ValueError, match="Unknown news provider"):
            NewsProviderFactory.get_provider("unknown")


class TestNewsAnalysisService:
    def test_analyze_article_positive_bias(self):
        provider = MockNewsProvider()
        service = NewsAnalysisService(provider_name="mock")

        article = NewsArticle(
            id="1",
            headline="Company beats earnings expectations with strong growth",
            summary="Revenue up 20% year over year",
            url="https://example.com",
            source="Test",
            published_at=datetime(2024, 1, 15),
            symbol="TEST",
            category=NewsCategory.EARNINGS,
            relevance_score=0.9,
        )

        event = service.analyze_article(article)

        assert event is not None
        assert event.symbol == "TEST"
        assert event.event_category == EventCategory.EARNINGS
        assert event.directional_bias == DirectionalBias.POSITIVE
        assert event.confidence > 0.5

    def test_analyze_article_negative_bias(self):
        provider = MockNewsProvider()
        service = NewsAnalysisService(provider_name="mock")

        article = NewsArticle(
            id="1",
            headline="Company misses earnings, revenue declines sharply",
            summary="Profit warning issued for next quarter",
            url="https://example.com",
            source="Test",
            published_at=datetime(2024, 1, 15),
            symbol="TEST",
            category=NewsCategory.EARNINGS,
        )

        event = service.analyze_article(article)

        assert event.directional_bias == DirectionalBias.NEGATIVE

    def test_analyze_article_neutral(self):
        provider = MockNewsProvider()
        service = NewsAnalysisService(provider_name="mock")

        article = NewsArticle(
            id="1",
            headline="Company announces new office location",
            summary="New headquarters to open in 2025",
            url="https://example.com",
            source="Test",
            published_at=datetime(2024, 1, 15),
            symbol="TEST",
            category=NewsCategory.GENERAL,
        )

        event = service.analyze_article(article)

        # Should be neutral or unknown for non-financial news
        assert event.directional_bias in [
            DirectionalBias.NEUTRAL,
            DirectionalBias.UNKNOWN,
        ]

    def test_summarize_events(self):
        provider = MockNewsProvider()
        service = NewsAnalysisService(provider_name="mock")

        events = [
            EventAnalysis(
                article_id="1",
                symbol="TEST",
                event_category=EventCategory.EARNINGS,
                headline="Earnings beat",
                summary="Beat",
                source="Test",
                published_at=datetime(2024, 1, 15),
                directional_bias=DirectionalBias.POSITIVE,
                confidence=0.8,
                relevance_score=0.9,
                key_factors=["earnings"],
                raw_article=Mock(),
            ),
            EventAnalysis(
                article_id="2",
                symbol="TEST",
                event_category=EventCategory.PRODUCT_LAUNCH,
                headline="Product launch",
                summary="Launch",
                source="Test",
                published_at=datetime(2024, 1, 16),
                directional_bias=DirectionalBias.NEUTRAL,
                confidence=0.6,
                relevance_score=0.7,
                key_factors=["product"],
                raw_article=Mock(),
            ),
        ]

        summary = service.summarize_events(events)

        assert summary["total_events"] == 2
        assert summary["by_category"]["earnings"] == 1
        assert summary["by_category"]["product_launch"] == 1
        assert summary["by_bias"]["positive"] == 1
        assert summary["by_bias"]["neutral"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
