"""Unit tests for storage layer."""

import tempfile
from datetime import date, datetime
from pathlib import Path

import polars as pl
import pytest

from orion_pulse.storage.database import Database
from orion_pulse.storage.repositories import (
    AnalysisRepository,
    BacktestRepository,
    ForecastRepository,
    MarketDataRepository,
    NewsRepository,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.duckdb"
        db = Database(db_path)
        yield db
        db.close()


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    return pl.DataFrame(
        {
            "date": [date(2024, 1, i) for i in range(1, 11)],
            "open": [100.0 + i for i in range(10)],
            "high": [102.0 + i for i in range(10)],
            "low": [99.0 + i for i in range(10)],
            "close": [101.0 + i for i in range(10)],
            "volume": [1000000 + i * 10000 for i in range(10)],
            "adjusted_close": [100.5 + i for i in range(10)],
            "symbol": ["TEST"] * 10,
        }
    )


class TestDatabase:
    def test_database_creation(self, temp_db):
        assert temp_db.db_path.exists()

    def test_execute_query(self, temp_db):
        result = temp_db.execute("SELECT 1 as test")
        df = result.df()
        assert len(df) == 1
        assert df["test"][0] == 1

    def test_tables_exist(self, temp_db):
        tables = temp_db.execute("SHOW TABLES").df()
        table_names = tables["name"].tolist()
        assert "market_data" in table_names
        assert "news_articles" in table_names
        assert "forecasts" in table_names
        assert "backtest_results" in table_names
        assert "analysis_results" in table_names


class TestMarketDataRepository:
    def test_save_and_get_market_data(self, temp_db, sample_market_data):
        repo = MarketDataRepository(temp_db)

        count = repo.save_market_data("TEST", sample_market_data, "test_provider")
        assert count == 10

        result = repo.get_market_data("TEST", provider="test_provider")
        assert len(result) == 10
        assert result["symbol"][0] == "TEST"

    def test_get_market_data_with_date_filter(self, temp_db, sample_market_data):
        repo = MarketDataRepository(temp_db)
        repo.save_market_data("TEST", sample_market_data, "test_provider")

        result = repo.get_market_data(
            "TEST",
            start_date=date(2024, 1, 5),
            end_date=date(2024, 1, 8),
            provider="test_provider",
        )
        assert len(result) == 4

    def test_get_latest_date(self, temp_db, sample_market_data):
        repo = MarketDataRepository(temp_db)
        repo.save_market_data("TEST", sample_market_data, "test_provider")

        latest = repo.get_latest_date("TEST", "test_provider")
        assert latest == date(2024, 1, 10)

    def test_get_symbols(self, temp_db, sample_market_data):
        repo = MarketDataRepository(temp_db)
        repo.save_market_data("TEST", sample_market_data, "test_provider")
        repo.save_market_data("AAPL", sample_market_data, "test_provider")

        symbols = repo.get_symbols("test_provider")
        assert "TEST" in symbols
        assert "AAPL" in symbols

    def test_delete_symbol(self, temp_db, sample_market_data):
        repo = MarketDataRepository(temp_db)
        repo.save_market_data("TEST", sample_market_data, "test_provider")

        count = repo.delete_symbol("TEST", "test_provider")
        assert count == 10

        result = repo.get_market_data("TEST", provider="test_provider")
        assert result.is_empty()


class TestNewsRepository:
    def test_save_and_get_article(self, temp_db):
        repo = NewsRepository(temp_db)

        article = {
            "id": "test-1",
            "symbol": "TEST",
            "headline": "Test Headline",
            "summary": "Test Summary",
            "url": "https://example.com/test",
            "source": "Test Source",
            "published_at": datetime(2024, 1, 15, 10, 0),
            "relevance_score": 0.9,
            "category": "earnings",
            "sentiment_score": 0.5,
            "entities": ["TEST", "earnings"],
        }

        article_id = repo.save_article(article)
        assert article_id == "test-1"

        articles = repo.get_articles(symbol="TEST")
        assert len(articles) == 1
        assert articles[0]["headline"] == "Test Headline"

    def test_save_multiple_articles(self, temp_db):
        repo = NewsRepository(temp_db)

        articles = [
            {
                "symbol": "TEST",
                "headline": f"Headline {i}",
                "summary": f"Summary {i}",
                "url": f"https://example.com/{i}",
                "source": "Test Source",
                "published_at": datetime(2024, 1, 15, 10, i),
            }
            for i in range(5)
        ]

        count = repo.save_articles(articles)
        assert count == 5

        results = repo.get_articles(symbol="TEST", limit=10)
        assert len(results) == 5


class TestForecastRepository:
    def test_save_and_get_forecast(self, temp_db):
        repo = ForecastRepository(temp_db)

        forecast = {
            "id": "forecast-1",
            "symbol": "TEST",
            "model_name": "test_model",
            "horizon_days": 5,
            "forecast_date": date(2024, 1, 15),
            "target_date": date(2024, 1, 22),
            "expected_return": 0.02,
            "prob_positive": 0.6,
            "p10": -0.05,
            "p25": -0.01,
            "p50": 0.02,
            "p75": 0.05,
            "p90": 0.09,
            "volatility": 0.15,
            "confidence": 0.7,
            "factors": {"method": "test"},
            "input_features": {"mean": 0.001},
        }

        forecast_id = repo.save_forecast(forecast)
        assert forecast_id == "forecast-1"

        result = repo.get_forecast(forecast_id)
        assert result is not None
        assert result["symbol"] == "TEST"
        assert result["expected_return"] == 0.02

    def test_get_forecasts_with_filters(self, temp_db):
        repo = ForecastRepository(temp_db)

        for i in range(3):
            repo.save_forecast(
                {
                    "id": f"f{i}",
                    "symbol": "TEST",
                    "model_name": "model_a" if i < 2 else "model_b",
                    "horizon_days": 5,
                    "forecast_date": date(2024, 1, 15),
                    "target_date": date(2024, 1, 22),
                    "expected_return": 0.01 * i,
                    "prob_positive": 0.5,
                }
            )

        results = repo.get_forecasts(symbol="TEST", model_name="model_a")
        assert len(results) == 2

    def test_update_evaluation(self, temp_db):
        repo = ForecastRepository(temp_db)

        repo.save_forecast(
            {
                "id": "eval-test",
                "symbol": "TEST",
                "model_name": "test",
                "horizon_days": 5,
                "forecast_date": date(2024, 1, 15),
                "target_date": date(2024, 1, 22),
                "expected_return": 0.02,
                "prob_positive": 0.6,
            }
        )

        repo.update_evaluation("eval-test", 0.03, "positive")

        result = repo.get_forecast("eval-test")
        assert result["actual_return"] == 0.03
        assert result["actual_outcome"] == "positive"
        assert result["evaluated_at"] is not None


class TestBacktestRepository:
    def test_save_and_get_backtest(self, temp_db):
        repo = BacktestRepository(temp_db)

        result = {
            "id": "bt-1",
            "symbol": "TEST",
            "model_name": "test_model",
            "start_date": date(2024, 1, 1),
            "end_date": date(2024, 6, 30),
            "horizon_days": 5,
            "total_predictions": 50,
            "directional_accuracy": 0.55,
            "mae": 0.02,
            "rmse": 0.03,
            "cumulative_return": 0.15,
            "sharpe_ratio": 1.2,
        }

        bt_id = repo.save_backtest(result)
        assert bt_id == "bt-1"

        saved = repo.get_backtest(bt_id)
        assert saved is not None
        assert saved["symbol"] == "TEST"
        assert saved["directional_accuracy"] == 0.55


class TestAnalysisRepository:
    def test_save_and_get_analysis(self, temp_db):
        repo = AnalysisRepository(temp_db)

        analysis = {
            "id": "analysis-1",
            "symbol": "TEST",
            "analysis_date": date(2024, 1, 15),
            "lookback_days": 365,
            "trend": "bullish",
            "last_price": 150.0,
            "ma_20": 145.0,
            "ma_50": 140.0,
            "ma_200": 130.0,
            "volatility": 0.25,
            "volume_ratio": 1.2,
            "signals": {"price_above_ma20": True},
        }

        analysis_id = repo.save_analysis(analysis)
        assert analysis_id == "analysis-1"

        result = repo.get_latest_analysis("TEST")
        assert result is not None
        assert result["trend"] == "bullish"
        assert result["last_price"] == 150.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
