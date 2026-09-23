"""Repository classes for data access."""

import json
import uuid
from datetime import date, datetime
from typing import Any

import polars as pl

from orion_pulse.storage.database import Database, get_database


class MarketDataRepository:
    """Repository for market data operations."""

    def __init__(self, db: Database | None = None):
        self.db = db or get_database()

    def save_market_data(
        self,
        symbol: str,
        df: pl.DataFrame,
        provider: str = "yfinance",
    ) -> int:
        """Save market data DataFrame to database. Returns number of rows inserted."""
        if df.is_empty():
            return 0

        # Prepare data for insertion
        records = []
        for row in df.iter_rows(named=True):
            records.append(
                (
                    symbol.upper(),
                    row["date"],
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    int(row["volume"]),
                    float(row["adjusted_close"])
                    if row.get("adjusted_close") is not None
                    else None,
                    provider,
                )
            )

        # Upsert using INSERT OR REPLACE
        with self.db.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO market_data
                (symbol, trading_date, open, high, low, close, volume, adjusted_close, provider)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                records,
            )

        return len(records)

    def get_market_data(
        self,
        symbol: str,
        start_date: date | None = None,
        end_date: date | None = None,
        provider: str = "yfinance",
        limit: int | None = None,
    ) -> pl.DataFrame:
        """Retrieve market data for a symbol."""
        query = """
            SELECT symbol, trading_date, open, high, low, close, volume, adjusted_close, provider
            FROM market_data
            WHERE symbol = ? AND provider = ?
        """
        params = [symbol.upper(), provider]

        if start_date:
            query += " AND trading_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND trading_date <= ?"
            params.append(end_date)

        query += " ORDER BY trading_date DESC"

        if limit:
            query += f" LIMIT {limit}"

        with self.db.connection() as conn:
            result = conn.execute(query, params).fetchall()

        if not result:
            return pl.DataFrame()

        # Convert to polars DataFrame directly from rows
        return pl.DataFrame(
            {
                "symbol": [r[0] for r in result],
                "date": [r[1] for r in result],
                "open": [r[2] for r in result],
                "high": [r[3] for r in result],
                "low": [r[4] for r in result],
                "close": [r[5] for r in result],
                "volume": [r[6] for r in result],
                "adjusted_close": [r[7] for r in result],
                "provider": [r[8] for r in result],
            }
        )

    def get_latest_date(
        self, symbol: str, provider: str = "yfinance"
    ) -> date | None:
        """Get the latest available date for a symbol."""
        result = self.db.fetchone(
            """
            SELECT MAX(trading_date) FROM market_data
            WHERE symbol = ? AND provider = ?
        """,
            (symbol.upper(), provider),
        )
        return result[0] if result and result[0] else None

    def get_symbols(self, provider: str = "yfinance") -> list[str]:
        """Get all symbols with data."""
        results = self.db.fetchall(
            """
            SELECT DISTINCT symbol FROM market_data WHERE provider = ? ORDER BY symbol
        """,
            (provider,),
        )
        return [r[0] for r in results]

    def delete_symbol(self, symbol: str, provider: str = "yfinance") -> int:
        """Delete all data for a symbol."""
        with self.db.connection() as conn:
            # First get count
            count_result = conn.execute(
                """
                SELECT COUNT(*) FROM market_data WHERE symbol = ? AND provider = ?
            """,
                (symbol.upper(), provider),
            ).fetchone()
            count = count_result[0] if count_result else 0

            # Then delete
            conn.execute(
                """
                DELETE FROM market_data WHERE symbol = ? AND provider = ?
            """,
                (symbol.upper(), provider),
            )
            return count


class NewsRepository:
    """Repository for news articles."""

    def __init__(self, db: Database | None = None):
        self.db = db or get_database()

    def save_article(self, article: dict[str, Any]) -> str:
        """Save a news article. Returns article ID."""
        article_id = article.get("id") or str(uuid.uuid4())

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO news_articles
                (id, symbol, headline, summary, url, source, published_at,
                 relevance_score, category, sentiment_score, entities, raw_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    article_id,
                    article.get("symbol", "").upper()
                    if article.get("symbol")
                    else None,
                    article["headline"],
                    article.get("summary"),
                    article["url"],
                    article["source"],
                    article["published_at"],
                    article.get("relevance_score"),
                    article.get("category"),
                    article.get("sentiment_score"),
                    json.dumps(article.get("entities", [])),
                    json.dumps(article.get("raw_data", {})),
                ),
            )

        return article_id

    def save_articles(self, articles: list[dict[str, Any]]) -> int:
        """Save multiple articles. Returns count saved."""
        count = 0
        for article in articles:
            self.save_article(article)
            count += 1
        return count

    def get_articles(
        self,
        symbol: str | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """Retrieve news articles."""
        query = """
            SELECT id, symbol, headline, summary, url, source, published_at,
                   relevance_score, category, sentiment_score, entities, raw_data, created_at
            FROM news_articles
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        if start_date:
            query += " AND published_at >= ?"
            params.append(start_date)

        if end_date:
            query += " AND published_at <= ?"
            params.append(end_date)

        query += " ORDER BY published_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results = self.db.fetchall(query, tuple(params))

        articles = []
        for row in results:
            # Handle entities - might be JSON string or already a list
            entities = row[10]
            if isinstance(entities, str):
                entities = json.loads(entities) if entities else []
            elif entities is None:
                entities = []

            # Handle raw_data - might be JSON string or already a dict
            raw_data = row[11]
            if isinstance(raw_data, str):
                raw_data = json.loads(raw_data) if raw_data else {}
            elif raw_data is None:
                raw_data = {}

            articles.append(
                {
                    "id": row[0],
                    "symbol": row[1],
                    "headline": row[2],
                    "summary": row[3],
                    "url": row[4],
                    "source": row[5],
                    "published_at": row[6],
                    "relevance_score": row[7],
                    "category": row[8],
                    "sentiment_score": row[9],
                    "entities": entities,
                    "raw_data": raw_data,
                    "created_at": row[12],
                }
            )

        return articles

    def get_latest_articles(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """Get latest articles for a symbol."""
        return self.get_articles(symbol=symbol, limit=limit)


class ForecastRepository:
    """Repository for forecasts."""

    def __init__(self, db: Database | None = None):
        self.db = db or get_database()

    def save_forecast(self, forecast: dict[str, Any]) -> str:
        """Save a forecast. Returns forecast ID."""
        forecast_id = forecast.get("id") or str(uuid.uuid4())

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO forecasts
                (id, symbol, model_name, horizon_days, forecast_date, target_date,
                 expected_return, prob_positive, p10, p25, p50, p75, p90,
                 volatility, confidence, factors, input_features)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    forecast_id,
                    forecast["symbol"].upper(),
                    forecast["model_name"],
                    forecast["horizon_days"],
                    forecast["forecast_date"],
                    forecast["target_date"],
                    forecast.get("expected_return"),
                    forecast.get("prob_positive"),
                    forecast.get("p10"),
                    forecast.get("p25"),
                    forecast.get("p50"),
                    forecast.get("p75"),
                    forecast.get("p90"),
                    forecast.get("volatility"),
                    forecast.get("confidence"),
                    json.dumps(forecast.get("factors", {})),
                    json.dumps(forecast.get("input_features", {})),
                ),
            )

        return forecast_id

    def get_forecast(self, forecast_id: str) -> dict[str, Any] | None:
        """Get a single forecast by ID."""
        result = self.db.fetchone(
            """
            SELECT id, symbol, model_name, horizon_days, forecast_date, target_date,
                   expected_return, prob_positive, p10, p25, p50, p75, p90,
                   volatility, confidence, factors, input_features,
                   actual_return, actual_outcome, evaluated_at, created_at
            FROM forecasts WHERE id = ?
        """,
            (forecast_id,),
        )

        if not result:
            return None

        return self._row_to_forecast(result)

    def get_forecasts(
        self,
        symbol: str | None = None,
        model_name: str | None = None,
        start_date: date | None = None,
        end_date: date | None = None,
        evaluated_only: bool = False,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve forecasts with filters."""
        query = """
            SELECT id, symbol, model_name, horizon_days, forecast_date, target_date,
                   expected_return, prob_positive, p10, p25, p50, p75, p90,
                   volatility, confidence, factors, input_features,
                   actual_return, actual_outcome, evaluated_at, created_at
            FROM forecasts
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        if start_date:
            query += " AND forecast_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND forecast_date <= ?"
            params.append(end_date)

        if evaluated_only:
            query += " AND actual_return IS NOT NULL"

        query += " ORDER BY forecast_date DESC LIMIT ?"
        params.append(limit)

        results = self.db.fetchall(query, tuple(params))
        return [self._row_to_forecast(r) for r in results]

    def get_pending_evaluation(self, before_date: date) -> list[dict[str, Any]]:
        """Get forecasts that need evaluation (target_date passed but not evaluated)."""
        results = self.db.fetchall(
            """
            SELECT id, symbol, model_name, horizon_days, forecast_date, target_date,
                   expected_return, prob_positive, p10, p25, p50, p75, p90,
                   volatility, confidence, factors, input_features,
                   actual_return, actual_outcome, evaluated_at, created_at
            FROM forecasts
            WHERE target_date <= ? AND actual_return IS NULL
            ORDER BY target_date
        """,
            (before_date,),
        )

        return [self._row_to_forecast(r) for r in results]

    def update_evaluation(
        self,
        forecast_id: str,
        actual_return: float,
        actual_outcome: str,
    ) -> None:
        """Update forecast with actual outcome."""
        with self.db.connection() as conn:
            conn.execute(
                """
                UPDATE forecasts
                SET actual_return = ?, actual_outcome = ?, evaluated_at = ?
                WHERE id = ?
            """,
                (actual_return, actual_outcome, datetime.now(), forecast_id),
            )

    def _row_to_forecast(self, row: tuple) -> dict[str, Any]:
        return {
            "id": row[0],
            "symbol": row[1],
            "model_name": row[2],
            "horizon_days": row[3],
            "forecast_date": row[4],
            "target_date": row[5],
            "expected_return": row[6],
            "prob_positive": row[7],
            "p10": row[8],
            "p25": row[9],
            "p50": row[10],
            "p75": row[11],
            "p90": row[12],
            "volatility": row[13],
            "confidence": row[14],
            "factors": json.loads(row[15]) if row[15] else {},
            "input_features": json.loads(row[16]) if row[16] else {},
            "actual_return": row[17],
            "actual_outcome": row[18],
            "evaluated_at": row[19],
            "created_at": row[20],
        }


class BacktestRepository:
    """Repository for backtest results."""

    def __init__(self, db: Database | None = None):
        self.db = db or get_database()

    def save_backtest(self, result: dict[str, Any]) -> str:
        """Save backtest result. Returns result ID."""
        result_id = result.get("id") or str(uuid.uuid4())

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO backtest_results
                (id, symbol, model_name, start_date, end_date, horizon_days,
                 total_predictions, directional_accuracy, mae, rmse,
                 calibration_error, cumulative_return, annualized_return,
                 volatility, max_drawdown, sharpe_ratio, sortino_ratio,
                 parameters, detailed_results)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    result_id,
                    result["symbol"].upper(),
                    result["model_name"],
                    result["start_date"],
                    result["end_date"],
                    result["horizon_days"],
                    result["total_predictions"],
                    result.get("directional_accuracy"),
                    result.get("mae"),
                    result.get("rmse"),
                    result.get("calibration_error"),
                    result.get("cumulative_return"),
                    result.get("annualized_return"),
                    result.get("volatility"),
                    result.get("max_drawdown"),
                    result.get("sharpe_ratio"),
                    result.get("sortino_ratio"),
                    json.dumps(result.get("parameters", {})),
                    json.dumps(result.get("detailed_results", {})),
                ),
            )

        return result_id

    def get_backtest(self, result_id: str) -> dict[str, Any] | None:
        """Get a single backtest result by ID."""
        result = self.db.fetchone(
            """
            SELECT id, symbol, model_name, start_date, end_date, horizon_days,
                   total_predictions, directional_accuracy, mae, rmse,
                   calibration_error, cumulative_return, annualized_return,
                   volatility, max_drawdown, sharpe_ratio, sortino_ratio,
                   parameters, detailed_results, created_at
            FROM backtest_results WHERE id = ?
        """,
            (result_id,),
        )

        if not result:
            return None

        return self._row_to_backtest(result)

    def get_backtests(
        self,
        symbol: str | None = None,
        model_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Retrieve backtest results."""
        query = """
            SELECT id, symbol, model_name, start_date, end_date, horizon_days,
                   total_predictions, directional_accuracy, mae, rmse,
                   calibration_error, cumulative_return, annualized_return,
                   volatility, max_drawdown, sharpe_ratio, sortino_ratio,
                   parameters, detailed_results, created_at
            FROM backtest_results
            WHERE 1=1
        """
        params = []

        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())

        if model_name:
            query += " AND model_name = ?"
            params.append(model_name)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        results = self.db.fetchall(query, tuple(params))
        return [self._row_to_backtest(r) for r in results]

    def _row_to_backtest(self, row: tuple) -> dict[str, Any]:
        return {
            "id": row[0],
            "symbol": row[1],
            "model_name": row[2],
            "start_date": row[3],
            "end_date": row[4],
            "horizon_days": row[5],
            "total_predictions": row[6],
            "directional_accuracy": row[7],
            "mae": row[8],
            "rmse": row[9],
            "calibration_error": row[10],
            "cumulative_return": row[11],
            "annualized_return": row[12],
            "volatility": row[13],
            "max_drawdown": row[14],
            "sharpe_ratio": row[15],
            "sortino_ratio": row[16],
            "parameters": json.loads(row[17]) if row[17] else {},
            "detailed_results": json.loads(row[18]) if row[18] else {},
            "created_at": row[19],
        }


class AnalysisRepository:
    """Repository for cached analysis results."""

    def __init__(self, db: Database | None = None):
        self.db = db or get_database()

    def save_analysis(self, analysis: dict[str, Any]) -> str:
        """Save analysis result."""
        analysis_id = analysis.get("id") or str(uuid.uuid4())

        with self.db.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO analysis_results
                (id, symbol, analysis_date, lookback_days, trend, last_price,
                 ma_20, ma_50, ma_200, volatility, volume_ratio, signals)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    analysis_id,
                    analysis["symbol"].upper(),
                    analysis["analysis_date"],
                    analysis["lookback_days"],
                    analysis["trend"],
                    analysis["last_price"],
                    analysis.get("ma_20"),
                    analysis.get("ma_50"),
                    analysis.get("ma_200"),
                    analysis.get("volatility"),
                    analysis.get("volume_ratio"),
                    json.dumps(analysis.get("signals", {})),
                ),
            )

        return analysis_id

    def get_latest_analysis(self, symbol: str) -> dict[str, Any] | None:
        """Get latest cached analysis for a symbol."""
        result = self.db.fetchone(
            """
            SELECT id, symbol, analysis_date, lookback_days, trend, last_price,
                   ma_20, ma_50, ma_200, volatility, volume_ratio, signals, created_at
            FROM analysis_results
            WHERE symbol = ?
            ORDER BY analysis_date DESC LIMIT 1
        """,
            (symbol.upper(),),
        )

        if not result:
            return None

        return {
            "id": result[0],
            "symbol": result[1],
            "analysis_date": result[2],
            "lookback_days": result[3],
            "trend": result[4],
            "last_price": result[5],
            "ma_20": result[6],
            "ma_50": result[7],
            "ma_200": result[8],
            "volatility": result[9],
            "volume_ratio": result[10],
            "signals": json.loads(result[11]) if result[11] else {},
            "created_at": result[12],
        }
