"""Database connection and schema management for Orion Pulse."""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb

from orion_pulse.config.settings import settings


class Database:
    """DuckDB database wrapper with schema management."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (settings.data_dir / "orion_pulse.duckdb")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None
        self._initialize_schema()

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    @contextmanager
    def connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Context manager for database connection."""
        conn = self._get_connection()
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _initialize_schema(self) -> None:
        """Initialize database schema."""
        with self.connection() as conn:
            # Market data table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol VARCHAR NOT NULL,
                    trading_date DATE NOT NULL,
                    open DOUBLE NOT NULL,
                    high DOUBLE NOT NULL,
                    low DOUBLE NOT NULL,
                    close DOUBLE NOT NULL,
                    volume BIGINT NOT NULL,
                    adjusted_close DOUBLE,
                    provider VARCHAR NOT NULL DEFAULT 'yfinance',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, trading_date, provider)
                )
            """)

            # Create index for common queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_market_data_symbol_date
                ON market_data (symbol, trading_date DESC)
            """)

            # News articles table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS news_articles (
                    id VARCHAR PRIMARY KEY,
                    symbol VARCHAR,
                    headline VARCHAR NOT NULL,
                    summary TEXT,
                    url VARCHAR NOT NULL,
                    source VARCHAR NOT NULL,
                    published_at TIMESTAMP NOT NULL,
                    relevance_score DOUBLE,
                    category VARCHAR,
                    sentiment_score DOUBLE,
                    entities VARCHAR[],
                    raw_data JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_news_symbol_published
                ON news_articles (symbol, published_at DESC)
            """)

            # Forecasts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    model_name VARCHAR NOT NULL,
                    horizon_days INT NOT NULL,
                    forecast_date DATE NOT NULL,
                    target_date DATE NOT NULL,
                    expected_return DOUBLE,
                    prob_positive DOUBLE,
                    p10 DOUBLE,
                    p25 DOUBLE,
                    p50 DOUBLE,
                    p75 DOUBLE,
                    p90 DOUBLE,
                    volatility DOUBLE,
                    confidence DOUBLE,
                    factors JSON,
                    input_features JSON,
                    actual_return DOUBLE,
                    actual_outcome VARCHAR,
                    evaluated_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_date
                ON forecasts (symbol, forecast_date DESC)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_forecasts_target_date
                ON forecasts (target_date)
            """)

            # Backtest results table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    model_name VARCHAR NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    horizon_days INT NOT NULL,
                    total_predictions INT NOT NULL,
                    directional_accuracy DOUBLE,
                    mae DOUBLE,
                    rmse DOUBLE,
                    calibration_error DOUBLE,
                    cumulative_return DOUBLE,
                    annualized_return DOUBLE,
                    volatility DOUBLE,
                    max_drawdown DOUBLE,
                    sharpe_ratio DOUBLE,
                    sortino_ratio DOUBLE,
                    parameters JSON,
                    detailed_results JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_backtest_symbol_model
                ON backtest_results (symbol, model_name)
            """)

            # Analysis results table (for caching)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id VARCHAR PRIMARY KEY,
                    symbol VARCHAR NOT NULL,
                    analysis_date DATE NOT NULL,
                    lookback_days INT NOT NULL,
                    trend VARCHAR NOT NULL,
                    last_price DOUBLE NOT NULL,
                    ma_20 DOUBLE,
                    ma_50 DOUBLE,
                    ma_200 DOUBLE,
                    volatility DOUBLE,
                    volume_ratio DOUBLE,
                    signals JSON,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_symbol_date
                ON analysis_results (symbol, analysis_date DESC)
            """)

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def execute(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute a query and return relation."""
        with self.connection() as conn:
            return conn.execute(query, params)

    def fetchall(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> list[tuple[Any, ...]]:
        """Execute query and fetch all results."""
        with self.connection() as conn:
            return conn.execute(query, params).fetchall()

    def fetchone(
        self, query: str, params: tuple[Any, ...] = ()
    ) -> tuple[Any, ...] | None:
        """Execute query and fetch one result."""
        with self.connection() as conn:
            return conn.execute(query, params).fetchone()

    def fetch_df(self, query: str, params: tuple[Any, ...] = ()) -> Any:
        """Execute query and return as DataFrame."""
        with self.connection() as conn:
            return conn.execute(query, params).df()


# Global database instance
_database: Database | None = None


def get_database() -> Database:
    """Get or create global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database


def reset_database() -> None:
    """Reset global database instance (useful for testing)."""
    global _database
    if _database is not None:
        _database.close()
        _database = None
