"""Yahoo Finance market data provider."""

import logging
from datetime import date, timedelta

import polars as pl
import yfinance as yf

from orion_pulse.config.settings import settings
from orion_pulse.data.providers.base import (
    DataNotFoundError,
    DataProviderError,
    MarketDataProvider,
    ProviderUnavailableError,
)

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance data provider implementation."""

    def __init__(self, timeout: int | None = None):
        self.timeout = timeout or settings.yfinance_timeout
        self._enabled = settings.yfinance_enabled

    def get_provider_name(self) -> str:
        return "yfinance"

    def is_available(self) -> bool:
        return self._enabled

    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> pl.DataFrame:
        """Fetch historical OHLCV data from Yahoo Finance."""
        if not self._enabled:
            raise ProviderUnavailableError(
                "Yahoo Finance provider is disabled",
                provider=self.get_provider_name(),
            )

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(
                start=start_date,
                end=end_date + timedelta(days=1),  # yfinance end is exclusive
                timeout=self.timeout,
            )

            if hist.empty:
                raise DataNotFoundError(
                    f"No data found for symbol {symbol}",
                    provider=self.get_provider_name(),
                    symbol=symbol,
                )

            # Convert to polars DataFrame with standardized columns
            df = pl.DataFrame(
                {
                    "date": hist.index.date,
                    "open": hist["Open"].values,
                    "high": hist["High"].values,
                    "low": hist["Low"].values,
                    "close": hist["Close"].values,
                    "volume": hist["Volume"].values,
                }
            )

            # Add adjusted close if available
            if "Adj Close" in hist.columns:
                df = df.with_columns(
                    pl.Series("adjusted_close", hist["Adj Close"].values)
                )
            else:
                df = df.with_columns(
                    pl.lit(None).cast(pl.Float64).alias("adjusted_close")
                )

            # Add symbol column
            df = df.with_columns(pl.lit(symbol.upper()).alias("symbol"))

            return df

        except DataNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            raise DataProviderError(
                f"Failed to fetch data from Yahoo Finance: {e}",
                provider=self.get_provider_name(),
                symbol=symbol,
            ) from e

    def get_latest_data(
        self,
        symbol: str,
        lookback_days: int = 365,
    ) -> pl.DataFrame:
        """Fetch latest available data for a symbol."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        return self.get_historical_data(symbol, start_date, end_date)
