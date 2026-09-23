"""Core data models for Orion Pulse."""

from datetime import date as date_type
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Trend(str, Enum):
    """Market trend classification."""

    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"


class OHLCV(BaseModel):
    """Canonical OHLCV market data record."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    symbol: str = Field(..., description="Market symbol (e.g., NVDA)")
    trading_date: date_type = Field(
        ..., description="Trading date", validation_alias="date"
    )
    open: float = Field(..., gt=0, description="Opening price")
    high: float = Field(..., gt=0, description="High price")
    low: float = Field(..., gt=0, description="Low price")
    close: float = Field(..., gt=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adjusted_close: float | None = Field(
        default=None, gt=0, description="Adjusted closing price"
    )

    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)."""
        return (self.high + self.low + self.close) / 3

    @property
    def price_change(self) -> float:
        """Calculate absolute price change."""
        return self.close - self.open

    @property
    def price_change_pct(self) -> float:
        """Calculate percentage price change."""
        return (self.close - self.open) / self.open * 100


class DailyReturn(BaseModel):
    """Daily return calculation."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    symbol: str
    trading_date: date_type = Field(..., validation_alias="date")
    return_pct: float
    log_return: float


class MovingAverage(BaseModel):
    """Moving average indicator."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    symbol: str
    trading_date: date_type = Field(..., validation_alias="date")
    period: int
    value: float
    ma_type: str = "SMA"


class Volatility(BaseModel):
    """Rolling volatility indicator."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    symbol: str
    trading_date: date_type = Field(..., validation_alias="date")
    window: int
    value: float
    annualized: bool = True


class VolumeRatio(BaseModel):
    """Volume ratio indicator."""

    model_config = ConfigDict(frozen=True, populate_by_name=True)

    symbol: str
    trading_date: date_type = Field(..., validation_alias="date")
    ratio: float
    avg_volume: float
    current_volume: int


class TrendAnalysis(BaseModel):
    """Complete trend analysis result."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    as_of: date_type
    trend: Trend
    last_price: float

    # Moving averages
    ma_20: float | None = None
    ma_50: float | None = None
    ma_200: float | None = None

    # Indicators
    volatility: float | None = None
    volume_ratio: float | None = None

    # Trend signals
    price_above_ma20: bool | None = None
    price_above_ma50: bool | None = None
    price_above_ma200: bool | None = None
    ma20_above_ma50: bool | None = None
    ma50_above_ma200: bool | None = None


class AnalysisReport(BaseModel):
    """Complete analysis report for a symbol."""

    model_config = ConfigDict(frozen=True)

    symbol: str
    generated_at: datetime
    trend_analysis: TrendAnalysis
    lookback_days: int
    data_points: int
