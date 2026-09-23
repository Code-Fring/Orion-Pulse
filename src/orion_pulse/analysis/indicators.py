"""Technical indicators for market analysis."""


import polars as pl

from orion_pulse.core.models import (
    Trend,
    TrendAnalysis,
)


def calculate_daily_returns(df: pl.DataFrame) -> pl.DataFrame:
    """Calculate daily returns from OHLCV data."""
    return df.with_columns(
        [
            pl.col("close").pct_change().alias("return_pct"),
            (pl.col("close").log() - pl.col("close").shift(1).log()).alias(
                "log_return"
            ),
        ]
    )


def calculate_moving_average(
    df: pl.DataFrame,
    period: int,
    column: str = "close",
) -> pl.DataFrame:
    """Calculate simple moving average."""
    return df.with_columns(
        pl.col(column).rolling_mean(window_size=period).alias(f"ma_{period}")
    )


def calculate_rolling_volatility(
    df: pl.DataFrame,
    window: int = 20,
    column: str = "return_pct",
    annualize: bool = True,
) -> pl.DataFrame:
    """Calculate rolling volatility (standard deviation of returns)."""
    annualization_factor = (252**0.5) if annualize else 1.0

    return df.with_columns(
        (pl.col(column).rolling_std(window_size=window) * annualization_factor).alias(
            f"volatility_{window}"
        )
    )


def calculate_volume_ratio(
    df: pl.DataFrame,
    window: int = 20,
) -> pl.DataFrame:
    """Calculate volume ratio (current volume / average volume)."""
    return df.with_columns(
        [
            pl.col("volume")
            .rolling_mean(window_size=window)
            .alias(f"avg_volume_{window}"),
            (
                pl.col("volume") / pl.col("volume").rolling_mean(window_size=window)
            ).alias(f"volume_ratio_{window}"),
        ]
    )


def classify_trend(
    close: float,
    ma_20: float | None,
    ma_50: float | None,
    ma_200: float | None,
) -> Trend:
    """Classify market trend based on price and moving averages."""
    signals = []

    if ma_20 is not None:
        signals.append(close > ma_20)
    if ma_50 is not None:
        signals.append(close > ma_50)
    if ma_200 is not None:
        signals.append(close > ma_200)

    if not signals:
        return Trend.SIDEWAYS

    bullish_count = sum(signals)
    total_signals = len(signals)

    if bullish_count == total_signals:
        return Trend.BULLISH
    elif bullish_count == 0:
        return Trend.BEARISH
    else:
        return Trend.SIDEWAYS


def analyze_trend(
    symbol: str,
    df: pl.DataFrame,
    ma_periods: list[int] = None,
    volatility_window: int = 20,
    volume_window: int = 20,
) -> TrendAnalysis:
    """Perform complete trend analysis on market data."""
    if ma_periods is None:
        ma_periods = [20, 50, 200]

    if df.is_empty():
        raise ValueError("No data provided for analysis")

    # Get latest row
    latest = df.tail(1).to_dicts()[0]
    last_price = latest["close"]
    as_of_date = latest["date"]

    # Calculate moving averages
    ma_values = {}
    for period in ma_periods:
        col_name = f"ma_{period}"
        if col_name in df.columns:
            ma_values[period] = latest.get(col_name)

    ma_20 = ma_values.get(20)
    ma_50 = ma_values.get(50)
    ma_200 = ma_values.get(200)

    # Get volatility
    vol_col = f"volatility_{volatility_window}"
    volatility = latest.get(vol_col)

    # Get volume ratio
    vol_ratio_col = f"volume_ratio_{volume_window}"
    volume_ratio = latest.get(vol_ratio_col)

    # Classify trend
    trend = classify_trend(last_price, ma_20, ma_50, ma_200)

    # Determine trend signals
    price_above_ma20 = last_price > ma_20 if ma_20 else None
    price_above_ma50 = last_price > ma_50 if ma_50 else None
    price_above_ma200 = last_price > ma_200 if ma_200 else None
    ma20_above_ma50 = ma_20 > ma_50 if ma_20 and ma_50 else None
    ma50_above_ma200 = ma_50 > ma_200 if ma_50 and ma_200 else None

    return TrendAnalysis(
        symbol=symbol,
        as_of=as_of_date,
        trend=trend,
        last_price=last_price,
        ma_20=ma_20,
        ma_50=ma_50,
        ma_200=ma_200,
        volatility=volatility,
        volume_ratio=volume_ratio,
        price_above_ma20=price_above_ma20,
        price_above_ma50=price_above_ma50,
        price_above_ma200=price_above_ma200,
        ma20_above_ma50=ma20_above_ma50,
        ma50_above_ma200=ma50_above_ma200,
    )


def prepare_analysis_data(
    df: pl.DataFrame,
    ma_periods: list[int] = None,
    volatility_window: int = 20,
    volume_window: int = 20,
) -> pl.DataFrame:
    """Prepare data with all indicators calculated."""
    if ma_periods is None:
        ma_periods = [20, 50, 200]

    # Calculate returns first
    df = calculate_daily_returns(df)

    # Calculate moving averages
    for period in ma_periods:
        df = calculate_moving_average(df, period)

    # Calculate volatility
    df = calculate_rolling_volatility(df, volatility_window)

    # Calculate volume ratio
    df = calculate_volume_ratio(df, volume_window)

    return df
