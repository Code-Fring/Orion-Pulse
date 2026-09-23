"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.fixture
def sample_ohlcv_data():
    """Sample OHLCV data for testing."""
    from datetime import date

    import polars as pl

    return pl.DataFrame(
        {
            "date": [date(2024, 1, i) for i in range(1, 21)],
            "open": [100.0 + i * 0.5 for i in range(20)],
            "high": [102.0 + i * 0.5 for i in range(20)],
            "low": [99.0 + i * 0.5 for i in range(20)],
            "close": [101.0 + i * 0.5 for i in range(20)],
            "volume": [1000000 + i * 10000 for i in range(20)],
            "symbol": ["TEST"] * 20,
        }
    )


@pytest.fixture
def sample_market_data():
    """Sample market data for testing."""
    from datetime import date, timedelta

    import numpy as np
    import polars as pl

    np.random.seed(42)
    n = 100
    dates = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    # Generate realistic price series with slight upward drift
    returns = np.random.normal(0.0005, 0.015, n)
    prices = 100 * np.exp(np.cumsum(returns))

    return pl.DataFrame(
        {
            "date": dates,
            "open": prices * (1 + np.random.normal(0, 0.002, n)),
            "high": prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            "low": prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            "close": prices,
            "volume": np.random.randint(1000000, 5000000, n),
            "symbol": ["TEST"] * n,
        }
    )


@pytest.fixture
def mock_yfinance_data():
    """Mock yfinance data structure."""
    from datetime import date

    class MockHist:
        def __init__(self):
            self.empty = False
            self.index = [date(2024, 1, i) for i in range(1, 11)]
            self._data = {
                "Open": [100.0 + i for i in range(10)],
                "High": [102.0 + i for i in range(10)],
                "Low": [99.0 + i for i in range(10)],
                "Close": [101.0 + i for i in range(10)],
                "Volume": [1000000 + i * 10000 for i in range(10)],
                "Adj Close": [100.5 + i for i in range(10)],
            }
            self.columns = list(self._data.keys())

        def __getitem__(self, key):
            return self._data[key]

    return MockHist()
