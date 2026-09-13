"""Market data retrieval and recursive EWMA standardization."""

from __future__ import annotations

import numpy as np
import pandas as pd


def fetch_asset_data(
    tickers: list[str], start_date: str, end_date: str
) -> pd.DataFrame:
    """Fetch daily adjusted-close returns for a cross-section of assets."""
    if not tickers:
        raise ValueError("At least one ticker is required")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "Fetching data requires the optional dependency: pip install yfinance"
        ) from exc

    prices = yf.download(
        tickers, start=start_date, end=end_date, auto_adjust=False, progress=False
    )["Adj Close"]
    if isinstance(prices, pd.Series):
        prices = prices.rename(tickers[0]).to_frame()
    return prices.sort_index().pct_change().dropna(how="all").dropna(axis=1, how="all")


def standardize_returns_ewma(
    returns: pd.DataFrame, decay_factor: float = 0.99
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize returns using an open-ended, recursively updated EWMA."""
    if returns.empty:
        raise ValueError("returns must contain at least one observation")
    if not 0.0 < decay_factor < 1.0:
        raise ValueError("decay_factor must be between 0 and 1")
    if returns.isna().any().any():
        raise ValueError("returns must not contain missing values")

    alpha = 1.0 - decay_factor
    mean = returns.ewm(alpha=alpha, adjust=False).mean()
    variance = (returns - mean).pow(2).ewm(alpha=alpha, adjust=False).mean()
    volatility = np.sqrt(variance).replace(0.0, np.nan).bfill()
    standardized = ((returns - mean) / volatility).dropna()
    volatility = volatility.loc[standardized.index]
    return standardized, volatility
