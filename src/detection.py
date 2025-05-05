# src/detection.py

import pandas as pd
from .utils import load_csv, compute_keltner


def _find_touches(df: pd.DataFrame, recompute_kc: bool = False) -> pd.DataFrame:
    """
    Internal helper: return rows where any OHLC price <= KC_lower.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexed by Date with at least ['Open','High','Low','Close'].
        If 'KC_lower' is absent or recompute_kc=True, bands will be recalculated.
    recompute_kc : bool
        If True, ignore existing KC_lower and recompute via compute_keltner().

    Returns
    -------
    pd.DataFrame
        Subset of df where any OHLC <= KC_lower.
    """
    # Recompute bands if requested or missing
    if recompute_kc or 'KC_lower' not in df.columns:
        df = compute_keltner(df)

    # Build boolean mask across OHLC vs. lower band
    mask = (
        (df['Open']  <= df['KC_lower']) |
        (df['High']  <= df['KC_lower']) |
        (df['Low']   <= df['KC_lower']) |
        (df['Close'] <= df['KC_lower'])
    )
    return df.loc[mask]


def find_daily_touches(ticker: str, recompute_kc: bool = False) -> pd.DataFrame:
    """
    Load daily CSV for a ticker and return all lower-band touch events.

    Parameters
    ----------
    ticker : str
        Stock symbol, expects file at:
        stock_historical_information/daily/{ticker}_daily.csv
    recompute_kc : bool
        If True, Keltner bands are recalculated instead of using existing columns.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily touch events with full OHLCV + indicators.
    """
    path = f"stock_historical_information/daily/{ticker}_daily.csv"
    df = load_csv(path)
    return _find_touches(df, recompute_kc=recompute_kc)


def find_weekly_touches(ticker: str, recompute_kc: bool = False) -> pd.DataFrame:
    """
    Load weekly CSV for a ticker and return all lower-band touch events.

    Parameters
    ----------
    ticker : str
        Stock symbol, expects file at:
        stock_historical_information/weekly/{ticker}_weekly.csv
    recompute_kc : bool
        If True, Keltner bands are recalculated instead of using existing columns.

    Returns
    -------
    pd.DataFrame
        DataFrame of weekly touch events with full OHLCV + indicators.
    """
    path = f"stock_historical_information/weekly/{ticker}_weekly.csv"
    df = load_csv(path)
    return _find_touches(df, recompute_kc=recompute_kc)
