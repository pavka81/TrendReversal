# src/utils.py

import pandas as pd

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV with a 'Date_' column, rename and parse dates, then set it as the index.
    
    Expected raw columns:
      - Date_
      - Open, High, Low, Close, Volume
      - MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
      - RSI_14, EFI_2
      - EMA_11, EMA_22
      - KCLs_20_3.0, KCBs_20_3.0, KCUs_20_3.0
    
    This function will:
      1. Parse 'Date_' → pandas datetime.
      2. Rename:
         - 'Date_' → 'Date'
         - 'KCLs_20_3.0' → 'KC_lower'
         - 'KCBs_20_3.0' → 'KC_middle'
         - 'KCUs_20_3.0' → 'KC_upper'
         - 'EFI_2' → 'Elder_Force_Index_2'
         - 'EMA_11' → 'EMA11'
         - 'EMA_22' → 'EMA22'
      3. Set the new 'Date' column as the DataFrame index.
    """
    df = pd.read_csv(path, parse_dates=['Date_'])
    df.rename(columns={
        'Date_'         : 'Date',
        'KCLs_20_3.0'   : 'KC_lower',
        'KCBs_20_3.0'   : 'KC_middle',
        'KCUs_20_3.0'   : 'KC_upper',
        'EFI_2'         : 'Elder_Force_Index_2',
        'EMA_11'        : 'EMA11',
        'EMA_22'        : 'EMA22',
    }, inplace=True)
    df.set_index('Date', inplace=True)
    return df

def compute_keltner(df: pd.DataFrame, period: int = 20, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute a 3-level Keltner Channel (lower, middle, upper) based on EMA and ATR.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input indexed-by-Date DataFrame with 'High', 'Low', 'Close'.
    period : int
        Look-back period for EMA and ATR.
    multiplier : float
        ATR multiplier for channel width.
    
    Returns
    -------
    pd.DataFrame
        Copy of df with added columns:
          - EMA20
          - ATR20
          - KC_lower
          - KC_middle
          - KC_upper
    """
    df = df.copy()
    df['EMA20']     = df['Close'].ewm(span=period, adjust=False).mean()
    df['ATR20']     = (df['High'] - df['Low']).rolling(window=period).mean()
    df['KC_lower']  = df['EMA20'] - multiplier * df['ATR20']
    df['KC_middle'] = df['EMA20']
    df['KC_upper']  = df['EMA20'] + multiplier * df['ATR20']
    return df
