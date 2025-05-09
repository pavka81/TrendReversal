# src/utils.py

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV with a date column ('Date_' or 'Date'), parse dates, rename, and set as index.

    Expected raw columns may include:
      - Date_ or Date
      - Open, High, Low, Close, Volume
      - MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
      - RSI_14, EFI_2
      - EMA_11, EMA_22
      - KCLs_20_3.0, KCBs_20_3.0, KCUs_20_3.0

    This function will:
      1. Detect the date column, parse it to datetime.
      2. Rename it to 'Date'.
      3. Rename indicator columns:
         - 'KCLs_20_3.0' → 'KC_lower'
         - 'KCBs_20_3.0' → 'KC_middle'
         - 'KCUs_20_3.0' → 'KC_upper'
         - 'EFI_2' → 'Elder_Force_Index_2'
         - 'EMA_11' → 'EMA11'
         - 'EMA_22' → 'EMA22'
      4. Set the 'Date' column as index.
    """
    # Read without parse_dates to inspect columns
    df = pd.read_csv(path)

    # Determine which date column to use
    if 'Date_' in df.columns:
        date_col = 'Date_'
    elif 'Date' in df.columns:
        date_col = 'Date'
    else:
        # Fallback: first column as index
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = 'Date'
        return df

    # Parse and rename date column
    df[date_col] = pd.to_datetime(df[date_col])
    if date_col != 'Date':
        df.rename(columns={date_col: 'Date'}, inplace=True)

    # Indicator column renames
    rename_map = {
        'KCLs_20_3.0': 'KC_lower',
        'KCBs_20_3.0': 'KC_middle',
        'KCUs_20_3.0': 'KC_upper',
        'EFI_2':       'Elder_Force_Index_2',
        'EMA_11':      'EMA11',
        'EMA_22':      'EMA22'
    }
    df.rename(columns=rename_map, inplace=True)

    # Set index
    df.set_index('Date', inplace=True)
    return df


def compute_keltner(df: pd.DataFrame, period: int = 20, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Compute a 3-level Keltner Channel (lower, middle, upper) based on EMA and ATR.
    """
    df = df.copy()
    df['EMA20']     = df['Close'].ewm(span=period, adjust=False).mean()
    df['ATR20']     = (df['High'] - df['Low']).rolling(window=period).mean()
    df['KC_lower']  = df['EMA20'] - multiplier * df['ATR20']
    df['KC_middle'] = df['EMA20']
    df['KC_upper']  = df['EMA20'] + multiplier * df['ATR20']
    return df



import pandas_ta as ta

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df['RSI_14'] = ta.rsi(df['Close'], length=period)
    return df

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    macd = ta.macd(df['Close'])
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Hist'] = macd['MACDh_12_26_9']
    return df

def compute_efi(df: pd.DataFrame, length: int = 2) -> pd.DataFrame:
    df = df.copy()
    df['EFI'] = ta.efi(close=df['Close'], volume=df['Volume'], length=length)
    return df

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    df = df.copy()
    df['ATR'] = ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=period)
    return df
