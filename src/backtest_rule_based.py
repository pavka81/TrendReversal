# backtest_rule_based.py
#
# Standalone script for rule-based backtesting of Keltner bounce strategy
# Includes toggleable filters: confirmation candle, RSI, MACD, Force Index, ATR, and MACD divergence
# This is independent of the main machine learning pipeline (main.py)
#

# backtest_keltner_2024.py (Refactored)

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fix import paths
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from src.utils import load_csv, compute_keltner, compute_rsi, compute_macd, compute_efi, compute_atr

# Configuration
YEAR = None  # Set to a specific year like 2024, or None to use all data
TICKERS_FILE = "src/tickers.txt"
DAILY_DIR = "stock_historical_information/daily"
RESULTS_DIR = "results"
ATR_THRESHOLD = 1.0

# Toggle filters
USE_CONFIRMATION_CANDLE = True
USE_RSI_FILTER = True
USE_MACD_HIST_FILTER = True
USE_FORCE_INDEX_FILTER = True
USE_ATR_FILTER = True
USE_MACD_DIVERGENCE = True
USE_TRAILING_EXIT = True


def is_macd_divergence(df, i):
    # Ensure we have enough data
    if i < 5:
        return False
    price_trend = df['Close'].iloc[i] > df['Close'].iloc[i - 5]
    macd_trend = df['MACD'].iloc[i] < df['MACD'].iloc[i - 5]
    return price_trend and macd_trend  # Bullish divergence


def backtest_keltner_2024(ticker: str) -> pd.DataFrame:
    file_path = os.path.join(DAILY_DIR, f"{ticker}_daily.csv")
    df = load_csv(file_path)

    if YEAR is not None:
        df = df[df.index.year == YEAR].copy()
    if df.empty:
        return pd.DataFrame()

    df = compute_keltner(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_efi(df)
    df = compute_atr(df)

    df['Touch'] = (
        (df['Open'] <= df['KC_lower']) |
        (df['High'] <= df['KC_lower']) |
        (df['Low']  <= df['KC_lower']) |
        (df['Close']<= df['KC_lower'])
    )

    trades = []
    for i in range(len(df) - 1):
        if df.iloc[i]['Touch']:

            # Filter 1: Confirmation Candle
            if USE_CONFIRMATION_CANDLE and df.iloc[i + 1]['Close'] <= df.iloc[i + 1]['Open']:
                continue

            # Filter 2-4: Momentum Filters
            rsi = df.iloc[i]['RSI_14']
            macd_hist = df.iloc[i]['MACD_Hist']
            macd_hist_prev = df.iloc[i - 1]['MACD_Hist'] if i > 0 else np.nan
            efi = df.iloc[i]['EFI']
            efi_prev = df.iloc[i - 1]['EFI'] if i > 0 else np.nan

            momentum_conditions = []
            if USE_RSI_FILTER:
                momentum_conditions.append(rsi < 30)
            if USE_MACD_HIST_FILTER:
                momentum_conditions.append(not np.isnan(macd_hist_prev) and macd_hist > macd_hist_prev)
            if USE_FORCE_INDEX_FILTER:
                momentum_conditions.append(not np.isnan(efi_prev) and efi > efi_prev)

            if not any(momentum_conditions):
                continue

            # Filter 5: ATR Threshold
            if USE_ATR_FILTER and df.iloc[i]['ATR'] < ATR_THRESHOLD:
                continue

            # Filter 6: MACD Divergence
            if USE_MACD_DIVERGENCE and not is_macd_divergence(df, i):
                continue

            # Entry
            entry_date = df.index[i]
            buy_price = df.iloc[i]['Close']

            # Trailing Exit Logic
            exit_price = None
            exit_date = None
            for j in range(i + 1, len(df)):
                if USE_TRAILING_EXIT:
                    if (df.iloc[j]['Close'] < df.iloc[j - 1]['Close']) or (df.iloc[j]['Close'] < df.iloc[j]['KC_middle']):
                        exit_price = df.iloc[j]['Close']
                        exit_date = df.index[j]
                        break
                else:
                    if j >= i + 5:
                        exit_price = df.iloc[j]['Close']
                        exit_date = df.index[j]
                        break

            if exit_price is None:
                continue

            trades.append({
                'Ticker': ticker,
                'Entry Date': entry_date,
                'Exit Date': exit_date,
                'Buy Price': buy_price,
                'Sell Price': exit_price,
                'Return %': 100 * (exit_price - buy_price) / buy_price
            })

    return pd.DataFrame(trades)
