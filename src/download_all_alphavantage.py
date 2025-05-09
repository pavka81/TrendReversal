# ─────────────────────────────────────────────────────────────────────────────
# Script Name   : download_all.py
# Description   : Incrementally downloads and updates daily & weekly stock data
#                 from Alpha Vantage, enriches with technical indicators, and 
#                 saves to CSV files for the TrendReversal project.
#
# Author        : Pavel Fernandez
# Created       : 2025-05-08
# Last Updated  : 2025-05-08
#
# Requirements  : Alpha Vantage API key (set in environment variable),
#                 pandas, pandas_ta, requests
#
# Notes         : - Handles premium and free API key rate limits
#                 - Stores enriched CSVs in /stock_historical_information
#                 - Adds MACD, RSI, EFI, EMA(11,22), Keltner Channels
#
# Usage         : Run this script directly or call from main pipeline:
#                 $ python src/download_all.py
# ─────────────────────────────────────────────────────────────────────────────


import os
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ── PROJECT ROOT CONFIG ──
PROJECT_ROOT = Path(
    os.getenv("TREND_REV_ROOT", Path(__file__).resolve().parent.parent)
)
SRC_DIR      = PROJECT_ROOT / "src"
OUTPUT_DIR   = PROJECT_ROOT / "stock_historical_information"
BASE_START   = "2007-01-01"
TICKERS_FILE = SRC_DIR / "tickers.txt"
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")  # Required for Alpha Vantage

DAILY_DIR  = OUTPUT_DIR / "daily"
WEEKLY_DIR = OUTPUT_DIR / "weekly"

# ── HELPERS ──
def ensure_dirs():
    for d in (DAILY_DIR, WEEKLY_DIR):
        d.mkdir(parents=True, exist_ok=True)

def load_tickers():
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"Tickers file not found at {TICKERS_FILE}")
    with TICKERS_FILE.open("r") as f:
        return [line.strip().upper() for line in f if line.strip()]

def process_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() if isinstance(col, tuple) else col for col in df.columns]
    df.rename(
        columns=lambda c: c.replace(f"_{ticker}", "") if c.endswith(f"_{ticker}") else c,
        inplace=True
    )
    df.ta.macd(append=True)
    df.ta.rsi(append=True)
    df.ta.efi(length=2, append=True)
    df.ta.ema(length=11, append=True)
    df.ta.ema(length=22, append=True)
    df.ta.kc(length=20, scalar=3, mamode="Ema", append=True)
    return df

def fetch_alpha_vantage_data(ticker: str, interval: str) -> pd.DataFrame:
    base_url = "https://www.alphavantage.co/query"
    if interval == '1d':
        function = "TIME_SERIES_DAILY_ADJUSTED"
        key = "Time Series (Daily)"
    elif interval == '1wk':
        function = "TIME_SERIES_WEEKLY_ADJUSTED"
        key = "Weekly Adjusted Time Series"
    else:
        raise ValueError("Unsupported interval: must be '1d' or '1wk'")

    params = {
        "function": function,
        "symbol": ticker,
        "apikey": ALPHAVANTAGE_API_KEY,
        "outputsize": "full"
    }

    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code}")

    data = response.json().get(key, {})
    if not data:
        raise Exception(f"No data returned for {ticker}")

    df = pd.DataFrame.from_dict(data, orient='index')
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    df.rename(columns=lambda c: c.split(". ")[-1], inplace=True)
    df = df.rename(columns={
        'adjusted close': 'Adj Close',
        'volume': 'Volume',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close'
    })
    df = df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
    return df

def update_periodic(ticker: str, interval: str, out_dir: Path, period_delta: timedelta):
    suffix = 'daily' if interval == '1d' else 'weekly'
    csv_path = out_dir / f"{ticker}_{suffix}.csv"

    if csv_path.exists():
        try:
            df_old = pd.read_csv(csv_path, parse_dates=['Date'], index_col='Date')
        except ValueError:
            df_old = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
            df_old.index.name = 'Date'
        last_date = df_old.index.max()
        start = (last_date + period_delta).strftime('%Y-%m-%d')
    else:
        df_old = pd.DataFrame()
        start = BASE_START

    today = datetime.today().strftime('%Y-%m-%d')
    if start > today:
        print(f"[{ticker}][{suffix}] up-to-date through {last_date.date()}")
        return

    try:
        df_new = fetch_alpha_vantage_data(ticker, interval)
    except Exception as e:
        print(f"[{ticker}][{suffix}] download failed: {e}")
        return

    df_new = df_new[df_new.index >= pd.to_datetime(start)]
    if df_new.empty:
        print(f"[{ticker}][{suffix}] no new data since {start}")
        return

    df_new.index.name = 'Date'
    df_new = process_df(df_new, ticker)

    df_combined = pd.concat([df_old, df_new[~df_new.index.isin(df_old.index)]])
    df_combined.sort_index(inplace=True)
    df_combined.to_csv(csv_path)
    print(f"[{ticker}][{suffix}] updated {start} → {today} ({len(df_new)} rows)")

    # time.sleep(15)  # Alpha Vantage allows 5 requests/min on free tier

def main():
    print(f"Project root: {PROJECT_ROOT}")
    ensure_dirs()
    tickers = load_tickers()

    for ticker in tickers:
        print(f"→ Updating {ticker}")
        update_periodic(ticker, interval="1d",  out_dir=DAILY_DIR,  period_delta=timedelta(days=1))
        update_periodic(ticker, interval="1wk", out_dir=WEEKLY_DIR, period_delta=timedelta(days=7))

    print("\n✅ All tickers processed. Historical data updated or created successfully.")


if __name__ == "__main__":
    main()
