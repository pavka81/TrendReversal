# src/download_all.py — incremental downloader for TrendReversal with clean session, cache wipe, retry, and throttling
import os
import time
import shutil
from pathlib import Path
from datetime import datetime, timedelta

import yfinance as yf
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

DAILY_DIR  = OUTPUT_DIR / "daily"
WEEKLY_DIR = OUTPUT_DIR / "weekly"



from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import os
import shutil


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
    df.ta.kc(length=20, scalar=3, mamode="sma", append=True)
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
        df_new = yf.download(
            tickers=ticker,
            start=start,
            end=today,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
    except Exception as e:
        print(f"[{ticker}][{suffix}] download failed: {e}")
        return

    if df_new.empty:
        print(f"[{ticker}][{suffix}] no new data since {start}")
        return
        return

    df_new.index.name = 'Date'
    df_new = process_df(df_new, ticker)

    df_combined = pd.concat([df_old, df_new[~df_new.index.isin(df_old.index)]])
    df_combined.sort_index(inplace=True)
    df_combined.to_csv(csv_path)
    print(f"[{ticker}][{suffix}] updated {start} → {today} ({len(df_new)} rows)")

    time.sleep(1.5)  # ⚠️ throttle requests

def main():
    print(f"Project root: {PROJECT_ROOT}")
    ensure_dirs()
    tickers = load_tickers()

    for ticker in tickers:
        print(f"→ Updating {ticker}")
        update_periodic(ticker, interval="1d",  out_dir=DAILY_DIR,  period_delta=timedelta(days=1))
        update_periodic(ticker, interval="1wk", out_dir=WEEKLY_DIR, period_delta=timedelta(days=7))

if __name__ == "__main__":
    main()