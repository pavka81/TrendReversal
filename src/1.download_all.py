# download_all.py

import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta


from pathlib import Path

# If your scripts live in project_root/src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── CONFIG ──────────────────────────────────────────────────────────────────────
TICKERS_FILE = "tickers.txt"
OUTPUT_DIR   = "historic_info"
START_DATE   = "2007-01-01"
# ────────────────────────────────────────────────────────────────────────────────


# Read the list of stock ticker symbols from the file
file_path = "/Users/pavferna/Desktop/finance/4.TrendReversal/src/"+TICKERS_FILE
with open(file_path, "r") as file:
    ticker_symbols = [line.strip() for line in file.readlines()]

# Define directories for daily and weekly data
base_dir = "/Users/pavferna/Desktop/finance/4.TrendReversal/stock_historical_information"
daily_dir = os.path.join(base_dir, "daily")
weekly_dir = os.path.join(base_dir, "weekly")

# Create directories if they don't exist
os.makedirs(daily_dir, exist_ok=True)
os.makedirs(weekly_dir, exist_ok=True)

# Loop through each ticker symbol
for ticker_symbol in ticker_symbols:
    # Download daily historical data starting on Jan 01, 2007
    daily_data = yf.download(ticker_symbol, start=START_DATE, interval='1d')
    daily_data.reset_index(inplace=True)  # Reset the index to ensure a single-level index

    # Flatten multi-level column names (if any)
    if isinstance(daily_data.columns, pd.MultiIndex):
        daily_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in daily_data.columns]

    # Remove ticker suffix from column names
    daily_data.rename(columns=lambda col: col.replace(f"_{ticker_symbol}", "") if f"_{ticker_symbol}" in col else col, inplace=True)

    # Calculate indicators for daily data
    daily_data.ta.macd(append=True)  # MACD
    daily_data.ta.rsi(append=True)  # RSI
    daily_data.ta.efi(append=True, length=2)  # Elder Force Index (2-day period)
    daily_data.ta.ema(append=True, length=11)  # Exponential Moving Average (11 days)
    daily_data.ta.ema(append=True, length=22)  # Exponential Moving Average (22 days)
    daily_data.ta.kc(append=True, length=20, scalar=3, mamode="sma")  # Keltner Channels (20,3,sma)

    # Save daily data to the "daily" directory
    daily_file_path = os.path.join(daily_dir, f"{ticker_symbol}_daily.csv")
    daily_data.to_csv(daily_file_path, index=False)
    print(f"Daily data saved to {daily_file_path}")

    # Download weekly historical data starting on Jan 01, 2007
    weekly_data = yf.download(ticker_symbol, start=START_DATE, interval='1wk')
    weekly_data.reset_index(inplace=True)  # Reset the index to ensure a single-level index

    # Flatten multi-level column names (if any)
    if isinstance(weekly_data.columns, pd.MultiIndex):
        weekly_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in weekly_data.columns]

    # Remove ticker suffix from column names
    weekly_data.rename(columns=lambda col: col.replace(f"_{ticker_symbol}", "") if f"_{ticker_symbol}" in col else col, inplace=True)

    # Calculate indicators for weekly data
    weekly_data.ta.macd(append=True)  # MACD
    weekly_data.ta.rsi(append=True)  # RSI
    weekly_data.ta.efi(append=True, length=2)  # Elder Force Index (2-week period)
    weekly_data.ta.ema(append=True, length=11)  # Exponential Moving Average (11 weeks)
    weekly_data.ta.ema(append=True, length=22)  # Exponential Moving Average (22 weeks)
    weekly_data.ta.kc(append=True, length=20, scalar=3, mamode="sma")  # Keltner Channels (20,3,sma)

    # Save weekly data to the "weekly" directory
    weekly_file_path = os.path.join(weekly_dir, f"{ticker_symbol}_weekly.csv")
    weekly_data.to_csv(weekly_file_path, index=False)
    print(f"Weekly data saved to {weekly_file_path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
def download_and_save(ticker):
    print(f"Downloading {ticker} …")
    df = yf.download(
        ticker,
        start=START_DATE,
        interval="1d",
        progress=False,
        auto_adjust=False
    ).reset_index()

    # Flatten any multi‐index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(col).strip() for col in df.columns]

    # Rename '{col}_{ticker}' → '{col}'
    df.rename(
        columns={c: c.replace(f"_{ticker}", "") for c in df.columns},
        inplace=True
    )

    # Ensure required cols
    for col in ["Open","High","Low","Close","Volume"]:
        if col not in df:
            raise KeyError(f"{ticker}: missing {col}")

    # Compute indicators
    df.ta.kc(append=True, length=20, scalar=3, mamode="sma")   # KCLs/KCBs/KCUs
    df.rename(columns={"KCLs_20_3.0":"KC_lower","KCUs_20_3.0":"KC_upper"}, inplace=True)
    df.ta.macd(append=True)   # gives MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    df.ta.rsi(append=True)    # gives RSI_14

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{ticker}_daily.csv")
    df.to_csv(out_path, index=False)
    print(f"  ↳ saved → {out_path}")

def main():
    with open(TICKERS_FILE) as f:
        tickers = [l.strip().upper() for l in f if l.strip() and not l.startswith("#")]

    for t in tickers:
        try:
            download_and_save(t)
        except Exception as e:
            print(f"‼️ Error processing {t}: {e}")

if __name__ == "__main__":
    main()

"""