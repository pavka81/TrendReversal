# download_all_delay.py

import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import time  # Import the time module


# ── CONFIG ──────────────────────────────────────────────────────────────────────
TICKERS_FILE = "tickers.txt"
OUTPUT_DIR   = "historic_info"
START_DATE   = "2007-01-01"
DOWNLOAD_DELAY = 20  # Set the delay in seconds between downloads
ERROR_DELAY = 60   # Set a longer delay after a potential error
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
    try:
        # Download daily historical data starting on Jan 01, 2007
        print(f"Downloading daily data for {ticker_symbol}...")
        daily_data = yf.download(ticker_symbol, start=START_DATE, interval='1d', progress=False)
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
        time.sleep(DOWNLOAD_DELAY)  # Add delay after successful daily download

        # Download weekly historical data starting on Jan 01, 2007
        print(f"Downloading weekly data for {ticker_symbol}...")
        weekly_data = yf.download(ticker_symbol, start=START_DATE, interval='1wk', progress=False)
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
        time.sleep(DOWNLOAD_DELAY)  # Add delay after successful weekly download

    except yfinance.exceptions.YFRateLimitError as e:
        print(f"Rate limit error for {ticker_symbol}: {e}. Waiting longer...")
        time.sleep(ERROR_DELAY)
    except Exception as e:
        print(f"An unexpected error occurred for {ticker_symbol}: {e}")
        time.sleep(ERROR_DELAY) # Wait longer even for other errors, in case they are related to network issues

os.makedirs(OUTPUT_DIR, exist_ok=True)



if __name__ == "__main__":
    main()