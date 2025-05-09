# backtest_keltner_2024.py

import os
import sys

# ── Fix the import path before any other imports ──────────────────────────────
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

# ── Now safely import from src/ ───────────────────────────────────────────────
from src.utils import load_csv, compute_keltner, compute_rsi, compute_macd, compute_efi, compute_atr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



YEAR = 2024
TICKERS_FILE = "src/tickers.txt"
DAILY_DIR = "stock_historical_information/daily"
RESULTS_DIR = "results"
ATR_THRESHOLD = 1.0  # Define your ATR threshold here

def backtest_keltner_2024(ticker: str) -> pd.DataFrame:
    # Load daily CSV
    file_path = os.path.join(DAILY_DIR, f"{ticker}_daily.csv")
    df = load_csv(file_path)

    print(f"{ticker}: {len(df)} rows loaded, index dtype = {df.index.dtype}")

    # Filter for the target year
    df = df[df.index.year == YEAR].copy()
    if df.empty:
        print(f"[{ticker}] No data found for {YEAR}.")
        return pd.DataFrame()

    # Compute technical indicators
    df = compute_keltner(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_efi(df)
    df = compute_atr(df)

    # Identify touches to KC_lower
    df['Touch'] = (
        (df['Open'] <= df['KC_lower']) |
        (df['High'] <= df['KC_lower']) |
        (df['Low']  <= df['KC_lower']) |
        (df['Close']<= df['KC_lower'])
    )

    # Simulate trades
    trades = []
    for i in range(len(df) - 1):
        if df.iloc[i]['Touch']:
            # Confirmation candle
            if df.iloc[i + 1]['Close'] <= df.iloc[i + 1]['Open']:
                continue  # Skip if next candle is not bullish

            # Momentum indicators
            rsi = df.iloc[i]['RSI_14']
            macd_hist_current = df.iloc[i]['MACD_Hist']
            macd_hist_prev = df.iloc[i - 1]['MACD_Hist'] if i > 0 else np.nan
            efi_current = df.iloc[i]['EFI']
            efi_prev = df.iloc[i - 1]['EFI'] if i > 0 else np.nan

            momentum_conditions = [
                rsi < 30,
                macd_hist_current > macd_hist_prev if not np.isnan(macd_hist_prev) else False,
                efi_current > efi_prev if not np.isnan(efi_prev) else False
            ]

            if not any(momentum_conditions):
                continue  # Skip if none of the momentum indicators agree

            # Volatility filter
            if df.iloc[i]['ATR'] < ATR_THRESHOLD:
                continue  # Skip if ATR is below threshold

            # Entry
            entry_date = df.index[i]
            buy_price = df.iloc[i]['Close']

            # Trailing exit
            exit_price = None
            exit_date = None
            for j in range(i + 1, len(df)):
                if df.iloc[j]['Close'] < df.iloc[j - 1]['Close']:
                    exit_price = df.iloc[j]['Close']
                    exit_date = df.index[j]
                    break
                if df.iloc[j]['Close'] > df.iloc[j]['KC_middle']:
                    exit_price = df.iloc[j]['Close']
                    exit_date = df.index[j]
                    break
            if exit_price is None:
                exit_price = df.iloc[-1]['Close']
                exit_date = df.index[-1]

            ret = (exit_price - buy_price) / buy_price
            trades.append({
                "ticker": ticker,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "buy_price": round(buy_price, 2),
                "sell_price": round(exit_price, 2),
                "return_pct": round(ret * 100, 2),
                "success": ret > 0
            })

    return pd.DataFrame(trades)

def run_all_backtests():
    print(f"Reading tickers from: {os.path.abspath(TICKERS_FILE)}")
    if not os.path.exists(TICKERS_FILE):
        print("❌ Ticker file not found.")
        return

    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    if not tickers:
        print("❌ No tickers loaded. Is the file empty?")
        return

    print(f"✅ Loaded {len(tickers)} tickers: {tickers}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_trades = []

    for ticker in tickers:
        print(f"Backtesting {ticker}...")
        try:
            trades = backtest_keltner_2024(ticker)
            if not trades.empty:
                all_trades.append(trades)
                trades.to_csv(os.path.join(RESULTS_DIR, f"{ticker}_trades.csv"), index=False)
        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Summary
    if all_trades:
        df_all = pd.concat(all_trades)
        summary = (
            df_all.groupby("ticker")
            .agg(
                trades=("success", "count"),
                win_rate_pct=("success", lambda x: round(100 * x.sum() / len(x), 2)),
                avg_return_pct=("return_pct", "mean"),
                total_return_pct=("return_pct", "sum")
            )
            .reset_index()
        )
        summary.to_csv(os.path.join(RESULTS_DIR, f"summary_backtest_{YEAR}.csv"), index=False)
        print("\n== Backtest Summary ==")
        print(summary)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.scatter(summary['trades'], summary['win_rate_pct'])
        plt.title('Win Rate vs Number of Trades')
        plt.xlabel('Number of Trades')
        plt.ylabel('Win Rate (%)')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f"winrate_vs_trades_{YEAR}.png"))
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.scatter(summary['avg_return_pct'], summary['total_return_pct'])
        plt.title('Average Return vs Total Return')
        plt.xlabel('Average Return (%)')
        plt.ylabel('Total Return (%)')
        plt.grid(True)
        plt.savefig(os.path.join(RESULTS_DIR, f"avg_vs_total_return_{YEAR}.png"))
        plt.close()

if __name__ == "__main__":
    run_all_backtests()
