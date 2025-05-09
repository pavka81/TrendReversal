# src/aggregate.py
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path("results")
TICKERS_FILE = Path("src") / "tickers.txt"


def aggregate_results():
    """
    Read per-ticker backtest CSVs, normalize date index, and produce a portfolio summary.
    """
    if not TICKERS_FILE.exists():
        raise FileNotFoundError(f"Tickers file not found: {TICKERS_FILE}")

    # Load tickers
    tickers = [
        t.strip().upper()
        for t in TICKERS_FILE.read_text().splitlines()
        if t.strip() and not t.startswith("#")
    ]

    all_trades = []
    for ticker in tickers:
        file_path = RESULTS_DIR / f"{ticker}_trades.csv"
        if not file_path.exists():
            print(f"Skipping {ticker}: no trades file found at {file_path}")
            continue

        # Load trades, flexibly parse dates
        df = pd.read_csv(file_path)
        # Determine date column for index
        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df.set_index('entry_date', inplace=True)
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        else:
            # no date index available
            pass

        df['ticker'] = ticker
        all_trades.append(df)

    if not all_trades:
        print("No trade data to aggregate.")
        return

    # Combine and summarize
    combined = pd.concat(all_trades)
    # Ensure return_pct exists
    if 'return_pct' not in combined.columns:
        raise KeyError("Column 'return_pct' not found in trades data")

    summary = combined.groupby('ticker')['return_pct'].agg(
        mean_return = 'mean',
        total_return = 'sum',
        num_trades = 'count'
    )
    summary_path = RESULTS_DIR / 'portfolio_summary.csv'
    summary.to_csv(summary_path)
    print(f"Saved portfolio summary to {summary_path}")


if __name__ == '__main__':
    aggregate_results()
