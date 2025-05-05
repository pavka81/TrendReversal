# src/aggregate.py

import pandas as pd
from pathlib import Path

def aggregate_results():
    results = []
    tickers = [p.stem.split("_")[0] 
               for p in Path("results").glob("*_trades.csv")]

    for ticker in tickers:
        # Load backtest trades
        trades = pd.read_csv(f"results/{ticker}_trades.csv", parse_dates=["entry_date"], index_col="entry_date")
        win_rate = trades["return_pct"].gt(0).mean()
        cum_ret  = (1 + trades["return_pct"]).prod() - 1

        # (Optionally) load CV metrics if you saved them, or recompute here

        results.append({
            "ticker":      ticker,
            "win_rate":    win_rate,
            "cum_return":  cum_ret,
            # "avg_f1":   avg_f1,
            # "precision": avg_precision, etc.
        })

    df = pd.DataFrame(results).set_index("ticker")
    df.to_csv("results/portfolio_summary.csv")
    print("Wrote portfolio summary to results/portfolio_summary.csv")

if __name__=="__main__":
    aggregate_results()
