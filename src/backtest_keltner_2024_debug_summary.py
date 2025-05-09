# backtest_keltner_2024_debug_summary.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nbformat import v4 as nbf
import nbformat
from src.utils import load_csv, compute_keltner
from src.labeling import label_reversals
from src.backtest import backtest_reversals

# Config
YEAR = 2024
RESULTS_DIR = "results"
NOTEBOOKS_DIR = "notebooks"
DAILY_DIR = "stock_historical_information/daily"
TICKERS_FILE = "src/tickers.txt"
SUMMARY_FILE = f"{RESULTS_DIR}/summary_backtest_{YEAR}.csv"
NOTEBOOK_FILE = f"{NOTEBOOKS_DIR}/backtest_summary_{YEAR}.ipynb"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def generate_report():
    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    all_trades = []
    for ticker in tickers:
        file_path = os.path.join(DAILY_DIR, f"{ticker}_daily.csv")
        if not os.path.exists(file_path):
            continue

        df = load_csv(file_path)
        df = compute_keltner(df)
        df_2024 = df[df.index.year == YEAR].copy()

        if df_2024.empty or 'KC_lower' not in df_2024.columns:
            continue

        touches = df_2024[(df_2024[['Open','High','Low','Close']] <= df_2024['KC_lower'].values[:, None]).any(axis=1)]
        if touches.empty:
            continue

        df_events = label_reversals(df_2024, touches.index.to_list(), lookahead=[1,2,3,4,5])
        df_events = df_events[df_events['Reversal']]
        if df_events.empty:
            continue

        df_trades = backtest_reversals(df_events, df_2024, hold_days=1)
        if not df_trades.empty:
            df_trades['ticker'] = ticker
            all_trades.append(df_trades)
            df_trades.to_csv(os.path.join(RESULTS_DIR, f"{ticker}_trades.csv"), index=False)

    if not all_trades:
        print("No trades to summarize.")
        return

    df_all = pd.concat(all_trades).reset_index()
    df_all['entry_date'] = pd.to_datetime(df_all['entry_date'])

    summary = (
        df_all.groupby("ticker")
        .agg(
            trades=('return_pct', 'count'),
            win_rate_pct=('return_pct', lambda x: round(100 * (x > 0).sum() / len(x), 2)),
            avg_return_pct=('return_pct', 'mean'),
            total_return_pct=('return_pct', 'sum')
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_FILE, index=False)

    # Plots
    def save_plot(fig, filename):
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, filename))
        plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary, x='trades', y='win_rate_pct')
    plt.title('Win Rate vs Number of Trades')
    plt.xlabel('Number of Trades')
    plt.ylabel('Win Rate (%)')
    save_plot(fig, 'debug_winrate_vs_trades.png')

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(data=summary, x='avg_return_pct', y='total_return_pct')
    plt.title('Average Return vs Total Return')
    plt.xlabel('Average Return (%)')
    plt.ylabel('Total Return (%)')
    save_plot(fig, 'debug_avg_vs_total_return.png')

    fig = plt.figure(figsize=(10, 6))
    sns.histplot(summary['avg_return_pct'], bins=15, kde=True)
    plt.title('Distribution of Average Trade Returns')
    plt.xlabel('Average Return (%)')
    plt.ylabel('Frequency')
    save_plot(fig, 'debug_return_distribution.png')

    df_all['cum_return'] = (1 + df_all['return_pct'] / 100).cumprod()
    fig = plt.figure(figsize=(10, 5))
    df_all['cum_return'].plot()
    plt.title('Simulated Portfolio Growth (Equity Curve)')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Return (×)')
    save_plot(fig, 'debug_equity_curve.png')

    df_all['Month'] = df_all['entry_date'].dt.to_period('M').astype(str)
    fig = plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_all, x='Month', y='return_pct')
    plt.xticks(rotation=45)
    plt.title('Monthly Distribution of Trade Returns')
    plt.xlabel('Month')
    plt.ylabel('Return (%)')
    save_plot(fig, 'debug_monthly_return_boxplot.png')

    equity = df_all['cum_return']
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    fig = plt.figure(figsize=(10, 5))
    drawdown.plot(color='red')
    plt.title('Drawdown Curve')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Trade Number')
    save_plot(fig, 'debug_drawdown.png')

    # Notebook
    def img(path):
        return f"from IPython.display import Image\nImage('../{path}')"

    nb = nbf.new_notebook()
    nb.cells = [
        nbf.new_markdown_cell("""
# Keltner Backtest Summary (2024)
This report summarizes the backtest results of a strategy that buys stocks when they touch the lower Keltner Channel band and meet reversal conditions. We analyze trade quality, consistency, returns, and risk.
"""),
        nbf.new_markdown_cell("### Summary Table of Backtest Results"),
        nbf.new_code_cell("import pandas as pd\nsummary = pd.read_csv('../results/summary_backtest_2024.csv')\nsummary.head()"),

        nbf.new_markdown_cell("### Win Rate vs Number of Trades\nThis chart shows whether stocks with more trades tend to perform better or worse."),
        nbf.new_code_cell(img('results/debug_winrate_vs_trades.png')),

        nbf.new_markdown_cell("### Average Return vs Total Return\nThis helps assess consistency. A stock with high average returns but few trades might be less reliable."),
        nbf.new_code_cell(img('results/debug_avg_vs_total_return.png')),

        nbf.new_markdown_cell("### Distribution of Average Returns\nThis histogram shows how average trade returns vary across all tickers. A tight distribution indicates stability."),
        nbf.new_code_cell(img('results/debug_return_distribution.png')),

        nbf.new_markdown_cell("### Simulated Equity Curve\nVisualizes how your portfolio would have grown by following all trades in sequence. Upward slope is good."),
        nbf.new_code_cell(img('results/debug_equity_curve.png')),

        nbf.new_markdown_cell("### Monthly Return Distribution\nBoxplots for each month help spot when the strategy performs best (e.g., rebounds in volatile months)."),
        nbf.new_code_cell(img('results/debug_monthly_return_boxplot.png')),

        nbf.new_markdown_cell("### Drawdown Curve\nTracks the largest decline from a previous peak. The flatter the curve, the less risk you took."),
        nbf.new_code_cell(img('results/debug_drawdown.png')),

        nbf.new_markdown_cell("### Per-Ticker Trade Detail Viewer\nYou can view trades for any ticker in the `results/` folder. Change the ticker to explore others."),
        nbf.new_code_cell(
            "ticker = 'AAPL'  # Change to another ticker if desired\ntry:\n    pd.read_csv(f'../results/{ticker}_trades.csv').head()\nexcept FileNotFoundError:\n    print(f'Trade file not found for {ticker}')"
        )
    ]

    with open(NOTEBOOK_FILE, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"✅ Report generated and saved as: {NOTEBOOK_FILE}")

if __name__ == "__main__":
    generate_report()
