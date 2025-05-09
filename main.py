
#!/usr/bin/env python
# main.py — orchestrates the full TrendReversal pipeline from one entry-point

import runpy
import subprocess
import sys
import argparse
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit

from src.utils import load_csv
from src.analysis import show_feature_importances
from src.detection import find_daily_touches, find_weekly_touches
from src.labeling import label_reversals
from src.features import build_feature_matrixwhite
from src.model import train_model
from src.backtest import evaluate_preds, backtest_reversals
from src.backtest_keltner_2024_debug_summary import generate_report
from src.generate_portfolio_report import generate_portfolio_notebook

# Base directories
SCRIPT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_ROOT / "results"
MODELS_DIR = SCRIPT_ROOT / "models"
NOTEBOOKS_DIR = SCRIPT_ROOT / "notebooks"

def process_ticker(ticker: str):
    print(f"\n→ Processing {ticker}…")
    df_full = load_csv(f"stock_historical_information/daily/{ticker}_daily.csv")
    daily_events = find_daily_touches(ticker)
    weekly_events = find_weekly_touches(ticker)

    touch_dates = daily_events.index.to_list()
    labeled = label_reversals(df_full, touch_dates, lookahead=[1, 2])

    events = daily_events.join(
        labeled[['Close_t', 'Reversal', 'First_Reversal_Day', 'First_Reversal_Date']]
    )

    trades = backtest_reversals(events, df_full, hold_days=1)
    trades["entry_date"] = trades.index
    print(
        f"  • Trades: {len(trades)}, Win%: {trades['return_pct'].gt(0).mean():.1%}, "
        f"AvgRet: {trades['return_pct'].mean():.1%}, "
        f"CumulRet: {(1 + trades['return_pct']).prod() - 1:.1%}"
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    trades_file = RESULTS_DIR / f"{ticker}_trades.csv"
    trades.to_csv(trades_file, index=False)
    print(f"  • Saved trades to {trades_file}")

    X, y = build_feature_matrix(events, weekly_events)
    print(f"  • Feature matrix: {X.shape}, label dist: {y.value_counts(normalize=True).to_dict()}")

    tscv = TimeSeriesSplit(n_splits=5)
    metrics = []
    print("  • Running 5-fold CV with consensus RF settings…")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        model = train_model(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        m = evaluate_preds(y.iloc[test_idx], preds)
        print(f"    – Fold {i}: Precision {m['precision']:.3f}, Recall {m['recall']:.3f}, F1 {m['f1']:.3f}")
        metrics.append(m)

    avg = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    print(
        f"  • CV average — Precision {avg['precision']:.3f}, Recall {avg['recall']:.3f}, F1 {avg['f1']:.3f}"
    )

    final_model = train_model(X, y)
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "trend_reversal_rf.pkl"

    import joblib
    joblib.dump(final_model, model_path)
    print(f"  • Saved model to {model_path}")

    import matplotlib.pyplot as plt
    imps = show_feature_importances(final_model, X.columns.tolist()).head(7).sort_values()
    fig, ax = plt.subplots()
    imps.plot.barh(ax=ax)
    ax.set_title(f"{ticker} — Top Feature Importances", loc='center')
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plot_file = RESULTS_DIR / f"feature_importances_{ticker}.png"
    fig.savefig(plot_file)
    plt.close(fig)
    print(f"  • Saved feature importances plot to {plot_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-report", action="store_true", help="Generate visual backtest summary and notebook")
    args = parser.parse_args()

    print("=== Step 1/5: Downloading historical data ===")
    dl1 = SCRIPT_ROOT / "src" / "download_all_yahoo.py"
    dl2 = SCRIPT_ROOT / "src" / "download_all_alphavantage.py"
    if dl1.exists():
        runpy.run_path(str(dl1), run_name="__main__")
    elif dl2.exists():
        runpy.run_path(str(dl2), run_name="__main__")
    else:
        raise FileNotFoundError(f"No downloader script at {dl1} or {dl2}")

    print("=== Step 2/5: Hyperparameter tuning (if present) ===")
    for cand in [SCRIPT_ROOT / "tune_hyperparams.py", SCRIPT_ROOT / "src" / "tune_hyperparams.py"]:
        if cand.exists():
            print(f"Running tuning: {cand}")
            runpy.run_path(str(cand), run_name="__main__")
            break
    else:
        print("No tuning script found; skipping.")

    print("=== Step 3/5: Processing tickers ===")
    tickers_path = SCRIPT_ROOT / "src" / "tickers.txt"
    if not tickers_path.exists():
        raise FileNotFoundError(f"Tickers file not found at {tickers_path}")
    tickers = [t.strip().upper() for t in tickers_path.read_text().splitlines() if t.strip() and not t.startswith("#")]
    print(f"Loaded {len(tickers)} tickers.")
    for tk in tickers:
        process_ticker(tk)

    print("=== Step 4/5: Aggregating backtests ===")
    agg = SCRIPT_ROOT / "src" / "aggregate.py"
    if agg.exists():
        runpy.run_path(str(agg), run_name="__main__")
    else:
        print("No aggregate script; skipping.")

    if args.generate_report:
        print("=== Extra Step: Generating strategic analysis report ===")
        generate_report()

    print("=== Step 5/5: Exporting HTML report ===")
    generate_portfolio_notebook()

    RESULTS_DIR.mkdir(exist_ok=True)
    subprocess.run([
        sys.executable, "-m", "nbconvert", "--to", "html",
        str(NOTEBOOKS_DIR / "portfolio_report.ipynb"),
        "--output", str(RESULTS_DIR / "portfolio_report.html")
    ], check=True)

    print("=== Final Step: Consolidating ML trades ===")
    import pandas as pd

    trade_files = list(RESULTS_DIR.glob("*_trades.csv"))
    all_trades = []
    for file in trade_files:
        df = pd.read_csv(file)
        df["Ticker"] = file.stem.replace("_trades", "")
        all_trades.append(df)

    if all_trades:
        ml_df = pd.concat(all_trades, ignore_index=True)
        ml_df.rename(columns={
            'entry_date': 'Entry Date',
            'exit_date': 'Exit Date',
            'entry_price': 'Buy Price',
            'exit_price': 'Sell Price',
            'return_pct': 'Return %'
        }, inplace=True)
        ml_df.to_csv(RESULTS_DIR / "ml_predictions.csv", index=False)
        print(f"  • Saved consolidated ML predictions to {RESULTS_DIR / 'ml_predictions.csv'}")

    print("\n🎉 Pipeline complete via main.py 🎉")

if __name__ == "__main__":
    main()
