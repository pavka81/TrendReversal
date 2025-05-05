#!/usr/bin/env python
# main.py — at project root

from pathlib import Path
import pandas as pd

from src.analysis import show_feature_importances, plot_feature_importances

from sklearn.model_selection import TimeSeriesSplit
from src.backtest import evaluate_preds, backtest_reversals
from src.utils    import load_csv
from src.detection import find_daily_touches, find_weekly_touches
from src.labeling import label_reversals
from src.features import build_feature_matrix
from src.model    import train_model

def process_ticker(ticker: str):
    # 1. Load full daily series
    df_full = load_csv(f"stock_historical_information/daily/{ticker}_daily.csv")

    # 2. Detect touch events
    daily_events  = find_daily_touches(ticker)
    weekly_events = find_weekly_touches(ticker)

    # 3. Label reversals
    touch_dates = daily_events.index.to_list()
    labeled     = label_reversals(df_full, touch_dates, lookahead=[1, 2])

    # 4. Merge labels back
    events = daily_events.join(
        labeled[['Close_t','Reversal','First_Reversal_Day','First_Reversal_Date']]
    )
    
    # —— Backtest the reversal signals ——
    trades  = backtest_reversals(events, df_full, hold_days=1)

    print("\nBacktest summary:")
    print(f"  Trades executed:     {len(trades)}")
    print(f"  Win rate:            {trades['return_pct'].gt(0).mean():.2%}")
    print(f"  Average return:      {trades['return_pct'].mean():.2%}")
    print(f"  Cumulative return:   {(1 + trades['return_pct']).prod() - 1:.2%}\n")

    # — Save the raw trade list to CSV —
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    trades.to_csv(results_dir / f"{ticker}_trades.csv")
    print(f"Saved trade log to {results_dir / (ticker + '_trades.csv')}")       

    # Debug: show first few events
    print(f"--- {ticker} sample events ---")
    print(events.head(5)[['Close','KC_lower','Reversal']])

    # 5. Build features
    X, y = build_feature_matrix(events, weekly_events)
    print(f"Feature matrix shape: {X.shape}")
    print("Label distribution:\n", y.value_counts(normalize=True))
    print("Feature preview:")
    print(X.head())

    # 6. 5‐fold CV with fixed parameters
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = []
    print(f"=== Evaluating {ticker} with consensus RF settings ===")
    for i, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        model = train_model(X_tr, y_tr)           # uses your consensus defaults
        preds  = model.predict(X_te)
        m      = evaluate_preds(y_te, preds)
        print(f"Fold {i} — Precision: {m['precision']:.3f}, Recall: {m['recall']:.3f}, F1: {m['f1']:.3f}")
        metrics.append(m)

    # 7. Retrain on the full X,y to get a final model
    final_model = train_model(X, y)

    # 8. Show & plot importances
    imps = show_feature_importances(final_model, X.columns.tolist())

    import matplotlib.pyplot as plt

    # grab & sort just the top 7
    top_imp = imps.head(7).sort_values()

    fig, ax = plt.subplots()
    top_imp.plot.barh(ax=ax)

    # Center your ticker name in the title
    ax.set_title(f"{ticker} \n Feature Importances", loc='center')
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.show()


    # 7. Average metrics
    avg = {k: sum(d[k] for d in metrics) / len(metrics) for k in metrics[0]}
    print(f"Average for {ticker}: Precision {avg['precision']:.3f}, Recall {avg['recall']:.3f}, F1 {avg['f1']:.3f}\n")

    # in main.py, after final_model = train_model(X,y)
    import joblib
    from pathlib import Path

    model_path = Path("models") / "trend_reversal_rf.pkl"
    model_path.parent.mkdir(exist_ok=True)
    joblib.dump(final_model, model_path)
    print(f"Saved model to {model_path}")

def main():
    tickers_path = Path("src/tickers.txt")
    if not tickers_path.exists():
        raise FileNotFoundError(f"No tickers file at {tickers_path}")

    tickers = [
        line.strip().upper()
        for line in tickers_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    print(f"Loaded {len(tickers)} tickers.")

    for ticker in tickers:
        print(f"\nProcessing {ticker}…")
        process_ticker(ticker)


if __name__ == "__main__":
    main()






