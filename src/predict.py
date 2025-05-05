#!/usr/bin/env python
# src/predict.py

import joblib
import pandas as pd
from pathlib import Path

# Absolute imports to work whether run as module or script
from src.utils       import load_csv
from src.detection   import find_daily_touches, find_weekly_touches
from src.labeling    import label_reversals
from src.features    import build_feature_matrix

# Configuration
MODEL_PATH   = Path("models") / "trend_reversal_rf.pkl"
TICKERS_FILE = Path("src") / "tickers.txt"
THRESHOLD    = 0.60
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load trained model
model = joblib.load(MODEL_PATH)

signals = []
for line in TICKERS_FILE.read_text().splitlines():
    ticker = line.strip().upper()
    if not ticker or ticker.startswith("#"):
        continue

    # 1) Load full history and detect today's touch events
    df_full      = load_csv(f"stock_historical_information/daily/{ticker}_daily.csv")
    daily_events = find_daily_touches(ticker)
    weekly_events= find_weekly_touches(ticker)
    if daily_events.empty:
        continue

    # 2) Check if the most recent touch is today
    today = daily_events.index.max()
    if today != df_full.index[-1]:
        continue

    # 3) Label reversals so we have the 'Reversal' column
    labeled = label_reversals(
        df_full,
        daily_events.index.to_list(),
        lookahead=[1, 2]
    )

    # 4) Merge labels onto daily events, then isolate today's row
    events = daily_events.join(
        labeled[['Close_t', 'Reversal', 'First_Reversal_Day', 'First_Reversal_Date']]
    )
    today_events = events.loc[[today]]

    # 5) Build feature matrix for today
    X, _ = build_feature_matrix(today_events, weekly_events)

    # 6) Predict probability of a reversal
    prob = model.predict_proba(X)[0, 1]
    if prob >= THRESHOLD:
        signals.append({
            'ticker':      ticker,
            'date':        today,
            'probability': prob
        })

# Save signals for tomorrow's open
out = pd.DataFrame(signals)
out_path = RESULTS_DIR / f"signals_{pd.Timestamp.today().date()}.csv"
out.to_csv(out_path, index=False)
print(f"Saved {len(signals)} signals to {out_path}")
print(out.to_string(index=False))
