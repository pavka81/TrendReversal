#!/usr/bin/env python
# Train an ML model using rule-based labels as ground truth

import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.utils import load_csv, compute_keltner
from src.features import build_feature_matrix
from src.model import train_model
from src.analysis import show_feature_importances

# ── CONFIG ─────────────────────────────────────────────────────────────
CONFIG_PATH = Path("configs/top_10_configs.json")
TRADES_PATH = Path("results/batch_backtest_results.csv")
WEEKLY_DIR = Path("stock_historical_information/weekly")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_OUT = MODELS_DIR / "trained_from_rule_labels.pkl"
PLOT_OUT = RESULTS_DIR / "feature_importances_rule_label_model.png"
# ───────────────────────────────────────────────────────────────────────

print("=== Rule-Based Label Training ===")

# ✅ Load top-performing rule-based config IDs
with open(CONFIG_PATH) as f:
    top_configs = pd.json_normalize(eval(f.read()))
top_config_ids = top_configs["Config_ID"].tolist()
print(f"Using {len(top_config_ids)} top config IDs:", top_config_ids)

# ✅ Load and label trade data
all_trades = pd.read_csv(TRADES_PATH, parse_dates=["Entry Date", "Exit Date"])
positives = all_trades[all_trades["Config_ID"].isin(top_config_ids)].copy()
positives["Label"] = 1
negatives = all_trades[~all_trades["Config_ID"].isin(top_config_ids)].copy()
negatives["Label"] = 0
labeled_events = pd.concat([positives, negatives], ignore_index=True)
print(
    f"Total trades: {len(labeled_events)}, "
    f"Positives: {labeled_events['Label'].sum()}, "
    f"Negatives: {(labeled_events['Label'] == 0).sum()}"
)

# ✅ Rename column for compatibility
labeled_events = labeled_events.rename(columns={'Label': 'Reversal'})

# ✅ Build feature matrix
all_features = []
all_labels = []

for ticker in labeled_events["Ticker"].unique():
    weekly_file = WEEKLY_DIR / f"{ticker}_weekly.csv"
    if not weekly_file.exists():
        print(f"Skipping {ticker} — no weekly file.")
        continue

    try:
        weekly_touches = load_csv(weekly_file)
        weekly_touches = compute_keltner(weekly_touches)
        weekly_touches.index = pd.to_datetime(weekly_touches.index)

        # Drop NaNs in required columns
        required_cols = ['KC_lower', 'KC_middle', 'KC_upper', 'MACD_12_26_9', 'RSI_14', 'ATR20']
        weekly_touches.dropna(subset=required_cols, inplace=True)

        print(f"{ticker} weekly columns: {weekly_touches.columns.tolist()}")

        ticker_events = labeled_events[labeled_events["Ticker"] == ticker].copy()

        # 🚨 Ensure Entry Dates are within available range
        min_date, max_date = weekly_touches.index.min(), weekly_touches.index.max()
        ticker_events = ticker_events[
            (ticker_events["Entry Date"] >= min_date) &
            (ticker_events["Entry Date"] <= max_date)
        ]

        if ticker_events.empty:
            print(f"  ⚠️  Skipping {ticker} — no matching events in weekly data range.")
            continue

        try:
            X, _ = build_feature_matrix(ticker_events, weekly_touches)
            y = ticker_events["Reversal"]
            all_features.append(X)
            all_labels.append(y)
        except Exception as e:
            print(f"  ⚠️  Error in build_feature_matrix for {ticker}: {e}")
            print("  ⚠️  Ticker Events sample:")
            print(ticker_events.head(2))
            print("  ⚠️  Weekly touches sample:")
            print(weekly_touches.head(2))

    except Exception as e:
        print(f"  ⚠️  Error processing {ticker}: {e}")

# ✅ Combine across all tickers
if not all_features:
    raise RuntimeError("❌ No features generated — check if input files are missing indicators or malformed.")

X_all = pd.concat(all_features, axis=0)
y_all = pd.concat(all_labels, axis=0)
print(f"Final dataset: {X_all.shape}, Positives: {y_all.sum()}")

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

# ✅ Train model
model = train_model(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred, output_dict=False)
print("\n📊 Classification Report:\n", report)

# ✅ Save model
MODELS_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_OUT)
print(f"✅ Model saved to {MODEL_OUT}")

# ✅ Save feature importances
imps = show_feature_importances(model, X_all.columns).head(10).sort_values()
fig, ax = plt.subplots()
imps.plot.barh(ax=ax)
ax.set_title("Top Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(PLOT_OUT)
plt.close(fig)
print(f"✅ Feature importance plot saved to {PLOT_OUT}")
