# src/save_model.py

import joblib
from pathlib           import Path

from src.utils         import load_csv
from src.detection     import find_daily_touches, find_weekly_touches
from src.labeling      import label_reversals
from src.features      import build_feature_matrix
from src.model         import train_model


# Pick one representative ticker (e.g. AAPL) or ensemble later
TICKER     = "AAPL"
MODEL_PATH = Path("models") / "trend_reversal_rf.pkl"

# 1. Build X, y on the full history
df_full      = load_csv(f"stock_historical_information/daily/{TICKER}_daily.csv")
daily_events = find_daily_touches(TICKER)
weekly_events= find_weekly_touches(TICKER)

labeled = label_reversals(
    df_full,
    daily_events.index.to_list(),
    lookahead=[1, 2]
)

events = daily_events.join(
    labeled[['Close_t','Reversal','First_Reversal_Day','First_Reversal_Date']]
)

X, y = build_feature_matrix(events, weekly_events)

# 2. Train on all available data
model = train_model(X, y)

# 3. Save to disk
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")
