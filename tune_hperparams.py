
"""
Phase 1: Hyperparameter tuning on a representative subset of tickers.

This script:
  1. Reads tickers from src/tickers.txt
  2. Samples a subset of N tickers (default 20)
  3. For each ticker:
     - Loads data
     - Detects touch events
     - Labels reversals
     - Builds features
     - Runs GridSearchCV (TimeSeriesSplit) on a trimmed grid
     - Collects best_params_
  4. Aggregates best parameters across tickers to find consensus defaults
  5. Saves individual results to results/tuning_results.csv and consensus to results/consensus_params.json

Run:
  python src/tune_hyperparams.py
"""
import json
import random
from pathlib import Path
from collections import Counter

import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from src.utils      import load_csv
from src.detection  import find_daily_touches, find_weekly_touches
from src.labeling   import label_reversals
from src.features   import build_feature_matrix
from src.model      import train_model
from src.hyperparams import param_grid  # your trimmed grid

# Configuration
SAMPLE_SIZE = 20    # number of tickers to tune on
CV_SPLITS   = 3     # folds for TimeSeriesSplit
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def tune_ticker(ticker: str):
    """
    Tune hyperparameters for a single ticker.

    Returns the best_params_ dict.
    """
    # Load and detect
    df_full      = load_csv(f"stock_historical_information/daily/{ticker}_daily.csv")
    daily_events = find_daily_touches(ticker)
    weekly_events= find_weekly_touches(ticker)

    # Label
    touch_dates = daily_events.index.to_list()
    labeled     = label_reversals(df_full, touch_dates, lookahead=[1,2])

    # Merge and build features
    events = daily_events.join(
        labeled[['Close_t','Reversal','First_Reversal_Day','First_Reversal_Date']]
    )
    X, y = build_feature_matrix(events, weekly_events)

    print(f"Tuning {ticker}: {len(X)} samples", flush=True)
    # Train + grid search
    model = train_model(
        X, y,
        param_grid=param_grid,
        cv_splits=CV_SPLITS
    )
    return model.best_params_


def main():
    # Read all tickers
    tickers_path = Path("src/tickers.txt")
    if not tickers_path.exists():
        raise FileNotFoundError(f"No tickers file at {tickers_path}")

    all_tickers = [
        line.strip().upper()
        for line in tickers_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    # Sample subset
    random.seed(42)
    sample_tickers = random.sample(all_tickers, min(SAMPLE_SIZE, len(all_tickers)))
    print(f"Sampled {len(sample_tickers)} tickers for tuning: {sample_tickers}")

    # Tune each
    results = []
    for t in sample_tickers:
        best = tune_ticker(t)
        results.append({'ticker': t, **best})

    # Save individual results
    df_res = pd.DataFrame(results).set_index('ticker')
    res_csv = RESULTS_DIR / 'tuning_results.csv'
    df_res.to_csv(res_csv)
    print(f"Saved per-ticker results to {res_csv}")

    # Compute consensus: most common value for each param
    consensus = {}
    for param in param_grid:
        vals = df_res[param].tolist()
        consensus[param] = Counter(vals).most_common(1)[0][0]

    # Save consensus
    cons_json = RESULTS_DIR / 'consensus_params.json'
    cons_json.write_text(json.dumps(consensus, indent=2))
    print(f"Saved consensus params to {cons_json}")

if __name__ == '__main__':
    main()
