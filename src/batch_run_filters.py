
# batch_run_filters.py
#
# Run all 128 combinations of USE_* filters defined in configs/filter_config_matrix.csv
# Executes src/backtest_rule_based.py for each config and saves results to results/batch_backtest_results.csv

import os
import sys
import pandas as pd
from tqdm import tqdm
from importlib import util as importlib_util

# Path setup
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "filter_config_matrix.csv")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "batch_backtest_results.csv")
RULE_BACKTEST_PATH = os.path.join(SRC_PATH, "backtest_rule_based.py")

# Dynamically import the rule-based backtest module
spec = importlib_util.spec_from_file_location("backtest_rule_based", RULE_BACKTEST_PATH)
brb = importlib_util.module_from_spec(spec)
spec.loader.exec_module(brb)

# Load config combinations
configs = pd.read_csv(CONFIG_PATH)

# Prepare to collect all results
all_results = []

# Minimal test set to keep things quick
tickers = ["AAPL", "MSFT", "NVDA"]

for _, row in tqdm(configs.iterrows(), total=len(configs), desc="Running backtests"):
    config_id = row["Config_ID"]

    # Inject each flag
    for flag in row.index[1:]:
        setattr(brb, flag, row[flag])

    for ticker in tickers:
        try:
            df = brb.backtest_keltner_2024(ticker)
            if not df.empty:
                df["Ticker"] = ticker
                df["Config_ID"] = config_id
                for flag in row.index[1:]:
                    df[flag] = row[flag]
                all_results.append(df)
        except Exception as e:
            print(f"Error with {ticker} under {config_id}: {e}")

# Save to CSV
if all_results:
    results_df = pd.concat(all_results, ignore_index=True)
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"✅ Results saved to {RESULTS_PATH}")
else:
    print("⚠️ No results were generated.")
