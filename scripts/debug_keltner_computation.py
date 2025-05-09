import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src.utils import load_csv, compute_keltner

df = load_csv("stock_historical_information/weekly/AAPL_weekly.csv")
print("Columns:", df.columns.tolist())

# Try applying compute_keltner
try:
    df = compute_keltner(df)
    print("✅ Keltner computed. Columns now include:", df.columns.tolist())
except Exception as e:
    print("❌ Keltner computation failed:", e)
