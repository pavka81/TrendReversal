# peek_summary.py
import pandas as pd

df = pd.read_csv('results/portfolio_summary.csv', index_col='ticker')
print(df)


