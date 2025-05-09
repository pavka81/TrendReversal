# src/features.py
import pandas as pd
import numpy as np


def build_feature_matrix(events: pd.DataFrame, weekly_df: pd.DataFrame):
    """
    Build feature matrix X and labels y from weekly stock data and trade events.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame with trade events (must include 'Entry Date', 'Ticker', 'Reversal').
    weekly_df : pd.DataFrame
        Weekly OHLCV + indicator data, indexed by date.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series or None
        Series of labels if 'Reversal' column exists, else None.
    """
    features = []
    labels = []

    for _, row in events.iterrows():
        entry_date = pd.to_datetime(row['Entry Date'])

        # Try to align entry_date with the nearest previous available weekly date
        if entry_date not in weekly_df.index:
            # Pick latest row before or equal to entry_date
            weekly_slice = weekly_df[weekly_df.index <= entry_date]
            if weekly_slice.empty:
                continue  # no data available before entry
            entry_row = weekly_slice.iloc[-1]
        else:
            entry_row = weekly_df.loc[entry_date]

        # Extract relevant features (skip NaNs)
        if entry_row.isna().any():
            continue

        feature = {
            'MACD': entry_row.get('MACD_12_26_9', np.nan),
            'MACDh': entry_row.get('MACDh_12_26_9', np.nan),
            'MACDs': entry_row.get('MACDs_12_26_9', np.nan),
            'RSI': entry_row.get('RSI_14', np.nan),
            'ForceIndex': entry_row.get('Elder_Force_Index_2', np.nan),
            'EMA11': entry_row.get('EMA11', np.nan),
            'EMA22': entry_row.get('EMA22', np.nan),
            'ATR20': entry_row.get('ATR20', np.nan),
            'KC_lower': entry_row.get('KC_lower', np.nan),
            'KC_middle': entry_row.get('KC_middle', np.nan),
            'KC_upper': entry_row.get('KC_upper', np.nan),
        }

        if np.isnan(list(feature.values())).any():
            continue

        features.append(feature)
        labels.append(row['Reversal'] if 'Reversal' in row else None)

    X = pd.DataFrame(features)
    y = pd.Series(labels, name='Reversal') if 'Reversal' in events.columns else None
    return X, y
