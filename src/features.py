# src/features.py
""" 
Use relative imports inside src/
Edit each of your modules in src/ so that they import from each other with . rather than trying to reach a top-level src. For example, at the top of detection.py:
"""
from .detection import find_daily_touches
from .utils     import compute_keltner

import pandas as pd
from typing import Tuple


def build_feature_matrix(
    events: pd.DataFrame,
    weekly_touches: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Construct the feature matrix X and label vector y for model training.

    Parameters
    ----------
    events : pd.DataFrame
        DataFrame of daily touch events (with indicators & labels) indexed by Date;
        must contain ['Close','KC_lower','EMA11','EMA22','MACD_12_26_9',
        'MACDs_12_26_9','RSI_14','Elder_Force_Index_2','Reversal'].
    weekly_touches : pd.DataFrame
        DataFrame of weekly touch events indexed by Date (only index used).

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with rows aligned to events.index.
    y : pd.Series
        Series of labels ('Reversal') aligned to events.index.
    """
    feats = []
    for dt, row in events.iterrows():
        feat = {
            'dist_pct':      (row['KC_lower'] - row['Close']) / row['KC_lower'],
            'ema11_diff':    row['Close'] - row['EMA11'],
            'ema22_diff':    row['Close'] - row['EMA22'],
            'macd_center':   row['MACD_12_26_9'] - row['MACDs_12_26_9'],
            'rsi':           row['RSI_14'],
            'efi':           row['Elder_Force_Index_2'],
            'weekly_touch':  (dt in weekly_touches.index)
        }
        feats.append(feat)

    X = pd.DataFrame(feats, index=events.index)
    y = events['Reversal']
    return X, y
