# src/backtest.py

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score


def evaluate_preds(y_true, y_pred):
    """
    Compute precision, recall, and F1-score for binary classification.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred : array-like
        Predicted binary labels.

    Returns
    -------
    dict
        Dictionary with keys 'precision', 'recall', 'f1'.
    """
    return {
        'precision': precision_score(y_true, y_pred),
        'recall':    recall_score(y_true, y_pred),
        'f1':        f1_score(y_true, y_pred)
    }


def backtest_reversals(
    df_events: pd.DataFrame,
    df_full:   pd.DataFrame,
    hold_days: int = 1
) -> pd.DataFrame:
    """
    Simulate buying at the next-day Open and selling after `hold_days` days.

    Parameters
    ----------
    df_events : pd.DataFrame
        Touch-event rows indexed by Date, must contain 'Reversal' boolean.
    df_full : pd.DataFrame
        Full OHLCV series indexed by Date (must have 'Open','Close').
    hold_days : int
        Number of days to hold (1 = same-day exit).

    Returns
    -------
    pd.DataFrame
        One row per executed trade, columns: entry_date, exit_date,
        entry_price, exit_price, return_pct.
    """
    trades = []
    for dt, ev in df_events.iterrows():
        if not ev.get('Reversal', False):
            continue
        try:
            idx      = df_full.index.get_loc(dt)
            entry_dt = df_full.index[idx + 1]
            exit_dt  = df_full.index[idx + hold_days]
        except (KeyError, IndexError):
            continue
        entry_p = df_full.at[entry_dt, 'Open']
        exit_p  = df_full.at[exit_dt,  'Close']
        ret     = (exit_p / entry_p) - 1
        trades.append({
            'entry_date':  entry_dt,
            'exit_date':   exit_dt,
            'entry_price': entry_p,
            'exit_price':  exit_p,
            'return_pct':  ret
        })
    # Return empty indexed DataFrame if no trades
    if trades:
        return pd.DataFrame(trades).set_index('entry_date')
    else:
        return pd.DataFrame(columns=[
            'entry_date','exit_date','entry_price','exit_price','return_pct'
        ])
