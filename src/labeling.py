# src/labeling.py

import pandas as pd
from typing import List


def label_reversals(
    df_full: pd.DataFrame,
    touch_dates: List[pd.Timestamp],
    lookahead: List[int] = [1, 2]
) -> pd.DataFrame:
    """
    Given full daily data and a list of touch-event dates, compute reversal labels.

    A reversal is marked True if, within the next `lookahead` trading days,
    the Close price exceeds the touch-day Close. Also returns when the bounce occurs.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full daily data indexed by Date with at least a 'Close' column.
    touch_dates : List[pd.Timestamp]
        Sorted list of dates where price touched/crossed KC_lower.
    lookahead : List[int]
        Offsets (in trading days) to check for a bounce (default [1,2]).

    Returns
    -------
    pd.DataFrame
        Indexed by touch_dates with columns:
        - 'Close_t': Touch-day Close price
        - 'Reversal': bool flag
        - 'First_Reversal_Day': int offset of first bounce day or None
        - 'First_Reversal_Date': Date of first bounce or None
    """
    records = []
    for dt in touch_dates:
        close_t = df_full.at[dt, 'Close']
        # position of touch date in full index
        loc = df_full.index.get_loc(dt)
        max_off = max(lookahead)
        # slice next max_off days
        future_index = df_full.index[loc+1: loc+1+max_off]
        future_closes = df_full.loc[future_index, 'Close']

        bounced = future_closes.gt(close_t)
        if bounced.any():
            # first date where True
            first_date = bounced.idxmax()
            first_offset = df_full.index.get_loc(first_date) - loc
            rec = {
                'Date': dt,
                'Close_t': close_t,
                'Reversal': True,
                'First_Reversal_Day': first_offset,
                'First_Reversal_Date': first_date
            }
        else:
            rec = {
                'Date': dt,
                'Close_t': close_t,
                'Reversal': False,
                'First_Reversal_Day': None,
                'First_Reversal_Date': None
            }
        records.append(rec)

    out = pd.DataFrame(records).set_index('Date')
    return out


