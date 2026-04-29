"""
feature_utils.py
================
Utility functions for the IEEE-CIS Fraud Detection project.

Key responsibilities
--------------------
  • load_and_merge   – load the four raw CSV files, fix id-column naming
                       inconsistency, and merge transaction + identity tables.
  • reduce_mem_usage – cast numeric columns to the smallest dtype that
                       holds the values (speeds up training significantly
                       on ~600 k-row datasets).
  • get_feature_groups – returns a dict of {group_name: [col_list]} so the
                         notebook can selectively inspect V-features, C-counts,
                         D-deltas, etc.
"""

import os
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load & Merge
# ─────────────────────────────────────────────────────────────────────────────

def load_and_merge(data_dir: str = ".") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the four IEEE-CIS CSV files and return (train_df, test_df).

    Parameters
    ----------
    data_dir : str
        Directory that contains the four files:
          train_transaction.csv, train_identity.csv,
          test_transaction.csv,  test_identity.csv

    Returns
    -------
    train_df, test_df : pd.DataFrame
        Merged DataFrames (transaction left-joined with identity on
        TransactionID).  The 'Unnamed: 0' index column is dropped.
        The test identity columns are renamed from 'id-XX' to 'id_XX'
        to match train naming.
    """
    resolved = os.path.normpath(data_dir)
    print(f"📂 Loading CSVs from: {resolved}")

    # ── File paths ────────────────────────────────────────────────────────
    train_tx_path  = os.path.join(resolved, 'train_transaction (1).csv')
    train_id_path  = os.path.join(resolved, 'train_identity (1).csv')
    test_tx_path   = os.path.join(resolved, 'test_transaction (1).csv')
    test_id_path   = os.path.join(resolved, 'test_identity (1).csv')

    # ── Load ──────────────────────────────────────────────────────────────
    train_tx = pd.read_csv(train_tx_path)
    train_id = pd.read_csv(train_id_path)
    test_tx  = pd.read_csv(test_tx_path)
    test_id  = pd.read_csv(test_id_path)

    # ── Drop unnamed index columns ─────────────────────────────────────────
    for df in [train_tx, train_id, test_tx, test_id]:
        df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')],
                inplace=True, errors='ignore')

    # ── Fix test identity column naming (id-XX  →  id_XX) ─────────────────
    test_id.columns = [c.replace('-', '_') for c in test_id.columns]

    # ── Merge ──────────────────────────────────────────────────────────────
    train_df = train_tx.merge(train_id, on='TransactionID', how='left')
    test_df  = test_tx.merge(test_id,  on='TransactionID', how='left')

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Memory reduction
# ─────────────────────────────────────────────────────────────────────────────

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to the smallest type that holds the values.
    Reduces RAM by ~50–70 % on this dataset.

    Parameters
    ----------
    df      : pd.DataFrame
    verbose : bool  – print before/after memory usage

    Returns
    -------
    df : pd.DataFrame  (modified in place and returned)
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type in [object, 'string']:
            # Leave string/categorical columns alone
            continue

        if col_type == bool:
            df[col] = df[col].astype(np.int8)
            continue

        c_min = df[col].min()
        c_max = df[col].max()

        if str(col_type).startswith('int'):
            if c_min > np.iinfo(np.int8).min  and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float32)   # float16 too lossy
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    if verbose:
        pct = 100 * (start_mem - end_mem) / start_mem
        print(f"Memory: {start_mem:.2f} MB  →  {end_mem:.2f} MB  "
              f"({pct:.1f} % reduction)")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Feature group catalogue
# ─────────────────────────────────────────────────────────────────────────────

def get_feature_groups(df: pd.DataFrame) -> dict:
    """
    Return a dictionary mapping semantic group names to column lists.
    Useful for targeted EDA and ablation studies.

    Groups
    ------
    transaction_core  : ID, time, amount, product
    card              : card1-6 + encoded variants
    address           : addr1, addr2
    distance          : dist1, dist2
    email             : purchaser and recipient domains
    count_features    : C1–C14
    timedelta_features: D1–D15
    match_flags       : M1–M9
    vesta_features    : V1–V339
    identity          : id_01–id_38 + DeviceType + DeviceInfo
    engineered        : columns created by TransactionFeatureEngineer
    """
    all_cols = df.columns.tolist()

    def _pick(prefix, n_range):
        return [f'{prefix}{i}' for i in n_range if f'{prefix}{i}' in all_cols]

    groups = {
        'transaction_core'  : [c for c in ['TransactionID', 'TransactionDT',
                                            'TransactionAmt', 'ProductCD',
                                            'isFraud'] if c in all_cols],
        'card'              : [c for c in all_cols if c.startswith('card')],
        'address'           : [c for c in ['addr1', 'addr2'] if c in all_cols],
        'distance'          : [c for c in ['dist1', 'dist2'] if c in all_cols],
        'email'             : [c for c in all_cols if 'emaildomain' in c
                               or c == 'email_match'],
        'count_features'    : _pick('C', range(1, 15)),
        'timedelta_features': _pick('D', range(1, 16)),
        'match_flags'       : _pick('M', range(1, 10)),
        'vesta_features'    : _pick('V', range(1, 340)),
        'identity'          : ([f'id_{i:02d}' for i in range(1, 39)
                                if f'id_{i:02d}' in all_cols]
                               + [c for c in ['DeviceType', 'DeviceInfo']
                                  if c in all_cols]),
        'engineered'        : [c for c in all_cols
                               if any(c.endswith(s) for s in
                                      ['_freq_enc', '_log', '_hour', '_dayofwk',
                                       '_day', '_freq', 'amt_mean', 'amt_std',
                                       'ratio', 'email_match'])],
    }

    return groups
        
