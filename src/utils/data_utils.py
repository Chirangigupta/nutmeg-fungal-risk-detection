import pandas as pd
import numpy as np

def make_lag_features(df: pd.DataFrame, cols, lags=(1,2,3)):
    for col in cols:
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
    df = df.dropna().reset_index(drop=True)
    return df

def scale_df(df: pd.DataFrame, scaler=None, fit=True, exclude_cols=None):
    from sklearn.preprocessing import StandardScaler
    exclude_cols = exclude_cols or []
    cols = [c for c in df.columns if c not in exclude_cols]
    if fit:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df[cols])
    else:
        scaled = scaler.transform(df[cols])
    out = df.copy()
    out[cols] = scaled
    return out, scaler
