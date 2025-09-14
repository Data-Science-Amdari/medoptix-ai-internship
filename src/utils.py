import pandas as pd

def add_lag_features(df, group_col, value_col, lags=[1]):
    for lag in lags:
        df[f'{value_col}_lag{lag}'] = df.groupby(group_col)[value_col].shift(lag)
    return df

def add_delta_features(df, group_col, value_col):
    df[f'{value_col}_delta'] = df.groupby(group_col)[value_col].diff()
    return df 