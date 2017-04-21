import pandas as pd
import numpy as np


def get_dummy(df, dummy_candidate):
    dummy_list = []
    for col in dummy_candidate:
        dummy_df = pd.get_dummies(df[col],prefix=col)
        dummy_list.append(dummy_df)
    for dl in dummy_list:
        df = pd.concat([df, dl],axis=1)
    return df

def restore_dummy(df, dummy_col):
    dummy_list = []
    for col in dummy_col:
        if col in df.columns:
            dummy_df = pd.get_dummies(df[col],prefix=col)
            dummy_list.append(dummy_df)
    for dl in dummy_list:
        df = pd.concat([df, dl],axis=1)
    return df
