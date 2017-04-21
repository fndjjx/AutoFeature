import pandas as pd


def alignment(train_df, test_df):
    train_col = list(train_df.columns)
    test_col = list(test_df.columns)
    all_col = set(train_col + test_col)
    train_after_aligment = pd.DataFrame()
    test_after_aligment = pd.DataFrame()
    for col in all_col:
        if col in train_col and col in test_col:
            train_after_aligment[col] = train_df[col].values
            test_after_aligment[col] = test_df[col].values
    return train_after_aligment, test_after_aligment


