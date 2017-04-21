import pandas as pd
def remove_same(df):
    return df.loc[:,~df.columns.duplicated()]
        
