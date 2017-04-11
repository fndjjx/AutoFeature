import pandas as pd
import numpy as np

class ColType():
    def __init__(self, df, custom=None):
        self.df = df
        self.discrete_df = pd.DataFrame()
        self.continuous_df = pd.DataFrame()
        if custom:
            for col,col_type in custom.items():
                if col_type == "c":
                    self.continuous_df[col] = self.df[col].values
                else:
                    self.discrete_df[col] = self.df[col].values
                self.df = self.df.drop(col,axis=1)
                 

    def run(self):
        for col_index in range(len(self.df.columns)):
            col_value = self.df[self.df.columns[col_index]].values
            if len(np.unique(col_value))>5:
                self.continuous_df[self.df.columns[col_index]] = col_value
            else:
                self.discrete_df[self.df.columns[col_index]] = col_value

        return self.discrete_df, self.continuous_df
                
            
