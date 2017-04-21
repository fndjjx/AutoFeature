from minepy import MINE
import pandas as pd
import numpy as np


def pre_select(df, target_label):
    target = df[target_label]
    feature = df.drop(target_label,axis=1)
    scores = []
    scores2 = []
    for col in feature.columns:
        col_value = feature[col].values
        mine = MINE(alpha=0.4)
        mine.compute_score(target.values, col_value)
        score=mine.mic()/len(np.unique(col_value))
        scores.append([col,score])
        scores2.append(score)
    mean_score = np.mean(scores2) 
    std_score = np.std(scores2) 
    new_df = pd.DataFrame()
    select_col = []
    for i in scores:
        if i[1]>mean_score-std_score:
            new_df[i[0]] = df[i[0]].values
            select_col.append(i[0])
    new_df = pd.concat([new_df, target],axis=1)
    return new_df

    

