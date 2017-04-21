import pandas as pd
import numpy as np

def blance_positive_negative(df, target_label):
    y = df[target_label].values
    df = df.drop(target_label, axis=1)
    x = df.values
    y_positive = y[np.where(y==1)]
    y_negative = y[np.where(y==0)]
    x_positive = x[np.where(y==1)]
    x_negative = x[np.where(y==0)]

    scale =  len(y_positive)/len(y_negative)
    if scale < 2 and scale > 0.5:
        return np.array(x), np.array(y)
    elif scale >= 2:
        y_need_add = y_negative
        x_need_add = x_negative
        scale_int = int(scale - 1)
    elif scale <= 0.5:
        y_need_add = y_positive
        x_need_add = x_positive
        scale_int = int(1//scale - 1)
    
    y_add = np.array(list(y_need_add)*scale_int)
    x_add = np.array(list(x_need_add)*scale_int)

    x = np.append(x, x_add, axis=0)
    y = np.append(y, y_add, axis=0)

    r = np.array(list(zip(x,y)))
    np.random.shuffle(r)
    x = r[:,:-1]
    x = [list(i[0]) for i in x]
    y = list(r[:,-1])
    x = np.array(x)
    new_df = pd.DataFrame()
    for col_index in range(len(df.columns)):
        new_df[df.columns[col_index]] = x.T[col_index]
    target = pd.DataFrame({target_label:y})
    new_df = pd.concat([new_df, target],axis=1)
    return new_df

