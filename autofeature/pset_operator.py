import numpy as np
from scipy import stats as st
from sklearn import preprocessing
from itertools import product


def ignore(x):
    return np.array(x)-np.array(x)

def add(x, y):
    return np.add(x, y)

def square(x):
    return np.array(x)**2

def minusmean(x,y):
    return np.array(x)-np.mean(y)

def log(x):
    if min(x)<0:
        x=np.array(x)+abs(min(x)*1.001)
    return np.log1p(x)
    

def subtract(x, y):
    return np.subtract(x, y)

def multiply(x, y):
    return np.multiply(x, y)

def xor(x,y):
    try:
        return [1 if x[i]*y[i]!=0 else 0 for i in range(len(x))]
    except:
        return np.array(x)


def xnor(x,y):
    try:
        return [1 if x[i]*y[i]==0 else 0 for i in range(len(x))]
    except:
        return np.array(x)

def logic_and(x,y):
    try:
        return [1 if x[i]==y[i] else 0 for i in range(len(x))]
    except:
        return np.array(x)

def logic_or(x,y):
    try:
        return [1 if x[i]!=y[i] else 0 for i in range(len(x))]
    except:
        return np.array(x)



def divide(x, y):
    try:
        if np.mean(y)!=0:
            y = [np.mean(y) if i==0 else i for i in y]
        else:
            y = [1 if i==0 else i for i in y]
        r = np.divide(x, y)
        r = [1 if str(i)=="inf" else i for i in r]
        r = [1 if str(i)=="-inf" else i for i in r]
        r = [1 if str(i)=="nan" else i for i in r]
        return np.array(r)
    except:
        return np.array(x)

def bagging_2(x):
    part_num = 2

    return bagging(x, part_num)

def bagging_3(x):
    part_num = 3

    return bagging(x, part_num)

def bagging_4(x):
    part_num = 4

    return bagging(x, part_num)

def bagging_5(x):
    part_num = 5

    return bagging(x, part_num)

def bagging_10(x):
    part_num = 10

    return bagging(x, part_num)

def bagging_q_5(x):
    return bagging_with_q(x,5)

def bagging_q_10(x):
    return bagging_with_q(x,10)

def bagging_q_20(x):
    return bagging_with_q(x,20)

def bagging_q_30(x):
    return bagging_with_q(x,30)

def bagging_q_40(x):
    return bagging_with_q(x,40)

def bagging_q_50(x):
    return bagging_with_q(x,50)

def bagging_with_q(x, n):
    x = np.array(x)
    q = st.scoreatpercentile(x,n) 
    x[x<=q]=0
    x[x>q]=1
    return x
    


def bagging(x, n):
    try:
        part_num = n
        interval = (max(x)-min(x))/part_num
        mapping = []
        for i in range(1, part_num+1):
            mapping.append([(i-1)*interval+min(x),i])
        mapping.sort(key=lambda x:x[0],reverse=True)
        new = []
        for i in x:
            for j in mapping:
                if i>=j[0]:
                    new.append(j[1])
                    break

        return new
    except:
        return x


def bigger(x):
    try:
        return np.max(x)
    except:
        return x
    

def smaller(x):
    try:
        return np.min(x)
    except:
        return x


def absolute(x):
    return np.abs(x)

def scale(x):
    try:
        return preprocessing.scale(x)
    except:
        return x

def min_max_scale(x):
    try:
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(x)
    except:
        return x

def normalize(x):
    try:
        return preprocessing.normalize(x, norm='l2')[0]
    except:
        return x

def cross(x, y):
    unique_value_x = np.unique(x)
    unique_value_y = np.unique(y)
    all_combinations = list(product(unique_value_x, unique_value_y))
    all_combinations_dict = dict(zip(all_combinations, range(len(all_combinations))))
    return [all_combinations_dict[(x[i],y[i])] for i in range(len(x))]
    

if __name__ == "__main__":
    from minepy import MINE
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.cross_validation import cross_val_score
    from sklearn.metrics import f1_score, make_scorer, accuracy_score, mutual_info_score, auc
    df = pd.read_csv("/tmp/train_after_etl2.csv")
    x = df["Fare"].values
    y = df["Survived"].values
    
    def calculate_mic(x,y):
        mine = MINE()
        mine.compute_score(x,y)
        value=mine.mic()
        value=np.corrcoef(x,y)[0][1]
        return value
    l = [0,5,10,20,30,40,50]
    for n in l:
        s=bagging_with_q(x,n)
        s = [[i] for i in s]
        scorer = make_scorer(mutual_info_score)
        scorer = make_scorer(accuracy_score)
     #   scorer = make_scorer(f1_score)
        #print(np.mean(cross_val_score(RandomForestClassifier(), x,y, n_jobs=-1, cv=5)))
        #print(np.mean(cross_val_score(RandomForestClassifier(), x,y, scoring=scorer,n_jobs=-1, cv=5)))
        #print(np.mean(cross_val_score(RandomForestClassifier(),s,y, scoring=scorer,n_jobs=-1, cv=5)))
        print(np.mean(cross_val_score(RandomForestClassifier(),s,y, n_jobs=-1, cv=5)))
    
    
    
