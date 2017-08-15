import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datacleaner import autoclean
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.cluster import KMeans
import traceback
import sys
import datetime
import time

clfs = [LogisticRegression, RandomForestClassifier, AdaBoostClassifier, XGBClassifier]
clfs = [RandomForestClassifier]
def sample_reduction(df, target_label, similarity):

    y = df[target_label]
    x = df.drop(target_label,axis=1)
    x_label = x.columns
    all_index = list(df.index)

    
    count = 0
    centers = cluster_centers(df, target_label)
    train_index = top_neighour(centers, x, 1)
    try:
        flag = True
        while count<1000 and flag:
            print("count")
            print(count)
            print(len(train_index))
            test_index = df.drop(train_index).index
            y_train = y.loc[train_index]
            y_test = y.loc[test_index]
            x_train = x.loc[train_index]
            x_test = x.loc[test_index]

            wrong_sample_index = None 
            for clf_index in range(len(clfs)):
                print(clf_index)
                clf = clfs[clf_index](n_jobs=-1)
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                pred = pd.DataFrame(pred, index=x_test.index, columns=["pred"])
                target_pred = pd.concat([x_test, y_test, pred],axis=1)
                pred_diff = target_pred["pred"]!=target_pred[target_label]
                wrong_sample = target_pred[pred_diff]
                if clf_index == 0:
                    wrong_sample_index = wrong_sample.index
                else:
                    wrong_sample_index = wrong_sample_index|wrong_sample.index 

            wrong_sample_index = list(set(wrong_sample_index))
            wrong_sample = df.loc[wrong_sample_index]
            centers = cluster_centers(wrong_sample, target_label)
            train_index_add = top_neighour(centers, wrong_sample.drop(target_label,axis=1), 1)
            train_index.extend(train_index_add)
            train_index = list(set(train_index))


            count += 1

            least_index = train_index
            least_df = df.loc[least_index]
            other_df = df.drop(least_index)
            flag = judge_loop(least_df, df, target_label, similarity)

    except Exception as err:
        print(err)
        ty, tv, tb = sys.exc_info()
        print(''.join(traceback.format_tb(tb)))


    return least_df, other_df, least_index

def judge_loop(least_df, df, target_label, similarity):

    auc_list = []
    for i in range(10):
        for clf in clfs:
            clf = clf(n_jobs=-1)
            y_train = least_df[target_label]
            x_train = least_df.drop(target_label,axis=1)
            clf.fit(x_train,y_train)
            y_test = df[target_label]
            x_test = df.drop(target_label,axis=1)
            pred = clf.predict_proba(x_test)[:,1]
            auc = roc_auc_score(y_test,pred)
            auc_list.append(auc)
    print(np.mean(auc_list))
    if np.mean(auc_list)>similarity:
        return False
    else:
        return True


def cluster_centers(df, target_label):
    y = df[target_label]
    x = df.drop(target_label, axis=1)
    unique_y = list(set(y.values))
    dfx_for_each_y = []
    for uy in unique_y:
        tmp = df[df[target_label] == uy]
        dfx_for_each_y.append(tmp.drop(target_label,axis=1))

    centers = []
    for dfx in dfx_for_each_y:
        kmeans = KMeans(n_clusters=5).fit(dfx)
        centers.extend(kmeans.cluster_centers_)
    return centers

def top_neighour(center_list, df, select_percentage):
    print("k")
    print(df.shape)
    k = int((df.shape[0]*select_percentage)/100)
    print(k)
    index = []
    for center in center_list:
        center_distance = df.apply(lambda x:sum((x-center)**2),axis=1)
        k_near_center = center_distance.sort_values().head(k).index
        index.extend(list(k_near_center))
    return index
    
if __name__=="__main__":
    df = pd.read_csv("train_tita.csv")
    df = pd.read_csv("cstrain.csv")
#    df = pd.read_csv("IsBad.train.csv")
    df = autoclean(df)
    
    target = "Survived"
    target = "SeriousDlqin2yrs"
#    target = "IsBadBuy"
    

    t1 = time.time()
    least_df, other_df = sample_reduction(df, target, 0.9)
    t2 = time.time()
    print(t2-t1)
    print(least_df.shape)
    clf1 = XGBClassifier()
    clf1 = RandomForestClassifier(n_jobs=-1)
###
    y_train = least_df[target]
    x_train = least_df.drop(target,axis=1)
###
    clf1.fit(x_train,y_train)
###
#
    clf2 = LogisticRegression()
    clf2 = RandomForestClassifier()
##
    y_train = df[target]
    x_train = df.drop(target,axis=1)
##
    clf2.fit(x_train,y_train)

    pred = clf1.predict_proba(x_train)[:,1]
    print(roc_auc_score(y_train,pred))

    pred = clf2.predict_proba(x_train)[:,1]
    print(roc_auc_score(y_train,pred))



