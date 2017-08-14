import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from datacleaner import autoclean
from imblearn.under_sampling import CondensedNearestNeighbour, ClusterCentroids
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

def sample_reduction(df, target_label):

    y = df[target_label]
    x = df.drop(target_label,axis=1)
    x_label = x.columns
    all_index = list(df.index)

    clfs = [LogisticRegression, RandomForestClassifier, AdaBoostClassifier, XGBClassifier]
    
    count = 0
    centers = cluster_centers(df, target_label)
    train_index = top_k_neighour(centers, x, 20)
    try:
        performance = []
        while count<50:

            test_index = [i for i in all_index if i not in train_index]
            y_train = y.loc[train_index]
            y_test = y.loc[test_index]
            x_train = x.loc[train_index]
            x_test = x.loc[test_index]

            wrong_sample_index = None 
            for i in range(10):  
                clf = clfs[np.random.randint(0,4)]()
                clf.fit(x_train, y_train)
                pred = clf.predict(x_test)
                pred = pd.DataFrame(pred, index=x_test.index, columns=["pred"])
                target_pred = pd.concat([x_test, y_test, pred],axis=1)
                pred_diff = target_pred["pred"]!=target_pred[target_label]
                wrong_sample = target_pred[pred_diff]
                if i == 0:
                    wrong_sample_index = wrong_sample.index
                else:
                    wrong_sample_index = wrong_sample_index|wrong_sample.index 

            wrong_sample = df.loc[wrong_sample_index]
            centers = cluster_centers(wrong_sample, target_label)
            train_index_add = top_k_neighour(centers, wrong_sample.drop(target_label,axis=1), 20)
            train_index.extend(train_index_add)
            train_index = list(set(train_index))

            print(count)
            performance.append(len(wrong_sample_index)/len(x_test))
            if len(performance)>10:
                print(np.mean(performance[-5:])/np.mean(performance[-10:-5]))
            if len(performance)>10 and np.mean(performance[-5:])/np.mean(performance[-10:-5])>0.95:
                break

            count += 1
    except Exception as err:
        print(err)
        ty, tv, tb = sys.exc_info()
        print(''.join(traceback.format_tb(tb)))

    least_index = train_index
    other_index = [i for i in all_index if i not in train_index]
    least_df = df.loc[least_index]
    other_df = df.loc[other_index]

    return least_df, other_df, least_index, other_index

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

def top_k_neighour(center_list, df, k):
    index = []
    for center in center_list:
        center_distance = df.apply(lambda x:sum((x-center)**2),axis=1)
        k_near_center = center_distance.sort_values().head(k).index
        index.extend(list(k_near_center))
    return index
    
if __name__=="__main__":
    df = pd.read_csv("train_tita.csv")
    df = autoclean(df)
    
    target = "Survived"
    print(sample_reduction(df,target))

