from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso, RidgeClassifier
from minepy import MINE
import numpy as np
import pandas as pd

class FeatureSelection():

    def __init__(self):
        self.estimators = [CorrelationEstimator(), RandomforestEstimator(), LassoEstimator(), MICEstimator(), RidgeEstimator()]
        self.scores = {}

    def run(self, df, target_label, k):
        for estimator in self.estimators:
            self.scores[estimator.get_name()] = estimator.run(df, target_label)
        merged_scores = self.merge_scores(df, self.scores, target_label)
        print("fs merge score")
        print(self.scores)
        print(merged_scores)
        select_feature_label = [i[0] for i in merged_scores[:k]]
        select_feature = df[select_feature_label]
        df = pd.concat([select_feature, df[target_label]],axis=1)
        return df, select_feature_label

    def merge_scores(self, df, scores, target_label):
        feature_importance = {}
        for col in  df.columns:
            if col != target_label:
                feature_importance[col] = 0
        for estimator, score in scores.items():
            for col,value in score.items():
                feature_importance[col] += value
        return sorted(feature_importance.items(), key=lambda x:x[1])

class Estimator():
    def get_name(self):
        return self.name

class CorrelationEstimator(Estimator):
    def __init__(self):
        self.name = "corr"
    def run(self, df, target_label):
        print("begin estimate")
        scores = {}
        for col in df.columns:
            print(df[col].values)
            print(df[target_label].values)
            print(df[col].values.shape)
            print(df[target_label].values.shape)
            if col != target_label:
                scores[col] = abs(np.corrcoef(df[col].values, df[target_label].values)[0][1])
        scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
        print(scores)
        position = {}
        i = 0
        for col,_ in scores:
            position[col] = i
            i+=1
        print(position) 
        return position

class MICEstimator(Estimator):
    def __init__(self):
        self.name = "mic"
    def run(self, df, target_label):
        scores = {}
        for col in df.columns:
            if col != target_label:
                scores[col] = self.calculate_mic(df[col].values,df[target_label].values)
        scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
        print(scores)
        position = {}
        i = 0
        for col,_ in scores:
            position[col] = i
            i+=1
        print(position)
        return position
    def calculate_mic(self, x, y):
        mine = MINE()
        mine.compute_score(x,y)
        score=mine.mic()/len(np.unique(x))
        return score


class RandomforestEstimator(Estimator):
    def __init__(self):
        self.name = "rf"
    def run(self, df, target_label):
        target = df[target_label]
        feature = df.drop(target_label,axis=1)
        clf = RandomForestClassifier()
        for col in feature.columns:
            if np.any(np.isnan(feature[col].values)) or np.any(np.isinf(feature[col].values)):
                print(list(feature[col].values))
        try:
            clf.fit(feature.values, target.values)
        except:
            for col in feature.columns:
                print(list(feature[col].values))
        scores = {}
        for col_index in range(len(feature.columns)):
            scores[feature.columns[col_index]] = abs(clf.feature_importances_[col_index])
        scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
        print(scores)
        position = {}
        i = 0
        for col,_ in scores:
            position[col] = i
            i+=1
        print(position)
        return position

class LassoEstimator(Estimator):
    def __init__(self):
        self.name = "la"
    def run(self, df, target_label):
        target = df[target_label]
        feature = df.drop(target_label,axis=1)
        clf = Lasso()
        clf.fit(feature.values, target.values)
        scores = {}
        for col_index in range(len(feature.columns)):
            scores[feature.columns[col_index]] = abs(clf.coef_[col_index])
        scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
        print(scores)
        position = {}
        i = 0
        for col,_ in scores:
            position[col] = i
            i+=1
        print(position)
        return position

class RidgeEstimator(Estimator):
    def __init__(self):
        self.name = "rg"
    def run(self, df, target_label):
        target = df[target_label]
        feature = df.drop(target_label,axis=1)
        clf = RidgeClassifier()
        clf.fit(feature.values, target.values)
        scores = {}
        for col_index in range(len(feature.columns)):
            scores[feature.columns[col_index]] = abs(clf.coef_[0][col_index])
        scores = sorted(scores.items(),key=lambda x:x[1],reverse=True)
        print(scores)
        position = {}
        i = 0
        for col,_ in scores:
            position[col] = i
            i+=1
        print(position)
        return position
        
            
if __name__ == "__main__":
    import pandas as pd
    from datacleaner import autoclean

    df = pd.read_csv("/tmp/middle.csv")
    #df.fillna(0,inplace=True)
    fs = FeatureSelection() 
    new_df, cols = fs.run(df, "flag", 300)
    print(cols)
