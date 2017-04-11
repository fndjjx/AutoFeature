from sklearn.ensemble import RandomForestClassifier
from minepy import MINE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import f1_score, make_scorer, accuracy_score,mutual_info_score
import autotune
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from pset_operator import bagging_2, bagging_3,bagging_4, bagging_5,bagging_10
import numpy as np
from sklearn.cross_validation import cross_val_score

bagging_list = [bagging_2, bagging_3,bagging_4, bagging_5, bagging_10]

class LoopBagging():
    def __init__(self, df, target_label):
        self.target = df[target_label]
        self.feature = df.drop(target_label,axis=1)

    def run(self):
        best_bagging = {}
        for col in self.feature.columns:
            col_value = self.feature[col].values
            record = []
            for i in range(len(bagging_list)):
                bagging_value1 = bagging_list[i](col_value)
                bagging_value2 = [[i] for i in bagging_value1]
                bagging_value2 = np.array(bagging_value2)

                #clf1 = GradientBoostingClassifier()
                #clf2 = LogisticRegression()
                #clf3 = XGBClassifier()
                #clf4 = RandomForestClassifier()
                #eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3),("rr",clf4)], voting='hard')
                #scorer = make_scorer(accuracy_score)
                #value = np.mean(cross_val_score(clf1, bagging_value2, self.target.values, scoring=scorer,n_jobs=-1, cv=5))

                mine = MINE()
                mine.compute_score(self.target.values, bagging_value1)
                value=mine.mic()/len(np.unique(bagging_value1))

                record.append([bagging_list[i], value, bagging_value1])
            record.sort(key=lambda x:x[1])
            print("haha")
            print(col)
            print(record)

            best_bagging[col] = record[-1]

        return best_bagging 
             
            
        
        
