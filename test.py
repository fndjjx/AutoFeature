import numpy as np
import pandas as pd
from datacleaner import autoclean
import time
from autofeature.sample_reduction import sample_reduction
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from autofeature.operator_config import config1, config2
from autofeature.autofeature import AutoFeature
from xgboost.sklearn import XGBClassifier

train_file = "/home/learner/project/dataset/titanic/train_tita.csv"
test_file = "/home/learner/project/dataset/titanic/test_tita.csv"
target = "Survived"

df = pd.read_csv(train_file)
df = autoclean(df)



t1 = time.time()
least_df, other_df, least_index = sample_reduction(df, target, 0.9)
least_df.to_csv("/tmp/least.csv",index=False)

test_data = pd.read_csv(test_file, error_bad_lines=False)
test_df = autoclean(test_data)


af = AutoFeature(df, "Survived", 20, XGBClassifier,roc_auc_score ,"classification", least_index, test_df, direction=1)
train_df = af.fit(config1,config2)


test_df,train_df = af.transform(config1,config2)
train_df.to_csv("/tmp/train_after_etl2.csv", sep=',', index=False)
test_df.to_csv("/tmp/test_after_etl2.csv", sep=',', index=False)



