from celery import Celery
from sklearn.cross_validation import KFold
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score
import time

app = Celery('tasks', broker='amqp://guest@localhost//', backend='amqp')
app.conf.update(
    accept_content = ['json','pickle'],
    result_serializer = 'pickle'
)


@app.task(serializer="pickle")
def get_result(x1, x2, y):

    kf = KFold(len(x1), n_folds=4)
    scores = []
    for train_index, test_index in kf:
        x1_train, x1_test = x1[train_index], x1[test_index]
        x2_train, x2_test = x2[train_index], x2[test_index]
        y_train, y_test = y[train_index], y[test_index]


        clf1 = RandomForestClassifier()
        clf1.fit(x1_train,y_train)
        r1 = clf1.predict(x1_test)
        score1 = accuracy_score(y_test,r1)

        clf2 = RandomForestClassifier()
        clf2.fit(x2_train,y_train)
        r2 = clf2.predict(x2_test)
        score2 = accuracy_score(y_test,r2)
        scores.append(score2-score1)

    
    return np.mean(scores)

