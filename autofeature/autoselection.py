from deap import creator, base, tools, algorithms
import multiprocessing
import random
import pandas as pd
from sklearn.cross_validation import cross_val_score, KFold
import numpy as np
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier





class AutoSelection():

    def __init__(self, df, target_label):
        self.target = df[target_label]
        self.feature = df.drop(target_label,axis=1)
        feature_length = self.feature.shape[1]

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                           toolbox.attr_bool, feature_length)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=10)

        self.toolbox = toolbox


 
#    def eval_performance(self, individual):
#        col_bool = [True if i==1 else False for i in individual]
#        col_bool_all = [True for i in individual]
#        df1 = self.feature.loc[:, col_bool]
#        df2 = self.feature.loc[:, col_bool_all]
#        
#        x1 = df1.values
#        x2 = df2.values
#        y = self.target.values
#
#        kf = KFold(n=len(x1))
#        scores = []
#        for train_index, test_index in kf:
#            x1_train, x1_test = x1[train_index], x1[test_index]
#            x2_train, x2_test = x2[train_index], x2[test_index]
#            y_train, y_test = y[train_index], y[test_index]
#
#            #clf = RandomForestClassifier(n_estimators=10)
#            #clf = XGBClassifier()
#            clf = LogisticRegression()
#            clf.fit(x2_train,y_train)
#            pred = clf.predict(x2_test)
#            base = f1_score(y_test, pred)
#
#            #clf = RandomForestClassifier(n_estimators=10)
#            #clf = XGBClassifier()
#            clf = LogisticRegression()
#            clf.fit(x1_train,y_train)
#            pred = clf.predict(x1_test)
#            candidate = f1_score(y_test, pred)
#            scores.append(candidate-base)
#        score = np.mean(scores)
#        print(score)
#        return score,

    def eval_performance(self, individual):
        col_bool = [True if i==1 else False for i in individual]
        print(col_bool)
        if not np.any(col_bool):
            return 0, 
        df = self.feature.loc[:, col_bool]

        x = df.values
        y = self.target.values
        #clf = GaussianProcessClassifier()
        #clf = LogisticRegression()
        scorer = make_scorer(accuracy_score)
        clf = XGBClassifier()
        score = np.mean(cross_val_score(clf, x, y, n_jobs=-1, scoring=scorer, cv=3))
        print(score)
        return score,


    def run(self, pop_num, cxpb, mutpb, gen_num):
        self.toolbox.register("evaluate", self.eval_performance)
        pop = self.toolbox.population(n=pop_num)
        self.pop, log = algorithms.eaSimple(pop, self.toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gen_num, verbose=True)

    def get_best(self):
        fits = [(ind.fitness.values[0],ind) for ind in self.pop]
        fits.sort(key=lambda x:x[0])

        best_ind = fits[-1]
        print(best_ind)
        col_bool = [True if i==1 else False for i in best_ind[1]]
        select_col = self.feature.loc[:, col_bool]
        return pd.concat([select_col,self.target],axis=1), select_col.columns


if __name__ == "__main__":

    df = pd.read_csv("/tmp/train_after_etl.csv")
    myas = AutoSelection(df, "Survived")
    myas.run(pop_num=1000, cxpb=0.6, mutpb=0.2, gen_num=10)
    df=myas.get_best()
    print(df)
    df.to_csv("/tmp/train_after_etl2.csv",index=False)
