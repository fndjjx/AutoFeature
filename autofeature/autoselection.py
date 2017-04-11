from deap import creator, base, tools, algorithms
import multiprocessing
import random
import pandas as pd
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.metrics import f1_score, make_scorer, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier





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

 
    def eval_performance(self, individual):
        col_bool = [True if i==1 else False for i in individual]
        df1 = self.feature.loc[:, col_bool]
        
        x = df1.values
        y = self.target.values
        
        scorer = make_scorer(accuracy_score)
        clf1 = GradientBoostingClassifier()
        clf2 = LogisticRegression()
        clf3 = XGBClassifier()
        clf4 = RandomForestClassifier(n_estimators=10)
        eclf = VotingClassifier(estimators=[('gb', clf1), ('lr', clf2), ('xgb', clf3),("rf",clf4)], voting='hard')

        score = np.mean(cross_val_score(clf4, x, y, scoring=scorer, cv=3))
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
        return pd.concat([self.feature.loc[:, col_bool],self.target],axis=1), col_bool


if __name__ == "__main__":

    df = pd.read_csv("/tmp/train_after_etl.csv")
    myas = AutoSelection(df, "Survived")
    myas.run(pop_num=1000, cxpb=0.6, mutpb=0.2, gen_num=10)
    df=myas.get_best()
    print(df)
    df.to_csv("/tmp/train_after_etl2.csv",index=False)
