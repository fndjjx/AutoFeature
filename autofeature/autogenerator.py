from deap import base, creator, gp
from deap import creator, base, tools, algorithms
from scoop import futures
from xgboost.sklearn import XGBClassifier
import hashlib
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score, mutual_info_score, roc_auc_score, calinski_harabaz_score, adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.cross_validation import KFold
from sklearn import linear_model
import pickle
from random import Random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression,RandomizedLogisticRegression
from minepy import MINE
from pset_operator import bagging_2, bagging_3,bagging_4, bagging_5
from loop_bagging import LoopBagging
from feature_selection import FeatureSelection
from tasks import get_result
from celery import group
from celery.result import allow_join_result
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def random_str(randomlength=8):
    string = ""
    chars = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789"
    length = len(chars)-1
    random = Random()
    for i in range(randomlength):
        string+=chars[random.randint(0, length)]
    return string



class AutoGenerator():

    def __init__(self, df, target_label, operator_dict, feature_all, clf_obj, metric):
           
        if target_label:
            self.target = df[target_label]
            self.feature = df.drop(target_label,axis=1)
        else:
            self.target = None
            self.feature = df
        self.feature_dict = {}
        for col in self.feature.columns:
            self.feature_dict[col] = self.feature[col].values

   
        self.clf_obj = clf_obj
        self.metric = metric
        self.feature_all = feature_all
        self.pset = self.init_pset(operator_dict)
        print("generator init finish")

    def init_pset(self, operator_dict):
    
        pset = gp.PrimitiveSet("MAIN", len(self.feature.columns))
        for op, op_num in operator_dict.items():
            pset.addPrimitive(op, op_num)

        for col_index in range(len(self.feature.columns)):
            para = "{'ARG%s':'%s'}"%(col_index, self.feature.columns[col_index])
            pset.renameArguments(**eval(para))

#        ephemeral = hashlib.md5(random_str().encode('utf-8')).hexdigest()
#        pset.addEphemeralConstant(ephemeral, lambda: np.random.uniform(-10, 10))


        return pset

    def run(self, popsize=100, matepb=0.6, mutpb=0.3, gensize=20, selectsize=10, kbest=1):
        print("begin run generate")
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,pset=self.pset)
        toolbox = base.Toolbox()
        toolbox.register("map", futures.map)

        toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=2)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluation, pset=self.pset)
        toolbox.register("select", tools.selTournament, tournsize=selectsize)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=1)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        pop = toolbox.population(n=popsize)
        hof = tools.HallOfFame(300)

        pop, log = algorithms.eaSimple(pop, toolbox, matepb, mutpb, gensize, halloffame=hof, verbose=True)

        fits = []
        for i in hof:
            fits.append([i, i.fitness.values[0], str(i)])
        fits.sort(key=lambda x:x[1])

        return self.get_best(fits, self.pset, k=kbest)


    def evaluation(self, individual, pset): 
        print(str(individual))
        func = gp.compile(individual,pset)
        
        new=func(**self.feature_dict)
        if np.std(new)<0.001 or (np.mean(new)!=0 and abs(np.std(new)/np.mean(new)) < 0.001):
            return -10,
        ### method 1
        x1 = self.feature_all.values
        x2 = np.column_stack([self.feature_all.values,new])
        y = self.target.values
        #x = [[i] for i in new]
     #   random_num = np.random.randint(0, len(model_list))
     #   random_num = len(model_list)-1
     #   clf=model_list[random_num]()
        #scores = []
        #for i in range(len(metric_list)):
        #    scorer = make_scorer(metric_list[i])
        #scorer = make_scorer(accuracy_score)
        #    scores.append(np.mean(cross_val_score(clf, x, self.target.values, scoring=scorer, n_jobs=-1, cv=3)))
        scorer = make_scorer(self.metric)
        clf = self.clf_obj()
        score1=np.mean(cross_val_score(clf, x1, self.target.values, scoring=scorer, n_jobs=-1, cv=10))
        score2=np.mean(cross_val_score(clf, x2, self.target.values, scoring=scorer, n_jobs=-1, cv=10))
        score = score2-score1


#        try:
#            #m = TSNE(n_components=2)
#            m = PCA(n_components=2)
#
#            x1 = m.fit_transform(x1)
#            x2 = m.fit_transform(x2)
#
#            km = KMeans(n_clusters=2)
#            y_pred = km.fit_predict(x1)
#            #score1 = calinski_harabaz_score(x1, y_pred) 
#            score1 = adjusted_rand_score(y,y_pred)
#            y_pred = km.fit_predict(x2)
#            #score2 = calinski_harabaz_score(x2, y_pred) 
#            score2 = adjusted_rand_score(y, y_pred) 
#            score = score2-score1
#        except:
#            score = 0
        #score = sum(abs(self.target.values-new))

        #x = np.column_stack([new])
        #clf=RandomForestClassifier()
        #clf.fit(self.x_train,self.y_train)
        #pred = clf.predict(self.x_test)
        #base_score2 = f1_score(self.y_test, pred)
        #scorer = make_scorer(accuracy_score)
        #scorer = make_scorer(f1_score)
        #base_score2 = np.mean(cross_val_score(clf, x, self.target.values, scoring=scorer, n_jobs=-1, cv=2))

      #  np.random.shuffle(new)
      #  x = [[i] for i in new]
      #  x = np.array(x)
      #  x = np.column_stack([self.feature.values,new])
      #  clf=model_list[random_num]()
        #clf = LogisticRegression()
     #   kf = KFold(len(x1), n_folds=3)
     ##   
     ## #  scores=[]
     #   x1_train_list = []
     #   x2_train_list = []
     #   x1_test_list = []
     #   x2_test_list = []
     #   y_train_list = []
     #   y_test_list = []
     #   for train_index, test_index in kf:
     #       x1_train_list.append(list(x1[train_index]))
     #       x1_test_list.append(list(x1[test_index]))
     #       x2_train_list.append(list(x2[train_index]))
     #       x2_test_list.append(list(x2[test_index]))
     #       y_train_list.append(list(y[train_index]))
     #       y_test_list.append(list(y[test_index]))
     #   for train_index, test_index in kf:
     #       x1_train, x1_test = x1[train_index], x1[test_index]
     #       x2_train, x2_test = x2[train_index], x2[test_index]
     #       y_train, y_test = y[train_index], y[test_index]


      #      clf1 = RandomForestClassifier()
      #      clf1.fit(x1_train,y_train)
      #      r1 = clf1.predict(x1_test)
      #      score1 = accuracy_score(y_test,r1)

      #      clf2 = RandomForestClassifier()
      #      clf2.fit(x2_train,y_train)
      #      r2 = clf2.predict(x2_test)
      #      score2 = accuracy_score(y_test,r2)

      #      scores.append(score2-score1)
      #  x1_list = [x1[:len(x1)/3],x1[len(x1)/3:2*len(x1)/3],x1[2*len(x1)/3:]]
      #  x2_list = [x2[:len(x1)/3],x2[len(x2)/3:2*len(x2)/3],x2[2*len(x2)/3:]]
      #  y_list = [y[:len(y)/3],y[len(y)/3:2*len(y)/3],y[2*len(y)/3:]]
      #  result_group = group(get_result.s(x1_list[index], x2_list[index], y_list[index]) for index in range(len(x1_list)))()
      #  #result_group = group(add.signature(i,i) for i in range(10))()
      #  #result_group = group(add.s(i, i) for i in range(100))()
      #  result = result_group.get()
      #  print(result)
      #  score = np.mean(result)


        #clf=RandomForestClassifier()
        #clf.fit(self.x_train,self.y_train)
        #pred = clf.predict(self.x_test)
        #adj_score = f1_score(self.y_test, pred)

        #scorer = make_scorer(accuracy_score)
        #scorer = make_scorer(f1_score)
        #adj_score = np.mean(cross_val_score(clf, x, self.target.values, scoring=scorer, n_jobs=-1, cv=2))

     #   score = score-score2 
        
        #### method 2
     #   score = mutual_info_score(self.target.values, new)/len(np.unique(new))
     #   #score = mutual_info_score(self.target.values, new)#/len(np.unique(new))
     #   score = abs(np.corrcoef(self.target.values, new)[0][1])
     #   scores = []
     #   for i in self.feature.columns:
     #       col_values = self.feature[i].values
     #       #scores.append(abs(np.corrcoef(col_values, new)[0][1]))
     #       scores.append(mutual_info_score(col_values, new)/len(np.unique(new)))
     #       #scores.append(mutual_info_score(col_values, new))
     #   mean_score = np.max(scores)
     #   std_score = np.std(scores)
     #   score = score / mean_score


       # mine = MINE()
       # mine.compute_score(self.target.values, new)
       # score=mine.mic()#/len(np.unique(new))
     #   scores = []
     #   for i in self.feature.columns:
     #       col_values = self.feature[i].values
     ##       scores.append(abs(np.corrcoef(col_values, new)[0][1]))
     #       mine = MINE()
     #       mine.compute_score(col_values, new)
     #       scores.append(mine.mic())
     #   score = score - np.mean(scores)
     #       scores.append(mutual_info_score(col_values, new)/len(np.unique(new)))
     #       #scores.append(mutual_info_score(col_values, new))
 
        #score=mutual_info_score(self.target.values, new)/len(np.unique(new))

#        np.random.shuffle(new)
        #mine = MINE()
        #mine.compute_score(self.target.values, new)
        #score=mine.mic()#/len(np.unique(new))
#        score = score-score2 
     #   scores = []
        #for i in self.feature.columns:
        #    col_values = self.feature[i].values
        #    mine = MINE()
        #    mine.compute_score(col_values, new)
        #    s=mine.mic()
        #    scores.append(s)
     #   score = score
        

        #### method 3
        #new = np.column_stack([self.feature.values,new])
        #clf = RandomizedLogisticRegression()
        #clf.fit(new, self.target.values)
        #rank = []
        ##for index in range(len(clf.feature_importances_)):
        #for index in range(len(clf.scores_)):
        #    #rank.append([clf.feature_importances_[index], index])
        #    rank.append([clf.scores_[index], index])
        #rank.sort(key=lambda x:x[0])
        #for index in range(len(rank)):
        #    #if rank[index][1] == len(clf.feature_importances_)-1 :
        #    if rank[index][1] == len(clf.scores_)-1 :
        #        score1 = index

        #clf = XGBClassifier()
        #clf.fit(new, self.target.values)
        #rank = []
        #for index in range(len(clf.feature_importances_)):
        #    rank.append([clf.feature_importances_[index], index])
        #rank.sort(key=lambda x:x[0])
        #for index in range(len(rank)):
        #    if rank[index][1] == len(clf.feature_importances_)-1 :
        #        score2 = index

        #clf = RandomForestClassifier()
        #clf.fit(new, self.target.values)
        #rank = []
        #for index in range(len(clf.feature_importances_)):
        #    rank.append([clf.feature_importances_[index], index])
        #rank.sort(key=lambda x:x[0])
        #for index in range(len(rank)):
        #    if rank[index][1] == len(clf.feature_importances_)-1 :
        #        score3 = index
            
        print(score)
        return score,

    def valid_new(self, new, new_list):
        scores = []
        for i in new_list:
            scores.append(abs(np.corrcoef(new,i)[0][1]))
        for col in self.feature.columns:
            scores.append(abs(np.corrcoef(new,self.feature[col].values)[0][1]))
        
        print("score")
        print(max(scores))
        print(max(scores)>0.9 or (np.any(np.isnan(new))) or (np.any(np.isinf(new))))
        if max(scores)>0.9 or (np.any(np.isnan(new))) or (np.any(np.isinf(new))):
            return False
        else:
            return True

    def get_best(self, fits, pset, k):
        new = []
        new_ins = []
        new_str = []
        new_score = []
        count = 0
        print("begin get best")
        for i in range(1, len(fits)):
            individual = fits[-i][0]
            print(i)
            print("count {}".format(count))
            print(str(individual))
            if str(individual) not in new_str:
                func = gp.compile(individual, pset)
                new_tmp = func(**self.feature_dict)
                if self.valid_new(new_tmp, new):
                    new.append(new_tmp)
                    new_str.append(str(individual))
                    new_score.append(fits[-i][1])
                    new_ins.append(individual)
                    count += 1
                    if count>=k:
                        break
              
        print("new_str")
        print(new)
        print(new_str)
        print(new_score)
        return new, new_ins

    def restore_ind_by_file(self, ind_file, predict_file, new_add_name):
        input_file = open(ind_file, 'rb')
        individual = pickle.load(input_file)
        input_file.close()
        func = gp.compile(individual, self.pset)
        print(str(individual))

        feature = pd.read_csv(predict_file)

        feature_dict = {}
        for col in feature.columns:
            feature_dict[col] = feature[col].values

        new_add = func(**feature_dict)
        feature[new_add_name] = new_add
        return feature 

    def restore_ind(self, ind, df):
        print(str(ind))
        func = gp.compile(ind, self.pset)
        print(str(func))
        feature_dict = {}
        for col in df.columns:
            feature_dict[col] = df[col].values
        new_add = func(**feature_dict)
        return new_add

        


if __name__ == "__main__":
    import pandas as pd
    from operator_config import config1
    from operator_config import config2
    from operator_config import config3
    from datacleaner import autoclean
    raw_data = pd.read_csv("/tmp/train.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)

    ag = AutoGenerator(clean_data, "Survived", config1)
    new_add_cols, transform_methods = ag.run(popsize=100, matepb=0.7, mutpb=0.2, gensize=10, selectsize=100, kbest=30)





