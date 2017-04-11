from deap import base, creator, gp
from deap import creator, base, tools, algorithms
from scoop import futures
import hashlib
import random
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import f1_score, make_scorer, accuracy_score, mutual_info_score, auc
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
from minepy import MINE
from pset_operator import bagging_2, bagging_3,bagging_4, bagging_5
from loop_bagging import LoopBagging

def random_str(randomlength=8):
    string = ""
    chars = "AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789"
    length = len(chars)-1
    random = Random()
    for i in range(randomlength):
        string+=chars[random.randint(0, length)]
    return string


class AutoGenerator():

    def __init__(self, df, target_label, operator_dict):
           
        if target_label:
            self.target = df[target_label]
            self.feature = df.drop(target_label,axis=1)
        else:
            self.target = None
            self.feature = df
        self.feature_dict = {}
        for col in self.feature.columns:
            self.feature_dict[col] = self.feature[col].values

   
        self.pset = self.init_pset(operator_dict)
        print("generator init finish")

    def init_pset(self, operator_dict):
    
        pset = gp.PrimitiveSet("MAIN", len(self.feature.columns))
        for op, op_num in operator_dict.items():
            pset.addPrimitive(op, op_num)

        for col_index in range(len(self.feature.columns)):
            para = "{'ARG%s':'%s'}"%(col_index, self.feature.columns[col_index])
            pset.renameArguments(**eval(para))

        ephemeral = hashlib.md5(random_str().encode('utf-8')).hexdigest()
        pset.addEphemeralConstant(ephemeral, lambda: np.random.uniform(-10, 10))


        return pset

    def run(self, popsize=100, matepb=0.6, mutpb=0.3, gensize=20, selectsize=10, kbest=1):
        print("begin run generate")
        creator.create("FitnessMin", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,pset=self.pset)
        toolbox = base.Toolbox()
        toolbox.register("map", futures.map)

        toolbox.register("expr", gp.genFull, pset=self.pset, min_=1, max_=3)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self.evaluation, pset=self.pset)
        toolbox.register("select", tools.selTournament, tournsize=selectsize)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)
        pop = toolbox.population(n=popsize)

        pop, log = algorithms.eaSimple(pop, toolbox, matepb, mutpb, gensize, verbose=True)

        fits = []
        for i in pop:
            fits.append([i, i.fitness.values[0], str(i)])
        fits.sort(key=lambda x:x[1])

        return self.get_best(fits, self.pset, k=kbest)


    def evaluation(self, individual, pset): 
        func = gp.compile(individual,pset)
        
        new=func(**self.feature_dict)
        print(str(individual))
        if np.std(new)<0.001 or (np.mean(new)!=0 and abs(np.std(new)/np.mean(new)) < 0.001):
            return -10,
        ### method 1
     #   x = self.feature.values
     #   x = np.column_stack([self.feature.values,new])
        #x = [[i] for i in new]
     #   random_num = np.random.randint(0, len(model_list))
     #   random_num = len(model_list)-1
     #   clf=model_list[random_num]()
        #scores = []
        #for i in range(len(metric_list)):
        #    scorer = make_scorer(metric_list[i])
        #scorer = make_scorer(f1_score)
        #    scores.append(np.mean(cross_val_score(clf, x, self.target.values, scoring=scorer, n_jobs=-1, cv=3)))
     #   scorer = make_scorer(accuracy_score)
     #   score=np.mean(cross_val_score(clf, x, self.target.values, scoring=scorer, n_jobs=-1, cv=3))
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
        #x = [[i] for i in new]
      #  x = np.column_stack([self.feature.values,new])
      #  clf=model_list[random_num]()
      #  scorer = make_scorer(accuracy_score)
      #  score2=np.mean(cross_val_score(clf, x, self.target.values, scoring=scorer, n_jobs=-1, cv=3))
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


        
        mine = MINE()
        mine.compute_score(self.target.values, new)
        score=mine.mic()/len(np.unique(new))
 
        #score=mutual_info_score(self.target.values, new)/len(np.unique(new))

#        np.random.shuffle(new)
#        mine = MINE()
#        mine.compute_score(self.target.values, new)
#        score2=mine.mic()/len(np.unique(new))
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
        #es = []
        #for i in range(5):
        #    clf = ExtraTreesClassifier()
        #    clf.fit(new, self.target.values)
        #    es.append(clf.feature_importances_[-1])
        #score = np.mean(es)
        print(score)
        return score,

    def get_best(self, fits, pset, k):
        new = []
        new_ins = []
        new_str = []
        new_score = []
        count = 0
        for i in range(1, len(fits)):
            individual = fits[-i][0]
            print(individual)
            if str(individual) not in new_str:
                func = gp.compile(individual, pset)
                new.append(func(**self.feature_dict))
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
    clean_data.to_csv("/tmp/train_after_etl.csv", sep=',', index=False)



    instances11 = []
    df = pd.read_csv("/tmp/train_after_etl.csv") 
    #l = ["Name","Cabin","Fare","Ticket", "Survived"]
    df1 = df.drop("PassengerId",axis=1)
    df1 = df1.drop("Sex",axis=1)
    df1 = df1.drop("Pclass",axis=1)
    df1 = df1.drop("Embarked",axis=1)
    df1 = df1.drop("Name",axis=1)
    af = AutoFeature(df1, "Survived", config1)
    new, instance = af.run(popsize=100, matepb=0.6, mutpb=0.3, gensize=20, selectsize=10, kbest=20)
    print(new)
    for i in range(len(new)):
        df["new{}".format(i)] = new[i]
    df.to_csv("/tmp/train_after_etl.csv",index=False)
    filename = "new11{}".format(i)
    output = open(filename, 'wb')
    pickle.dump(instance[0], output)
    output.close()
    instances11.append(filename)


    instances12=[]
    df = pd.read_csv("/tmp/train_after_etl.csv")
    #l = ["Name","Cabin","Fare","Ticket", "Survived"]
    df1 = df.drop("PassengerId",axis=1)
    df1 = df1.drop("Sex",axis=1)
    df1 = df1.drop("Pclass",axis=1)
    df1 = df1.drop("Embarked",axis=1)
    df1 = df1.drop("Name",axis=1)
    df1 = df1.drop("Cabin",axis=1)
    lb = LoopBagging(df1, "Survived")
    new = lb.run()
    print(new)
    for key,value in new.items():
        df["bagging{}".format(key)] = value[2]
        instances12.append([key, value[0], value[1]])
    print("nnn")
    print(instances12)
    df.to_csv("/tmp/train_after_etl.csv",index=False)


    raw_data = pd.read_csv("/tmp/test.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/test_after_etl.csv", sep=',', index=False)


    for i in instances11:
        print("restore")
        print(i)
        df = pd.read_csv("/tmp/test_after_etl.csv") 
        af = AutoFeature(df, None, config1)
        df = af.restore_ind(i, "/tmp/test_after_etl.csv", "new0")
        df.to_csv("/tmp/test_after_etl.csv",index=False)

    for i in range(len(instances12)):
        print("restore")
        print(i)
        df = pd.read_csv("/tmp/test_after_etl.csv")
    #    af = AutoFeature(df, None, config2)
    #    df = af.restore_ind(i, "/tmp/test_after_etl.csv","new0")
        col = instances12[i][0]
        f=instances12[i][1]
        df["new1{}".format(col)]=f(df[col].values)
        df.to_csv("/tmp/test_after_etl.csv",index=False)

