
from deap import base, creator, gp
from deap.gp import PrimitiveSet
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
def ignore(x):
    return x-x
pset = gp.PrimitiveSet("MAIN", 4)
pset.addPrimitive(np.add, 2)
pset.addPrimitive(np.subtract, 2)
pset.addPrimitive(np.multiply, 2)
#pset.addPrimitive(np.log, 1)
pset.addPrimitive(ignore, 1)
expr = gp.genFull(pset, min_=1, max_=10)

import pandas as pd
df=pd.read_csv("/tmp/train_after_etl.csv")
df=df.drop("id",axis=1)
df=df.drop("color",axis=1)
target = df["type"].values
feature = df.drop("type",axis=1)
feature
count=0
pset.renameArguments(ARG0='bone_length')
pset.renameArguments(ARG1='rotting_flesh')
pset.renameArguments(ARG2='hair_length')
pset.renameArguments(ARG3='has_soul')

tree = gp.PrimitiveTree(expr)
"rotting_flesh" in str(tree)
str(tree)
tree = gp.PrimitiveTree(expr)

str(tree)

import random
from deap import creator, base, tools, algorithms
from sklearn.cross_validation import train_test_split


creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin,pset=pset)
toolbox = base.Toolbox()

toolbox.register("expr", gp.genFull, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.expr)
expr = toolbox.individual()
print(expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
#toolbox.register("compile", gp.compile, pset=pset)

def eval1(individual,pset): 
    print(dir(pset))
    print(individual)
    func = gp.compile(individual,pset)
    print("haha")
    bone_length = df["bone_length"].values
    rotting_flesh = df["rotting_flesh"].values
    hair_length = df["hair_length"].values
    has_soul = df["has_soul"].values
    print(dir(func))
    
    new=func(bone_length=bone_length,rotting_flesh=rotting_flesh,hair_length=hair_length,has_soul=has_soul)
    new = [[i] for i in new]
    print(len(new))
    print(len(target))
    x_train, x_test, y_train, y_test = train_test_split(new, target)
    clf=RandomForestClassifier()
    clf.fit(x_train,y_train)
    pred=clf.predict(x_test)
    f1 = f1_score(y_test,pred)
    return f1,

toolbox.register("evaluate", eval1,pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
pop = toolbox.population(n=300)
hof = tools.HallOfFame(1)
pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, halloffame=hof, verbose=True)

fits = [ind.fitness.values[0] for ind in pop]
#for i in pop:
#    print(dir(i))
#    print(str(i))
fits.sort()
fits[-10:]
