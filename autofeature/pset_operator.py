import numpy as np
from scipy import stats as st
from sklearn import preprocessing


def ignore(x):
    return np.array(x)-np.array(x)

def add(x, y):
    return np.add(x, y)

def subtract(x, y):
    return np.subtract(x, y)

def multiply(x, y):
    return np.multiply(x, y)

def xor(x,y):
    try:
        return [1 if x[i]*y[i]!=0 else 0 for i in range(len(x))]
    except:
        return x


def xnor(x,y):
    try:
        return [1 if x[i]*y[i]==0 else 0 for i in range(len(x))]
    except:
        return x

def divide(x, y):
    try:
        if np.mean(y)!=0:
            y = [np.mean(y) if i==0 else i for i in y]
        else:
            y = [1 if i==0 else i for i in y]
        print(y)
        r = np.divide(x, y)
        r = [1 if str(i)=="inf" else i for i in r]
        r = [1 if str(i)=="-inf" else i for i in r]
        r = [1 if str(i)=="nan" else i for i in r]
        return np.array(r)
    except:
        return x

def bagging_2(x):
    part_num = 2

    return bagging(x, part_num)

def bagging_3(x):
    part_num = 3

    return bagging(x, part_num)

def bagging_4(x):
    part_num = 4

    return bagging(x, part_num)

def bagging_5(x):
    part_num = 5

    return bagging(x, part_num)

def bagging_10(x):
    part_num = 10

    return bagging(x, part_num)


def bagging(x, n):
    try:
        part_num = n
        interval = (max(x)-min(x))/part_num
        mapping = []
        for i in range(1, part_num+1):
            mapping.append([(i-1)*interval+min(x),i])
        mapping.sort(key=lambda x:x[0],reverse=True)
        new = []
        for i in x:
            for j in mapping:
                if i>=j[0]:
                    new.append(j[1])
                    break

        return new
    except:
        return x


def bigger(x):
    try:
        return np.max(x)
    except:
        return x
    

def smaller(x):
    try:
        return np.min(x)
    except:
        return x


def absolute(x):
    return np.abs(x)

def scale(x):
    try:
        return preprocessing.scale(x)
    except:
        return x

def min_max_scale(x):
    try:
        min_max_scaler = preprocessing.MinMaxScaler()
        return min_max_scaler.fit_transform(x)
    except:
        return x

def normalize(x):
    try:
        return preprocessing.normalize(x, norm='l2')[0]
    except:
        return x

if __name__ == "__main__":
    x= [1,1,1,1,1,1,1,8,9,10,11]
    print(min_max_scale(x))
    
