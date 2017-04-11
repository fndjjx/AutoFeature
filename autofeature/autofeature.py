from loop_bagging import LoopBagging
from autogenerator import AutoGenerator
from autoselection import AutoSelection
from coltype import ColType
import pandas as pd
import numpy as np

class AutoFeature():

    def __init__(self):
        self.transform_methods = {}
        self.bagging_methods = {}
        self.select_bitmap = None


    def fit(self, train_df, target_label, custom_col_type, generate_config):
        target = train_df[target_label]
        feature = train_df.drop(target_label,axis=1) 
        #step1
        discrete_df, continuous_df = ColType(feature).run()
        print(discrete_df.columns)
        print(continuous_df.columns)
        #step2
        print("begin generate")
        input_df = pd.concat([continuous_df, target],axis=1)
        ag = AutoGenerator(input_df, target_label, generate_config)
        new_add_cols, transform_methods = ag.run(popsize=100, matepb=0.6, mutpb=0.3, gensize=20, selectsize=10, kbest=5)
        for i in range(len(new_add_cols)):
            col_name = "new{}".format(i)
            train_df[col_name] = new_add_cols[i]
            continuous_df[col_name] = new_add_cols[i]
            self.transform_methods[col_name] = transform_methods[i]
        continuous_df = pd.concat([continuous_df, target],axis=1)
        #step3
        #lb = LoopBagging(train_df, target_label)
        lb = LoopBagging(continuous_df, target_label)
        new_add_bagging = lb.run()
        for col,value in new_add_bagging.items():
            col_name = "bagging{}".format(col)
            train_df[col_name] = value[2]
            self.bagging_methods[col] = value[0]
            train_df = train_df.drop(col,axis=1)
        
        #step4
        myas = AutoSelection(train_df, target_label)
        myas.run(pop_num=500, cxpb=0.6, mutpb=0.2, gen_num=10)
        train_df, self.select_bitmap=myas.get_best()
        return train_df

    def transform(self, test_df, generate_config):
        #step1
        ag = AutoGenerator(test_df, None, generate_config)
        new_add = {}
        for col, transform_method in self.transform_methods.items():
            new_add[col] = ag.restore_ind(transform_method, test_df)
        for col, new_add_value in new_add.items():
            test_df[col] = new_add_value
     
        #step2
        for col, bagging_method in self.bagging_methods.items():
            test_df["bagging{}".format(col)] = bagging_method(test_df[col].values)
            test_df = test_df.drop(col, axis=1)

        #step3
        test_df = test_df.loc[:,self.select_bitmap]
        return test_df


if __name__=="__main__":
    from operator_config import config1
    from datacleaner import autoclean

    raw_data = pd.read_csv("/tmp/train2.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/train_after_etl.csv", sep=',', index=False)
    df = pd.read_csv("/tmp/train_after_etl.csv")

    af = AutoFeature()
    #df = af.fit(df,"Survived", {"SibSp":"c","Parch":"c"}, config1)
    df = af.fit(df,"Survived", None, config1)
    df.to_csv("/tmp/train_after_etl2.csv", sep=',', index=False)

    raw_data = pd.read_csv("/tmp/test2.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/test_after_etl.csv", sep=',', index=False)

    df = pd.read_csv("/tmp/test_after_etl.csv")
    df = af.transform(df,config1)
    df.to_csv("/tmp/test_after_etl2.csv", sep=',', index=False)



