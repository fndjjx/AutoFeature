from loop_bagging import LoopBagging
from autogenerator import AutoGenerator
from autoselection import AutoSelection
from coltype import ColType
from get_dummy import get_dummy, restore_dummy
import pandas as pd
import numpy as np
from data_alignment import alignment
from pre_select import pre_select
from balance import blance_positive_negative
from data_manager import DataManager
from feature_selection import FeatureSelection
from remove_same import remove_same

class AutoFeature():

    def __init__(self, train_df, target_label, final_feature_number, test_df=None):
        self.c_transform_methods = {}
        self.d_transform_methods = {}
        self.bagging_methods = {}
        self.select_col = None
        self.final_feature_number = final_feature_number

        dm = DataManager(train_df, test_df, target_label)
        self.train_df, self.test_df, self.dummy_candidate_col = dm.run()
        self.train_df.to_csv("/tmp/train_after_etl.csv",index=False)
        self.target_label = target_label


    def fit(self, c_config, d_config):
        target = self.train_df[self.target_label]
        feature = self.train_df.drop(self.target_label,axis=1) 

        #step1
        discrete_df, continuous_df = ColType(feature).run()
        print("cshape")
        print(continuous_df.shape)
        print(discrete_df.shape)
        self.continuous_df_col = continuous_df.columns
        self.discrete_df_col = discrete_df.columns

        #step2
        print("begin generate")
        continuous_df = pd.concat([continuous_df, target],axis=1)
        ag = AutoGenerator(continuous_df, self.target_label, c_config)
        new_add_cols, transform_methods = ag.run(popsize=500, matepb=0.7, mutpb=0.2, gensize=20, selectsize=100, kbest=50)
        for i in range(len(new_add_cols)):
            col_name = "new{}".format(i)
            continuous_df[col_name] = new_add_cols[i]
            self.c_transform_methods[col_name] = transform_methods[i]


        #step3
        lb = LoopBagging(continuous_df, self.target_label)
        new_add_bagging = lb.run()
        print("new add baggung")
        print(new_add_bagging)
        for col,value in new_add_bagging.items():
            col_name = "bagging{}".format(col)
            discrete_df[col_name] = value[2]
            self.bagging_methods[col] = value[0]

        discrete_df = pd.concat([discrete_df, target],axis=1)
        ag = AutoGenerator(discrete_df, self.target_label, d_config)
        new_add_cols, transform_methods = ag.run(popsize=500, matepb=0.7, mutpb=0.2, gensize=20, selectsize=100, kbest=50)
        for i in range(len(new_add_cols)):
            col_name = "newd{}".format(i)
            discrete_df[col_name] = new_add_cols[i]
            self.d_transform_methods[col_name] = transform_methods[i]

         
        #step4
        train_df = pd.concat([discrete_df, continuous_df],axis=1)
        train_df = get_dummy(train_df, self.dummy_candidate_col)


        train_df = remove_same(train_df)
        train_df.to_csv("/tmp/middle.csv",index=False)
        fs = FeatureSelection()
        train_df, self.select_col = fs.run(train_df, self.target_label, self.final_feature_number)

#        train_df_without_target = train_df.drop(target_label,axis=1)
#        train_df_without_target_with_dummy, self.dummy_col = get_dummy(train_df_without_target)
#        train_df = pd.concat([train_df_without_target_with_dummy, target],axis=1)

        #step
#        train_df = pre_select(train_df, target_label)
#
#        #step5
#        myas = AutoSelection(train_df, target_label)
#        myas.run(pop_num=100, cxpb=0.6, mutpb=0.2, gen_num=10)
#        train_df, select_col=myas.get_best()
        train_df = remove_same(train_df)
        
        return train_df

    def transform(self, c_config, d_config):
        #step1
        test_continuous_df = self.test_df[self.continuous_df_col]
        test_discrete_df = self.test_df[self.discrete_df_col]

        ag = AutoGenerator(test_continuous_df, None, c_config)
        new_add = {}
        for col, transform_method in self.c_transform_methods.items():
            new_add[col] = ag.restore_ind(transform_method, test_continuous_df)
        for col, new_add_value in new_add.items():
            test_continuous_df[col] = new_add_value
     
        #step2
        for col, bagging_method in self.bagging_methods.items():
            test_discrete_df["bagging{}".format(col)] = bagging_method(test_continuous_df[col].values)

        ag = AutoGenerator(test_discrete_df, None, d_config)
        new_add = {}
        for col, transform_method in self.d_transform_methods.items():
            new_add[col] = ag.restore_ind(transform_method, test_discrete_df)
        for col, new_add_value in new_add.items():
            test_discrete_df[col] = new_add_value

        test_df = pd.concat([test_discrete_df, test_continuous_df],axis=1)

        test_df = get_dummy(test_df, self.dummy_candidate_col)
        test_df = remove_same(test_df)
        test_df.to_csv("middle2.csv",index=False)

        #step3
        new_test_df = pd.DataFrame()
        for col in self.select_col:
            if col in test_df.columns:
                new_test_df[col] = test_df[col].values
        new_test_df = remove_same(new_test_df)
        return new_test_df


if __name__=="__main__":
    from operator_config import config1, config2
    from datacleaner import autoclean

    raw_data = pd.read_csv("/tmp/train.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/train_after_etl.csv", sep=',', index=False)
    train_df = pd.read_csv("/tmp/train_after_etl.csv")

    raw_data = pd.read_csv("/tmp/test.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/test_after_etl.csv", sep=',', index=False)
    test_df = pd.read_csv("/tmp/test_after_etl.csv")


    af = AutoFeature(train_df, "Survived", 20, test_df)
    train_df = af.fit(config1,config2)
    train_df.to_csv("/tmp/train_after_etl2.csv", sep=',', index=False)


    test_df = af.transform(config1,config2)
    test_df.to_csv("/tmp/test_after_etl2.csv", sep=',', index=False)



