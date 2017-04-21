import numpy as np
import pandas as pd
from feature_selection import FeatureSelection
from remove_same import remove_same


class DataManager():
    def __init__(self, train_df, test_df, target_label):
        self.train_df = train_df
        self.test_df = test_df
        self.target_label = target_label

    def pre_select(self, df, target_label=None):
        cols = list(df.columns)
        if target_label!=None:
            target = df[target_label]
            cols.remove(target_label)

        #method1

#        col_std_list = []
#        for col in cols:
#            col_value = df[col].values
#            min_value = min(col_value)
#            max_value = max(col_value)
#            min_max_col_value = [(v-min_value)/(max_value-min_value) for v in col_value]
#            col_std = np.std(min_max_col_value)
#            print(col_std)
#            if str(col_std) == "nan":
#                col_std_list.append([col,col_value,0])
#            else:
#                col_std_list.append([col,col_value,col_std])
#
#        col_std_mean = np.mean([i[2] for i in col_std_list])
#        col_std_std = np.std([i[2] for i in col_std_list])
#        print("mean std")
#        print(col_std_mean)
#        print(col_std_std)
#        print(col_std_list)
#        
#        new_df = pd.DataFrame()
#        for i in col_std_list:
#            print("haha")
#            print(i[2])
#            print(col_std_mean-col_std_std)
#            if i[2]>col_std_mean:#-col_std_std:
#                new_df[i[0]] = i[1]

        #method2
        new_df = pd.DataFrame()
        for col in cols:
            col_value = df[col].values
            if np.std(col_value)!=0:
                new_df[col] = col_value



        if target_label!=None:
            new_df = pd.concat([new_df, target],axis=1)
        return new_df

    def pre_select2(self, train_df, test_df, target_label):
        fs = FeatureSelection()
        train_df, select_col = fs.run(train_df, target_label, 150)

        new_test_df = pd.DataFrame()
        for col in select_col:
            if col in test_df.columns:
                new_test_df[col] = test_df[col].values
        new_test_df = remove_same(new_test_df)
        return train_df, new_test_df


            
    def alignment(self, train_df, test_df):
        train_col = list(train_df.columns)
        test_col = list(test_df.columns)
        all_col = set(train_col + test_col)
        train_after_aligment = pd.DataFrame()
        test_after_aligment = pd.DataFrame()
        for col in all_col:
            if col in train_col and col in test_col:
                train_after_aligment[col] = train_df[col].values
                test_after_aligment[col] = test_df[col].values
        train_after_aligment = pd.concat([train_after_aligment, train_df[self.target_label]], axis=1)
        return train_after_aligment, test_after_aligment

    def blance_positive_negative(self, df, target_label):
        target = df[target_label]
        y = df[target_label].values
        df = df.drop(target_label, axis=1)
        x = df.values
        y_positive = y[np.where(y==1)]
        y_negative = y[np.where(y==0)]
        x_positive = x[np.where(y==1)]
        x_negative = x[np.where(y==0)]

        scale =  len(y_positive)/len(y_negative)
        if scale < 2 and scale > 0.5:
            return pd.concat([df,target],axis=1)
        elif scale >= 2:
            y_need_add = y_negative
            x_need_add = x_negative
            scale_int = int(scale - 1)
        elif scale <= 0.5:
            y_need_add = y_positive
            x_need_add = x_positive
            scale_int = int(1//scale - 1)

        y_add = np.array(list(y_need_add)*scale_int)
        x_add = np.array(list(x_need_add)*scale_int)

        x = np.append(x, x_add, axis=0)
        y = np.append(y, y_add, axis=0)

        r = np.array(list(zip(x,y)))
        np.random.shuffle(r)
        x = r[:,:-1]
        x = [list(i[0]) for i in x]
        y = list(r[:,-1])
        x = np.array(x)
        new_df = pd.DataFrame()
        for col_index in range(len(df.columns)):
            new_df[df.columns[col_index]] = x.T[col_index]
        target = pd.DataFrame({target_label:y})
        new_df = pd.concat([new_df, target],axis=1)
        return new_df

    def dummy_candidate(self, train_df, test_df):
        dummy_candidate = []
        for col in train_df.columns:
            if col!= self.target_label:
                unique_value_train = set(list(np.unique(train_df[col])))
                unique_value_test = set(list(np.unique(test_df[col])))
                if unique_value_train == unique_value_test:
                    dummy_candidate.append(col)
        return dummy_candidate
            

    def run(self):
        train_df = self.pre_select(self.train_df, self.target_label)
        test_df = self.pre_select(self.test_df)
        train_df, test_df = self.alignment(train_df, test_df)
        print("after aligment")
        print(train_df.shape)
        print(test_df.shape)
        train_df = self.blance_positive_negative(train_df, self.target_label)
        train_df, test_df = self.pre_select2(train_df, test_df, self.target_label)
        print("after pre select")
        print(train_df.shape)
        print(test_df.shape)
        dummy_candidate_col = self.dummy_candidate(train_df, test_df)
        return train_df, test_df, dummy_candidate_col



if __name__ == "__main__":
    from datacleaner import autoclean
    import pandas as pd

    raw_data = pd.read_csv("/tmp/train.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/train_after_etl.csv", sep=',', index=False)
    train_df = pd.read_csv("/tmp/train_after_etl.csv")

    raw_data = pd.read_csv("/tmp/test.csv", error_bad_lines=False)
    clean_data = autoclean(raw_data)
    clean_data.to_csv("/tmp/test_after_etl.csv", sep=',', index=False)
    test_df = pd.read_csv("/tmp/test_after_etl.csv")

    dm = DataManager(train_df, test_df, "Survived")
    train_df, test_df, dummy_candidate_col = dm.run()
    print(dummy_candidate_col)

            
        



