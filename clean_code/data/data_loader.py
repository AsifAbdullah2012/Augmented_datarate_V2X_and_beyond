import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut

# imports for the ml algo (gradientBoostingRegression and randomForest)

import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
import numpy as np
import re
import math 
from scipy import fftpack
from math import sqrt
from csv import reader
import datetime as dt
from datetime import datetime
import pdb
import seaborn as sns
import dask
import scipy as sp
import joblib


# for dask as a single machine 
from dask.distributed import Client, LocalCluster

# for ml computation
# from dask_jobqueue import SLURMCluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.compose import make_column_transformer
from sklearn.impute import  SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import LeaveOneGroupOut

#---------------------finishes here ---------------------------------------------



def _package_path():
    """
    Returns
    -------
    package_path : str
        directory path of this package (data directories will be subdirectories of this path)
    """
    return os.path.dirname(sys.modules[__name__].__file__) + '/'


class S3Measurements:
    def __init__(self):
        self.save_all_values = self.load_measurements_ml_data()

    # cluster ON
    def start_cluster(self):
        cluster1 = SLURMCluster(scheduler_options={'dashboard_address': ':8784'}, cores=4, memory="160 GB", n_workers=150, walltime="15000000")   
        client = Client(cluster1)
        print(client.cluster)


    # load the DF
    # return a list of dfs 0 is for round 1, 1 is for round 2, ......, 5 is for round 6
    @staticmethod
    def load_df(filter_unreliable_measurements=True):
        # file 1 
        print('# ----> ', _package_path())
        filename1 = _package_path() + "/ML_data_v2/ml_data_1.csv"
        df_round_1 = pd.read_csv(filename1)
        df_round_1 = df_round_1.dropna()
        # file 2 
        filename2 = _package_path() + "/ML_data_v2/ml_data_2.csv"
        df_round_2 = pd.read_csv(filename2)
        df_round_2 = df_round_2.dropna()
        # file 3 
        filename3 = _package_path() + "/ML_data_v2/ml_data_3.csv"
        df_round_3 = pd.read_csv(filename3)
        df_round_3 = df_round_3.dropna()
        # file 4
        filename4 = _package_path() + "/ML_data_v2/ml_data_4.csv"
        df_round_4 = pd.read_csv(filename4)
        df_round_4 = df_round_4.dropna()
        # file 5
        filename5 = _package_path() + "/ML_data_v2/ml_data_5.csv"
        df_round_5 = pd.read_csv(filename5)
        df_round_5 = df_round_5.dropna()
        # file 6
        filename6 = _package_path() + "/ML_data_v2/ml_data_6.csv"
        df_round_6 = pd.read_csv(filename6)
        df_round_6 = df_round_6.dropna()

        lis = list()
        lis.append(df_round_1)
        lis.append(df_round_2)
        lis.append(df_round_3)
        lis.append(df_round_4)
        lis.append(df_round_5)
        lis.append(df_round_6)

        return lis


    # read all the data from ML_data_v2 folder 
    # ml_data_1.csv to .. ml_data_6.csv
    # train test split 
    @staticmethod
    def load_measurements_ml_data(filter_unreliable_measurements=True):
        # file 1 
        filename1 =  "/mnt/my_dataset/ml_data_1.csv"
        df_round_1 = pd.read_csv(filename1)
        df_round_1 = df_round_1.dropna()
        # file 2 
        filename2 =  "/mnt/my_dataset/ml_data_2.csv"
        df_round_2 = pd.read_csv(filename2)
        df_round_2 = df_round_2.dropna()
        # file 3 
        filename3 =  "/mnt/my_dataset/ml_data_3.csv"
        df_round_3 = pd.read_csv(filename3)
        df_round_3 = df_round_3.dropna()
        # file 4
        filename4 =  "/mnt/my_dataset/ml_data_4.csv"
        df_round_4 = pd.read_csv(filename4)
        df_round_4 = df_round_4.dropna()
        # file 5
        filename5 =  "/mnt/my_dataset/ml_data_5.csv"
        df_round_5 = pd.read_csv(filename5)
        df_round_5 = df_round_5.dropna()
        # file 6
        filename6 = "/mnt/my_dataset/ml_data_6.csv"
        df_round_6 = pd.read_csv(filename6)
        df_round_6 = df_round_6.dropna()

        #-----------------------------------------------train test generation --------------
        # train test generation 
        X_df_round_1 = df_round_1
        X_df_round_1 = X_df_round_1.drop(['MCS'], axis=1)
        print(len(X_df_round_1))
        y_df_round_1 = df_round_1['MCS']

        X_df_round_2 = df_round_2
        X_df_round_2 = X_df_round_2.drop(['MCS'], axis=1)
        print(len(X_df_round_2))
        y_df_round_2 = df_round_2['MCS']

        X_df_round_3 = df_round_3
        X_df_round_3 = X_df_round_3.drop(['MCS'], axis=1)
        print(len(X_df_round_3))
        y_df_round_3 = df_round_3['MCS']

        X_df_round_4 = df_round_4
        X_df_round_4 = X_df_round_4.drop(['MCS'], axis=1)
        print(len(X_df_round_4))
        y_df_round_4 = df_round_4['MCS']

        X_df_round_5 = df_round_5
        X_df_round_5 = X_df_round_5.drop(['MCS'], axis=1)
        print(len(X_df_round_5))
        y_df_round_5 = df_round_5['MCS']

        X_df_round_6 = df_round_6
        X_df_round_6 = X_df_round_6.drop(['MCS'], axis=1)
        print(len(X_df_round_6))
        y_df_round_6 = df_round_6['MCS']
        
        df_X = pd.concat([X_df_round_1, X_df_round_2, X_df_round_3, X_df_round_4, X_df_round_5, X_df_round_6])
        df_Y = pd.concat([y_df_round_1, y_df_round_2, y_df_round_3, y_df_round_4, y_df_round_5, y_df_round_6])
        
        print(len(df_X))
        
        lis = list()
        lis.append(df_X)
        lis.append(df_Y)
        
        return lis


    def get_all_measurements(self):
        return self.save_all_values.copy()

    # get the data rate ---
    # test_pred_val is a list 0 -- contains the ideal data rate and 1 contains predicted data rate 
    def get_datarate_measurements(self, pred_val, test):
        test_pred_val = list()
        ln = len(pred_val)
        pred_val_floor = list()
        for i in range(ln):
            pred_val_floor.append(math.floor(pred_val[i]))
            
        mcs_data = [1352, 1800, 2216, 2856, 3496, 4392, 5160, 5992, 6968, 7736, 8504, 8504, 9912, 11064, 12576, 14112, 15264, 15840, 17568, 19080]
        
        lth = len(pred_val_floor)
        predict_data_rate = 0.0
        test_data_rate = 0.0
        for i in range(lth):
            if int(pred_val_floor[i]) < 0:
                pred_val_floor[i] = 0
            if int(pred_val_floor[i]) > 19:
                pred_val_floor[i] = 19
            d1 = mcs_data[int(pred_val_floor[i])]
            d2 = mcs_data[int(test[i])]
            test_data_rate = test_data_rate + d2
            if pred_val_floor[i] <= test[i]:
                predict_data_rate = predict_data_rate + d1

        predict_data_rate = predict_data_rate/1000
        test_data_rate = test_data_rate/1000
        predict_data_rate = predict_data_rate/lth
        test_data_rate = test_data_rate/lth
        test_pred_val.append(test_data_rate)
        test_pred_val.append(predict_data_rate)
        
        return test_pred_val   

    def over_predicted_mcs(self, pred_mcs, test_mcs):
        siz = len(pred_mcs)
        cnt = 0
        for i in range(siz):
            if pred_mcs[i] > test_mcs[i]:
                cnt = cnt + 1
        return ((cnt/siz)*100)


    def generate_graphs(self, df):
        print('generating graphs ....')
        pass

    def dummy_method_from_s3(self):
        print('in the data folders file ...  ')



