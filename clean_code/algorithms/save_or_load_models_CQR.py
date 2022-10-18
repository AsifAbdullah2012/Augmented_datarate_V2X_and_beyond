# import 
from importlib.resources import path
from wsgiref.util import request_uri
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
from functools import partial
# import seaborn as sns
# import dask
import scipy as sp
# import joblib
# from dask.distributed import Client, LocalCluster
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
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import dask
# import keras_tuner as kt
from datetime import datetime
from tensorflow.keras.models import Sequential, model_from_json

class Save_or_Load_model_plus_CQR():
    def save_model(self, model):
        model_json = model.to_json()
        
        now_time = datetime.now()
        path_model = 'algorithms/saved_models/Random_forest/' + 'RF_.20' + '_saved_model.json'
        with open(path_model, "w") as json_file:
            json_file.write(model_json)
        path_model_weight = 'algorithms/saved_models/Random_forest/' + 'RF_.20' + '_saved_model_weights.h5'
        model.save_weights(path_model_weight)
        print('model saved successfully ..')
    
    def load_model(self, path_model, path_model_data):
        json_file = open(path_model, 'r')
        loaded_model_lo = json_file.read()
        json_file.close()
        loaded_model_lo = model_from_json(loaded_model_lo)
        loaded_model_lo.load_weights(path_model_data)
        print('loaded model successfully ..')
        return loaded_model_lo
    
    # send me scalled data for neural network, other algo dont send me scalled data 
    # divide data ---> train, test 
    # here the test data is being sent for further dividation 
    # divide test ---> test, calibrated 
    # -- return a list where 0 prediction lo and hi for prediction hi  
    def CQR(self, X_test, y_test, quantile, path_for_model_lo, path_for_model_hi, path_for_model_data_lo, path_for_model_data_hi):
        # divide 40% in ratio 
        test_x, calibrated_x, test_y, calibrated_y = train_test_split(X_test, y_test, test_size=0.4)
        # test_y = test_y.iloc[: , 1:]
        # calibrated_y = calibrated_y.iloc[: , 1:]
        # calibrated_x = calibrated_x.iloc[:, 1:]
        loaded_model_lo = self.load_model(path_for_model_lo, path_for_model_data_lo)
        loaded_model_hi = self.load_model(path_for_model_hi, path_for_model_data_hi)
        pred_lo = loaded_model_lo.predict(calibrated_x)
        pred_hi = loaded_model_hi.predict(calibrated_x)
        calibrated_y = calibrated_y.to_numpy()
        print('TILL Here ------>')
                # print('pred_lo ', type(pred_lo), len(pred_lo), ' pred_hi ', type(pred_hi), len(pred_hi) )
        pred_lo = pred_lo.flatten()
        pred_hi = pred_hi.flatten()
        print(' # pred_lo ', pred_lo)
        print(' # pred_hi ', pred_hi)
        print(' # calibrated y ', calibrated_y)
        errors = np.maximum(pred_lo - calibrated_y, calibrated_y - pred_hi)

        significance = (((quantile) * 2) / 100.00)
        correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(calibrated_y) + 1) / len(calibrated_y)))
        # test_x = test_x.iloc[:, 1:]
        print('TILL Here ------>')
        test_pred_lo = loaded_model_lo.predict(test_x)
        test_pred_hi = loaded_model_hi.predict(test_x)
        test_pred_lo = test_pred_lo - correction
        test_pred_hi = test_pred_hi + correction
        lis_for_cqr = list()
        lis_for_cqr.append(test_pred_lo)
        lis_for_cqr.append(test_pred_hi)
        lis_for_cqr.append(test_x)
        lis_for_cqr.append(calibrated_x)
        lis_for_cqr.append(test_y)
        lis_for_cqr.append(calibrated_y)
        # 0 for the lo, and 1 for the hi, 2 for test_x, 3 for calibrated_x, 4 for test_y, 5 for calibrated_y
        return lis_for_cqr
        

    def graph_plot(self, cqr_lis):
        pred_lo_cqr = cqr_lis[0]
        pred_hi_cqr = cqr_lis[1]
        real_val = cqr_lis[4]
        fig, ax = plt.subplots()
        lo = pred_lo_cqr[40:80]
        hi = pred_hi_cqr[40:80]
        x = np.arange(len(lo))
        y1 = lo.flatten()
        y2 = hi.flatten()
        print('lo shape ', y1.shape, 'hi shape ', y2.shape)
        ax.fill_between(x, y1, y2, alpha=.2, linewidth=0)
        real = real_val[40:80]
        # real = real.flatten()
        ax.plot(x, real, linewidth=2)    

    def CQR_m(self, X_test, y_test, quantile, significance, path_for_model_lo, path_for_model_hi, path_for_model_mid, path_for_model_data_lo, path_for_model_data_hi, path_for_model_data_mid):
        test_x, calibrated_x, test_y, calibrated_y = train_test_split(X_test, y_test, test_size=0.4)
        loaded_model_lo = self.load_model(path_for_model_lo, path_for_model_data_lo)
        loaded_model_hi = self.load_model(path_for_model_hi, path_for_model_data_hi)
        loaded_model_mid = self.load_model(path_for_model_mid, path_for_model_data_mid)
        pred_lo = loaded_model_lo.predict(test_x)
        pred_hi = loaded_model_hi.predict(test_x)
        # pred_mid = loaded_model_mid.predict(test_x)

        pred_mid = loaded_model_mid.predict(calibrated_x)
        eps = 1e-6
        pred_lo = pred_lo.flatten()
        pred_hi = pred_hi.flatten()
        errors = np.maximum((pred_lo - calibrated_y) / (pred_mid - pred_lo + eps), (calibrated_y - pred_hi) / (pred_hi - pred_mid + eps))
        correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(calibrated_y) + 1) / len(calibrated_y)))
        test_pred_mid = loaded_model_mid.predict(test_x)
        test_pred_lo = test_pred_lo - correction * (test_pred_mid - test_pred_lo + eps)
        test_pred_hi = test_pred_hi + correction * (test_pred_hi - test_pred_mid + eps)
        lis = list()
        lis.append(test_pred_lo)
        lis.append(test_pred_hi)
        return lis 

    def CQR_r(self, X_test, y_test, quantile, significance, path_for_model_lo, path_for_model_hi, path_for_model_mid, path_for_model_data_lo, path_for_model_data_hi, path_for_model_data_mid):
        test_x, calibrated_x, test_y, calibrated_y = train_test_split(X_test, y_test, test_size=0.4)
        loaded_model_lo = self.load_model(path_for_model_lo, path_for_model_data_lo)
        loaded_model_hi = self.load_model(path_for_model_hi, path_for_model_data_hi)
        eps = 1e-6
        pred_lo = loaded_model_lo.predict(test_x)
        pred_hi = loaded_model_hi.predict(test_x)
        scaling_factor = pred_hi - pred_lo + eps
        errors = np.maximum((pred_lo - calibrated_y) / scaling_factor, (calibrated_y - pred_hi) / scaling_factor)
        correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(calibrated_y) + 1) / len(calibrated_y)))
        scaling_factor = pred_hi -  pred_lo + eps
        test_pred_lo = pred_lo - correction * scaling_factor
        test_pred_hi = pred_hi + correction * scaling_factor
        lis = list()
        lis.append(test_pred_lo)
        lis.append(test_pred_hi)
        return lis

    def In_boundary(self, real_val, test_pred_lo, test_pred_hi):
        count = 0
        siz = len(real_val)
        for i in range(siz):
            if real_val[i] >= test_pred_lo[i] and real_val[i] <= test_pred_hi[i]:
                count = count + 1
        
        return ((count/siz)*100.00)


    def empirical_quantile(self, test_y, pred_y):
        siz = len(test_y)
        cnt = 0
        for i in range(siz):
            if pred_y[i] > test_y[i]:
                cnt = cnt + 1
        
        return (float)(cnt/siz)
    

