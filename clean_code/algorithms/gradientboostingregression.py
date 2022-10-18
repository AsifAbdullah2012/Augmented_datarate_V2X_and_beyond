import numpy as np
import scipy as sp
import scipy.stats
from sklearn.base import BaseEstimator, RegressorMixin
from algorithms.framework import Framework
from data.data_loader import S3Measurements
from sklearn.ensemble import GradientBoostingRegressor
# from dask_jobqueue import SLURMCluster
import dask
import joblib
from dask.distributed import Client, LocalCluster
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
import tensorflow as tf

class GradientBoostingRegression():
    @classmethod
    def pipeline_steps(cls):
        return [('impute', cls.get_per_column_imputation()),
                ('regr', GradientBoostingRegressor())]

    @classmethod
    def params_to_optimize(cls):
        return {
            'n_estimators': sp.stats.randint(low=10, high=300),
            'loss': ['quantile'],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_depth': sp.stats.randint(low=4, high=20),
            'min_samples_split': sp.stats.randint(low=2, high=20),
            'min_samples_leaf': sp.stats.randint(low=1, high=20),
            'max_features': sp.stats.uniform(loc=0, scale=1),
            'alpha': np.arange(.1, .31, 0.01, dtype=float), 
        }

    @classmethod
    def optimal_params(cls):
        return {
            'n_estimators': 298,
           #  'loss': ['quantile'],
            'criterion': 'squared_error',
            'max_depth': 17,
            'min_samples_split': 9,
            'min_samples_leaf': 15,
            'max_features': 0.7805291762864555,
           #  'alpha': .27, 
        }


    def start_cluster_in_CPU_Cluster(self, jobs):
        cluster = SLURMCluster(
        header_skip=['--mem', 'another-string', '-A'],
        queue='generic',
        project="myproj",
        cores=24,
        memory='400GB',
        walltime="60:00:00",
        n_workers=120,
        )
        cluster.scale(jobs)
        client = Client(cluster)

    def start_local_cluster(self):
        cluster = LocalCluster(
           n_workers = 60, 
           threads_per_worker = 1,
            memory_limit='30GB',
           dashboard_address = 'localhost:8787', 
        )
        client = Client(cluster)
        print('the client location is ', client)
    



class HyperparameterSearch_Gradient_Boosting_Regression(GradientBoostingRegressor, RandomizedSearchCV):
    def gethyperparameter_result(self, param, data, target):
        gbr = GradientBoostingRegressor()
        clf = RandomizedSearchCV(gbr, param, random_state=0)
        search = clf.fit(data, target)
        return search.best_params_



class GradientBoostingQuantileRegressor(GradientBoostingRegressor):


    def __init__(self, n_estimators=298, criterion="squared_error", max_depth=17, min_samples_split=9, min_samples_leaf=15,
                 min_weight_fraction_leaf=0., max_features=0.7805291762864555, max_leaf_nodes=None, min_impurity_decrease=0., random_state=None, verbose=0,
                 warm_start=False, alpha = .20, loss = 'quantile', min_sample_weight=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, random_state=random_state, loss = loss, alpha = alpha, verbose=verbose, warm_start=warm_start)
        self.min_sample_weight = min_sample_weight

    def fit(self, X, y):
        super().fit(X, y)

    def predict(self, X):
        return super().predict(X)

    def check_access(self):
        print('I am inside the random forest regression')






        
        





