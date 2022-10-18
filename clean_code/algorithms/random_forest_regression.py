import numpy as np
import scipy as sp
import scipy.stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib
import tensorflow as tf

from .framework import Framework


class RandomForest(Framework):
    @classmethod
    def pipeline_steps(cls):
        return [('impute', cls.get_per_column_imputation()),
                ('regr', RandomForestQuantileRegressor())]

    @classmethod
    def params_to_optimize(cls):
        # return {
        #     'regr__n_estimators': np.arange(10, 300, 1), # sp.stats.randint(low=10, high=300),
        #     'regr__criterion': ['mse', 'mae'],
        #     'regr__max_depth': np.arange(4, 20, 1), # sp.stats.randint(low=4, high=20),
        #     'regr__min_samples_split': np.arange(2, 20, 1), # sp.stats.randint(low=2, high=20),
        #     'regr__min_samples_leaf': np.arange(5, 20, 1), # sp.stats.randint(low=1, high=20),
        #     'regr__max_features': sp.stats.uniform(loc=0, scale=1),
        # }
        return {
            'n_estimators': sp.stats.randint(low=10, high=300),
            'criterion': ['squared_error'],
            'max_depth': sp.stats.randint(low=4, high=20),
            'min_samples_split': sp.stats.randint(low=2, high=20),
            'min_samples_leaf': sp.stats.randint(low=1, high=20),
            'max_features': sp.stats.uniform(loc=0, scale=1),
        }


    @classmethod
    def optimal_params(cls):
        return {
            # 'criterion': 'squared_error',
            # 'max_depth': 16,
            # 'max_features': 0.5928446182250183,
            # 'min_samples_leaf': 1,
            # 'min_samples_split': 5,
            # 'n_estimators': 261

            # 'criterion': 'squared_error',
            # 'max_depth': 13,
            # 'max_features': 0.4539141500147891,
            # 'min_samples_leaf': 2,
            # 'min_samples_split': 3,
            # 'n_estimators': 245

            'criterion': 'squared_error',
            'max_depth': 18,
            'max_features': 0.5865129348100832,
            'min_samples_leaf': 15,
            'min_samples_split': 9,
            'n_estimators': 296
        }


class RandomForestClassi(Framework):
    @classmethod
    def pipeline_steps(cls):
        return [('regr', RandomForestClassifier())]

    @classmethod
    def params_to_optimize(cls):
        return {
            'regr__n_estimators': sp.stats.randint(low=10, high=300),
            'regr__criterion': ['gini', 'entropy'],
            'regr__max_depth': sp.stats.randint(low=4, high=20),
            'regr__min_samples_split': sp.stats.randint(low=2, high=20),
            'regr__min_samples_leaf': sp.stats.randint(low=1, high=20),
            'regr__max_features': sp.stats.uniform(loc=0, scale=1),
        }

    @classmethod
    def optimal_params(cls):
        return {
            'regr__criterion': 'entropy',
            'regr__max_depth': 16,
            'regr__max_features': 0.08928606191827915,
            'regr__min_samples_leaf': 3,
            'regr__min_samples_split': 6,
            'regr__n_estimators': 109
        }

class HyperparameterSearch(RandomForestRegressor, RandomizedSearchCV):
    def gethyperparameter_result(self, param, data, target):
        ran = RandomForestQuantileRegressor()
        clf = RandomizedSearchCV(ran, param, n_iter=100, n_jobs=-1, random_state=0)
        search = clf.fit(data, target)
        return search.best_params_

class RandomForestQuantileRegressor(RandomForestRegressor):
    """
    Modified version of random forest, that uses the sorted predictions from the individual trees to get quantile
    estimations. If no quantile os given, it behaves like a normal random forest.
    """

    def __init__(self, n_estimators=296, criterion="squared_error", max_depth=18, min_samples_split=15, min_samples_leaf=9,
                 min_weight_fraction_leaf=0., max_features=0.5865129348100832, max_leaf_nodes=None, min_impurity_decrease=0.,
                 bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0,
                 warm_start=False, min_sample_weight=None):
        super().__init__(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                         min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                         min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                         max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                         bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs,
                         random_state=random_state, verbose=verbose, warm_start=warm_start)
        self.min_sample_weight = min_sample_weight

    def fit(self, X, y, sample_weights=None):
        if sample_weights is None and self.min_sample_weight is not None:
            sample_weights = 1 / np.maximum(y, self.min_sample_weight)
            sample_weights /= np.mean(sample_weights)
        super().fit(X, y, sample_weights)

    def predict(self, X, quantile=None):
        if quantile is None:
            return super().predict(X)
        else:
            per_tree_pred = [tree.predict(X) for tree in self.estimators_]
            predictions = np.stack(per_tree_pred)
            predictions.sort(axis=0)
            return predictions[int(round(len(per_tree_pred) * quantile / 100)), :]
    def check_access(self):
        print('I am inside the random forest regression')
