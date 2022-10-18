import logging
import re

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
# from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_validate, cross_val_predict, RandomizedSearchCV, KFold, LeaveOneGroupOut, \
    train_test_split
from sklearn.pipeline import Pipeline

class Framework:
    def __init__(self):
        pass
        
    @classmethod
    def _generate_test_data(cls, filename, features, data_source, cv_mode, classification_threshold, num_train_samples,
                            category):
        # noinspection PyTypeChecker
        df: pd.DataFrame = pd.read_hdf(filename, 'data')

        if category and category != 'All':
            df = df[df['Category'] == category]
        if 'Category' in features:
            unique_categories = df['Category'].unique()
            category_mapping = dict(zip(unique_categories, range(len(unique_categories))))
            df['Category'] = df['Category'].replace(category_mapping)

        if num_train_samples:
            assert cv_mode == 'shuffle', "Selecting number of training samples only works with shuffle CV mode"
            num_samples = int(num_train_samples * 1.5)  # 1/3 of samples is used for training
            assert num_samples <= df.shape[0], "More samples requested than available"
            df = df.sample(n=num_samples, axis='index')

        if data_source.startswith('romes'):
            gps_features = [x for x in ['Latitude', 'Longitude', 'Altitude', 'Speed', 'Heading'] if x in df.columns]
            df[gps_features] = df[gps_features].interpolate(limit_direction='both')
            latitudes = df['Latitude']
            longitudes = df['Longitude']
            datetimes = df.index
            if data_source == 'romes':
                y = df['Current Data Rate']
            elif data_source == 'romes_ul':
                y = df['UL Current Data Rate']
            else:
                raise ValueError("Unknown data source.")
        elif data_source == 'bosch':
            latitudes = df['latitude']
            longitudes = df['longitude']
            datetimes = df.index
            y = df['measuredrxdatarateavg1sec'] / 1000
        elif data_source == 'mobile_insight':
            latitudes = df['latitude']
            longitudes = df['longitude']
            datetimes = df['chipsettime']
            y = df['tp_cleaned'] * 1000
        else:
            raise ValueError("Unknown data source.")
        groups = cls.get_groups(df, cv_mode)
        x = df[features]
        if classification_threshold:
            y = (y > classification_threshold).astype(int)
        return x, y, latitudes, longitudes, datetimes, groups

    def get_pipeline(self):
        pipeline = Pipeline(self.pipeline_steps())
        pipeline.set_params(**self.optimal_params())
        pipeline.set_params(**self.override_params)
        return pipeline

    @classmethod
    def get_per_column_imputation(cls):
        return ConstantPerColumnImputer([
            (r".*RSSI", -80),
            (r".*Power", -100),
            (r".*RSRP", -140),
            (r"rsrp", -87.7),  # mean value for Bosch data set
            (r".*RSRQ", -50),
            (r"rsrq", -13.8),  # mean value for Bosch data set
            (r".*SINR", -10),
            (r"sinr", 7.5),  # mean value for Bosch data set
            (r".*Tx Power", 0),
            (r".*Frequency", 0),
            (r".*Bandwidth", 0),
            (r"Num\. Carriers", 0),
            (r".*RB Usage.*", 0),
            (r".*F-Factor", 0),
            (r".*4G-Drift", 0),
            (r".*TimeOfArrival", 0),
            (r".*RI", 0),
            (r".*MCS Level", 0),
            (r".*RB Number", 0),
            (r".*Num RB", 0),
            (r".*MIMO Layers Average", 0),
            (r".*Retransmission Rate", 0),
            (r".*TP", 0),
            (r".*Pathloss", 100),
            (r".*CQI", 0),
            (r".*Timing Advance", 0),
            (r".*Num Antenna eNodeB", 2),
            (r".*cellIdentity", 0),
            (r".*TAC", 0),
            (r"precipIntensity", 0),
            (r"precipProbability", 0),
            (r"temperature", 0),
            (r"apparentTemperature", 0),
            (r"dewPoint", 0),
            (r"humidity", 0.5),
            (r"pressure", 1000),
            (r"windSpeed", 0),
            (r"cloudCover", 0),
            (r"uvIndex", 0),
            (r"visibility", 0),
            (r"Weather Humidity", 0.5),
            (r"Weather Pressure", 1000),
            (r"Weather .*", 0),
            (r"Traffic Jam Factor", 0),
        ])

    @classmethod
    def pipeline_steps(cls):
        raise NotImplementedError()

    @classmethod
    def params_to_optimize(cls):
        raise NotImplementedError()

    @classmethod
    def optimal_params(cls):
        return {}

    def get_cv(self):
        if self.cv_mode == 'shuffle':
            return KFold(3, shuffle=True)
        else:
            return LeaveOneGroupOut()

    @classmethod
    def get_groups(cls, x, cv_mode):
        if cv_mode == 'shuffle':
            return None
        elif cv_mode == 'drive_test':
            return x.index.to_period('d').astype('category').codes
        elif cv_mode == 'region_lat':
            return pd.qcut(x['Latitude'], 3).cat.codes
        elif cv_mode == 'region_long':
            return pd.qcut(x['Longitude'], 3).cat.codes

    @classmethod
    def get_verbose(cls):
        if logging.getLogger().isEnabledFor(logging.INFO):
            return 1
        else:
            return 0

    def score(self):
        pipeline = self.get_pipeline()
        scores = cross_validate(pipeline, self.x, self.y, cv=self.get_cv(), groups=self.groups,
                                scoring=self.scoring, verbose=self.get_verbose(), return_train_score=True)
        return {k: np.mean(v) for k, v in scores.items()}

    def predict(self):
        pipeline = self.get_pipeline()
        return self.y, cross_val_predict(pipeline, self.x, self.y, cv=self.get_cv(), groups=self.groups,
                                         verbose=self.get_verbose())

    def predict_bootstrap(self, n_samples):
        pipeline = self.get_pipeline()
        test_indices = []
        predictions = []
        std_devs = []

        for train_idx, test_idx in self.get_cv().split(self.x, self.y, groups=self.groups):
            train_x = self.x.iloc[train_idx]
            train_y = self.y.iloc[train_idx]
            test_x = self.x.iloc[test_idx]
            test_indices.append(test_idx)
            bootstrap_predictions = []

            for _ in range(n_samples):
                bootstrap_pipeline = clone(pipeline)
                bootstrap_idx = np.random.randint(train_x.shape[0], size=(train_x.shape[0],))
                bootstrap_x = train_x.iloc[bootstrap_idx]
                bootstrap_y = train_y.iloc[bootstrap_idx]

                bootstrap_pipeline.fit(bootstrap_x, bootstrap_y)
                bootstrap_predictions.append(bootstrap_pipeline.predict(test_x))

                self._cleanup_memory()

            bootstrap_predictions = np.array(bootstrap_predictions)
            predictions.append(bootstrap_predictions.mean(axis=0))
            std_devs.append(bootstrap_predictions.std(axis=0))

        test_indices = np.concatenate(test_indices)
        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))
        predictions = np.concatenate(predictions)[inv_test_indices]
        std_devs = np.concatenate(std_devs)[inv_test_indices]
        return self.y, predictions, std_devs

    def predict_proba(self):
        pipeline = self.get_pipeline()
        if self.prob_calibration:
            classifier = pipeline.steps[-1][1]
            calibrator = CalibratedClassifierCV(classifier, method=self.prob_calibration, cv=3)
            pipeline.steps[-1] = ('regr', calibrator)
        return self.y, cross_val_predict(pipeline, self.x, self.y, cv=self.get_cv(), groups=self.groups,
                                         verbose=self.get_verbose(), method='predict_proba')[:, 1]

    def predict_quantiles(self, quantiles=(5, 50, 95), bootstrap_samples=None):
        pipeline = self.get_pipeline()

        predictions = [[] for _ in range(len(quantiles))]
        std_devs = [[] for _ in range(len(quantiles))]
        test_indices = []
        for train_idx, test_idx in self.get_cv().split(self.x, self.y, groups=self.groups):
            train_x = self.x.iloc[train_idx]
            train_y = self.y.iloc[train_idx]
            test_x = self.x.iloc[test_idx]
            test_indices.append(test_idx)

            if bootstrap_samples:
                bootstrap_predictions = [[] for _ in range(len(quantiles))]
                for _ in range(bootstrap_samples):
                    bootstrap_idx = np.random.randint(train_x.shape[0], size=(train_x.shape[0],))
                    bootstrap_x = train_x.iloc[bootstrap_idx]
                    bootstrap_y = train_y.iloc[bootstrap_idx]

                    new_predictions = self._predict_quantiles_with_cqr(clone(pipeline), quantiles, bootstrap_x,
                                                                       bootstrap_y, test_x)
                    for i in range(len(quantiles)):
                        bootstrap_predictions[i].append(new_predictions[i])

                    self._cleanup_memory()

                for i in range(len(quantiles)):
                    bootstrap_predictions_arr = np.array(bootstrap_predictions[i])
                    predictions[i].append(bootstrap_predictions_arr.mean(axis=0))
                    std_devs[i].append(bootstrap_predictions_arr.std(axis=0))
            else:
                new_predictions = self._predict_quantiles_with_cqr(clone(pipeline), quantiles, train_x, train_y, test_x)
                for i in range(len(quantiles)):
                    predictions[i].append(new_predictions[i])

        # Concatenate the predictions
        test_indices = np.concatenate(test_indices)
        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))

        predictions = [np.concatenate(prediction)[inv_test_indices] for prediction in predictions]
        if bootstrap_samples:
            std_devs = [np.concatenate(std_dev)[inv_test_indices] for std_dev in std_devs]
            return self.y, predictions, std_devs
        else:
            return self.y, predictions

    # noinspection PyUnboundLocalVariable
    def _predict_quantiles_with_cqr(self, pipeline, quantiles, train_x, train_y, test_x):
        # paper from https://arxiv.org/pdf/1909.05433.pdf
        # code inspired by https://github.com/msesia/cqr-comparison/blob/master/cqr_comparison/cqr.py
        if self.cqr:
            assert quantiles[0] == 100 - quantiles[-1], "Quantiles have to be symmetric"
            significance = quantiles[0] * 2 / 100
            train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
        pipeline.fit(train_x, train_y)
        predictions = self._sort_predictions([pipeline.predict(test_x, quantile=quantile) for quantile in quantiles])

        if self.cqr:
            pred_lo = pipeline.predict(val_x, quantile=quantiles[0])
            pred_hi = pipeline.predict(val_x, quantile=quantiles[-1])
            if self.cqr == 'CQR':
                errors = np.maximum(pred_lo - val_y, val_y - pred_hi)
                correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(val_y) + 1) / len(val_y)))
                predictions[0] -= correction
                predictions[-1] += correction
            elif self.cqr == 'CQR-m':
                eps = 1e-6
                pred_med = pipeline.predict(val_x, quantile=50)
                errors = np.maximum((pred_lo - val_y) / (pred_med - pred_lo + eps),
                                    (val_y - pred_hi) / (pred_hi - pred_med + eps))
                correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(val_y) + 1) / len(val_y)))
                predictions[0] -= correction * (predictions[1] - predictions[0] + eps)
                predictions[-1] += correction * (predictions[-1] - predictions[1] + eps)
            elif self.cqr == 'CQR-r':
                eps = 1e-6
                scaling_factor = pred_hi - pred_lo + eps
                errors = np.maximum((pred_lo - val_y) / scaling_factor, (val_y - pred_hi) / scaling_factor)
                correction = np.quantile(errors, np.minimum(1.0, (1.0 - significance) * (len(val_y) + 1) / len(val_y)))
                scaling_factor = predictions[-1] - predictions[0] + eps
                predictions[0] -= correction * scaling_factor
                predictions[-1] += correction * scaling_factor
            elif self.cqr == 'dummy':
                pass  # For comparing the results with the same number of training samples
            else:
                raise ValueError(f"Unknown CQR mode: {self.cqr}")
        return predictions

    @classmethod
    def _sort_predictions(cls, predictions):
        array = np.array(predictions)
        array.sort(axis=0)
        return [array[i, :] for i in range(len(predictions))]

    def evaluate_quantiles(self, quantiles=(5, 50, 95)):
        measurement, predictions = self.predict_quantiles(quantiles)
        result = []
        for quantile, prediction in zip(quantiles, predictions):
            empirical_quantile = np.count_nonzero(prediction > measurement) / len(measurement)
            result.append((quantile, empirical_quantile))
        return result

    def optimize_parameters(self, n_iter=20):
        clf = RandomizedSearchCV(self.get_pipeline(), self.params_to_optimize(), n_iter=n_iter, iid=False,
                                 cv=self.get_cv(), verbose=self.get_verbose(), return_train_score=True)
        clf.fit(self.x, self.y, groups=self.groups)
        return clf.best_params_, clf.cv_results_

    # def select_features(self, method='importance'):
    #     # 'backward' only works with version from https://github.com/rasbt/mlxtend (will be fixed in mlextend 0.16)

    #     cv = list(self.get_cv().split(self.x, self.y, self.groups))
    #     # Wrap last step (regressor) into the feature selector
    #     pipeline = self.get_pipeline()
    #     regressor = pipeline.steps[-1][1]
    #     if method == 'importance':
    #         selector = FullRFE(regressor, verbose=self.get_verbose(), n_features_to_select=1, cv=cv)
    #     elif method == 'forward':
    #         selector = SequentialFeatureSelector(regressor, forward=True, floating=False, cv=cv,
    #                                              scoring=self.scoring, verbose=self.get_verbose(),
    #                                              k_features=len(self.features))
    #     elif method == 'backward':
    #         selector = SequentialFeatureSelector(regressor, forward=False, floating=False, cv=cv,
    #                                              scoring=self.scoring, verbose=self.get_verbose(),
    #                                              k_features=1)
    #     else:
    #         raise ValueError("Unknown method")
    #     pipeline.steps[-1] = ('regr', selector)

    #     pipeline.fit(self.x, self.y)
    #     if method == 'importance':
    #         ranking = selector.ranking_
    #         grid_scores = selector.scores_[::-1]
    #         sorted_features = [x for _, x in sorted(zip(ranking, self.x.columns))]
    #     else:
    #         df = pd.DataFrame.from_dict(selector.get_metric_dict()).T
    #         if method == 'backward':
    #             df = df[::-1]
    #         grid_scores = df['avg_score']
    #         sorted_features = []
    #         old_set = set()
    #         for list_ in df['feature_idx']:
    #             new_set = set(list_)
    #             sorted_features.append((new_set - old_set).pop())
    #             old_set = new_set
    #         sorted_features = [self.features[idx] for idx in sorted_features]
    #     return sorted_features, grid_scores

    def predict_full_optimize(self):
        logging.warning("Start optimizing parameters (first round)")
        best_params, _ = self.optimize_parameters(100)
        self.override_params.update(best_params)

        logging.warning("Start selecting optimal features")
        sorted_features, grid_scores = self.select_features('forward')
        opt_num_features = np.argmax(grid_scores.to_list())
        opt_features = self.features[:opt_num_features + 1]
        self.features = opt_features

        logging.warning("Start optimizing parameters (second round)")
        best_params, _ = self.optimize_parameters(1000)
        self.override_params.update(best_params)

        logging.warning("Start final prediction")
        y, y_pred = self.predict()
        return y, y_pred, best_params, opt_features

    def _cleanup_memory(self):
        pass


class ConstantPerColumnImputer(BaseEstimator, TransformerMixin):
    def __init__(self, rules):
        self.rules = rules

    def fit(self, _x, _y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame), "ConstantPerColumnImputer only works with DataFrame input!"
        x = x.copy()
        rules = self._compile_rules(self.rules)
        for c in x.columns:
            for rule, value in rules:
                if rule.match(c):
                    x[c].fillna(value, inplace=True)
                    break
        return x

    @classmethod
    def _compile_rules(cls, rules):
        return [(re.compile(pattern), value) for pattern, value in rules]


# noinspection PyPep8Naming
class FullRFE(RFE):
    """
    Modified version of RFECV, that orders all features instead of a determining an "optimal" unsorted feature set and
    only ordering the remaining features (like regular RFECV does).
    """

    def __init__(self, estimator, n_features_to_select=None, step=1, verbose=0, cv='warn'):
        super().__init__(estimator=estimator, n_features_to_select=n_features_to_select, step=step, verbose=verbose)
        self.cv = cv

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self._fit(X, y, lambda estimator, features:
        cross_validate(estimator, X[:, features], y, cv=self.cv, scoring='neg_mean_squared_error')[
            'test_score'].mean())


# noinspection PyPep8Naming
class log_uniform:
    """
    Generates values that are uniformly distributed on a log scale between base^a and base^b
    """

    # copied from https://stackoverflow.com/a/49538913/9072188
    def __init__(self, a=-1, b=0, base=10):
        self.loc = a
        self.scale = b - a
        self.base = base

    def rvs(self, size=None, random_state=None):
        uniform = sp.stats.uniform(loc=self.loc, scale=self.scale)
        if size is None:
            return np.power(self.base, uniform.rvs(random_state=random_state))
        else:
            return np.power(self.base, uniform.rvs(size=size, random_state=random_state))


