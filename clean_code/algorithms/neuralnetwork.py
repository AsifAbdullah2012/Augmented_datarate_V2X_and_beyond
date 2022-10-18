import functools
import logging
import gc

import keras
import tensorflow as tf
# noinspection PyPep8Naming
import keras.backend as K
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import StandardScaler
from .framework import Framework, log_uniform
from tensorflow.keras import layers
from functools import partial

# only use a single thread, because usually multiple runs will be performed in parallel
# K.set_session(tf.compat.v1.Session(config=K.tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)))



class NeuralNetwork(Framework):
    def pipeline_steps(self):
        return [('impute', self.get_per_column_imputation()),
                ('cat', CategoricalTransformer(self.categorical_rules(self.x))),
                ('regr', NeuralNetworkRegressor())]

    @classmethod
    def params_to_optimize(cls):
        return {
            'regr__batch_size': [32, 64, 128, 256, 512],
            'regr__decay': log_uniform(a=-8, b=-2),
            'regr__epochs': sp.stats.randint(low=50, high=1000),
            'regr__hidden_layers': sp.stats.randint(low=1, high=6),
            'regr__l1_reg': log_uniform(a=-8, b=-2),
            'regr__l2_reg': log_uniform(a=-8, b=-2),
            'regr__learning_rate': log_uniform(a=-8, b=0),
            # 'regr__loss_func': 'mean_absolute_error',  # TODO
            'regr__units_per_layer': sp.stats.randint(low=16, high=100),
        }

    @classmethod
    def optimal_params(cls):
        return {
            # 'batch_size': 32,
            # 'decay': 1.8235192793002707e-05,
            'epochs': 5,
            'hidden_layers': 6,
            # 'regr__l1_reg': 1.7196267293135723e-08,
            # 'regr__l2_reg': 2.9119620679970727e-08,
            'learning_rate': 0.00001,
            'units_per_layer': 500,
            'activation': 'relu',
        }

    def _cleanup_memory(self):
        K.clear_session()
        gc.collect()


# noinspection PyAttributeOutsideInit
class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, hidden_layers=3, units_per_layer=32, l1_reg=0, l2_reg=0, learning_rate=0.001, decay=0.0001,
                 epochs=500, batch_size=256, loss_func='mean_absolute_error', min_sample_weight=None):
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.decay = decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.min_sample_weight = min_sample_weight

    def build_and_compile_model_v2(self, tau, unit, activation, lr, layer, train_dataset, test_dataset):
        tf.keras.backend.clear_session()
        model = keras.Sequential()
        model.add(layers.Input(shape=(14,), name='input_layer'))
        for i in range(layer):
            nam = 'internal_' + str(i)
            model.add(layers.Dense(units=unit, activation=activation, name=nam))
        model.add(layers.Dense(1, name = 'output_layer'))
        loss_func = partial(NeuralNetworkRegressor._pinball_loss, tau=tau)
        model.compile(loss = loss_func, optimizer=tf.keras.optimizers.Adam(learning_rate=lr))
        return model

    def nn_model_fit(self, epoch, Q, units, activation, lr, layer, X_train, X_test, y_train, y_test):
        NN_model = self.build_and_compile_model_v2(Q, units, activation, lr, layer, X_train, X_test)
        NN_model.fit(X_train, y_train, epochs=epoch, verbose=0, validation_split = 0.2)
        return NN_model
        # return self.build_and_compile_model_v2(train_dataset, y_train, epochs=epoch, verbose=0, validation_split = 0.2)
    
    def predict_data(self, nn_model, X_test):
        return nn_model.predict(X_test)
# ------------------------------------------------------------------------------
    def fit(self, x, y, sample_weights=None):
        self.train_x_ = x
        self.train_y_ = y
        self.model_ = self._fit_model(x, y, self.loss_func, sample_weights)
        return self

    def predict(self, x, quantile=None):
        input_ = self._preprocess_input(x)
        if quantile is None:
            return np.squeeze(self.model_.predict(input_))
        else:
            loss_func = functools.partial(NeuralNetworkRegressor._pinball_loss, tau=quantile / 100)
            model = self._fit_model(self.train_x_, self.train_y_, loss_func, None)
            return np.squeeze(model.predict(input_))

    def _fit_model(self, x, y, loss_func, sample_weights=None):
        model = self._create_model(x, loss_func)
        input_ = self._preprocess_input(x)
        verbose = 0
        if logging.getLogger().isEnabledFor(logging.INFO):
            verbose = 2
        if sample_weights is None and self.min_sample_weight is not None:
            sample_weights = 1 / np.maximum(y, self.min_sample_weight)
            sample_weights /= np.mean(sample_weights)
        self.history_ = model.fit(input_, y, epochs=self.epochs, batch_size=self.batch_size,
                                  sample_weight=sample_weights, verbose=verbose)
        return model

    def _create_model(self, x, loss_func):
        self.cat_features_ = [c for c in x.columns if x[c].dtype.name == 'category']
        self.contin_features_ = [c for c in x.columns if x[c].dtype.name != 'category']

        input_layers = []
        embedding_layers = []
        for feature in self.cat_features_:
            input_layer, embedding_layer = self._create_embedding_layer(feature, len(x[feature].cat.categories))
            input_layers.append(input_layer)
            embedding_layers.append(embedding_layer)

        # concatenate the output with continuous variables
        contin_input = keras.layers.Input(shape=(len(self.contin_features_),), name='contin_input')
        embedding_layers.append(contin_input)
        input_layers.append(contin_input)
        if len(input_layers) == 1:
            x = input_layers[0]
        else:
            x = keras.layers.concatenate(embedding_layers)

        for _ in range(self.hidden_layers):
            x = keras.layers.Dense(self.units_per_layer, activation='relu',
                                   kernel_regularizer=keras.regularizers.L1L2(self.l1_reg, self.l2_reg))(x)

        main_output = keras.layers.Dense(1, activation='relu')(x)
        model = keras.models.Model(inputs=input_layers, outputs=main_output)
        optimizer = keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay)
        model.compile(optimizer=optimizer, loss=loss_func)
        return model

    @classmethod
    def _create_embedding_layer(cls, name, dim):
        input_layer = keras.layers.Input(shape=(1,), name=f'{name}_input')
        embedding_length = (dim + 1) // 2
        embedding_layer = keras.layers.Embedding(input_dim=dim, output_dim=embedding_length, input_length=1,
                                                 name=f'embedding_{name}')(input_layer)
        output_layer = keras.layers.Reshape((embedding_length,), name=f'reshape_{name}')(embedding_layer)
        return input_layer, output_layer

    def _preprocess_input(self, x):
        inputs = {}
        for feature in self.cat_features_:
            inputs[f'{feature}_input'] = x[feature].cat.codes
        inputs['contin_input'] = x[self.contin_features_]
        return inputs

    @staticmethod
    def _pinball_loss(y_true, y_pred, tau):
        err = y_true - y_pred
        return K.mean(K.maximum(tau * err, (tau - 1) * err), axis=-1)


class NeuralNetworkClassi(Framework):
    def pipeline_steps(self):
        return [('cat', CategoricalTransformer(self.categorical_rules(self.x))),
                ('regr', NeuralNetworkClassifier())]

    @classmethod
    def params_to_optimize(cls):
        return {
            'regr__batch_size': [32, 64, 128, 256, 512],
            'regr__decay': log_uniform(a=-8, b=-2),
            'regr__epochs': sp.stats.randint(low=50, high=1000),
            'regr__hidden_layers': sp.stats.randint(low=1, high=6),
            'regr__l1_reg': log_uniform(a=-8, b=-2),
            'regr__l2_reg': log_uniform(a=-8, b=-2),
            'regr__learning_rate': log_uniform(a=-8, b=0),
            'regr__units_per_layer': sp.stats.randint(low=16, high=100),
        }

    @classmethod
    def optimal_params(cls):
        return {
            'regr__batch_size': 128,
            'regr__decay': 1.1813518060529857e-08,
            'regr__epochs': 965,
            'regr__hidden_layers': 1,
            'regr__l1_reg': 1.6225591052778203e-07,
            'regr__l2_reg': 5.791078316323209e-06,
            'regr__learning_rate': 5.356350952662053e-05,
            'regr__units_per_layer': 93
        }

    @classmethod
    def categorical_rules(cls, x):
        rules = {
            'Longitude': 0.001,
            'Latitude': 0.001,
        }
        bins = {}
        for c in x.columns:
            if c in rules:
                min_val = np.floor(np.min(x[c]) / rules[c] - 1) * rules[c]
                max_val = np.ceil(np.max(x[c]) / rules[c] + 1) * rules[c]
                bins[c] = np.arange(min_val, max_val, rules[c]).tolist()
        return bins

    def _cleanup_memory(self):
        K.clear_session()
        gc.collect()


# noinspection PyAttributeOutsideInit
class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=3, units_per_layer=32, l1_reg=0, l2_reg=0, learning_rate=0.001, decay=0.0001,
                 epochs=500, batch_size=256, loss_func='binary_crossentropy', min_sample_weight=None):
        self.hidden_layers = hidden_layers
        self.units_per_layer = units_per_layer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.learning_rate = learning_rate
        self.decay = decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss_func = loss_func
        self.min_sample_weight = min_sample_weight

    def fit(self, x, y, sample_weights=None):
        self.train_x_ = x
        self.train_y_ = y
        self.model_ = self._fit_model(x, y, self.loss_func, sample_weights)
        return self

    def predict(self, x):
        input_ = self._preprocess_input(x)
        y = (np.squeeze(self.model_.predict(input_)) > 0.5).astype(int)
        return y

    def predict_proba(self, x):
        input_ = self._preprocess_input(x)
        probas = np.squeeze(np.clip(self.model_.predict(input_), 0, 1))
        self.classes_ = [0, 1]
        y = np.vstack([1 - probas, probas]).T
        return y

    def _fit_model(self, x, y, loss_func, sample_weights=None):
        model = self._create_model(x, loss_func)
        input_ = self._preprocess_input(x)
        verbose = 0
        if logging.getLogger().isEnabledFor(logging.INFO):
            verbose = 2
        if sample_weights is None and self.min_sample_weight is not None:
            sample_weights = 1 / np.maximum(y, self.min_sample_weight)
            sample_weights /= np.mean(sample_weights)
        self.history_ = model.fit(input_, y, epochs=self.epochs, batch_size=self.batch_size,
                                  sample_weight=sample_weights, verbose=verbose)
        return model

    def _create_model(self, x, loss_func):
        self.cat_features_ = [c for c in x.columns if x[c].dtype.name == 'category']
        self.contin_features_ = [c for c in x.columns if x[c].dtype.name != 'category']

        input_layers = []
        embedding_layers = []
        for feature in self.cat_features_:
            input_layer, embedding_layer = self._create_embedding_layer(feature, len(x[feature].cat.categories))
            input_layers.append(input_layer)
            embedding_layers.append(embedding_layer)

        # concatenate the output with continuous variables
        contin_input = keras.layers.Input(shape=(len(self.contin_features_),), name='contin_input')
        embedding_layers.append(contin_input)
        input_layers.append(contin_input)
        if len(input_layers) == 1:
            x = input_layers[0]
        else:
            x = keras.layers.concatenate(embedding_layers)

        for _ in range(self.hidden_layers):
            x = keras.layers.Dense(self.units_per_layer, activation='relu',
                                   kernel_regularizer=keras.regularizers.L1L2(self.l1_reg, self.l2_reg))(x)

        main_output = keras.layers.Dense(1, activation='sigmoid')(x)
        model = keras.models.Model(inputs=input_layers, outputs=main_output)
        optimizer = keras.optimizers.Adam(lr=self.learning_rate, decay=self.decay)
        model.compile(optimizer=optimizer, loss=loss_func)
        return model

    @classmethod
    def _create_embedding_layer(cls, name, dim):
        input_layer = keras.layers.Input(shape=(1,), name=f'{name}_input')
        embedding_length = (dim + 1) // 2
        embedding_layer = keras.layers.Embedding(input_dim=dim, output_dim=embedding_length, input_length=1,
                                                 name=f'embedding_{name}')(input_layer)
        output_layer = keras.layers.Reshape((embedding_length,), name=f'reshape_{name}')(embedding_layer)
        return input_layer, output_layer

    def _preprocess_input(self, x):
        inputs = {}
        for feature in self.cat_features_:
            inputs[f'{feature}_input'] = x[feature].cat.codes
        inputs['contin_input'] = x[self.contin_features_]
        return inputs


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rules):
        self.rules = rules

    def fit(self, _x, _y=None):
        return self

    def transform(self, x):
        assert isinstance(x, pd.DataFrame), "CategoricalTransformer only works with DataFrame input!"
        x = x.copy()
        for c in x.columns:
            if c in self.rules:
                x[c] = pd.cut(x[c], self.rules[c])
        return x
