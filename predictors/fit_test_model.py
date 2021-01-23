import numpy as np
from sklearn.base import clone
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from importlib import import_module

def fit_test(model_instance, params, X_train, y_train, X_test, y_test):
    try:
        model = clone(model_instance)
        model.set_params(**params)
        model.fit(X_train, y_train)
        test_pred = model.predict(X_test)
        train_pred = model.predict(X_train)
        metrics = {"r_test" : pearsonr(test_pred, y_test)[0], 
                   "r_train" : pearsonr(train_pred, y_train)[0],
                   "mse_test" : mse(test_pred, y_test),
                   "mse_train" : mse(train_pred, y_train)}
    except:
        model = import_module(model_instance, package=None)
        _, _, train_pred, test_pred, metrics, _ = model.train_test(X_train, y_train, X_test, y_test)

    return train_pred, test_pred, metrics
