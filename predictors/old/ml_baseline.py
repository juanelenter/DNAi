#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 23:31:39 2020

@author: dnai
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV

from sklearn.linear_model import Lasso, RidgeCV

def train_n_predict_lasso(X_train, X_test, y_train):
    alphas = [200, 1500, 1500, 1750, 2375, 2750, 1000, 1000, 1000, 1250, 2750, 2750]
    model = Lasso(alpha = 1000).fit(X_train, y_train)
    y_pred = model.predict( X_test)
    return y_pred

def train_n_predict_ridge(X_train, X_test, y_train):
    alfas = [1500, 1650, 1750, 2000, 2250, 2500, 2750, 3000, 3500, 4000]
    #p = np.mean(X_train, axis=0) / 2
    #d = 2 * np.sum(p * (1 - p))
    #alfa_blup = (1 - h2) / (h2 / d)
    #X_mu0 = X_train.copy()-(2 * p)
    #alfas = np.append(alfas, np.linspace(alfa_blup/5, 10*alfa_blup, num=40))
    model = RidgeCV(alphas = alfas, cv = 3).fit(X_train, y_train)
    y_pred = model.predict( X_test)
    return y_pred
    

def train_n_predict_svr(X_train, X_test, y_train):
    svr = SVR(kernel = "rbf", gamma = "auto", C = 1).fit(X_train, y_train)
    y_pred = svr.predict(X_test)
    return y_pred

def train_n_predict_rf(X_train, X_test, y_train):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 300, stop = 800, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(75, 150, num = 11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 4, 6]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 3, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    
      # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                    'max_features': max_features,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split,
                    'min_samples_leaf': min_samples_leaf,
                    'bootstrap': bootstrap}
    rf = RandomForestRegressor()
      # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 15, \
    cv = 3, verbose=10, random_state=42, n_jobs = -1)
    # Fit the random search model
    rf_random.fit(X_train, y_train)   
    #params.append((pheno_names[i], rf_random.best_params_))
    best_random = rf_random.best_estimator_
    y_pred = best_random.predict(X_test)
    return y_pred

def get_metrics(y_true, y_pred):
    expl_var = sklearn.metrics.explained_variance_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    return expl_var, mae, mse, r2