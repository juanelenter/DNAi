#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 09:36:50 2020

@author: ignaciohounie
"""
import numpy as np
from sklearn.linear_model import RidgeCV
import config
import os
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

###########################
#   Load phenotype  matrix X
##########################
X = np.loadtxt(config.snps_matrix_path, delimiter = ",")
i=0
idx_train = np.loadtxt(os.path.abspath(os.path.join(config.sim_output_path,str(i)+'_'+'idx_train.csv')), dtype=np.int32)
idx_test  = np.loadtxt(os.path.abspath(os.path.join(config.sim_output_path,str(i)+'_'+'idx_test.csv')), dtype=np.int32)
X_train = X[idx_train, :]
X_test = X[idx_test, :]

#####################
# Alfa BLUP
#####################
y =  np.loadtxt(os.path.join(config.sim_output_path,"y_n_1000_clust_add.csv"), delimiter = ",")
y_train = y[idx_train]
y_test = y[idx_test]
h2 = 0.7

alfas = [10, 50, 100, 200, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000]

p = np.mean(X_train, axis=0) / 2
d = 2 * np.sum(p * (1 - p))
alfa_blup = (1 - h2) / (h2 / d)
X_mu0 = X_train.copy()-(2 * p)

alfas = np.append(alfas, np.linspace(alfa_blup/5, 10*alfa_blup, num=40))
var = np.var(y)
ridge = RidgeCV(alphas = alfas, cv = None).fit(X_mu0, y_train)
y_pred = ridge.predict(X_test-(2 * p))
out_path = os.path.abspath(os.path.join(config.ridge_output_path,str(i)+'_'+"y_n_1000_clust_add.csv"))
np.savetxt(out_path, y_pred,  delimiter = ",")
alfa = ridge.alpha_
mse = mean_squared_error(y_pred, y_test)
r2 = r2_score( y_test, y_pred)
print('r2: ', r2)
print('var(y): ', var)
print('mse: ', mse)
print('mse_norm: ', mse/var)