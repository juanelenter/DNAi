#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:32:25 2020

@author: ignaciohounie
"""

from bayes_c import *
import numpy as np

X = np.load("../../Input/jersey/no_codif_del_rows/X.npy")
Y = np.load("../../Input/jersey/no_codif_del_rows/Y.npy")
idx_train = np.load("../../Input/jersey/no_codif_del_rows/2020-05-27/index_train0.npy").astype(int)
idx_test = np.load("../../Input/jersey/no_codif_del_rows/2020-05-27/index_test0.npy").astype(int)
X_train = X[idx_train] 
X_test = X[idx_test]
Y_train = Y[idx_train] 
Y_test = Y[idx_test]
X2, Y2, i2, i22 = split_to_idxs(X_train, Y_train, X_test, Y_test)
check_r_dependencies()
weight, prior, y_pred_train, y_pred_test = bayes_c(X, Y, idx_train, idx_test, normalize_pheno = False, normalize_geno = False, niter=10)