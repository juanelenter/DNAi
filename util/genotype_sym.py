#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:35:19 2020

@author: ignaciohounie
"""

import numpy as np

def shuffle_gen(X):
    X_new = np.empty([2*X.shape[0], X.shape[1]])
    for i in range(X.shape[1]):
        idx = np.random.permutation(X.shape[0])
        X_new [:X.shape[0],i] = X[idx, i]
    X_new[X.shape[0]:,:]=X
    return X_new