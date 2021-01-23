#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 09:13:01 2020

@author: ignaciohounie
"""
import sys
#import argparse
import numpy as np
import sklearn.metrics
import config
import os.path
from os import listdir
import pandas as pd

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ os.path.join(path_to_dir,filename) for filename in filenames if filename.endswith( suffix ) ]

def get_metrics(y_true, y_pred):
    expl_var = sklearn.metrics.explained_variance_score(y_true, y_pred)
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2 = sklearn.metrics.r2_score(y_true, y_pred)
    return expl_var, mae, mse, r2



if  len(sys.argv)>1:
    file =  sys.argv[1]
else:
    file = '*'

print(file+".csv")   
predictions_path  = find_csv_filenames(config.bayes_b_output_path, suffix=file+".csv")
print(predictions_path)
ground_truth  = np.loadtxt(os.path.join(config.sim_output_path, file+".csv") , delimiter = ",")
test_idx = np.loadtxt(config.idx_test_output_path, delimiter = ",").astype(int)
train_idx = np.loadtxt(config.idx_train_output_path, delimiter = ",").astype(int)
y_gt = ground_truth[test_idx]
#y_train =  ground_truth[train_idx]
#y_gt= (y_gt-np.mean(y_train))/np.std(y_train)
results = []
print(np.var(ground_truth))
for pred in  predictions_path:
    print(pred)
    y_pred = np.loadtxt(pred, delimiter = ",")[test_idx]
    expl_var, mae, mse, r2 = get_metrics(y_gt, y_pred)
    results.append([expl_var, mae, mse, r2])
    print("expl var ", expl_var )
    print("mae ", mae )
    print("mse ", mse)
    print("r2 ", r2)
    
df = pd.DataFrame(results, columns =['Explained var', 'mae', 'mse', 'r2'])
df.to_csv(os.path.join(config.eval_output_path, file+".csv"))