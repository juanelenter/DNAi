#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:07:38 2020

@author: DNAi
"""

import ml_baseline as mlb
import os
import numpy as np
import pandas as pd
import os
from pathlib import Path

#####################################
#   IN/out PATHS
#####################################
pred_dir = "../output/pred/yeast"
input_dir ="../output/real/yeast"
# Create output dir if it doesnt exist
Path(pred_dir).mkdir(parents=True, exist_ok=True)
#Each folder in input represents an environment
all_environ = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,f)) ]
for e in all_environ:   
    print("-"*20)
    print(f"processing {e}")
    print("-"*20)
    env_dir = os.path.join(input_dir,e)
    #Output path
    pred_pth = os.path.join(pred_dir,e)
    Path(pred_pth).mkdir(parents=True, exist_ok=True)
    
    print("loading data...")
    geno_pth = os.path.join(env_dir,"geno.csv")
    pheno_pth = os.path.join(env_dir,"pheno.csv")
    pheno = np.loadtxt(pheno_pth, delimiter = ",")
    geno = np.loadtxt(geno_pth, delimiter = ",")
    
    all_test_idx = [f for f in os.listdir(env_dir) if f.endswith('_test.csv')]
    splits = [filename.split('_')[0] for filename in all_test_idx]
    for s in splits:
        print("."*20)
        print(f"train/test split {s}")
        idx_train = np.loadtxt(os.path.join(env_dir, s+"_idx_train.csv"), delimiter = ",", dtype = int)
        idx_test = np.loadtxt(os.path.join(env_dir, s+"_idx_test.csv"), delimiter = ",", dtype = int)
        X_train = geno[idx_train] 
        X_test = geno[idx_test]
        y_train = pheno[idx_train]
        y_test = pheno[idx_test]
        y_train_std = (y_train-np.mean(y_train))/np.std(y_train)
        y_test_std = (y_test-np.mean(y_train))/np.std(y_train)
        print("Ridge Regression")
        y_pred_ridge = mlb.train_n_predict_ridge(X_train, X_test, y_train)
        expl_var, mae, mse, r2 = mlb.get_metrics(y_pred_ridge, y_test_std)
        print("Expl var {}, r2 {}, mse , {}".format(expl_var, r2, mse))        
        np.savetxt(os.path.join(pred_pth, s+'_ridge.csv'), y_pred_ridge ,delimiter = ",")
    #Save metriics (AGREGATE SPLITS?)-----TODO