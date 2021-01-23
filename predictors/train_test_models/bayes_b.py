#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 18:28:06 2020

@author: ignaciohounie
"""
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#R Stuff
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()

def check_r_dependencies():
    r('''
       list.of.packages <- c("doRNG", "doMC", "BGLR")
       new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
       if(length(new.packages)>0){install.packages(new.packages, repos = "http://cran.us.r-project.org")}
       library(doMC)
       require(BGLR)
        ''')
    return
    
def split_to_idxs(X_train, Y_train, X_test, Y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((Y_train, Y_test))
    idx_train = np.arange(len(Y_train))+1
    idx_test = len(Y_train)+np.arange(len(Y_test))+1
    return X, y, idx_train, idx_test

def bayes_b(X, y, idx_train, idx_test, normalize_pheno = False, normalize_geno = False, niter=30000, R2 = 0.3):
    r.assign('X',X)
    r.assign('y',y)
    r.assign('whichNa',idx_test)
    r.assign('train_idx', idx_train)
    r.assign('test_idx', idx_test)
    r.assign('normalize_phenotype', normalize_pheno)
    r.assign('normalize_genotype', normalize_geno)
    r.assign('n_it', niter)
    r.assign('R2', R2)
    r('sink("/dev/null")')
    r('''
    if(normalize_phenotype){
        print("scaling phenotypes IDK")
        y<-(y-mean(y[train_idx]))/sd(y[train_idx])
    }
    print("done scaling")

    if(normalize_genotype){
        print("Normalizing Genotypes, A Murky Business ") 
        for(i in ncol(X))
        {
            std_xi<-sd(X[,i])
            if (std_xi==0) {
                X[,i]<-(X[,i]-mean(X[,i]))/std_xi
            } else {
                X[,i]<-(X[,i]-mean(X[,i]))
            }
            
        }
    }
            y[test_idx]<-NA
    print("Starting Gibbs Sampling. Hold on to your hat!")
    print("Bayes C is my middle name")
    nIter=n_it;
    burnIn=10;
    thin=3;
    saveAt='';
    S0=NULL;
    weights=NULL;
    R2=0.5;
    ETA<-list(list(X=X,model='BayesB'))
      
    fit_BC=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)

    y_pred_BC<-fit_BC$yHat
    prior<-fit_BC$prior
    y_pred_train<-y_pred_BC[train_idx] 
    y_pred_test<-y_pred_BC[test_idx] 
    ''')
    return np.array(r['y_pred_train']), np.array(r['y_pred_test'])
    

def train_test(X_train, Y_train, X_test, Y_test, cv_params, custom_grid = False):
    check_r_dependencies()
    X, y, idx_train, idx_test = split_to_idxs(X_train, Y_train, X_test, Y_test)
    if custom_grid:
        h2=custom_grid["heritability"]
    else:
        h2=0.3
    train_predictions, test_predictions = bayes_c(X, y, idx_train, idx_test, normalize_pheno = False, normalize_geno = False, R2=h2)
    #metrics
    print(test_predictions.shape)
    print(Y_test.shape)
    r_train = pearsonr(Y_train, train_predictions)
    r_test = pearsonr(Y_test, test_predictions)
    mse_train = mse(Y_train, train_predictions)
    mse_test = mse(Y_test, test_predictions)
    metrics = {"r_train" : r_train,
               "r_test" : r_test,
               "mse_train" : mse_train,
               "mse_test" : mse_test}
    print(f"pearsonr train: {r_train}")
    print(f"pearsonr test: {r_test}")
    print(f"mse train: {mse_train}")
    print(f"mse test: {mse_test}")
    return {}, {}, train_predictions, test_predictions, metrics, {}
