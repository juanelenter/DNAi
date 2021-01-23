# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:51:24 2019

Non additive phenotype simulation module.

@author: Tecla
"""
#%%

import numpy as np
import pandas as pd
import config
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def snp_to_qtn(X_snp, n_qtn = 100, sampling_method = "random", neighbour=3):
    '''
    Samples QTN from SNP matrix.
    
    Input:
        X_snp: numpy array
            SNP matrix, each row is a different individual
        n_qtn: int
            number of QTN to sample from the SNP matrix
        sampling_method: string
            One of "random" or "cluster". (Gianola)
        neighbour:
            how many surrounding qtn to consider when sampling clustered
            
    
    Output:
        X: numpy array
            Additive QTN matrix
        D: numpy array
            Dominance QTN matrix
    '''
    
    if sampling_method == "random":        
        qtns = np.random.randint( X_snp.shape[0], size = n_qtn )
    
    else:
        cluster= np.random.randint( X_snp.shape[0], size = n_qtn//neighbour )
        qtns = np.ndarray.flatten(np.asarray([cluster-1, cluster, cluster+1]))
    
    X = X_snp[:, qtns]   
    D = X.copy()
    D[ D == 2 ] = 0
    
    return X, D, qtns
    
    

def epistasis(X, D, n_interactions = 3, interaction_method = "product"):
    '''
    Builds the 4 epistasis matrices.
    
    Input: 
        X: numpy array
            Additive snp matrix
        D: numpy array
            Domincance snp matrix
        n_interactions: int
            Number of surrounding QTNs that interact with every QTN.
        interaction_method: string
            One of ["product", ..]. Gianola dice que es una multiplicacion .
    
    Output: 
        Epistasis Matrices XoX, XoD, DoX, DoD: numpy arrays.
    '''
    
    # Not generalized for other number of interactions. Only n_interactions = 3.
    n_qtns = X.shape[1]
    # Calculate number of columns in epistasis matrices
    n_qtns_epistasis = ( n_qtns - (n_interactions - 1) ) * n_interactions
    
    XoX = np.zeros((X.shape[0], n_qtns_epistasis))
    XoD = np.zeros(XoX.shape)
    DoX = np.zeros(XoX.shape)
    DoD = np.zeros(XoX.shape)
    
    for j, col in enumerate(X.transpose()):

        if j < (n_qtns-n_interactions):
            XoX[:, 3*j] = col * X[:, j+1]
            XoX[:, 3*j+1] = col * X[:, j+2]
            XoX[:, 3*j+2] = col * X[:, j+3]
            
            XoD[:, 3*j] = col * D[:, j+1]
            XoD[:, 3*j+1] = col * D[:, j+2]
            XoD[:, 3*j+2] = col * D[:, j+3]
            
        elif j == (n_qtns-n_interactions):
            XoX[:, 3*j] = col * X[:, j+1]
            XoX[:, 3*j+1] = col * X[:, j+2] 
            
            XoD[:, 3*j] = col * D[:, j+1]
            XoD[:, 3*j+1] = col * D[:, j+2] 
        
        elif j == (n_qtns-n_interactions+1):
            XoX[:, 3*j-1] = col * X[:, j+1]
            XoD[:, 3*j-1] = col * D[:, j+1]
        
            
    for j, col in enumerate(D.transpose()):

        if j < (n_qtns-n_interactions):
            DoX[:, 3*j] = col * X[:, j+1]
            DoX[:, 3*j+1] = col * X[:, j+2]
            DoX[:, 3*j+2] = col * X[:, j+3]
            
            DoD[:, 3*j] = col * D[:, j+1]
            DoD[:, 3*j+1] = col * D[:, j+2]
            DoD[:, 3*j+2] = col * D[:, j+3]
            
        elif j == (n_qtns-n_interactions):
            DoX[:, 3*j] = col * X[:, j+1]
            DoX[:, 3*j+1] = col * X[:, j+2] 
            
            DoD[:, 3*j] = col * D[:, j+1]
            DoD[:, 3*j+1] = col * D[:, j+2] 
        
        elif j == (n_qtns-n_interactions+1):
            DoX[:, 3*j-1] = col * X[:, j+1]
            DoD[:, 3*j-1] = col * D[:, j+1]
        
    return XoX, XoD, DoX, DoD

def simulate_phenotype(X, D, XoX, XoD, DoX, DoD, ha = 0.10,  hd=0.1, he=0.5, hb=0.7 ):
    '''
    Adds additive, dominance, 4 types of epistasis effects and noise to simulate
    a phenotype.
    
    Input: 
        X, D, XoX, XoD, DoX, DoD: arrays
            QTN and QTN_epistasis matrices.
            according to heredability values
        ha(float): additive heritability
        hd(float):Dominance heritability
        hepi(float): Epistasis heritability
 
            
    Output:
        y: array
            Simulated phenotype.
    '''
    
    n_ind = XoX.shape[0]
    n_qtns = X.shape[1]
    n_interactions = XoX.shape[1]
    
    y = np.zeros(n_ind)
    
    ### Creation of gaussian and gamma vectors.
    alfa = np.random.normal(0, 1, n_qtns)
    delta = np.random.normal(0, 0.5, n_qtns)
    aa = np.random.choice([-1,1], n_interactions) * np.random.gamma(0.1, 0.1, n_interactions)
    ad = np.random.choice([-1,1], n_interactions) * np.random.gamma(0.1, 0.1, n_interactions)
    da = np.random.choice([-1,1], n_interactions) * np.random.gamma(0.1, 0.1, n_interactions)
    dd = np.random.choice([-1,1], n_interactions) * np.random.gamma(0.1, 0.1, n_interactions)
    add = X@alfa
    add = ha*add/np.var(add)
    epi = XoX@aa + XoD@ad + DoX@da + DoD@dd
    epi = he*epi/np.var(epi)
    dom =  D@delta 
    dom  = hd*dom/np.var(dom)
    
    var_err = (1/hb - 1)
    err = np.random.normal(0, var_err, n_ind)
    
#    if normalization:
#       err /= np.linalg.norm(err) 
#       dd /= np.linalg.norm(dd)
#       da /= np.linalg.norm(da)
#       ad /= np.linalg.norm(ad)
#       aa /= np.linalg.norm(aa)
#       delta /= np.linalg.norm(delta)
#       alfa /= np.linalg.norm(alfa)
       
    #Phenotype formula according to Gianola.
    y = add+ epi+ dom+err
    
    return y

def train_test_ridge(X, y, t_size, regularization_alphas = [1e-2, 5e-1, 1e-1, 1, 5, 8, 10, 12, 25, 50, 100, 500, 1000, 100000]):
    '''
    Trains Ridge Regression with efficient Leave One Out Cross Validation, 
    returns model results.
    
    Input:
        X: Numpy array
            Cycle times matrix.
        y: Numpy array or list
            IDDQ measurements (ground truth).
        t_size: int
            Test size (0..1)
        regularization_alphas: list  
            List of alphas (regularization parameter) to try in RidgeCV.
    Returns:
        Model parameters and test results.
    '''
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size)
    
    scaler = StandardScaler()
    y_train_std = scaler.fit_transform(y_train.reshape(-1, 1))
    y_test_std = scaler.transform(y_test.reshape(-1,1))
    
    ridge = RidgeCV(alphas = regularization_alphas, cv = None).fit(X_train, y_train_std)
    y_pred = ridge.predict(X_test) 
    
    print("train mse= ", mean_squared_error(ridge.predict(X_train), y_train_std))
    
    mse = mean_squared_error(y_pred, y_test_std)
    print("test mse = ", mse)
    
    sq = np.std(y_pred - y_test_std) / np.std(y)
    
    return ridge.coef_, ridge.alpha_, sq, mse, scaler.mean_, scaler.var_

from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

def train_test_ridge_RFECV(X, y, t_size, regularization_alphas = [1e-2, 5e-1, 1e-1, 1, 5, 8, 10, 12, 25, 50, 100, 500, 1000, 100000], std_y=True):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = t_size)
    
    if std_y:
        scaler = StandardScaler()
        y_train_std = scaler.fit_transform(y_train.reshape(-1, 1))
        y_test_std = scaler.transform(y_test.reshape(-1,1))
    else:
        y_train_std = y_train.reshape(-1, 1)
        y_test_std = y_test.reshape(-1, 1)
    ridge = RidgeCV(alphas = regularization_alphas, cv = None).fit(X_train, y_train_std)
    y_pred = ridge.predict(X_test) 
    
    print("train mse= ", mean_squared_error(ridge.predict(X_train), y_train_std))
    
    mse = mean_squared_error(y_pred, y_test_std)
    print("test mse = ", mse)
    
    clf = RidgeCV(alphas = regularization_alphas, cv = None)
    selector = RFECV(clf, step=100, cv=10, min_features_to_select=100, n_jobs = -1)
    selector.fit(X_train, y_train_std)
    qtn_estim = np.asarray(np.where(selector.ranking_==1))[0]
    print(len(qtn_estim))
    
    X_train = X_train[:,qtn_estim]
    X_test = X_test[:,qtn_estim]
    ridge = RidgeCV(alphas = regularization_alphas, cv = None).fit(X_train, y_train_std)
    y_pred = ridge.predict(X_test) 
    
    
    print("train mse= ", mean_squared_error(ridge.predict(X_train), y_train_std))
    
    mse = mean_squared_error(y_pred, y_test_std)
    print("test mse = ", mse)
    
    sq = np.std(y_pred - y_test_std) / np.std(y)
    
    return ridge.coef_, ridge.alpha_, sq, mse, scaler.mean_, scaler.var_

if __name__ == "__main__":
        
    snps_matrix = np.loadtxt(config.snps_matrix_path, delimiter = ",")
    ####################################
    # Random QTN
   ###################################### 
   #  n = 100
    x, d, qtns = snp_to_qtn(snps_matrix, 100, sampling_method='random')
    np.savetxt(os.path.join(config.sim_output_path, "qtn_n_100_rand.csv"),  qtns, delimiter = ",",  fmt='%d')
    xox, xod, dox, dod = epistasis(x, d, 3, "product")
    
    y = simulate_phenotype(x, d, xox, xod, dox, dod,  ha = 0.1,  hd=0.1, he=0.5, hb=0.7 )
    np.savetxt(os.path.join(config.sim_output_path, "y_n_100_rand.csv"),y, delimiter = ",",  fmt='%d')
    
    #n = 1000
    x, d, qtns = snp_to_qtn(snps_matrix, 1000, sampling_method='random')
    np.savetxt(os.path.join(config.sim_output_path, "qtn_n_1000_rand.csv"),  qtns, delimiter = ",",  fmt='%d')
    xox, xod, dox, dod = epistasis(x, d, 3, "product")
    y = simulate_phenotype(x, d, xox, xod, dox, dod,  ha = 0.02,  hd=0.02, he=0.68, hb=0.7 )
    np.savetxt(os.path.join(config.sim_output_path, "y_n_1000_rand.csv"),y, delimiter = ",",  fmt='%d')
    
    
    ####################################
    # Clustered QTN
   ###################################### 
   ###########ADITIVE#####################################################
       #  n = 100
    x, d, qtns = snp_to_qtn(snps_matrix, 100, sampling_method='cluster')
    np.savetxt(os.path.join(config.sim_output_path, "qtn_n_100_clust.csv"),  qtns, delimiter = ",",  fmt='%d')
    xox, xod, dox, dod = epistasis(x, d, 3, "product")
    y = simulate_phenotype(x, d, xox, xod, dox, dod,  ha = 0.3,  hd=0, he=0, hb=0.3 )
    np.savetxt(os.path.join(config.sim_output_path, "y_n_100_clust_add.csv"),y, delimiter = ",")
    
    #n = 1000
    x, d, qtns = snp_to_qtn(snps_matrix, 1000, sampling_method='cluster')
    np.savetxt(os.path.join(config.sim_output_path, "qtn_n_1000_clust_add.csv"),  qtns, delimiter = ",",  fmt='%d')
    xox, xod, dox, dod = epistasis(x, d, 3, "product")
    y = simulate_phenotype(x, d, xox, xod, dox, dod,  ha = 0.3,  hd=0, he=0, hb=0.3 )
    np.savetxt(os.path.join(config.sim_output_path, "y_n_1000_clust_add.csv"),y, delimiter = ",")   
   
   
   ########## NON ADDITIVE#################################################
       #  n = 100
    x, d, qtns = snp_to_qtn(snps_matrix, 100, sampling_method='cluster')
    np.savetxt(os.path.join(config.sim_output_path, "qtn_n_100_clust.csv"),  qtns, delimiter = ",",  fmt='%d')
    xox, xod, dox, dod = epistasis(x, d, 3, "product")
    y = simulate_phenotype(x, d, xox, xod, dox, dod,  ha = 0.1,  hd=0.1, he=0.5, hb=0.7 )
    np.savetxt(os.path.join(config.sim_output_path, "y_n_100_clust.csv"),y, delimiter = ",")
    
    #n = 1000
    x, d, qtns = snp_to_qtn(snps_matrix, 1000, sampling_method='cluster')
    np.savetxt(os.path.join(config.sim_output_path, "qtn_n_1000_clust.csv"),  qtns, delimiter = ",",  fmt='%d')
    xox, xod, dox, dod = epistasis(x, d, 3, "product")
    y = simulate_phenotype(x, d, xox, xod, dox, dod,  ha = 0.02,  hd=0.02, he=0.68, hb=0.7 )
    np.savetxt(os.path.join(config.sim_output_path, "y_n_1000_clust.csv"),y, delimiter = ",")
#    coefs, alpha, sq, mse_out, y_mean, y_var = train_test_ridge(snps_matrix, phenotype, 0.10)
    idx = np.arange(len(y))
    #################################################
    #   TRAIN TEST SPLIT
    #################################################
    X_train, X_test, y_train, y_test = train_test_split(snps_matrix, y, test_size = 0.1)
    n_exp = 5
    for i in range(n_exp):
        idx_train, idx_test = train_test_split(idx, test_size = 0.1)
        np.savetxt(os.path.abspath(os.path.join(config.sim_output_path,str(i)+'_'+'idx_train.csv')), idx_train ,delimiter = ",",  fmt='%d' )
        np.savetxt(os.path.abspath(os.path.join(config.sim_output_path,str(i)+'_'+'idx_test.csv')), idx_test ,delimiter = "," ,  fmt='%d')
        
#    np.savetxt(config.X_train_output_path, X_train ,delimiter = "," )
#    np.savetxt(config.X_test_output_path, X_test ,delimiter = "," )
#    np.savetxt(config.y_test_output_path, y_train ,delimiter = "," )
#    np.savetxt(config.y_train_output_path, y_test ,delimiter = "," )

    
