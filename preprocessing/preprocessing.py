# -*- coding: utf-8 -*-
"""
Created on Sat May 16 13:02:17 2020

@author: Juan
"""

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode
import sys
import json
from datetime import date
import os

# Local Imports
from switchers import *
from standarizers import *
from codifiers import *
from imputators import *

#%%

def make_splits( n, m, test_ratio ):
    """

    Parameters
    ----------
    n : int
        Number of train test splits
    m : int
        Number of individuals in the genotype matrix.
    test_ratio : int
        Fraction of the data set to be used for testing.

    Returns
    -------
    train_indxs : numpy array
        Array of index to be used in training.
    test_indxs : numpy array
        Array of index to be used in testing.

    """
    x = np.empty((n, m))
    for i in range(n):
        x[i] = np.random.permutation( m )
    
    train_size = int( (1-test_ratio) * m)
    train_indxs = x[:, :train_size]
    test_indxs = x[:, train_size:]
    
    return train_indxs, test_indxs
#%%

def preprocessing( data, original_format, imput, codif, n_feno = None, imput_flag = 0, n_splits = 1, ind_split = 0, test_ratio = 0.10, overwrite_idxs=False):
    """
    
    Input
    -------

        data: string
            Database to preprocess. 
            Available data sets as of 16/05
                1) "yeast": As described in: 
                2) "example": Toy .ped database to test .ped preprocessing module.
                3) "crossa_wheat": Whatever Crossa sends us.
                4) "jersey": Jersey SCR 
    
        original_format: string
            Original format of the data base.
            One of:
                1) "ped": .ped format as described in https://zzz.bwh.harvard.edu/plink/data.shtml#ped
                2) "yeast_format": Yeast data set format as described in . 
                3) "no_format": No formatting required: X and y vectors ready. 
    
        codif: string
            Codification used in the preprocessing.
            Available codifications as of 16/05:
                1) "additive":  Alphabet = { 0, 1, 2 }
                2) "bio_ohe" : One hot encoding for A C T G data set . Alphabet = { 0, 1 }
                3) "ohe_add": One hot encoding after coding additively SNPs. Alphabet = {0, 1}
                4) "no_codif": No codification needed (ej: haploid organisms).
                5) "ohe": One hot encoding for ALREADY additively coded bases. Alphabet = { 0, 1 }
        
        imput: string
            Imputation method used to deal with missing values.
            Available imputation methods as of 16/05:
                1) "mode": Mode (most frequent value) imputation.
                2) "del_rows": Delete observations with missing values.
                3) "del_cols": Delete snp with missing values.
        
        n_feno: int
            Index of the phenotype involved.
            (Only for bases with multiple phenotypes like yeast or crossa wheat, else should be None).
            

        imput_flag: object
            Char, int or string that represents a missing value.
            Deafults to .ped standard: 0. 
            In jersey imput_flag = 5.
        
        n_splits: int
            Number of train-test splits to compute.
            
    Output
    -------    
        json_path: string
            Path to config.py that contains the paths to the preprocessed base
            and meta-data about the data base.
            
    """
    #### Domingo 32 de Otoño
    calendar = str(date.today())
    
    ####Setup paths
    ci = codif + "_" + imput
    base_dir= '..'
    config_name = '_'.join([data, ci, calendar])
    if n_feno is not None:
        config_name = '_'.join([config_name, str(n_feno)])
    config_path = os.path.join(base_dir, "Input", "config", config_name )
    
    
    #### Data Set Look up
    base_path = base( data )
    print('Loading data ...')
    
    #### Format Standarization
    geno, feno = format_std( original_format, base_path, n_feno )    

    #### Imputation
    geno = imputation( geno, imput, imput_flag )
    print(f"{imput} imputation: DONE.")
    
    #### Codification
    geno = codification( geno, codif )
    print(f"{codif} codification: DONE.")
    print(f"tipo = {type(geno)}")
    
    #### Write geno and feno as csv to INPUT > BASE > CI
    X_path = os.path.join(base_dir,'Input', data, ci ) 
    Y_path = os.path.join(base_dir,'Input', data, ci )
    if n_feno is not None:
        X_path = os.path.join(X_path, str(n_feno))
        Y_path = os.path.join(Y_path, str(n_feno))
    try:
        np.save( os.path.join(Y_path, "Y.npy"), feno )
    except FileNotFoundError:
        os.makedirs( Y_path )
        np.save( os.path.join(Y_path, "Y.npy"), feno)
        
    try:
        np.save( os.path.join(X_path, "X.npy") , geno )
    except FileNotFoundError:
        os.makedirs( X_path )
        np.save( os.path.join(X_path, "X.npy"), geno )
    
    print(f"Data sets saved in {X_path} ")
    
    #### Splits 
    n_obs = geno.shape[0]
    train_indexs, test_indexs = make_splits( n_splits, n_obs, test_ratio )
    
    #### Save indexes split
    indexs_path = os.path.join( base_dir,'Input', data, 'indexes')
    if n_feno is not None:
        indexs_path = os.path.join(indexs_path, str(n_feno))
        
    if not os.path.exists(indexs_path):
        os.makedirs( indexs_path )
    for i in range(n_splits):
        idx_train_path = os.path.join(indexs_path, "index_train" + str(i) + ".npy")
        idx_test_path = os.path.join(indexs_path, "index_test" + str(i) + ".npy")
        if os.path.exists(idx_train_path):
            if overwrite_idxs:
                np.save( idx_train_path, train_indexs[i] )
                np.save( idx_test_path , test_indexs[i] )
        else:
            np.save( idx_train_path, train_indexs[i] )
            np.save( idx_test_path , test_indexs[i] )
      
    print(f"Splits saved in {indexs_path}")
    
    #### Config file creation
    data_out = { "X_path": os.path.join(X_path, "X.npy"),
                "Y_path": os.path.join(Y_path, "Y.npy"),	
                "index_train": os.path.join(indexs_path, "index_train"), 
                "index_test": os.path.join(indexs_path,  "index_test"),	
                "meta_data": { "base": data,
                              "imputation": imput,
                              "codification": codif,
                              "date": calendar,
                              "num_splits": n_splits,
			      "n_feno/env" : n_feno
                              }
                }
	
    with open(config_path + '.json', "w+") as write_file:
        json.dump(data_out, write_file, indent = 2)
        
    print(f"JSON ledger written in {config_path}")
    print("PREPROCESSING SUCCESSFUL")
    return config_path

#%%
if __name__ == "__main__":
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 1:
        print('Use: preprocessing.py <base_name> -f <format> -e <encoding> -i <imputation> -nan_flag <imputation flag> -ns  <nsplits> -nf <nfeno/env>')
    else:

        # default parameters
        opt_args = {"encoding": "ohe",
                    "n_feno": None,
                    "nan_flag": 5,
                    "n_splits" :10,
                    "overwrite_idx": False,
                    "format": "no_format",
                    "imput": "mode",
            }

        # command line arguments
        if "-e" in argv:
            opt_args["encoding"] = argv[argv.index("-e") + 1]
        if "-nan_flag" in argv:
            opt_args["nan_flag"] = argv[argv.index("-nan_flag") + 1]
        if "-ns" in argv:
            opt_args["n_splits"] = int(argv[argv.index("-ns") + 1])
        if "-nf" in argv:
            opt_args["n_feno"] = int(argv[argv.index("-nf") + 1])
        if "-o_idx" in argv:
            opt_args["overwrite_idx"] = True  
        if "-i" in argv:
            opt_args["imput"] = argv[argv.index("-i") + 1]
        if "-f" in argv:
            opt_args["format"] = argv[argv.index("-f") + 1]
            
    c = preprocessing(sys.argv[1], opt_args["format"], opt_args["imput"], opt_args["encoding"], n_feno = opt_args["n_feno"],
                      imput_flag = opt_args["nan_flag"], n_splits = opt_args["n_splits"], overwrite_idxs=opt_args["overwrite_idx"])
