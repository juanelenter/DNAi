# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:24:07 2020

@author: Juan
"""

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode

# Local Imports
from standarizers import *
from codifiers import *
from imputators import *

#%% Case Switch a manopla

def base( b ):
    
    bases = {
        "jersey": ["../data/jersey/geno.npy", "../data/jersey/Y_weight_mean.npy"],
        "jersey_max":["../data/jersey/geno.npy", "../data/jersey/y_max.npy"],
        "yeast": ["../data/yeast/geno.txt", "../data/yeast/feno.txt"],
        "wheat": ["../data/wheat_geno.csv", "../data/wheat_feno.csv"],
        "example":"../data/extra.ped",
        "crossa_wheat": ["../data/crossa_wheat/geno_crossa.npy", "../data/crossa_wheat/feno_crossa.npy"],
        "holstein": ["../data/holstein/geno.npy", "../data/holstein/feno.npy"]
        }
    
    return bases.get(b, "Wrong dataset name.")

def format_std( of, bp, fenos = None ):
    
    standarizations = {
        "ped": ped_standarization,
        "yeast_format": yeast_standarization,
        "no_format": no_format
        }
    
    func = standarizations.get( of , "Wrong original format.")
    
    assert type(func) != str, func
    
    if (of == "yeast_format") or (of == "no_format"):
        return func(fenos, bp)
    else:
        return func(bp)

def codification( x, cod_method ):
    
    codifications = {
        "additive" : additive_cod,
        "bio_ohe" : ohe_cod_atcg,
        "ohe_add"  : ohe_cod_add,
        "ohe"      : ohe,
        "no_codif" : identidad
        }
    
    func = codifications.get(cod_method, "Wrong codification method.")
    
    assert type(func) != str, func
    
    return func( x )

def imputation( x, imput_method, i_flag ):
    
    imputations = {
        "mode": mode_imputation,
        "del_rows": delete_rows_imputation,
        "del_cols": delete_cols_imputation,
        "no_imput": identidad
        }
    
    func = imputations.get(imput_method, "Wrong imputation method.")
    assert type(func) != str, func
    
    return func( x, i_flag )
