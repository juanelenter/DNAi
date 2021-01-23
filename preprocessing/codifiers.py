# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:26:08 2020

@author: Juan
"""
# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode

# Local Imports
from switchers import *
from standarizers import *
from imputators import *

#%% Codification methods

def identidad( x ):
    return x

def additive_cod( x ):
    """
    { aa, aA, Aa, AA } -> { 0 , 1 , 1 , 2 }   

    Parameters
    ----------
    x : numpy array
        Genotype matrix.

    Returns
    -------
    x_add : numpy array
        Codified genotype matrix.

    """
    # Compute reference alleles as the mode of each column.
    reference_alleles1 = [ mode(locus)[0] for locus in x[:,::2].T ]
    reference_alleles1 = np.asarray(reference_alleles1)
    
    # Additive codification of the genotype matrix.
    x_add = np.empty( x[:, ::2].shape )
    
    for i, ref in enumerate( reference_alleles1 ):
        x_add[ :, i ] = (x[:, 2*i ] != ref)
        x_add[ :, i ] += (x[:, 2*i + 1 ] != ref)
        
    return x_add

def ohe_cod_atcg( x ):
    """
    { A, T, C, G } -> { 1 0 0 0, 0 1 0 0, 0 0 1 0, 0 0 0 1 }
    A particular SNP {A, T} -> {1 0 0 0, 0 1 0 0}
    
    Parameters
    ----------
    x : numpy array
        Genotype matrix.

    Returns
    -------
    x_add : numpy array
        Codified genotype matrix.
    """
    cats = [[1,2,3,4]] * x.shape[1]
    enc = OneHotEncoder(categories = cats, drop = None)
    x_ohe = enc.fit_transform(x).toarray()
    return x_ohe

def ohe_cod_add( x ):
    
    """
    {aa = 0, aA or Aa = 1, AA = 2} -> { 0 0 1, 0 1 0, 0 1 0, 1 0 0 }

    Parameters
    ----------
    x : numpy array
        Genotype matrix.

    Returns
    -------
    x_add : numpy array
        Codified genotype matrix.

    """
    
    x_ohe = additive_cod( x ) 
    
    # Define possible categories aa -> 0, aA or Aa -> 1, AA -> 2
    cats = [[0, 1, 2]] * x_ohe.shape[1]
    enc = OneHotEncoder(categories = cats, drop = "first")
    x_ohe = enc.fit_transform(x_ohe).toarray()
    return x_ohe

def ohe( x ):
    """
    One hot encoding ofr the genotypes that are already in the format {0,1,2}
    {aa = 0, aA or Aa = 1, AA = 2} -> { 0 0 1, 0 1 0, 0 1 0, 1 0 0 }

    Parameters
    ----------
    x : numpy array
        Genotype matrix.

    Returns
    -------
    x_add : numpy array
        Codified genotype matrix.

    """
    
    # Define possible categories aa -> 0, aA or Aa -> 1, AA -> 2
    cats = [[0, 1, 2]] * x.shape[1]
    enc = OneHotEncoder(categories = cats, drop = "first")
    x_ohe = enc.fit_transform(x).toarray()
    return x_ohe