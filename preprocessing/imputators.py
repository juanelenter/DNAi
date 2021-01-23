# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:25:48 2020

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
from codifiers import *

#%% Imputation methods 

def mode_imputation( x, iflag ):
    """
    Mode imputation (reference allele imputation) for genotype matrix. 
    
    Parameters
    ----------
    x: Numpy array
        Genotype matrix to impute.
        
    iflag: object
        Symbol that indicates missing value.

    Returns
    -------        
    x_i : numpy array
        Imputated genotype matrix.
        Each row is an individual, each is column is a different SNP.

    """
    xi = x.copy()
    reference_alleles = [ mode(locus, nan_policy = 'omit')[0] for locus in x.T ]
    sparse_columns = np.where( reference_alleles == iflag )[0] #columns where the mode is the nan_flag
    
    for i, ref in enumerate( reference_alleles ):
        xi[ :, i ][ xi[:, i] == iflag ] = ref
    
    #Delete sparse columns
    xi = np.delete(xi, sparse_columns, axis = 1)
    
    return xi

def delete_rows_imputation( x, iflag ):
    """
    Delete observations where SNPs are missing ( = 0 in .ped) .

    Parameters
    ----------
    x : numpy array
        Genotype matrix.
    
    iflag: object
        Symbol that indicates missing value.

    Returns
    -------
    xi : numpy array
        Imputated genotype matrix.

    """
    xi = x.copy()
    rows_zero = np.where(xi == iflag)[0]
    xi = np.delete(xi, rows_zero, axis = 0)
    
    return xi

def delete_cols_imputation( x, iflag ):
    
    """
    Delete columns where SNPs are missing ( = 0 in .ped).
    If original format is .ped has to delete a pair number of columns.

    Parameters
    ----------
    x : numpy array
        Genotype matrix.
    
    iflag: object
        Symbol that indicates missing value.

    Returns
    -------
    xi : numpy array
        Imputated genotype matrix.

    """
    xi = x.copy()
    cols_zero = np.where(xi == iflag)[1]
    xi = np.delete(xi, cols_zero, axis = 1)
    
    return xi

