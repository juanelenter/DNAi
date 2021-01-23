# -*- coding: utf-8 -*-
"""
Created on Mon May 18 19:25:09 2020

@author: Juan
"""

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import mode

# Local Imports
from switchers import *
from codifiers import *
from imputators import *

#%% Standarization methods

def ped_standarization( p ):
    """
    Parameters
    ----------
    p : string
        Path to the data base.

    Returns
    -------
    x : numpy array
        Genotype matrix.
    y : numpy array
        Phenotype vector.

    """
    ped = np.genfromtxt(p, delimiter = " ", dtype = str)
    
    y = ped[:, 5].astype(int)
    x = ped[:, 6:]
    
    x[ x == "A"] = 1
    x[ x == "T"] = 2
    x[ x == "C"] = 3
    x[ x == "G"] = 4
    x[ x == "0"] = 0
    x = np.array(x, dtype = int)
    return x, y

def yeast_standarization( environments, paths ):
    """
    
    Parameters
    ----------
    environments : list
        Index of phenotypes to include in the phenotype matrix.
    paths : list of strings
        Paths to the data set.

    Returns
    -------
    x : numpy array
        Genotype matrix.
    y : numpy array
        Phenotype matrix.

    """
    path_geno = paths[0]
    path_feno = paths[1]
        
    y = pd.read_csv(path_feno)
    x = pd.read_csv(path_geno)
    print( x.shape )
    
    feno_names = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide', 
                  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
                  'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]
    
    y = y[ feno_names[environments] ]
    
    missing_phenos = y[ y.isnull() ].index.values
    x = x.drop(columns = ["Unnamed: 0"])
    x = x.drop( missing_phenos, axis = 0).values
    y = y.drop( missing_phenos, axis = 0).values
    print(x.shape)
    return x, y

def no_format(environments, paths):
    
    path_geno = paths[0]
    path_feno = paths[1]
    
    y = np.load(path_feno, allow_pickle = True)
    if environments != None:
        y = y[:, environments]
    x = np.load(path_geno, allow_pickle = True)
    
    return x, y
    
    
    