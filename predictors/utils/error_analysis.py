# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:52:51 2020

@author: Juan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
from utils.indexes import sort_by_error
from utils.logging import experiment_logger
from sklearn.decomposition import PCA
#import plotly.express as px

def normality_test( error ):
    """
    Is normality assumption consistent with experiments?
    Performs Shapiro-Wilk test for normality in the error vector.

    Parameters
    ----------
    error_dict : dict
        Dictionnary returned by indexes.py/.

    Returns
    -------
    p_value: float
        Result of the normality test.

    """
    
    p_value = shapiro( error )[1] 
    return p_value

def hist_error( error_dict, n_split, logger ):
    """

    Parameters
    ----------
    error_dict : dict
        Dictionnary returned by indexes.py/.
    n_split : int
        Number of split.


    """
    plt.figure()
    plt.hist( error_dict["raw_error"])
    logger.log_figure( plt, figure_name = f"Error Histogram {n_split}" )
    logger.log_hist3d(  values=error_dict["raw_error"], figure_name ="Error Histogram", step=n_split )
    return None


def plot_PCA(X, plot_idx, error,  error_idx, logger, n_split, ndim=2):
    
    '''
    Parameters
    ----------
    X : ndarray
        SNP matrix
    idx : int array
        samples' indexes
    error : int array
        error at idxs
    ndim: int
        Number of pca components. The  default is 2.

    Returns
    -------
    '''
    pca  = PCA(n_components=ndim)
    pca.fit(X)
    X = pca.transform(X)
    if ndim ==2:
        plt.figure()
        for key, val in plot_idx.items():
            if val is not None:
                print(key)
                e = error[error_idx[key]]
                plt.scatter(X[val,0], X[val,1], s=e*1000, alpha=0.5)
                #plt.legend(key)
                plt.xlabel("PCA 1")
                plt.xlabel("PCA 2")
                '''
                fig = px.scatter(x=X[val,0], y=X[val,1], size = e)
                fig.show()   
                img_bytes = fig.to_image(format="png")
                '''
        logger.log_figure(plt, figure_name = f"PCA Error {n_split}" )

def error_PCA(error_dict, n_split, X, logger, n_top=20, n_bottom=20, n_middle=0):
    '''
    Parameters
    ----------
    error_dict : TYPE
        DESCRIPTION.
    n_split : TYPE
        DESCRIPTION.
    logger : TYPE
        DESCRIPTION.
    ntop : TYPE, optional
        DESCRIPTION. The default is 20.
    nbottom : TYPE, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    None.

    '''
    idx = error_dict['sorted_idx']
    error = error_dict['sorted_error']
    idx_rel = np.arange(len(idx))
    plot_idx = {'top': None, 'middle': None, 'bottom': None}
    error_idx = {'top': None, 'middle': None, 'bottom': None}
    if n_top!=0:
        plot_idx['top'] = idx[-n_top:]
        error_idx['top'] =idx_rel[-n_top:]
    if n_bottom!=0:
        plot_idx['bottom'] = idx[:n_bottom]
        error_idx['bottom'] =idx_rel[:n_bottom]
    if n_middle!=0:
        plot_idx['middle'] = idx[len(idx)-int(n_middle//2):len(idx)+int(n_middle//2)]
        error_idx['middle'] = idx_rel[len(idx_rel)-int(n_middle//2):len(idx_rel)+int(n_middle//2)]

    plot_PCA(X, plot_idx, error, error_idx, logger, n_split, ndim=2)
    
    
    