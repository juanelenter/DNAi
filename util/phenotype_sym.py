#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:02:39 2020

@author: ignaciohounie
"""

import numpy as np
from sklearn.impute import SimpleImputer

def txt_to_numpy(filename, imputation = False):
    array = np.loadtxt(filename)
    if imputation:
        imp = SimpleImputer(missing_values=[3,4,5], strategy='most_frequent')
        imp.fit_transform(array)
    else:
        array = array[:, np.all(array < 3, axis=0)]
    return array

def generate_alpha(n_snps, n_qtl, clustered = 0, mean_effect = 0.0, std_effect = 1.0, normalize = False):
    #Generate alpha vector
    alpha = np.zeros(n_snps)
    if clustered == 0:
        # sample n_qtl positions from  n_snps
        qtl = np.random.permutation(n_snps)[:n_qtl]
    # Sample effect sizes
    weights = np.random.normal(loc=mean_effect, scale=std_effect, size=n_qtl)
   #Assigning weights to qtl
    alpha[qtl] = weights
    if normalize:
        alpha = alpha/np.linalg.norm(alpha)
    return alpha

def simulate_additive(gen_ar, alfa, h=0.3, mean_pheno = 0, mean_noise = 0.0, std_noise = 1.0):
    gen = gen_ar@alfa
    gen_var = np.var(gen)
    std_noise = np.sqrt(gen_var*(1/h-1))
    noise =  np.random.normal(loc=mean_noise, scale=std_noise, size=gen_ar.shape[0])
    return gen+noise


