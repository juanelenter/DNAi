#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 19:41:59 2020

@author: ignaciohounie
"""
from sklearn.linear_model import Ridge
import numpy as np

def snp_blup_ridge(X, y, h2):
    """
    Do SNP-BLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.
    :param X: genotype matrix
    :param y: phenotype vector
    :return: float, prediction accuracy.
    """

    p = np.mean(X, axis=0) / 2
    d = 2 * np.sum(p * (1 - p))
    r = (1 - h2) / (h2 / d)

    X_copy = X.copy()-(2 * p)

    clf = Ridge(alpha=r)
    clf.fit(X_copy, y)
    return clf 

#def make_grm(geno):
#    """
#    Make the genomic relationship matrix.
#    - Expects rows as individuals and columns as markers.
#    :param geno: np.ndarray, 2D, genotype marker matrix. (N x P)
#    :return: np.ndarray, 2D, genomic relationship matrix. (N X N)
#    """
#    p = np.mean(geno, axis=0) / 2  # Row means over 2.
#    P = 2 * (p - 0.5)
#    W = (geno - 1) - P  # Subtract P from each column of G, where G is put into {-1, 0, 1} format from {0, 1, 2}.
#    WtW = np.matmul(W, np.transpose(W))
#    return WtW / (2 * np.sum(p * (1 - p)))
#
#def gblup(X, y, h2):
#    """
#    Do GBLUP on the provided data. Assumes self.data is SNP data in {0, 1, 2} format.
#    :param indices: list, list of ints corresponding to the features indices to use.
#    :param train_indices: list, list of ints corresponding to which samples to use for training.
#    :param validation_indices: list, list of ints corresponding to which samples to use for validation.
#    :return: float, prediction accuracy.
#    """
#    G = make_grm(X)
#
#    r = (1 - h2) / h2
#
#    # Inverse the matrix 
#    G_inv = G
#    G_inv.flat[:: G_inv.shape[0] + 1] += r  # Add regularization term to the diagonal of G.
#    G_inv = np.linalg.inv(G_inv)
#    return np.matmul(G, G_inv)