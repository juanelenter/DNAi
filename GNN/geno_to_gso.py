# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 16:26:39 2020

@author: Juan
"""

from snp_correlation import corr
from sparsers import sparse_thresh, sparse_knn, sparse_cluster
from normalizers import normalize
from coarsen import coarsen_gso, coarsen_signal
from clustering import drop_markers_clusters
from fermat_distance import fermat_distance_matrix
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import argparse
import os
import joblib
from tqdm import tqdm
#%%

def geno_to_gso(X, dist = "corr", norm_method = "eigenvalue", sparse_method = "thresh", 
                del_rate = 0.8, criterion = "maxclust", max_clusters = 35, k = 100, std_geno = True, std_pheno=True):
    """

    Parameters
    ----------
    X : numpy array
        SNP Matrix.
    dist : str, optional
        Distance to be used. The default is "corr".
    norm_method : str, optional
        Normalization method. The default is "eigenvalue".
    sparse_method : str, optional
        Sparsification method. The default is "thresh".
    del_rate : float, optional
        Deletion rate for sparsification (only used with sparse_method = "thresh"). The default is None.
    criterion : TYPE, optional
        Clustering criterion (only used with sparse_method = "clust"). The default is "maxclust".
    max_clusters : TYPE, optional
        The default is 35.
    k : TYPE, optional
        Number of neighbors. Only used with sparse_method = "knn". The default is None.

    Returns
    -------
    gso: numpy array
        Graph Shift Operator
    """
    meta = {"sparse_method" : sparse_method,
            "dist" : dist,
            "norm_method" : norm_method, 
            "std_geno" : std_geno,
            "std_pheno" : std_pheno}

    if dist == "corr":
        gso = corr(X)
    elif dist == "fermat":
        gso = fermat_distance_matrix(X)
    else:
        print("Wrong distance dude.. choose either corr or fermat.")
    
    if sparse_method == "thresh":
        gso = sparse_thresh(gso, del_rate)
        meta["del_rate"] = del_rate
    elif sparse_method == "cluster":
        gso = sparse_cluster(gso, criterion, max_clusters)
        meta["criterion"] = criterion
        meta["max_clusters"] = max_clusters
    elif sparse_method == "knn":
        gso = sparse_knn(gso, k)
        meta["k"] = k
    else:
        print("Wrong sparsification method dude.. choose either thresh, clust or knn.")
    
    gso = normalize(gso, norm_method)
    
    return gso, meta

    
if __name__ == "__main__":
    
    print("Usage example: python geno_to_gso.py  data/yeast/geno_yeast_congored.npy data/yeast/pheno_yeast_congored.npy ")

    parser = argparse.ArgumentParser()
    parser.add_argument("geno_path", help = "path to the genotype matrix")
    parser.add_argument("pheno_path", help = "path to the phenotype vector")
    parser.add_argument("-ph", "--std_pheno", help = "standardize phenotype", action = "store_false")
    parser.add_argument("-dr", "--del_rate", type=float, help = "sparcification ratio", default=0.97)
    parser.add_argument("-g", "--std_geno", help = "standardize genotype", action = "store_false")
    parser.add_argument("-c", "--coarsening", help = "graph coarsening", action = "store_true")
    parser.add_argument("-cr", "--coarsening_ratio", type=float, help = "graph coarsening", default=0.5 )
    parser.add_argument("-rs", "--random_sampling", help = "sample snps randomly", action = "store_true")
    parser.add_argument("-cs", "--cluster_sampling", help = "sample snps via clustering", action = "store_true")
    parser.add_argument("-pres", "--pre_sampling", help = "sample snps before constructing graph", action = "store_true")
    parser.add_argument("-sr", "--sampling_ratio", type=float, help = "graph coarsening", default=0.1 )
    args = parser.parse_args()


    ### Load data
    path_to_geno = args.geno_path
    path_to_pheno = args.pheno_path
    X = np.load(path_to_geno, allow_pickle = True)
    y = np.load(path_to_pheno, allow_pickle = True)
    
    # Sample graph randomly to reduce dim
    if args.random_sampling and args.pre_sampling:
        N = X.shape[1]
        num_samples = int(args.sampling_ratio*X.shape[1])
        print(f"sampling {num_samples} out of {N} SNPs")
        indexes = np.random.choice(N, size=num_samples, replace=False)
        X = X[:, indexes]

    if args.cluster_sampling:
        # falta implementar otros métodos, lo pongo acá por ahora
        X = drop_markers_clusters(X, args.sampling_ratio)
        print(f"Markers deleted via clustering. Markers remaining: {X.shape[1]}")

    ### Train Val Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15)
        
    if args.std_geno:
        print("Standardizing markers.")
        mean = np.mean(X_train, axis = 0).reshape((1, -1))
        std = np.std(X_train, axis = 0).reshape((1, -1))
        X_train = (X_train - mean)/std
        X_test = (X_test - mean)/std
        X_val = (X_val - mean)/std

    if args.std_pheno:
        print("Standardizing phenotypes.")
        mean = np.mean(y_train, axis = 0)
        std = np.std(y_train, axis = 0)
        y_train = (y_train - mean)/std
        y_test = (y_test - mean)/std
        y_val = (y_val - mean)/std

    ### Build GSO
    #X_train = X_train[:10,:10] ###########for debugging####################
    print("Computing GSO")
    gso, _ = geno_to_gso(X_train, del_rate = args.del_rate)
    print(f"The shape of the GSO is: {gso.shape}")

    ### Coarsening for dim reduction
    if args.coarsening:
        print(f"Coarsening gso with ratio {args.coarsening_ratio}")
        C, Gc, Call, Gall = coarsen_gso(gso, args.coarsening_ratio)
        X_train = coarsen_signal(X_train, C)
        X_val = coarsen_signal(X_val, C)
        X_test = coarsen_signal(X_test, C)
        gso = Gc.W.todense()
        print(f"GSO resulting shape {gso.shape}")
    
    # Sample graph randomly to reduce dim
    if args.random_sampling and not args.pre_sampling:
        N = X_train.shape[1]
        num_samples = int(args.sampling_ratio*X_train.shape[1])
        print(f"sampling {num_samples} out of {N} SNPs")
        indexes = np.random.choice(N, size=num_samples, replace=False)
        gso = gso[indexes, indexes]
        X_train = X_train[:, indexes]
        X_val = X_val[:, indexes]
        X_test = X_test[:, indexes]
    
    ### Build Bengio's dictionnaries
    train_dict = {"GSO": gso, "X": X_train, "y": y_train}
    val_dict = {"GSO": gso, "X": X_val, "y": y_val}
    test_dict = {"GSO": gso, "X": X_test, "y": y_test}
    
    ### Dump pickle-Ricks
    print("Dumping pickles...")
    pickle_out = open(os.path.join("3.0","train.pickle"),"wb")
    pickle.dump(train_dict, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join("3.0","val.pickle"),"wb")
    pickle.dump(val_dict, pickle_out)
    pickle_out.close()
    pickle_out = open(os.path.join("3.0","test.pickle"),"wb")
    pickle.dump(test_dict, pickle_out)
    pickle_out.close()

    # metadata
    pickle_out = open(os.path.join("3.0", "meta.pickle"),"wb")
    pickle.dump(vars(args), pickle_out)
    pickle_out.close()
