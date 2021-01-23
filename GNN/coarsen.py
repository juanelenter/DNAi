from coarsening_lib.graph_coarsening.coarsening_utils import *
import coarsening_lib.graph_coarsening.graph_utils
import os
import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
import pygsp as gsp
from pygsp import graphs
from  scipy.sparse import coo_matrix

def coarsen_gso(gso, coarsening_ratio, method = 'variation_neighborhood', r= 0.5, k=10):    
    S = (gso > 0).astype(int)
    np.fill_diagonal(S, 0)
    S = np.triu(S)+np.triu(S).T
    np.fill_diagonal(S, 0)
    S= coo_matrix(S)
    G = graphs.Graph(W = S)
    G.compute_fourier_basis()
    C, Gc, Call, Gall = coarsen(G, K=k, r=r, method=method)
    return C, Gc, Call, Gall
def coarsen_signal(signal, C):
    signal_out = []
    for x in signal:
        signal_out.append(coarsen_vector(x, C))
    return np.array(signal_out)