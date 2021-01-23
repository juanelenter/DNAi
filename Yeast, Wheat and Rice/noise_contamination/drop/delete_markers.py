import numpy as np

def delete_markers(geno, ratio):
    
    geno_n = geno.copy()
    N_or = geno.shape[1]
    N_random = np.random.permutation(N_or)[:int(N_or*ratio)]
    geno_n = np.delete(geno_n, N_random, axis = 1)

    return geno_n