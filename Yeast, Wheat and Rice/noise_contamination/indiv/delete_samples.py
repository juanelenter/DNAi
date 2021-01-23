import numpy as np

def delete_samples(geno, feno, ratio):
    
    geno_n = geno.copy()
    feno_n = feno.copy()
    N_or = feno.shape[0]
    N_random = np.random.permutation(N_or)[:int(N_or*ratio)]
    geno_n = np.delete(geno_n, N_random, axis = 0)
    feno_n = np.delete(feno_n, N_random)

    return geno_n, feno_n