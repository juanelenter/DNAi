import numpy as np

def add_noise(feno, ratio):
    
    feno_n = feno.copy()
    N_random = np.random.permutation(feno.shape[0])[:int(ratio*feno_n.shape[0])]
    std = np.std(feno_n)
    for n in N_random:
        if np.random.choice([0, 1]):
            feno_n[n] = feno_n[n] + 2*std
        else:
            feno_n[n] = feno_n[n] - 2*std
    
    return feno_n