import numpy as np

def drop_markers_clusters(geno, clusters, ratio):
    cluster_num = np.max(clusters)
    _, cluster_counts = np.unique(clusters, return_counts = True)
    arg = np.flip(np.argsort(cluster_counts)) + 1
    cluster_indxs = [np.where(clusters == i)[0] for i in arg]

    buf = 0
    for cl_indxs in cluster_indxs:
        if round(ratio*cl_indxs.size) == cl_indxs.size:
            buf += 1
    
    drop_indxs = []
    rng = np.random.default_rng()

    for cl_indxs in cluster_indxs:
        sample_size = round(ratio*cl_indxs.size)
        if sample_size >= cl_indxs.size: 
            sample_size -= 1
        else:
            if buf > 0 and sample_size - cl_indxs.size > 1:
                buf -= 1
                sample_size += 1

        drop_indxs += list(rng.choice(cl_indxs, size = int(sample_size), replace = False))
        
    geno_drop = np.delete(geno, drop_indxs, axis = 1)
    return geno_drop
