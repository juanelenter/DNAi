import numpy as np

def drop_markers_clusters(geno, clusters, ratio):
    cluster_num = np.max(clusters)
    cluster_indxs = [np.where(clusters == i)[0] for i in range(1, cluster_num + 1)]
    #_, cluster_counts = np.unique(clusters, return_counts = True)

    drop_indxs = []
    rng = np.random.default_rng()
    
    # Ensure Clusters get at least one sample remaining
    # Count number of notremoved samples to correct ratio 
    num_clust_1_sample = 0
    for cl_indxs in cluster_indxs:
        sample_size = round(ratio*cl_indxs.size)
        if sample_size == cl_indxs.size:
            num_clust_1_sample+=1
    
     # update ratio so that the total number of removed SNPs remains constant
    ratio = ratio*geno.size[1]/(geno.size[1]-num_clust_1_sample)  
        
    for cl_indxs in cluster_indxs:
        sample_size = round(ratio*cl_indxs.size)
        if sample_size == cl_indxs.size: sample_size -= 1
        drop_indxs += list(rng.choice(cl_indxs, size = int(sample_size), replace = False))
    geno_drop = np.delete(geno, drop_indxs, axis = 1)

    
    return geno_drop
