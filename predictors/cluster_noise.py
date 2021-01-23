import numpy as np 
import os
import json
import sys
from pandas import read_csv
from importlib import import_module
from scipy.cluster import hierarchy
from scipy.stats import pearsonr
from utils.drop2 import drop_markers_clusters
from utils.split_class import split_data
from utils.logging import experiment_logger


def drop(input_cfg, model_name, std_geno = True, std_pheno = True, n_jobs = -1,
         n_iter = 50, cv = 5, log_exp = None, log_proj = None, split = "all",
         custom_grid = False, debug = False):
    
    # load input config
    base_dir = ".."
    config_path = os.path.join(base_dir,"Input", "config")
    with open(os.path.join(config_path, input_cfg), "r") as read_file:
        cfg = json.load(read_file)
    
    # custom grid
    if custom_grid:
        grid_path = os.path.join(base_dir, "Input", "grid", custom_grid)
        with open(grid_path, "r") as read_file:
            grid = json.load(read_file)

    # cfg metadata
    base = cfg["meta_data"]["base"]
    imput = cfg["meta_data"]["imputation"]
    codif = cfg["meta_data"]["codification"]
    n_feno = cfg["meta_data"]["n_feno/env"]
    cv_params = {"n_jobs" : n_jobs, "n_iter" : n_iter, "cv" : cv}

    # load data
    print("Loading data... ")
    X = np.load(cfg["X_path"])
    y = np.load(cfg["Y_path"])

    # load split indxs
    if split is "all":
        split_indxs = np.arange(cfg["meta_data"]["num_splits"])
    else:
        split_indxs = [split]

    # hierarchical clustering
    Z = hierarchy.linkage(X.T, method = "average", metric = "correlation")

    # get clusters
    max_clusters = 35
    fc = hierarchy.fcluster(Z, max_clusters, criterion = "maxclust")

    # drop markers
    ratios = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.1, 99.2,
                      99.3, 99.4, 99.5, 99.6, 99.7, 99.8, 99.9])*0.01
    print("Deleting markers...")
    Xs = [drop_markers_clusters(X, fc, del_ratio) for del_ratio in ratios]
    ys = [y.copy() for i in range(len(Xs))] 
    print("Done. Drop ratios: ", ratios)

    # comet
    if log_exp == "comet":
        project_name = log_proj if log_proj else base
        logger = experiment_logger(log_exp, project_name)
        logger.add_params({"model": model_name, "base": base,
                           "imputation": imput, "coding": codif,
                           "noise_ratios" : np.round(ratios, 3), 
		           "n_feno/env" : n_feno, "cluster_num" : max_clusters,
			   "n_snps" : [G.shape[1] for G in Xs]})    

    # create splits for each noise ratio
    print("Creating splits...")
    splits_n = []
    for n, (X_n, y_n) in enumerate(zip(Xs, ys)):
        splits = []
        for i, split_indx in enumerate(split_indxs): 
            idx_test = np.load(cfg["index_test"] + str(split_indx) + ".npy").astype(np.int)
            idx_train = np.load(cfg["index_train"] + str(split_indx) + ".npy").astype(np.int)
            split = split_data(X_n,  y_n, idx_test, idx_train)
            # standardize
            if std_geno:
                split.standardize_geno()
            if std_pheno:
                split.standardize_pheno()
            if debug:
                split.debug_mode()
            splits.append(split)
        splits_n.append(splits)

    from sklearn.metrics import mean_squared_error as mse
    from sklearn.base import clone
    # import selected model   
    model = import_module(model_name, package=None)
    for n, _ in enumerate(ratios):
        print("Fitting step {}/{}.".format(n + 1, len(ratios)))
        # metrics
        metrics = {"r_test" : [],
                   "r_train" : [],
                   "mse_test" : [],
                   "mse_train" : []}
        # select splits for this noise factor
        splits = splits_n[n]
        # choose random split
        grid_split = np.random.choice(splits) 
        # gridsearch for best model params on random split (chosen above)
        grid_params, _, _, _, _, model_instance = model.train_test(grid_split.X_train, grid_split.Y_train,
                                                                      grid_split.X_test, grid_split.Y_test,
                                                                      cv_params, grid)
        for i, split in zip(split_indxs, splits):
            best_model = clone(model_instance)
            best_model.set_params(**grid_params)
            best_model.fit(split.X_train, split.Y_train)
            train_pred = best_model.predict(split.X_train)
            test_pred = best_model.predict(split.X_test)
            metrics["r_test"].append(pearsonr(test_pred, split.Y_test))
            metrics["r_train"].append(pearsonr(train_pred, split.Y_train))
            metrics["mse_test"].append(mse(test_pred, split.Y_test))
            metrics["mse_train"].append(mse(train_pred, split.Y_train))

        if log_exp:
            for metric_name, value in metrics.items():
                if metric_name.startswith("r"):
                    logger.log_metrics(metric_name + "_mean", np.mean(np.array(value)[:, 0]), epoch = n)
                else:
                    logger.log_metrics(metric_name + "_mean", np.mean(value), epoch = n)

if __name__ == "__main__":
    sys.path.append("train_test_models")
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 2:
        print("Usage: cluster_noise.py <config_name.json> <model_name> <-optional_arguments>")
    else:
        debug = False
        std_geno = True
        std_pheno = True
        n_jobs = -1
        n_iter = 50
        cv = 5
        log_exp = None
        log_proj = None
        split = "all"
        custom_grid = False
        # command line arguments
        if "-d" in argv:
            debug = True
        if "-g" in argv:
            std_geno = False
        if "-f" in argv:
            std_pheno = False
        if "-n_jobs" in argv:
            n_jobs = int(argv[argv.index("-n_jobs") + 1])
        if "-n_iter" in argv:
            n_iter = int(argv[argv.index("-n_iter") + 1])
        if "-cv" in argv:
            cv = int(argv[argv.index("-cv") + 1])
        if "-log_exp" in argv:
            log_exp = argv[argv.index("-log_exp") + 1]
        if "-log_proj" in argv:
            log_proj = argv[argv.index("-log_proj") + 1]
        if "-s" in argv:
            split = argv[argv.index("-s") + 1]
        if "-cg" in argv:
            custom_grid = argv[argv.index("-cg") + 1]

    drop(input_cfg = sys.argv[1], model_name = sys.argv[2], std_geno = std_geno, 
         std_pheno = std_pheno, n_jobs = n_jobs, n_iter = n_iter, cv = 5, 
         log_exp = log_exp, log_proj = log_proj, split = split, custom_grid = custom_grid, 
         debug = debug)
