import numpy as np
import pandas as pd
import sys
import json
from utils.logging import experiment_logger
from importlib import import_module
#from train_test_models import *
from datetime import date
from scipy.stats import pearsonr
#from sklearn.metrics import mean_squared_error as mse
import os
from sklearn.metrics import mean_squared_error as mse
from sklearn.base import clone
#sys.path.append(".")
#from utils.logging import experiment_logger
# Noise contamination functions
def add_noise(pheno, ratio, std_factor = 2):
    
    pheno_n = pheno.copy()
    N_random = np.random.permutation(pheno.shape[0])[:int(ratio*pheno_n.shape[0])]
    std = np.std(pheno_n)
    for n in N_random:
        pheno_n[n] = pheno_n[n] + np.random.choice([-1, 1])*std_factor*std
    
    return pheno_n

def drop_markers(geno, ratio):
    
    geno_n = geno.copy()
    N_or = geno.shape[1]
    N_random = np.random.permutation(N_or)[:int(N_or*ratio)]
    geno_n = np.delete(geno_n, N_random, axis = 1)
    print(str(ratio) + "%" + "of markers deleted.")
    return geno_n

def drop_samples(geno, pheno, ratio):
    
    geno_n = geno.copy()
    pheno_n = pheno.copy()
    N_or = pheno.shape[0]
    N_random = np.random.permutation(N_or)[:int(N_or*ratio)]
    geno_n = np.delete(geno_n, N_random, axis = 0)
    pheno_n = np.delete(pheno_n, N_random)

    return geno_n, pheno_n, N_random

class split_data:

    def __init__(self, X,  y, idx_test, idx_train, std_geno = False, std_pheno = True):

        self.X_train = X[idx_train].astype(np.float32)
        self.Y_train = y[idx_train]
        self.X_test = X[idx_test].astype(np.float32)
        self.Y_test = y[idx_test]
        self.idx_test = idx_test
        self.idx_train = idx_train
        self.std_geno = std_geno
        self.std_pheno = std_pheno

    def standardize_geno(self):

        if self.std_geno:
            self.std_geno = 1
            x_train_mean = np.mean(self.X_train)
            x_train_std = np.std(self.X_train)
            self.X_train -= x_train_mean
            self.X_train /= x_train_std
            self.X_test -= x_train_mean
            self.X_test /= x_train_std

    def standardize_pheno(self):

        if self.std_pheno:
            self.std_pheno = 1
            y_train_mean = np.mean(self.Y_train)
            y_train_std = np.std(self.Y_train)
            self.Y_train -= y_train_mean
            self.Y_train /= y_train_std
            self.Y_test -= y_train_mean
            self.Y_test /= y_train_std

    def debug_mode(self):

        self.X_train = self.X_train[::50]
        self.Y_train = self.Y_train[::50]
        self.X_test = self.X_test[::50]
        self.Y_test = self.Y_test[::50]
        print("Debug mode ON.")

def model_eval(input_cfg, model_name, opt_args):

    cv_params = {key : opt_args[key] for key in ["n_jobs", "n_iter", "cv"]}

    # load input config
    base_dir = '..'
    config_path = os.path.join(base_dir,'Input', 'config')
    with open( os.path.join(config_path, input_cfg), "r") as read_file:
        cfg = json.load(read_file)
    # custom grid
    custom_grid = opt_args["custom_grid"]
    # noise type
    noise_type = opt_args["noise_type"]
    # metadata
    base = cfg['meta_data']['base']
    imput = cfg['meta_data']['imputation']
    codif = cfg['meta_data']['codification']
    n_feno = cfg["meta_data"]["n_feno/env"]
    # load data
    print("Loading data... ")
    X = np.load(cfg['X_path'])
    y = np.load(cfg['Y_path'])
    # load split indxs
    if opt_args["split"] == "all":
        split_indxs = np.arange(cfg["meta_data"]["num_splits"])
    else:
        split_indxs = [opt_args["split"]]

    # evaluate noise type
    if noise_type == "std":
        ratios = np.arange(0, 100, step = 10)*0.01
        print("Adding std noise...")
        ys = [add_noise(y, noise_ratio) for noise_ratio in ratios]
        Xs = [X.copy() for i in range(len(ys))] 
        print("Done! Noise ratios: ", ratios)

    elif noise_type == "drop_markers":
        #ratios = np.arange(0, 100, step = 10)*0.01
        ratios = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])*0.01
        print("Deleting markers...")
        Xs = [drop_markers(X, del_ratio) for del_ratio in ratios]
        ys = [y.copy() for i in range(len(Xs))] 
        print("Done! Drop ratios: ", ratios)

    elif noise_type == "drop_samples":
        ratios = np.arange(0, 100, step = 10)*0.01
        #print("Deleting samples...")
        #aux = [drop_samples(X, y, del_ratio) for del_ratio in ratios]
        #Xs = [tup[0] for tup in aux]
        #ys = [tup[1] for tup in aux]
        #print("Done! Drop ratios: ", ratios)
        Xs = [X.copy() for i in range(ratios.size)]
        ys = [y.copy() for i in range(ratios.size)]

    # Logging stuff
    print("log_exp:", opt_args["log_exp"])
    print("log_proj:", opt_args["log_proj"])
    if opt_args["log_exp"] is not None:
        # boolean flag indicating wether to log experiments
        log_experiment = True
        # if project is passed through command line 
        if opt_args["log_proj"] is not None:
             project_name = opt_args["log_proj"]
        else:    
            # use database project, or debug
            if opt_args["debug"]:
                project_name =  'debug'
            else:
                project_name = cfg["meta_data"]["base"]                      
    else:
        log_experiment = False
        # start logging
    if log_experiment:
        print("INIT LOGGING")
        logger = experiment_logger(opt_args["log_exp"], project_name)
        logger.add_params({'model': model_name, 'base': base,
                          'imputation': imput, 'coding': codif,
                          "noise_type" : noise_type, "noise_ratios" : ratios,
                          "n_feno/env" : n_feno, **cv_params})    
    # create splits for each noise ratio
    splits_n = []
    for n, (X_n, y_n) in enumerate(zip(Xs, ys)):
        splits = []
        for i, split_indx in enumerate(split_indxs): 
            idx_test = np.load(cfg['index_test'] + str(split_indx) + ".npy").astype(np.int)
            idx_train = np.load(cfg['index_train'] + str(split_indx) + ".npy").astype(np.int)
            if noise_type == "drop_samples":
                idx_test = np.random.choice(idx_test, size = int((1 - ratios[n])*idx_test.size))
                idx_train = np.random.choice(idx_train, size = int((1 - ratios[n])*idx_train.size))
    
            split = split_data(X_n,  y_n, idx_test, idx_train)
            #standardize
            if opt_args["std_geno"]:
                print('Standardizing split {} genotypes... '.format(i))
                split.standardize_geno()
                print("Done!")
            if opt_args["std_pheno"]:
                print('Standardizing split {} phenotypes... '.format(i))
                split.standardize_pheno()
                print("Done!")
            if opt_args["debug"]:
                split.debug_mode()
            splits.append(split)
        splits_n.append(splits)

    # import selected model   
    model = import_module(model_name, package=None)
    results = np.zeros((ratios.size, cfg["meta_data"]["num_splits"], 4))
    for n, _ in enumerate(ratios):
        metrics = {"r_test" : [],
                   "r_train" : [],
                   "mse_test" : [],
                   "mse_train" : []
        }
        # select splits for this noise factor
        splits = splits_n[n]
        # choose random split
        grid_split = np.random.choice(splits) 
        # gridsearch for best model params on random split (chosen above)
        grid_params, _, _, _, _, model_instance = model.train_test(grid_split.X_train, grid_split.Y_train,
                                                           grid_split.X_test, grid_split.Y_test,
                                                           cv_params, custom_grid)
        for i, split in zip(split_indxs, splits):
            print("Split {}/{}.".format(int(i)+1, len(splits)))
            best_model = clone(model_instance)
            best_model.set_params(**grid_params)
            best_model.fit(split.X_train, split.Y_train)
            train_pred = best_model.predict(split.X_train)
            test_pred = best_model.predict(split.X_test)
            metrics["r_test"].append(pearsonr(test_pred, split.Y_test))
            metrics["r_train"].append(pearsonr(train_pred, split.Y_train))
            metrics["mse_test"].append(mse(test_pred, split.Y_test))
            metrics["mse_train"].append(mse(train_pred, split.Y_train))
            results[n, i, 0] = metrics["r_test"][-1][0]
            results[n, i, 1] = metrics["r_train"][-1][0]
            results[n, i, 2] = metrics["mse_test"][-1]
            results[n, i, 3] = metrics["mse_train"][-1]

        # log mean metrics for this noise factor
        if log_experiment:
            for metric_name, value in metrics.items():
                if metric_name.startswith("r"):
                    logger.log_metrics(metric_name + "_mean", np.mean(np.array(value)[:, 0]), epoch = n)
                else:
                    logger.log_metrics(metric_name + "_mean", np.mean(value), epoch = n)

   # np.save(os.path.join("..", "output", "yeast_noise", "mod_{}_env_{}_nt_{}.npy".format(model_name, n_feno, noise_type)), results)
#############################################################################################################################

if __name__ == "__main__":
    sys.path.append('train_test_models')
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 2:
        print('Use: train_test.py <config_name.json> <model_name> <-g> <-f> <-h> <hyperparameters.json>')
    else:

        # default parameters
        opt_args = {"debug" : False,
                    "std_geno" : True,
                    "std_pheno" : True,
                    "custom_grid" : False,
                    "n_jobs" : -1,
                    "n_iter" : 50,
                    "cv" : 5,
                    "log_exp": None,
                    "log_proj": None,
                    "split" : "all",
                    "noise_type" : "std"}

        # command line arguments
        if "-d" in argv:
            opt_args["debug"] = True
        if "-g" in argv:
            opt_args["std_geno"] = False
        if "-f" in argv:
            opt_args["std_pheno"] = False
        if "-n_jobs" in argv:
            opt_args["n_jobs"] = int(argv[argv.index("-n_jobs") + 1])
        if "-n_iter" in argv:
            opt_args["n_iter"] = int(argv[argv.index("-n_iter") + 1])
        if "-cv" in argv:
            opt_args["cv"] = int(argv[argv.index("-cv") + 1])
        # log experiment in neptune/comet/both
        if "-log_exp" in argv:
            opt_args["log_exp"] = argv[argv.index("-log_exp") + 1]
        # set manually which project to log. 
        # Otherwise it will be set to database default project.
        if "-log_proj" in argv:
            opt_args["log_proj"] = argv[argv.index("-log_proj") + 1]
        if "-s" in argv:
            opt_args["split"] = argv[argv.index("-s") + 1]
        if "-nt" in argv:
            opt_args["noise_type"] = argv[argv.index("-nt") + 1]

    model_eval(sys.argv[1], sys.argv[2], opt_args)
