"""
-._    _.--'"`'--._    _.--'"`'--._    _.--'"`'--._    _   
    '-:`.'|`|"':-.  '-:`.'|`|"':-.  '-:`.'|`|"':-.  '.` : '.   
  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '.
  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `.
  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `.
         `-..,..-'       `-..,..-'       `-..,..-'       `         `

                            
                            dnai
    train_test.py:

      load data as indicated in config file, train a model 
      and save predictions.

-._    _.--'"`'--._    _.--'"`'--._    _.--'"`'--._    _   
    '-:`.'|`|"':-.  '-:`.'|`|"':-.  '-:`.'|`|"':-.  '.` : '.   
  '.  '.  | |  | |'.  '.  | |  | |'.  '.  | |  | |'.  '.:   '.  '.
  : '.  '.| |  | |  '.  '.| |  | |  '.  '.| |  | |  '.  '.  : '.  `.
  '   '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.:_ | :_.' '.  `.'   `.
         `-..,..-'       `-..,..-'       `-..,..-'       `         `
"""

print(__doc__)

import numpy as np
import pandas as pd
import sys
import json
from importlib import import_module
#from train_test_models import *
from datetime import date
import os
sys.path.append(".")
from utils.logging import experiment_logger
from utils.indexes import sort_by_error
from utils.error_analysis import normality_test, hist_error, error_PCA


def train_test(cfg_name, model_name, opt_args):
    '''
    Available models:
    gbm.py
    lasso.py
    rf.py
    ridge.py
    svm.py 

    '''
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
                print('Standardizing genotypes... ')
                self.std_geno = 1
                x_train_mean = np.mean(self.X_train)
                x_train_std = np.std(self.X_train)
                self.X_train -= x_train_mean
                self.X_train /= x_train_std
                self.X_test -= x_train_mean
                self.X_test /= x_train_std
                print("Done!")

        def standardize_pheno(self):

            if self.std_pheno:
                print('Standardizing phenotypes... ')
                self.std_pheno = 1
                y_train_mean = np.mean(self.Y_train)
                y_train_std = np.std(self.Y_train)
                self.Y_train -= y_train_mean
                self.Y_train /= y_train_std
                self.Y_test -= y_train_mean
                self.Y_test /= y_train_std
                print("Done!")

        def debug_mode(self):

            self.X_train = self.X_train[::100]
            self.Y_train = self.Y_train[::100]
            self.X_test = self.X_test[::100]
            self.Y_test = self.Y_test[::100]
            print("Debug mode ON.")

    cv_params = {"n_jobs" : opt_args["n_jobs"],
                 "n_iter" : opt_args["n_iter"],
                 "cv" : opt_args["cv"]} 

    custom_grid = opt_args["custom_grid"]

    #load input config
    base_dir = '..'
    config_path = os.path.join(base_dir,'Input', 'config')
    with open( os.path.join(config_path, cfg_name), "r") as read_file:
        cfg = json.load(read_file)

    #load data
    print("Loading data... ", end = '')
    X = np.load(cfg['X_path'])
    y = np.load(cfg['Y_path'])
    print("Done!")
    #splits
    if opt_args["split"] == "all":
        split_indxs = np.arange(cfg["meta_data"]["num_splits"])
    else:
        split_indxs = [opt_args["split"]]
    splits = [] 
    for split_indx in split_indxs: 
        idx_test = np.load(cfg['index_test'] + str(split_indx) + ".npy").astype(np.int)
        idx_train = np.load(cfg['index_train'] + str(split_indx) + ".npy").astype(np.int)
        split = split_data(X,  y, idx_test, idx_train)
        #standardize
        if opt_args["std_geno"]:
            split.standardize_geno()
        if opt_args["std_pheno"]:
            split.standardize_pheno()
        if opt_args["debug"]:
            split.debug_mode()
        splits.append(split)
    
    # metadata
    base = cfg['meta_data']['base']
    imput = cfg['meta_data']['imputation']
    codif = cfg['meta_data']['codification']
    # train & test output paths
    output_cfg = os.path.join(base_dir, 'output', 'config')
    output_pred = os.path.join(base_dir,'output', 'pred')
    output_train = os.path.join(base_dir,'output', 'train')
        
    #####################
    ## Logging stuff
    if opt_args["log_exp"] is not None:
        # boolean flag indicating wether to log experiments
        log_experiment = True
         # if project is passed through command line 
        if opt_args["log_proj"] is not None:
             project_name=opt_args["log_proj"]
        else:# use database project, or debug
            if opt_args["debug"]:
                project_name =  'debug'
            else:
                project_name = cfg["meta_data"]["base"]                      
    else:
        log_experiment = False
    
    model_params_list = []
    if model_name != "all": #train one model only
        
        ## Initialise Logging
        if log_experiment:
            logger = experiment_logger(opt_args["log_exp"], project_name)
            logger.add_params({'model': model_name, 'base': base,
                              'imputation': imput, 'coding': codif,
                              'custom_grid': custom_grid, **cv_params})
         
        model = import_module(model_name, package=None)
        cfg_folder = os.path.join(output_cfg, base, model_name, str(date.today()))
        train_pred_folder = os.path.join(output_train, base, model_name, str(date.today()))
        test_pred_folder = os.path.join(output_pred, base, model_name, str(date.today()))
        results = []
        
            
        #normality_pvalues = [] # array to store pvalues to check normality
        n = len(split.Y_test) # Len of Not Top 10 
        n_worse = np.empty((len(splits)*n, 2)) #column one: indices, column two: errors
        
        for i, split in zip(split_indxs, splits):
            print("Split {}/{}.".format(int(i)+1, len(splits)))                
            grid_params, model_params, train_preds, test_preds, metrics, _ = model.train_test(split.X_train, split.Y_train,
                                                                                           split.X_test, split.Y_test,
                                                                                           cv_params, custom_grid)
            
            e_dict = sort_by_error(split.Y_test, test_preds, split.idx_test)
            
            
            
            n_worse[i*n : (i+1)*n, 0] = e_dict['sorted_idx'][ : n]
            n_worse[i*n : (i+1)*n, 1] = e_dict['sorted_error'][ : n]

            pvalue =  normality_test( e_dict )

            ###########
            # Logging metrics
            if log_experiment:
                error_PCA( e_dict, i, X, logger )
                hist_error( e_dict , i , logger )
                for metric_name, value in metrics.items():
                    logger.log_metrics(metric_name, value, epoch=i)
                logger.log_metrics("pvalue", pvalue, epoch=i)
                #logger.add_params({**grid_params, **model_params})
                model_params_list.append(model_params)
             
            ###############
            # Logging indexs
            np.savetxt( "idx_test.csv", split.idx_test)
            np.savetxt( "idx_train.csv", split.idx_train)
            if log_experiment:
                logger.log_table( "idx_test", split.idx_test )
                logger.log_table( "idx_train", split.idx_train )

            cfg_path = os.path.join(cfg_folder, cfg_name[:-5] + "_{}_".format(i) + '.json')
            train_pred_path = os.path.join(train_pred_folder, cfg_name[:-5] + "_{}_".format(i) + '.npy')
            test_pred_path = os.path.join(test_pred_folder, cfg_name[:-5] + "_{}_".format(i) + '.npy')

            try:
                np.save(train_pred_path, train_preds)
            except FileNotFoundError:
                os.makedirs(train_pred_folder)
                np.save(train_pred_path, train_preds)

            try:
                np.save(test_pred_path, test_preds)
            except FileNotFoundError:
                os.makedirs(test_pred_folder)
                np.save(test_pred_path, test_preds)

            data_out = {
                "train_pred_path" : train_pred_path,
                "test_pred_path" : test_pred_path,
                "model" : model_name,
                "split" : str(i),
                "metrics" : metrics,
                "grid_search_params" : grid_params,
                #"model_coefs" : model_params,
                "input_cfg" : cfg,
                "cv_params" : cv_params,
                "custom_grid" : custom_grid
            }
            ###########
            # Logging metrics
            if log_experiment:
                logger.add_params({**data_out})
            ###############

            try:
                with open(cfg_path, "w") as write_file:
                    json.dump(data_out, write_file, indent = 2)
            except FileNotFoundError:
                os.makedirs(cfg_folder)
                with open(cfg_path, "w") as write_file:
                    json.dump(data_out, write_file, indent = 2)

            metrics_log = {"r_train":metrics["r_train"][0],
                           "r_test":metrics["r_test"][0],
                           "mse_train":metrics["mse_train"],
                           "mse_test":metrics["mse_test"],
                }
            results.append(metrics_log)
            
        n_worse_df = pd.DataFrame(n_worse, columns = ["indexes", "errors"])
        n_worse_df_mean = n_worse_df.groupby("indexes").mean()
        n_worse_df_mean = n_worse_df_mean.sort_values("errors", ascending = False)
        n_worse_df_mean.to_csv("n_worse.csv")
        if log_experiment:
            logger.log_table( "n_worse", n_worse_df_mean.to_numpy, ["indexes", "mean errors"] )
    
        df = pd.DataFrame(results)
        columns = list(df)
        for i in columns:
            stats = df[i].describe()
            for stat_name, value in stats.items():
                if log_experiment:
                    logger.log_metrics("{} {}".format(i, stat_name), value)
                print("{} {} {}".format(i, stat_name, value))
        
        final_dict = {}
        for dict_params in model_params_list:
        	for key, val in dict_params.items():
        		try:
        			final_dict[key].append(val)
        		except:
        			final_dict[key] = [val]
        if log_experiment:            
            logger.add_params(final_dict)
            
    else:
        #train all models
        models_all = ["lasso", "rf", "gbm", "ridge", "svm"]
        for model_all in models_all:
            ## Initialise a new experiment for each model (logging)
            if log_experiment:
                logger = experiment_logger(opt_args["log_exp"], project_name)
                logger.add_params({'model': model_name, 'base': base,
                                  'imputation': imput, 'coding': codif,
                                  'custom_grid': custom_grid, **cv_params})
            print('Training model: {}.'.format(model_all))
            model = import_module(model_all, package=None)
            cfg_folder = os.path.join(output_cfg, base, model_all, str(date.today()))
            train_pred_folder = os.path.join(output_train, base, model_all, str(date.today()))
            test_pred_folder = os.path.join(output_pred, base, model_all, str(date.today()))
            
            
            for i, split in enumerate(splits):
                print("Split {}/{}.".format(i+1, len(splits)))
                grid_params, model_params, train_preds, test_preds, metrics = model.train_test(split.X_train, split.Y_train, split.X_test, 
                                                                                               split.Y_test, cv_params)
                
                
                cfg_path = os.path.join(cfg_folder, cfg_name[:-5] + "_{}_".format(i) + '.json')
                train_pred_path = os.path.join(train_pred_folder, cfg_name[:-5] + "_{}_".format(i))
                test_pred_path = os.path.join(test_pred_folder, cfg_name[:-5] + "_{}_".format(i))

                try:
                    np.save(train_pred_path, train_preds)
                except FileNotFoundError:
                    os.makedirs(train_pred_folder)
                    np.save(train_pred_path, train_preds)

                try:
                    np.save(test_pred_path, test_preds)
                except FileNotFoundError:
                    os.makedirs(test_pred_folder)
                    np.save(test_pred_path, test_preds)
                
                data_out = {
                    "train_pred_path" : train_pred_path,
                    "test_pred_path" : test_pred_path,
                    "model" : model_all,
                    "split" : i,
                    "metrics" : metrics,
                    "grid_search_params" : grid_params,
                    "model_coefs" : model_params,
                    "input_cfg" : cfg,
                    "cv_params" : cv_params,
                    "custom_grid" : custom_grid
                }

                try:
                    with open(cfg_path, "w") as write_file:
                        json.dump(data_out, write_file, indent = 2)
                except FileNotFoundError:
                    os.makedirs(cfg_folder)
                    with open(cfg_path, "w") as write_file:
                        json.dump(data_out, write_file, indent = 2)
                results.append(metrics_log)
            df = pd.DataFrame(results)
            columns = list(df)
            for i in columns:
                stats = df[i].describe()
                for stat_name, value in stats.items():
                    if log_experiment:
                        logger.log_metrics("{} {}".format(i, stat_name), value)
                    print("{} {} {}".format(i, stat_name, value))
    if log_experiment:
        logger.stop() # ends logging                

if __name__ == "__main__":
    sys.path.append('train_test_models')
    argv = sys.argv[1:]
    argc = len(argv)
    if argc < 2:
        print('Use: train_test.py <config_name.json> <model_name> <-g> <-f> <-cg> <grid_search_params.json>')
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
            "split" : "all"}

        # command line arguments
        if "-d" in argv:
            opt_args["debug"] = True
        if "-g" in argv:
            opt_args["std_geno"] = False
        if "-f" in argv:
            opt_args["std_pheno"] = False
        if "-cg" in argv:
            grid_name = argv[argv.index("-cg") + 1]
            with open(os.path.join("..", "Input", "grid", grid_name), "r") as read_file:
                grid_params = json.load(read_file)
            opt_args["custom_grid"] = grid_params
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



        #start training
        train_test(sys.argv[1], sys.argv[2], opt_args)
