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
from fit_test_model import *
from utils.split_class import *
from time import sleep

def train_test(cfg_name, model_name, opt_args):
    '''
    Available models:
    gbm.py
    lasso.py
    rf.py
    ridge.py
    svm.py 

    '''
    cv_params = {key : opt_args[key] for key in ["n_jobs", "n_iter", "cv"]} 
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
    n_feno = cfg["meta_data"]["n_feno/env"]    
    # train & test output paths
    #output_cfg = os.path.join(base_dir, 'output', 'config')
    #output_pred = os.path.join(base_dir,'output', 'pred')
    #output_train = os.path.join(base_dir,'output', 'train')
        
    #################################################################
    # Logging stuff
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
        
        # Initialise Logging
        if log_experiment:
            logger = experiment_logger(opt_args["log_exp"], project_name)
            logger.add_params({'model': model_name, 'base': base,
                              'imputation': imput, 'coding': codif,
                              'custom_grid': custom_grid, "n_feno/env" : n_feno,
			      **cv_params})

        from sklearn.base import clone # import here for comet compatibiliy
        model = import_module(model_name, package=None)
        #cfg_folder = os.path.join(output_cfg, base, model_name, str(date.today()))
        #train_pred_folder = os.path.join(output_train, base, model_name, str(date.today()))
        #test_pred_folder = os.path.join(output_pred, base, model_name, str(date.today()))
        results = []
        
        normality_pvalues = [] # array to store pvalues to check normality
        n = len(split.Y_test) # Len of Not Top 10 
        n_worse = np.empty((len(splits)*n, 2)) #column one: indices, column two: errors
        best_r_mean = -1
        best_split = -1
        global_metrics = {"r_train" : [],
                          "r_test" : [],
                          "mse_train" : [],
                          "mse_test" : []}
        grid_metrics_list = []
        grid_params_list = []
        p_values = [] #
        e_dict = {} #
        for i, split in zip(split_indxs, splits):

            print("Split {}/{}.".format(int(i)+1, len(splits)))                 
            grid_params, model_params, _, test_pred, grid_metrics, model_inst = model.train_test(split.X_train, split.Y_train,
                                                                                    split.X_test, split.Y_test,
                                                                                    cv_params, custom_grid)
            
            e_dict = sort_by_error( e_dict, i, split.Y_test, test_pred, split.idx_test )
            p_values.append( normality_test( e_dict[f"raw_error{i}"] ) )
            
            grid_params_list.append(grid_params)
            split_metrics = {met : [grid_metrics[met]] for met in grid_metrics}           
            splits_c = splits.copy()
            del(splits_c[i])
            for split_eval in splits_c:
                _, _, eval_metrics = fit_test(model_inst, grid_params, split_eval.X_train, split_eval.Y_train,
                                                            split_eval.X_test, split_eval.Y_test)
                for met, val in eval_metrics.items(): split_metrics[met].append(val)
            grid_metrics_list.append(split_metrics)
            for met in global_metrics: global_metrics[met].append(np.mean(split_metrics[met]))

            print("global met ", global_metrics["r_test"])
            print("global met -1", global_metrics["r_test"][-1])

            if global_metrics["r_test"][-1] > best_r_mean:
                best_split = i

            '''            
            n_worse[i*n : (i+1)*n, 0] = e_dict['sorted_idx'][ : n]
            n_worse[i*n : (i+1)*n, 1] = e_dict['sorted_error'][ : n]
            '''
            ###########
            # Logging metrics
            #if log_experiment:
                #error_PCA( e_dict, i, X, logger )
                #hist_error( e_dict , i , logger )
                #for metric_name, value in metrics.items():
                    #logger.log_metrics(metric_name, value, epoch=i)
                #logger.log_metrics("pvalue", pvalue, epoch=i)
                #logger.add_params({**grid_params, **model_params})
                #model_params_list.append(model_params)
             
            ###############
            # Logging indexs
            #np.savetxt( "idx_test.csv", split.idx_test)
            #np.savetxt( "idx_train.csv", split.idx_train)
            #if log_experiment:
                #logger.log_table( "idx_test_" + str(i), split.idx_test )
                #logger.log_table( "idx_train_" + str(i), split.idx_train )

            '''
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
            '''
            data_out = {
                "model" : model_name,
                "split" : str(i),
                #"metrics" : metrics,
                #"grid_search_params" : grid_params,
                #"model_coefs" : model_params,
                "input_cfg" : cfg,
                "cv_params" : cv_params,
                "custom_grid" : custom_grid}
            ###########
            # Logging metrics
            #if log_experiment:
                #logger.add_params({**data_out})
            ###############
            '''
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
                           "mse_test":metrics["mse_test"]}
            results.append(metrics_log)
            '''
        
        print(e_dict)
        print(p_values)
        print("best_split = ", best_split)
        print("grid_metrics_list = ", grid_metrics_list)
        #best_metrics = grid_metrics_list[best_split]
        
        best_metrics = grid_metrics_list[best_split]
        if log_experiment: 
            for met, val in best_metrics.items():
                for i in range(len(val)):
                    logger.log_metrics(met, val[i], epoch = i)
                    logger.log_metrics(met + "_mean", np.round(global_metrics[met][i], 3), epoch = i)
                    sleep(5)
            print("----------  All metrics were logged to comet. ----------")

        final_dict = {}
        for dict_params in grid_params_list:
            for key, val in dict_params.items():
                try:
                    final_dict[key].append(val)
                except:
                    final_dict[key] = [val]
        if log_experiment:
            print("FINAL DICT:")
            print(final_dict)
            logger.add_params(final_dict)
            logger.add_params({"best_split" : int(best_split)})
            logger.add_params({"p_values": p_values})
            #logger.stop() # ends logging    
            print("----------  All params were logged to comet. ----------")

            
        e_dict_df = pd.DataFrame.from_dict(e_dict)
        e_dict_df.to_csv("e_dict_df.csv")
        if log_experiment:
            logger.log_table( "e_dict_df", e_dict_df.to_numpy(), e_dict_df.columns )
            print("----------  Error csv was logged to comet. ----------")
            logger.stop()

        #n_worse_df = pd.DataFrame(n_worse, columns = ["indexes", "errors"])
        #n_worse_df_mean = n_worse_df.groupby("indexes").mean()
        #n_worse_df_mean = n_worse_df_mean.sort_values("errors", ascending = False)
        #n_worse_df_mean.to_csv("n_worse.csv")
        #if log_experiment:
            #logger.log_table( "n_worse", n_worse_df_mean.to_numpy, ["indexes", "mean errors"] )
    
        #df = pd.DataFrame(results)
        #columns = list(df)
        #for i in columns:
            #stats = df[i].describe()
            #for stat_name, value in stats.items():
                #if log_experiment:
                    #logger.log_metrics("{} {}".format(i, stat_name), value)
                #print("{} {} {}".format(i, stat_name, value))

        #if log_experiment:
           # for i, r_mean in enumerate(r_means_new):
                #logger.log_metrics("r_means_new", r_mean, epoch = i)
            #logger.add_params(best_grid_params)
        
          

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
                    "log_exp": "comet",
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
