import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from keras.optimizers import adam
from utils.logging import experiment_logger
from time import sleep

def test_results(model, X_test, y_test, max_env = 46):
    
    y_pred = model.predict({"geno" : X_test.reshape(X_test.shape[0], X_test.shape[1], 1), "env" : env_test})
    metrics = {"r2" : [],
               "mse" : []}
    for env in range(max_env):
        y_pred_e = y_pred[np.where(env_test[:, env] == 1)].reshape((-1,))
        y_test_e = y_test[np.where(env_test[:, env] == 1)].reshape((-1,))
        metrics["r2"].append(pearsonr(y_pred_e, y_test_e)[0]**2)
        metrics["mse"].append(np.mean((y_pred_e - y_test_e)**2))
        
    return metrics

def train_test_cnn(cfg)

	geno = pd.read_csv("geno.txt")
	feno = pd.read_csv("feno.txt")
	max_env = 46
	geno_np = geno.to_numpy()[:, 1:].astype(np.int)
	feno_np = feno.to_numpy()[:, 1:].astype(np.float32)

	tot_samples = np.count_nonzero(~np.isnan(feno_np))
	X = np.zeros((tot_samples, geno_np.shape[1])).astype(np.int)
	y = np.zeros(tot_samples).astype(np.float32)
	env = np.zeros(tot_samples).astype(np.int)
	indx = 0

	for geno, fenos in zip(geno_np, feno_np):
	    not_nans = ~np.isnan(fenos)
	    not_nans_Q = np.count_nonzero(not_nans)
	    X[indx : indx + not_nans_Q, :] = geno
	    y[indx : indx + not_nans_Q] = fenos[np.where(not_nans == True)]
	    env[indx : indx + not_nans_Q] = np.arange(max_env)[np.where(not_nans == True)] 
	    indx += not_nans_Q

	enc = OneHotEncoder()
	env_ohe = enc.fit_transform(env.reshape((-1, 1))).toarray().astype(np.int)

	X_train, X_test, y_train, y_test, env_train, env_test = train_test_split(X, y, env_ohe, test_size = .1)

	for env in range(max_env):
	    feno_train = y_train[np.where(env_train[:, env] == 1)]  
	    mean = np.mean(feno_train)
	    std = np.std(feno_train)
	    y_train[np.where(env_train[:, env] == 1)] -= mean
	    y_train[np.where(env_train[:, env] == 1)] /= std
	    y_test[np.where(env_test[:, env] == 1)] -= mean
	    y_test[np.where(env_test[:, env] == 1)] /= std

	if cfg["log_exp"]:
		logger = experiment_logger(cfg["log_exp"], cfg["log_proj"])
		logger.add_params({"lr" : cfg["lr"], "test_split" : cfg["test_split"], "val_split" : cfg["val_split"]})

    ########################################### CREATE MODEL #####################################################

	geno_input = keras.Input(shape = (X.shape[1], 1), name = "geno")       
	env_input = keras.Input(shape = (env_ohe.shape[1], ), name = "env")

	geno_features_1 = layers.Conv1D(kernel_size =  16, filters = 3 , padding = "valid",
	                                      strides =  8, activation = "relu", name = "conv_geno_1")(geno_input) 
	geno_features_2 = layers.Conv1D(kernel_size =  32, filters = 3 , padding = "valid",
	                                      strides =  16, activation = "relu", name = "conv_geno_2")(geno_features_1)

	geno_features_drop = layers.Dropout(rate = 0.33)(geno_features_2)
	geno_features_flat = layers.Flatten()(geno_features_drop)


	dense_1 = layers.Dense(units = 128, activation = "relu", name = "dense_1")(geno_features_flat)
	merge = layers.concatenate([dense_1, env_input], name = "merge")
	dense_2 = layers.Dense(units = 32, activation = "relu", name = "dense_2")(merge)
	dense_3 = layers.Dense(units = 16, activation = "relu", name = "dense_3")(dense_2)
	output = layers.Dense(1, name = "feno")(dense_3)

	model = keras.Model(inputs = [geno_input, env_input],
	                    outputs = output)

	model.compile(optimizer = keras.optimizers.Adam(learning_rate = 0.01), loss = "mse")

	############################################## FIT MODEL #########################################################

	hist = model.fit({"geno" : X_train.reshape(X_train.shape[0], X_train.shape[1], 1), "env" : env_train}, 
	           y_train, batch_size = 32, epochs = 10, validation_split = 0.1)

	metrics = test_results(model, X_test, y_test)

	############################################ PUSH METRICS #####################################################

	for met, met_list in metrics.items():
		for i, val in met_list:
			logger.log_metrics(met + "_" + str(i), val)
			sleep(.1)

	for met, met_list in hist.history.items():
		for i, val in met_list:
			logger.log_metrics(met, val, epoch = i)
			sleep(.1)


if __name__ == "__main__":
    argv = sys.argv[1:]
    argc = len(argv)

    # default parameters
    cfg = {"test_split" : 0.1,
     	   "val_split" : 0.1,
     	   "lr" : 0.001,
     	   "log_exp" : False,
     	   "proj_name" : "yeast_cnn"}

    # command line arguments
    if "-ts" in argv:
        opt_args["test_split"] = float(argv[argv.index("-ts") + 1])
    # log experiment in neptune/comet/both
    if "-vs" in argv:
        opt_args["val_split"] = float(argv[argv.index("-vs") + 1])
    # learning rate
    if "-lr" in argv:
        opt_args["lr"] = float(argv[argv.index("-lr") + 1])
    # set manually which project to log. 
    # Otherwise it will be set to database default project.
    if "-log_exp" in argv:
        opt_args["log_exp"] = True
    if "-log_proj" in argv:
        opt_args["proj_name"] = argv[argv.index("-log_proj") + 1]

	#start training
    train_test_cnn(cfg)
