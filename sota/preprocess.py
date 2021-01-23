import pandas as pd 
import numpy as np 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


def load_n_pre(geno_path, pheno_path, env_indxs = "all", test_size = .1, norm_mode = "og",
               seed = 42):
    
    print("Loading data...")

    if geno_path.endswith(".npy"):
        geno_np = np.load(geno_path)
        feno_np = np.load(pheno_path)
        feno_np = feno_np.T
        env_keys = np.arange(min(feno_np.shape[0], feno_np.shape[1]))
    else:
        geno = pd.read_csv(geno_path)
        feno = pd.read_csv(pheno_path)

        geno_np = geno.to_numpy()[:, 1:].astype(np.int)
        feno_np = feno.to_numpy()[:, 1:].astype(np.float32)
        feno_np = feno_np.T

        env_keys = np.array(feno.keys()[1:])

        del(geno)
        del(feno)

    if env_indxs != "all":
    	env_keys = env_keys[env_indxs]
    else:
    	env_indxs = np.arange(len(env_keys))

    # initialize arrays
    X_train = np.zeros((0, geno_np.shape[1])).astype(np.int)
    X_test = X_train.copy()

    y_train = np.zeros(0)
    y_test = y_train.copy()

    env_train = y_train.copy()
    env_test = y_train.copy()

    print("Preprocessing...")
    for env_num, fenos_env in enumerate(feno_np[env_indxs]):
        print("{} of {} environments processed ({}) | Normalization type : {}.".format(env_num + 1, len(env_keys),
        																			   env_keys[env_num], norm_mode))
        not_nans = ~np.isnan(fenos_env)
        genos_env = geno_np[not_nans, :]
        fenos_env = fenos_env[not_nans]
        
        X_env_train, X_env_test, y_env_train, y_env_test = train_test_split(genos_env, fenos_env, 
                                                                            test_size = test_size, random_state = seed)
        
        X_train = np.append(X_train, X_env_train, axis = 0)
        X_test = np.append(X_test, X_env_test, axis = 0)
        
        # standarize phenotype (envirorment)
        if norm_mode == "MinMax_groups":
            min_ = np.min(y_env_train)
            max_ = np.max(y_env_train)
            y_env_train = (y_env_train - min_)/(max_ - min_) 
            y_env_test = (y_env_test - min_)/(max_ - min_) 

        elif norm_mode == "groups":
            mean = np.mean(y_env_train)
            std = np.std(y_env_train)
            y_env_train -= mean
            y_env_train /= std
            y_env_test -= mean
            y_env_test /= std

        # append phenotype
        y_train = np.append(y_train, y_env_train)
        y_test = np.append(y_test, y_env_test)
        
        # append envirorment values
        env_train = np.append(env_train, np.ones(y_env_train.shape)*env_num)
        env_test = np.append(env_test, np.ones(y_env_test.shape)*env_num)

    if norm_mode == "og":
        mean = np.mean(y_train)
        std = np.std(y_train)
        y_train -= mean
        y_train /= std
        y_test -= mean
        y_test /= std
    elif norm_mode == "MinMax":
	    min_ = np.min(y_train)
	    max_ = np.max(y_train)
	    y_train = (y_train - min_)/(max_ - min_) 
	    y_test = (y_test - min_)/(max_ - min_) 
    # one-hot encode envirorment values
    enc = OneHotEncoder()
    env_train_ohe = enc.fit_transform(env_train.reshape((-1, 1))).toarray().astype(np.int)
    env_test_ohe = enc.fit_transform(env_test.reshape((-1, 1))).toarray().astype(np.int)

    return X_train, X_test, y_train, y_test, env_train_ohe, env_test_ohe