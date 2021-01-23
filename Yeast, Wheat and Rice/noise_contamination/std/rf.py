import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import clear_output

import os.path

my_path = os.path.abspath(os.path.dirname(os.path.abspath("__file__")))

import sys
sys.path.append(my_path)

from add_noise import add_noise

path_pheno = os.path.join(my_path, "../feno.txt")
path_geno = os.path.join(my_path, "../geno.txt")

pheno = pd.read_csv(path_pheno)
geno = pd.read_csv(path_geno)

pheno_names = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide',  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
              'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]

pheno_12 = pheno[pheno_names]
params_path = os.path.join(my_path, "../params/rf/rf.pickle")

with open(params_path, 'rb') as f:
    results = pickle.load(f)
#%%
i = 0
M = 10
noise_ratios = np.array([5, 10, 20, 30, 40, 50, 75, 90])*0.01
for name, y in pheno_12.iteritems():

    print('Analyzing environment: ' + name + '.')
    params = results[i][1]
    n_estimators = params['n_estimators']
    min_samples_split = params['min_samples_split']
    min_samples_leaf = params['min_samples_leaf']
    max_features = params['max_features']
    max_depth = params['max_depth']
    bootstrap = params['bootstrap']
    
    missing_phenos = y[ y.isnull() ].index.values
    y = y.drop(missing_phenos, axis = 0)
    geno_c = geno.copy()
    geno_c = geno_c.drop(missing_phenos, axis = 0)
    r2s = []
    y = y.to_numpy()
    for (j, noise_ratio) in enumerate(noise_ratios):
        r2s_n = []
        for k in np.arange(M):
        
            y_n = add_noise(y, noise_ratio)
            
            X_train, X_test, y_train, y_test = train_test_split(geno_c, y_n, test_size=0.3)
            X_train = X_train.drop(columns = ["Unnamed: 0"]).values
            X_test = X_test.drop(columns = ["Unnamed: 0"]).values
            
            # ESTANADRIZANDO COMO SE DEBE
            y_train_std = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test_std = (y_test - np.mean(y_train)) / np.std(y_train)
        
            gbm = RandomForestRegressor(n_estimators = n_estimators, min_samples_split = min_samples_split,\
                                        min_samples_leaf = min_samples_leaf, max_features = max_features,\
                                        max_depth = max_depth, bootstrap = bootstrap)
            
            gbm.fit(X_train, y_train_std)
            
            gbm_predictions = gbm.predict(X_test)
            r2 = r2_score(y_test_std, gbm_predictions)

            r2s_n.append(r2)

        r2s.append(np.mean(np.array(r2s_n))) #promedio
        print('Noise ratio = {} complete.'.format(noise_ratio))
        
    clear_output()
    with open('results/rf/r2_rf_std_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(r2s, f)
        
    i+=1