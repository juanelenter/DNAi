import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import clear_output
from add_noise import add_noise

pheno = pd.read_csv(r"../feno.txt")
geno = pd.read_csv(r"../geno.txt")

pheno_names = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide',  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
              'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]

pheno_12 = pheno[pheno_names]

results = []
for filename in os.listdir('../params/gbm/'):
    if filename[-6:] == 'pickle':
        with open('../params/gbm/' + filename, 'rb') as f:
            results.append(pickle.load(f))
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
    loss = params['loss']
    learning_rate = params['learning_rate']
    
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
        
            rf = GradientBoostingRegressor(n_estimators = n_estimators, min_samples_split = min_samples_split,\
                                        min_samples_leaf = min_samples_leaf, max_features = max_features,\
                                        max_depth = max_depth, loss = loss, learning_rate = learning_rate,
                                        subsample = 1)
            
            rf.fit(X_train, y_train_std)
            
            rf_predictions = rf.predict(X_test)
            r2 = r2_score(y_test_std, rf_predictions)

            r2s_n.append(r2)

        r2s.append(np.mean(np.array(r2s_n))) #promedio
        print('Noise ratio = {} complete.'.format(noise_ratio))
        
    clear_output()
    with open('results/gbm/r2_gbm_std_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(r2s, f)
        
    i+=1 