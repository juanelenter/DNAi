import pandas as pd
import numpy as np
import os
import pickle

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import clear_output
from delete_markers import delete_markers

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
del_ratios = np.arange(5, 99, step = 1)*0.01
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
    y_c = y.copy()
    y_c = y_c.drop(missing_phenos, axis = 0)
    y_c = y_c.drop(columns = ["Unnamed: 0"]).values
    geno_c = geno.copy()
    geno_c = geno_c.drop(missing_phenos, axis = 0)
    geno_c = geno_c.drop(columns = ["Unnamed: 0"]).values
    r2s = []
    for (j, del_ratio) in enumerate(del_ratios):
        r2s_n = []
        for k in np.arange(M):
            
            geno_n = delete_markers(geno_c, del_ratio)
            X_train, X_test, y_train, y_test = train_test_split(geno_n, y_c, test_size=0.3)
            np.savetxt('X_train.txt', X_train)
            np.savetxt('X_test.txt', X_test)
            np.savetxt('Y_train.txt', y_train)
            np.savetxt('Y_test.txt', y_test)
            #X_train = X_train.drop(columns = ["Unnamed: 0"]).values
            #X_test = X_test.drop(columns = ["Unnamed: 0"]).values
            print('done saving')
            y_train_std = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test_std = (y_test - np.mean(y_train)) / np.std(y_train)
        
            gbm = GradientBoostingRegressor(n_estimators = n_estimators, min_samples_split = min_samples_split,\
                                        min_samples_leaf = min_samples_leaf, max_features = max_features,\
                                        max_depth = max_depth, loss = loss, learning_rate = learning_rate,
                                        subsample = 1)
            
            gbm.fit(X_train, y_train_std)

            gbm_predictions = gbm.predict(X_test)
            r2 = r2_score(y_test_std, gbm_predictions)

            r2s_n.append(r2)

        r2s.append(np.mean(np.array(r2s_n)))
        print('Deletion ratio = {} complete.'.format(del_ratio))
    
    clear_output()
    with open('results/gbm/r2_gbm_drop_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(r2s, f)
        
    i+=1 