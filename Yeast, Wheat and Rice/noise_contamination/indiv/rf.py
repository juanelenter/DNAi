import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import clear_output
from delete_samples import delete_samples

pheno = pd.read_csv(r"../feno.txt")
geno = pd.read_csv(r"../geno.txt")

pheno_names = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide',  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
              'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]

pheno_12 = pheno[pheno_names]

with open('../params/rf/rf.pickle', 'rb') as f:
    results = pickle.load(f)
#%%
i = 0
M = 10
del_ratios = np.array([10, 25, 50, 60, 70, 80, 85])*0.01
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
            
            geno_n, y_n = delete_samples(geno_c, y_c, del_ratio)
            X_train, X_test, y_train, y_test = train_test_split(geno_n, y_n, test_size=0.3)

            #X_train = X_train.drop(columns = ["Unnamed: 0"]).values
            #X_test = X_test.drop(columns = ["Unnamed: 0"]).values

            y_train_std = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test_std = (y_test - np.mean(y_train)) / np.std(y_train)
        
            rf = RandomForestRegressor(n_estimators = n_estimators, min_samples_split = min_samples_split,\
                                        min_samples_leaf = min_samples_leaf, max_features = max_features,\
                                        max_depth = max_depth, bootstrap = bootstrap)
            
            rf.fit(X_train, y_train_std)
            
            rf_predictions = rf.predict(X_test)
            r2 = r2_score(y_test_std, rf_predictions)

            r2s_n.append(r2)

        r2s.append(np.mean(np.array(r2s_n)))
        print('Deletion ratio = {} complete.'.format(del_ratio))
    
    clear_output()
    with open('results/rf/r2_rf_drop_samples_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(r2s, f)
        
    i+=1