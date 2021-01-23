import pandas as pd
import numpy as np
import pickle

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import clear_output

import os.path

my_path = os.path.abspath(os.path.dirname(os.path.abspath("__file__")))

import sys
sys.path.append(my_path)

from delete_samples import delete_samples

path_pheno = os.path.join(my_path, "../feno.txt")
path_geno = os.path.join(my_path, "../geno.txt")

pheno = pd.read_csv(path_pheno)
geno = pd.read_csv(path_geno)

pheno_names = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide',  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
              'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]

pheno_12 = pheno[pheno_names]

# PARAMS
kernel = 'rbf'
C = 4
gamma = 'auto'
#%%
i = 0
M = 10
del_ratios = np.array([10, 25, 50, 60, 70, 80, 85])*0.01
for name, y in pheno_12.iteritems():

    print('Analyzing environment: ' + name + '.')
    
    missing_phenos = y[ y.isnull() ].index.values
    y_c = y.copy()
    y_c = y_c.drop(missing_phenos, axis = 0)
    y_c = y_c.drop(columns = ["Unnamed: 0"]).values
    geno_c = geno.copy()
    geno_c = geno_c.drop(missing_phenos, axis = 0)
    geno_c = geno_c.drop(columns = ["Unnamed: 0"]).values
    r2s = []
    for del_ratio in del_ratios:
        r2s_n = []
        for k in np.arange(M):
            
            geno_n, y_n = delete_samples(geno_c, y_c, del_ratio)
            X_train, X_test, y_train, y_test = train_test_split(geno_n, y_n, test_size=0.3)

            y_train_std = (y_train - np.mean(y_train)) / np.std(y_train)
            y_test_std = (y_test - np.mean(y_train)) / np.std(y_train)
        
            svr = SVR(kernel = kernel, gamma = gamma, C = C)
            
            svr.fit(X_train, y_train_std)
            
            svr_predictions = svr.predict(X_test)
            r2 = r2_score(y_test_std, svr_predictions)

            r2s_n.append(r2)

        r2s.append(np.mean(np.array(r2s_n)))
        print('Deletion ratio = {} complete.'.format(del_ratio))
    
    clear_output()
    with open('results/svr/r2_svr_drop_samples_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(r2s, f)
        
    i+=1