import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import Lasso

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from IPython.display import clear_output
from add_noise import add_noise

pheno = pd.read_csv(r"../feno.txt")
geno = pd.read_csv(r"../geno.txt")

pheno_names = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide',  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
              'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]

pheno_12 = pheno[pheno_names]

alphas = [200, 1500, 1500, 1750, 2375, 2750, 1000, 1000, 1000, 1250, 2750, 2750]
#%%
i = 0
M = 10
noise_ratios = np.array([5, 10, 20, 30, 40, 50, 75, 90])*0.01
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
        
            las = Lasso(alphas[i])
            
            las.fit(X_train, y_train_std)
            
            las_predictions = las.predict(X_test)
            r2 = r2_score(y_test_std, las_predictions)

            r2s_n.append(r2)

        r2s.append(np.mean(np.array(r2s_n))) #promedio
        print('Noise ratio = {} complete.'.format(noise_ratio))
        
    clear_output()
    with open('results/lasso/r2_lasso_std_{}.pickle'.format(name), 'wb') as f:
        pickle.dump(r2s, f)
        
    i+=1 