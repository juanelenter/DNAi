import numpy as np
from bayes import BayesRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from load_grid import *

def train_test(X_train, Y_train, X_test, Y_test, cv_params, custom_grid = False):

    if custom_grid:
        random_grid = load_grid(custom_grid)
    else:
        h2=np.linspace(0.1, 0.7, 10) 
        prior = ["BayesC", "BayesA"]
        std_feno=[True, False]
        std_geno=[True, False]        
        # Create the random grid
        random_grid = { 'h2' : h2,
                        'prior' : prior,
                        'std_feno': std_feno,
                        'std_geno': std_geno}
    
    print_grid(random_grid)
    bayes = BayesRegressor(n_iter=3000)
    # Random search of parameters, using 3 fold cross validation, 
      # search across 100 different combinations, and use all available cores
    bayes_random = RandomizedSearchCV(estimator = bayes, param_distributions = random_grid, n_iter = cv_params["n_iter"], 
                                      cv = cv_params["cv"], verbose=10, random_state=42, n_jobs = cv_params["n_jobs"], 
                                      pre_dispatch='2*n_jobs')
    bayes_random.fit(X_train, Y_train)
      
    best_grid_params = bayes_random.best_params_
    best_random = bayes_random.best_estimator_
    best_model_params = best_random.get_params()
    train_predictions = best_random.predict(X_train)
    test_predictions = best_random.predict(X_test)
    #metrics
    r_train = pearsonr(Y_train, train_predictions)[0]
    r_test = pearsonr(Y_test, test_predictions)[0]
    mse_train = mse(Y_train, train_predictions)
    mse_test = mse(Y_test, test_predictions)
    metrics = {"r_train" : r_train,
               "r_test" : r_test,
               "mse_train" : mse_train,
               "mse_test" : mse_test}
    print(f"pearsonr train: {r_train}")
    print(f"pearsonr test: {r_test}")
    print(f"mse train: {mse_train}")
    print(f"mse test: {mse_test}")
    print(best_model_params)
    return best_grid_params, best_model_params, train_predictions, test_predictions, metrics, bayes
