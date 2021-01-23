import numpy as np
from sklearn.linear_model import Ridge
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from load_grid import *

def train_test(X_train, Y_train, X_test, Y_test, cv_params, custom_grid = False):

    if custom_grid:
        random_grid = load_grid(custom_grid)
    else:
        alpha = np.linspace(30000, 20000, 500)
        #solver = ['svd', 'cholesky', 'lsqr']
        
        # Create the random grid
        random_grid = { 'alpha' : alpha}
                        #'solver' : solver}
    print_grid(random_grid)
    estimator = Ridge(alpha=90000)
    ridge_random = RFECV(estimator, step=500, cv=5, verbose=10)
    # Random search of parameters, using 3 fold cross validation, 
      # search across 100 different combinations, and use all available cores
    #ridge_random = RandomizedSearchCV(selector, param_distributions = random_grid, n_iter = cv_params["n_iter"], 
#                                      cv = cv_params["cv"], verbose=10, random_state=42, n_jobs = cv_params["n_jobs"], 
#                                      pre_dispatch='2*n_jobs')
    ridge_random.fit(X_train, Y_train)
      
    best_grid_params = {'alpha':30000}
    best_random = ridge_random.get_support()
    best_model_params = ridge_random.get_params()
    train_predictions = ridge_random.predict(X_train)
    test_predictions = ridge_random.predict(X_test)
    #metrics
    r_train = pearsonr(Y_train, train_predictions)
    r_test = pearsonr(Y_test, test_predictions)
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
    return best_grid_params, best_model_params, train_predictions, test_predictions, metrics, {}
