import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from load_grid import *

def train_test(X_train, Y_train, X_test, Y_test, cv_params, custom_grid = False):

	if custom_grid:
		random_grid = load_grid(custom_grid)
	else:
		alpha = np.linspace(50, 3000, 50)
		max_iter = np.linspace(1000, 5000, 4)

		# Create the random grid
		random_grid = { 'alpha' : alpha,
						'max_iter' : max_iter}

	#print random grid
	print_grid(random_grid)

	lasso = Lasso()
	# Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
	lasso_random = RandomizedSearchCV(estimator = lasso, param_distributions = random_grid, n_iter = cv_params["n_iter"], 
									  cv = cv_params["cv"], verbose = 10, n_jobs = cv_params["n_jobs"], pre_dispatch='2*n_jobs')
	lasso_random.fit(X_train, Y_train)
  
	best_grid_params = lasso_random.best_params_
	best_random = lasso_random.best_estimator_
	best_model_params = best_random.get_params()
	train_predictions = best_random.predict(X_train)
	test_predictions = best_random.predict(X_test)
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
	return best_grid_params, best_model_params, train_predictions, test_predictions, metrics
