import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from load_grid import *


def train_test(X_train, Y_train, X_test, Y_test, cv_params, custom_grid = False):

	#load custom grid
	if custom_grid:
		random_grid = load_grid(custom_grid)
	else:
		# Learning rates
		learning_rates = [x for x in np.linspace(start = 0.0001, stop = 0.05, num = 20)]
		#Number of estimators
		n_estimators = [int(x) for x in np.linspace(start = 1200, stop = 2000, num = 15)]
		# Number of features to consider at every split
		max_features = ['sqrt', 'log2']
		# Maximum number of levels in tree
		max_depth = [int(x) for x in np.linspace(1, 3, num = 3)]
		#max_depth.append(None)
		# Minimum number of samples required to split a node
		min_samples_split =  [int(x) for x in np.linspace(40, 100 , num = 10)]
		# Minimum number of samples required at each leaf node
		min_samples_leaf = [int(x) for x in np.linspace(20, 500, num = 10)]
		# Method of selecting samples for training each tree
		losses = ['ls', 'lad']
		#impurity_decrease
		min_impurity_decrease = np.linspace(start = 0, stop = 0.5, num = 10)	
		# Create the random grid
		subsample = np.linspace(start = 0, stop = 1, num = 10)		

		random_grid = { 'learning_rate' : learning_rates,
		                'n_estimators': n_estimators,
		                'max_features': max_features,
		                'max_depth': max_depth,
		                'min_samples_split': min_samples_split,
		                'min_samples_leaf': min_samples_leaf,
		                'loss': losses,
				'min_impurity_decrease':min_impurity_decrease,
				'subsample': subsample}

	#print random grid
	print_grid(random_grid)
	gbm = GradientBoostingRegressor()
	# Random search of parameters, using 3 fold cross validation, 
  	# search across 100 different combinations, and use all available cores
	gbm_random = RandomizedSearchCV(estimator = gbm, param_distributions = random_grid, n_iter = cv_params["n_iter"], 
									cv = cv_params["cv"], verbose=10, random_state=42, n_jobs = cv_params["n_jobs"], 
									pre_dispatch='2*n_jobs')
	gbm_random.fit(X_train, Y_train)
  	
	best_grid_params = gbm_random.best_params_
	best_random = gbm_random.best_estimator_
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
	return best_grid_params, best_model_params, train_predictions, test_predictions, metrics, gbm
