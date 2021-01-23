import numpy as np
try:
    from thundersvm import SVR
    print("Using GPU accelerated SVR!")
    thunder_available=True
except:
    from sklearn.svm import SVR
    print("Using Sklearn SVR, be patient")
    thunder_available=False
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse
from load_grid import *

def train_test(X_train, Y_train, X_test, Y_test, cv_params, custom_grid = False):
	
	if custom_grid:
		random_grid = load_grid(custom_grid)
	else:
		# Kernels
		kernel = ['rbf', 'linear', 'poly', 'sigmoid']
		# gammas
		gamma = ['scale', 'auto']
		# C
		C = np.linspace(start = 0.5, stop = 4, num = 30)
		# Epsilon
		epsilon = np.linspace(start = 0.05, stop = 0.2, num = 4)
		
		# Create the random grid
		random_grid = {'kernel' : kernel,
					   'gamma' : gamma,
					   'C' : C,
					   'epsilon': epsilon}
	
	print_grid(random_grid)
	if thunder_available:
		svr = SVR(max_mem_size=500)
	else:
		svr = SVR()
	# Random search of parameters, using 3 fold cross validation, 
	# search across 100 different combinations, and use all available cores
	svr_random = RandomizedSearchCV(estimator = svr, param_distributions = random_grid, n_iter = cv_params["n_iter"], 
	                                cv = cv_params["cv"], verbose = 15, random_state = 42, n_jobs = cv_params["n_jobs"], 
	                                pre_dispatch='n_jobs')
	# Fit the random search model
	svr_random.fit(X_train, Y_train)

	best_grid_params = svr_random.best_params_
	best_random = svr_random.best_estimator_
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
	return best_grid_params, best_model_params, train_predictions, test_predictions, metrics, svr
