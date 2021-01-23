import numpy as np 

def load_grid(grid_params):

	non_int_params = ["learning_rate", "min_impurity", "C", "epsilon", "h2", "subsample"]
	custom_grid = {} 
	for feature, value in grid_params.items():										
		if feature.endswith("linspace"):												 # LINSPACE PARAMETERS
			aux = list(np.linspace(start = value["start"],
							  stop = value["stop"],
							  num = value["num"]))											 
			if not any(feature.startswith(name) for name in non_int_params): 		     # INTEGER PARAMETERS																	 # INTEGER PARAMETERS
				aux = list(map(int, aux))												 # change list type to int
				if feature.startswith("max_depth"):								    	 # append none if feature = max_depth
					aux.append(None)
			custom_grid[feature[:-9]] = aux
				
		elif feature.endswith("list"): 													 # LIST PARAMETERS
			aux = []
			for element, include in value.items():
				if include:
					if feature.startswith("bootstrap"):  								 # bootstrap is loaded differently from other lists
						if element == "True":
							aux.append(True)
						else:
							aux.append(False)
					else:
						aux.append(element)
			custom_grid[feature[:-5]] = aux

	return custom_grid

def print_grid(grid):

	print("CV grid:")
	for key, value in grid.items():
		print("-> {} : {}".format(key, value))
		'''
		if type(value[0]) == str:
			print("-> {} : {}".format(key, value))
		else:
			print("-> {} : {:0.4f}".format(key, value))
		'''
