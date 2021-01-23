Parameters: Loci (Causal or all), n_qtns (100 or 1000), heredability , genotype normalization (boolean), phenotype normalization (boolean)

1) Objective: Reproduce Gianolas causal loci results. 
Parameters- Causal, 1000, 0.3, False, True

Results with Ridge Regression with Leave One Out Cross Validation.
train mse=  0.99
test mse =  1.32

gianolas best mse = 3.

Conclusion: Our ridge regression on 1000 causal loci works better than gianolas ensemble or ML techniques.

