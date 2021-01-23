#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:37:57 2020

@author: ignaciohounie
"""
import genotype_sym as geno
import phenotype_sym as sym
import phenotype_pred as pred
import numpy as np
#%% Load file
array = sym.txt_to_numpy("snps_idless.txt")
#%% Verify no missing data
#np.sum(np.all(array < 3, axis=0))
#array = array[:,np.all(array < 3, axis=0)]


#%% Simulate phenotype
h2 = 0.3
a = sym.generate_alpha(array.shape[1], 1000)
phenotype = sym.simulate_additive(array, a, h = h2)

phenotype = phenotype - np.mean(phenotype)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
#phenotype = scaler.fit_transform(phenotype)

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(array, phenotype, test_size=0.33, random_state=42)
clf_ridge = pred.snp_blup_ridge(X_train, y_train, h2)#0.5 es el parametro de heredabilidad, ni idea lo inventamos
X_test = X_test-np.mean(X_train, axis=0)
y_ridge = clf_ridge.predict(X_test)
from sklearn.metrics import mean_squared_error
corr_ridge = np.corrcoef(y_test, y_ridge)
print(corr_ridge)
mse_ridge = mean_squared_error(y_test, y_ridge)
print(mse_ridge)

#%%
from matplotlib import pyplot as plt
plt.figure()
plt.plot(y_test)
plt.plot(y_ridge)
plt.show()
#plt.plot(phenotype)







#%%Symulate new genotypes
X_sim = geno.shuffle_gen(array)
X_sim_2 = geno.shuffle_gen(X_sim)
#%% Simulate new phenotypes
a = sym.generate_alpha(X_sim_2.shape[1], 1000)
y = sym.simulate_additive(X_sim_2, a, h=h2)
#%% PREDICT
X_train, X_test, y_train, y_test = train_test_split(X_sim_2, y, test_size=0.33, random_state=42)
clf_ridge = pred.snp_blup_ridge(X_train, y_train, h2)#0.5 es el parametro de heredabilidad, ni idea lo inventamos
X_test = X_test-np.mean(X_train, axis=0)
y_ridge = clf_ridge.predict(X_test)
from sklearn.metrics import mean_squared_error

mse_ridge = mean_squared_error(y_test, y_ridge)
print(mse_ridge)
plt.figure()
plt.plot(y_test)
plt.plot(y_ridge)
plt.show()