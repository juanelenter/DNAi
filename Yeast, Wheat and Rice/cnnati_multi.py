# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:11:03 2020

@author: DNAti
"""
#%%

import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Dense, Flatten, Input
from tensorflow.keras.models import Model   

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os.path

    
#%%

path_pheno = './feno.txt'
path_geno = './geno.txt'

pheno = pd.read_csv(path_pheno)
geno = pd.read_csv(path_geno)
traits = ["Cadmium_Chloride", 'Congo_red', 'Cycloheximide', 'Diamide',  'Ethanol', 'Hydroquinone', 'Lithium_Chloride',
          'Maltose', 'Neomycin', 'Tunicamycin', "Galactose", "YNB:ph3"]
group_a = ['Lactate', 'Lactose', 'Xylose']


#################               Data preprocessing     ########################
n_feno = 1 #fenotipo sobre el cual entrenar
Y = pheno[group_a]
X = geno

missing_phenos = [ Y[group_a[i]][Y[group_a[i]].isnull() ].index.values for i in range(len(group_a))]
mis = np.hstack(missing_phenos)
X = geno.drop(columns = ["Unnamed: 0"])
X = X.drop( mis, axis = 0).values
Y = Y.drop(mis, axis = 0).values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.15)
Y_train_std = (Y_train - np.mean(Y_train)) / np.std(Y_train)
Y_test_std = (Y_test - np.mean(Y_train)) / np.std(Y_train)
#%%%%%

xt = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)


#%%
#################          	    Model                  ########################
	
input_snps = Input(shape = (X_train.shape[1], 1) )
	
output_1 = Conv1D( kernel_size =  5, filters = 3 , padding = "valid",
			    strides =  3, activation = "relu", name = "conv_snps" )( input_snps )

output_2 = Conv1D( kernel_size =  5, filters = 3 , padding = "valid",
			    strides =  3, activation = "relu", name = "conv_snps2" )( output_1 )

output_3 = Dropout( rate = 0.5 )( output_2 )

output_4 = Flatten()( output_3 )

output_5_1 = Dense( units = 4, activation = "relu", kernel_regularizer = regularizers.l2(0.001), name = "fully_connected11" )( output_4 )
fenotipos1 = Dense( units = 1, activation = "linear", kernel_regularizer = regularizers.l2(0.01), name = "fully_connected21" )( output_5_1 )

output_5_2 = Dense( units = 4, activation = "relu", kernel_regularizer = regularizers.l2(0.001), name = "fully_connected12" )( output_4 )
fenotipos2 = Dense( units = 1, activation = "linear", kernel_regularizer = regularizers.l2(0.01), name = "fully_connected22" )( output_5_2 )

output_5_3 = Dense( units = 4, activation = "relu", kernel_regularizer = regularizers.l2(0.001), name = "fully_connected13" )( output_4 )
fenotipos3 = Dense( units = 1, activation = "linear", kernel_regularizer = regularizers.l2(0.01), name = "fully_connected23" )( output_5_3 )
	
cnnati = Model( inputs = input_snps, outputs = [fenotipos1, fenotipos2, fenotipos3] )
cnnati.compile( optimizer = "adam", loss = "mse" )

hist = cnnati.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1),[Y_train_std[:,0],Y_train_std[:,1], Y_train_std[:,2] ], 
					validation_split = 0.12, batch_size= 32, shuffle = True, epochs = 30)

cnnati.summary()
#%%
#################          	    Test                  ########################
	
y_pred_test = np.hstack(cnnati.predict(X_test.reshape(X_test.shape[0],  X_test.shape[1] , 1), verbose = True))#.reshape(len(Y_test)) 

y_pred_train = np.hstack(cnnati.predict(X_train.reshape(X_train.shape[0],  X_train.shape[1], 1), verbose = True))#.reshape(len(Y_train))
#%%	
print(f"-----------         Model Results        ----------- ") 
print(f" MSE_out = {mean_squared_error(Y_test_std, y_pred_test) } ")
print(f" MSE_in = {mean_squared_error(Y_train_std, y_pred_train) } ")
print(f" R2 test = {r2_score(Y_test_std, y_pred_test)}" )
print(f" R2 train = {r2_score(Y_train_std, y_pred_train)}" )
	
plt.figure(1)
plt.title("CNN Ein y Eval")
plt.ylabel("MSE")
plt.xlabel("epoch")
plt.scatter(range(len(hist.history["loss"])), hist.history["loss"], label = "loss", c= "r", marker = "+" )
plt.scatter(range(len(hist.history["val_loss"])), hist.history["val_loss"], label = "val_loss", c = "b", marker = "+")
plt.legend()


#%%	
print(f"-----------         Model Results        ----------- ") 
print(group_a[0])
print(f" MSE_out = {mean_squared_error(Y_test_std[:,0], y_pred_test[:,0]) } ")
print(f" MSE_in = {mean_squared_error(Y_train_std[:,0], y_pred_train[:,0]) } ")
print(f" R2 test = {r2_score(Y_test_std[:,0], y_pred_test[:,0])}" )
print(f" R2 train = {r2_score(Y_train_std[0], y_pred_train[0])}" )
#%%	print(f"-----------         Model Results        ----------- ") 
print(group_a[1])
print(f" MSE_out = {mean_squared_error(Y_test_std[:,1], y_pred_test[:,1]) } ")
print(f" MSE_in = {mean_squared_error(Y_train_std[:,1], y_pred_train[:,1]) } ")
print(f" R2 test = {r2_score(Y_test_std[:,1], y_pred_test[:,1])}" )
print(f" R2 train = {r2_score(Y_train_std[:,1], y_pred_train[:,1])}" )
#%%	print(f"-----------         Model Results        ----------- ") 
print(group_a[2])
print(f" MSE_out = {mean_squared_error(Y_test_std[:,2], y_pred_test[:,2]) } ")
print(f" MSE_in = {mean_squared_error(Y_train_std[:,2], y_pred_train[:,2]) } ")
print(f" R2 test = {r2_score(Y_test_std[:,2], y_pred_test[:,2])}" )
print(f" R2 train = {r2_score(Y_train_std[:,2], y_pred_train[:,2])}" )