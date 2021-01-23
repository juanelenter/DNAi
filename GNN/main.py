# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 10:51:49 2020

@author: Luana Ruiz
"""

"""
LAB 2: SOURCE LOCALIZATION
"""

#\\\ Standard libraries:
import comet_ml
import numpy as np
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import copy
import pickle
#\\\ Own libraries:
import data as data
import myModules as myModules
from dnai_logging import experiment_logger
from scipy.stats import pearsonr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="linear fcn mfgf gnn gnn2")
parser.add_argument("-l", "--log_experiment", help="wether to log on comet",
                    action="store_true")
parser.add_argument("-e", "--epochs",type=int, help="number of epochs",default=30)
parser.add_argument("-lr", "--learning_rate",type=float, help="learning rate",default=0.05)
parser.add_argument("-vi", "--validation_interval",type=int, help="how much batches to validate",default=10)
parser.add_argument("-bs", "--batch_size",type=int, help="GNN require batch sizes close to 1 due to memory constraints",default=10)
args = parser.parse_args()

#################
# COMET logging
#################
if args.log_experiment:
    logger = experiment_logger("comet", "gnn-yeast")
    logger.add_params(vars(args))

################################
####### DATA GENERATION ########
################################

file_buffer = open('train.pickle', 'rb')
train = pickle.load(file_buffer)
file_buffer.close()

file_buffer = open('test.pickle', 'rb')
test = pickle.load(file_buffer)
file_buffer.close()

xTrain = train["X"]
yTrain = train["y"]
xTest = test["X"]
yTest = test["y"]

S = train["GSO"]

N = S.shape[0] # number of nodes
nTrain = xTrain.shape[0]
nTest = xTest.shape[0]

xTrain = torch.LongTensor(xTrain)
xTrain = xTrain.reshape([-1,1,N])
yTrain = torch.LongTensor(yTrain)
yTrain = yTrain.reshape([-1,1,1])

xTest = torch.LongTensor(xTest)
xTest = xTest.reshape([-1,1,N])
yTest = torch.LongTensor(yTest)
yTest = yTest.reshape([-1,1,1])

xTrain = xTrain.long()
xTest = xTest.long()



################################
######## LOSS FUNCTION #########
################################

loss = nn.MSELoss()

################################
######## ARCHITECTURES #########
################################

if args.model_name == 'linear':
# Linear parametrization    
    model = torch.nn.Linear(N,1, bias = True)
elif args.model_name == 'fcn':
# Fully connected neural network   
    model = nn.Sequential(torch.nn.Linear(N,25, bias=True), nn.ReLU(), torch.nn.Linear(25,1, bias=True), nn.ReLU())
        

elif args.model_name == 'mfgf':
# Multi-feature graph filter
    model = nn.Sequential(myModules.GraphFilter(S,8,1,32,True), myModules.GraphFilter(S,1,32,1,True), torch.nn.Linear(N,1, bias = True))

elif args.model_name == 'gnn':
# GNN, 1 layer
    model = nn.Sequential( myModules.GNN(S,2,[8,1],[1,32,1],nn.ReLU(),True), torch.nn.Linear(N,1, bias=True))

# GNN, 2 layers
elif args.model_name == 'gnn2':
    model = nn.Sequential(myModules.GNN(S,3,[5,5,1],[1,16,4,1],nn.ReLU(),True), torch.nn.Linear(N,1, bias=True))



################################
########### TRAINING ###########
################################

validationInterval = args.validation_interval

nEpochs = args.epochs
batchSize = args.batch_size
learningRate = args.learning_rate

nValid = int(np.floor(0.01*nTrain))
xValid = xTrain[0:nValid,:,:]
yValid = yTrain[0:nValid,:,:]
xTrain = xTrain[nValid:,:,:]
yTrain = yTrain[nValid:,:,:]
nTrain = xTrain.shape[0]

# Declaring the optimizers for each architectures
optimizer = optim.Adam(model.parameters(), lr=learningRate)

if nTrain < batchSize:
    nBatches = 1
    batchSize = [nTrain]
elif nTrain % batchSize != 0:
    nBatches = np.ceil(nTrain/batchSize).astype(np.int64)
    batchSize = [batchSize] * nBatches
    while sum(batchSize) != nTrain:
        batchSize[-1] -= 1
else:
    nBatches = np.int(nTrain/batchSize)
    batchSize = [batchSize] * nBatches
batchIndex = np.cumsum(batchSize).tolist()
batchIndex = [0] + batchIndex

epoch = 0 # epoch counter

# Store the training...
lossTrain = []
costTrain = []
lossValid = []
costValid = []





epoch = 0 # epoch counter
while epoch < nEpochs:
    randomPermutation = np.random.permutation(nTrain)
    idxEpoch = [int(i) for i in randomPermutation]
    print("")
    print("Epoch %d" % (epoch+1))

    batch = 0 

    while batch < nBatches:
        # Determine batch indices
        thisBatchIndices = idxEpoch[batchIndex[batch] : batchIndex[batch+1]]

        # Get the samples in this batch
        xTrainBatch = xTrain[thisBatchIndices,:,:]
        yTrainBatch = yTrain[thisBatchIndices,:,:]

        if (epoch * nBatches + batch) % validationInterval == 0:
            print("")
            print("    (E: %2d, B: %3d)" % (epoch+1, batch+1),end = ' ')
            print("")


        #print(f"Arch: {key}")
        # Reset gradients
        model.zero_grad()

        # Obtain the output of the architectures
        yHatTrainBatch = model(xTrainBatch.double())

        # Compute loss
        lossValueTrain = loss(yHatTrainBatch.squeeze().double(), yTrainBatch.squeeze().double()).double()

        # Compute gradients
        lossValueTrain.backward()

        # Optimize
        optimizer.step()

        costValueTrain = lossValueTrain.item()

        # Print:
        if (epoch * nBatches + batch) % validationInterval == 0:
            with torch.no_grad():
                # Obtain the output of the GNN
                yHatValid = model(xValid.double())

            # Compute loss
            lossValueValid = loss(yHatValid.squeeze().double(), yValid.squeeze().double()).double()

            # Compute accuracy:
            costValueValid = lossValueValid.item()
        
            lossValid += [lossValueValid.item()]
            costValid += [costValueValid]
            
            print("\t" + args.model_name + ": %6.4f [T]" % (
                    costValueTrain) + " %6.4f [V]" % (
                    costValueValid))
            if args.log_experiment:
                logger.log_metrics("loss Train", costValueTrain, epoch = epoch)
                logger.log_metrics("loss Validation", costValueValid, epoch = epoch)
            # Saving the best model so far
            if len(costValid) > 1:
                if costValueValid <= min(costValid):
                    bestModel=  copy.deepcopy(model)
            else:
                bestModel =  copy.deepcopy(model)

        batch+=1

    epoch+=1
###############
# Evaluation
###############
with torch.no_grad():
    yHatTest = model(xTest.double())
lossTestLast = loss(yHatTest.squeeze().double(), yTest.squeeze().double()).double()
costTestLast = lossTestLast.item()
with torch.no_grad():
    yHatTest = bestModel(xTest.double())
lossTestBest = loss(yHatTest.squeeze().double(), yTest.squeeze().double()).double()
costTestBest = lossTestBest.item()
with torch.no_grad():
    yHatTrain = bestModel(xTrain.double())

print(" " + args.model_name + "MSE : %6.4f [Best]" % (costTestBest) + " %6.4f [Last]" % (costTestLast))

r_train = pearsonr(yTrain.squeeze(), yHatTrain.squeeze())[0]
r_test = pearsonr(yTest.squeeze(), yHatTest.squeeze())[0]

print("r: %6.4f [Train]" % (r_train) + " %6.4f [Test]" % (r_test))
if args.log_experiment:
    logger.log_metrics("MSE Test Best", costTestBest, epoch = epoch)
    logger.log_metrics("MSE Test Last", costTestLast, epoch = epoch)
    logger.log_metrics("r train", r_train, epoch = epoch)
    logger.log_metrics("r test", r_test, epoch = epoch)
    
    
