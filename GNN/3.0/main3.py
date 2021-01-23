#Standard libraries
import comet_ml
import os
import numpy as np
import pickle
import datetime
from copy import deepcopy
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim
import argparse

#Personal libraries
import Utils.graphTools as graphTools
import Utils.dataTools
import Utils.graphML as gml
from Utils.miscTools import writeVarValues
from Utils.miscTools import saveSeed
import Modules.architectures as archit
import Modules.model as model
import Modules.training as training
import Modules.evaluation as evaluation
import Modules.loss as loss
from dnai_logging import experiment_logger
from sklearn.metrics import r2_score, mean_squared_error

today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
useGPU = True

################################
####### Argument Parsing #######
################################
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="linear fcn mfgf gnn gnn2")
parser.add_argument("-l", "--log_experiment", help="wether to log on comet", action="store_true")
parser.add_argument("-n", "--experiment_name",type=str, help="comet project name", default="gnn-wheat")
parser.add_argument("-e", "--epochs",type=int, help="number of epochs",default=10)
parser.add_argument("-lr", "--learning_rate",type=float, help="learning rate",default=0.000935)
parser.add_argument("-vi", "--validation_interval",type=int, help="how much batches to validate",default=50)
parser.add_argument("-bs", "--batch_size",type=int, help="GNN require batch sizes close to 1 due to memory constraints",default=2)
parser.add_argument("-lrd", "--learning_rate_decay",type=float, help="learning rate decay",default=1)
parser.add_argument("-es", "--early_stopping",type=int, help="early stopping tolerance in epochs",default=10)
parser.add_argument("-b1", "--beta_1",type=float, help="ADAM beta 1",default=0.9)
parser.add_argument("-b2", "--beta_2",type=float, help="ADAM beta 2",default=0.999)
parser.add_argument("-lm", "--log_model", help="wether to save final weights on comet", action="store_true")
args = parser.parse_args()

################################
####### Comet Logging ##########
################################

file_buffer = open('meta.pickle', 'rb')
meta = pickle.load(file_buffer)
file_buffer.close()

if args.log_experiment:
    logger = experiment_logger("comet", args.experiment_name)
    logger.add_params(vars(args))
    logger.add_params(meta)

################################
####### SYSTEM SETUP ###########
################################

if useGPU and torch.cuda.is_available():
    device = 'cuda:0'
    torch.cuda.empty_cache()
else:
    device = 'cpu'
# Notify:
print("Device selected: %s" % device)

################################
####### DATA GENERATION ########
################################

print("Loading Data...")
file_buffer = open('train.pickle', 'rb')
train = pickle.load(file_buffer)
file_buffer.close()

file_buffer = open('val.pickle', 'rb')
valid = pickle.load(file_buffer)
file_buffer.close()

file_buffer = open('test.pickle', 'rb')
test = pickle.load(file_buffer)
file_buffer.close()

xTrain = train["X"]
yTrain = train["y"]
xValid = valid["X"]
yValid = valid["y"]
xTest = test["X"]
yTest = test["y"]

print(f"Train size = {xTrain.shape}")
print(f"Val size = {xValid.shape}")
print(f"Test size = {xTest.shape}")

S = train["GSO"]
if device == 'cuda:0':
    S = torch.cuda.DoubleTensor(S)
elif device == 'cpu':
    S = torch.DoubleTensor(S)

nNodes = S.shape[0] # number of nodes
N = nNodes # jalo aca esto
nTrain = xTrain.shape[0]
nValid = xValid.shape[0]
nTest = xTest.shape[0]

xTrain = torch.Tensor(xTrain)
#print(f"xtrain[0][0] = {xTrain[0][0]}")
xTrain = xTrain.reshape([-1,1,N])
yTrain = torch.Tensor(yTrain)
yTrain = yTrain.reshape([-1,1,1])

xValid = torch.Tensor(xValid)
xValid = xValid.reshape([-1,1,N])
yValid = torch.Tensor(yValid)
yValid = yValid.reshape([-1,1,1])

xTest = torch.Tensor(xTest)
xTest = xTest.reshape([-1,1,N])
yTest = torch.Tensor(yTest)
yTest = yTest.reshape([-1,1,1])

data = Utils.dataTools.genomicData(nTrain, nValid, nTest, xTrain, yTrain, xValid, yValid, xTest, yTest, device)

################################
####### TRAINING SETUP #########
################################

lossFunction = nn.MSELoss

trainer = training.Trainer
evaluator = evaluation.evaluate

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'  #siempre es adam igual, lineal al pedo.
beta1 = args.beta_1 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = args.beta_2 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss

#\\\ Overall training options
nEpochs = args.epochs
batchSize = args.batch_size
learningRate = args.learning_rate
validationInterval = args.validation_interval
if args.learning_rate_decay<1:
    doLearningRateDecay = True # Learning rate decay
    learningRateDecayRate = args.learning_rate_decay # Rate
    learningRateDecayPeriod = 1 # How many epochs after which update the lr
else:
    doLearningRateDecay = False
    
    
################################
####### DEFINE MODEL ###########
################################

if args.model_name == 'linear':
# Linear parametrization    
    arch = torch.nn.Linear(N,1, bias = True)  

elif args.model_name == 'fcn':
# Fully connected neural network   
    arch = nn.Sequential(torch.nn.Linear(N,25, bias=True), nn.ReLU(), torch.nn.Linear(25,1, bias=True))

elif args.model_name == 'mfgf':
# Multi-feature graph filter
    arch = nn.Sequential(archit.LocalGNN([1, 64], [5], True, gml.NoActivation, [N], gml.NoPool, [1], [1], S), torch.nn.Linear(N,1, bias = True))

elif args.model_name == 'gnn':
# GNN, 1 layer
    arch = nn.Sequential(archit.LocalGNN([1, 16], [8], True, nn.ReLU, [N], gml.NoPool, [1], [1], S, batchnorm=True), torch.nn.Linear(N,1, bias=True))
        
# GNN, 2 layers
elif args.model_name == 'gnn2':
    arch = nn.Sequential(archit.LocalGNN([1, 32, 32], [15, 5], True, nn.ReLU, [N, N], gml.NoPool, [1, 1], [1], S, batchnorm=True), torch.nn.Linear(N,1, bias=True))
# GNN, 1 layer
    arch = nn.Sequential(archit.LocalGNN([1, 16], [8], True, nn.ReLU, [N], gml.NoPool, [1], [1], S, batchnorm=True), torch.nn.Linear(N,1, bias=True))

################################
#### INIT OPTIM AND MODEL ######
################################

#\\\ Optimizer
thisOptim = optim.Adam(arch.parameters(), lr = learningRate, betas = (beta1,beta2))

#\\\ Model
trainable_model = model.Model(arch.to(device), lossFunction(), thisOptim, trainer, evaluator, device, args.model_name, '')


#######################################
####### TRAINING DE VERDURA ###########
#######################################

print("Training model %s..." % trainable_model, end = ' ', flush = True)
    
#Train
trainingOptions = {}
trainingOptions['saveDir'] = ''
trainingOptions['printInterval'] = 1
trainingOptions['validationInterval'] = validationInterval
trainingOptions['doEarlyStopping'] = True
trainingOptions['earlyStoppingLag'] = args.early_stopping
trainingOptions['doLearningRateDecay'] = doLearningRateDecay
if doLearningRateDecay:
    trainingOptions['learningRateDecayRate'] = learningRateDecayRate
    trainingOptions['learningRateDecayPeriod'] = learningRateDecayPeriod
if args.log_experiment:
    trainingOptions['cometLogger'] = logger
    


thisTrainVars = trainable_model.train(data, nEpochs, batchSize, **trainingOptions)

# Save the variables
lossTrain = thisTrainVars['lossTrain']
costTrain = thisTrainVars['costTrain']
lossValid = thisTrainVars['lossValid']
costValid = thisTrainVars['costValid']

print("OK", flush = True)

##############
# Log model stuff
################
num_total_params = sum(p.numel() for p in arch.parameters())
print(f"Total parameters: {num_total_params}")
if args.log_experiment:
    logger.add_params({"num params": num_total_params})
#######################################
####### EVALUATION       ##############
#######################################

thisEvalVars = trainable_model.evaluate(data)
costBest = thisEvalVars['costBest']
costLast = thisEvalVars['costLast']
print("Last Evaluation \t%s: %6.2f%% [Best] %6.2f%% [Last]" % (trainable_model, costBest * 100, costLast * 100))

print("------------ Testing has begun. -------------")
yHatTest = torch.empty(yTest.shape)
lossBest = 0
costBest = 0
with torch.no_grad():    
    for i, x in enumerate(xTest):
        x = x.reshape(1,x.shape[0],x.shape[1]) #sino me chillaba .loss()
        yHatTest[i] = trainable_model.archit(x.to(device))

yTestn = yTest.squeeze().numpy()
yHatTestn = yHatTest.squeeze().numpy()

print(yHatTestn)
r_test = r2_score(yTestn, yHatTestn)
mse_test = mean_squared_error(yTestn, yHatTestn)

print(f"R2 test = {r_test}")
print(f"MSE test = {mse_test}")
if args.log_experiment:
    logger.log_metrics("MSE Test Best", mse_test, epoch = 0)
    logger.log_metrics("r test", r_test, epoch = 0)
    if args.log_model:
        logger.log_model_torch(args.model_name, trainable_model.archit)


