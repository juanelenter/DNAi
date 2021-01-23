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
import optuna
today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
useGPU = True
import neptunecontrib.monitoring.optuna as opt_utils
from optuna.samplers import TPESampler

################################
####### Argument Parsing #######
################################
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="linear fcn mfgf gnn gnn2")
parser.add_argument("-l", "--log_experiment", help="wether to log on comet", action="store_true")
parser.add_argument("-e", "--epochs",type=int, help="number of epochs",default=10)
parser.add_argument("-lr", "--learning_rate",type=float, help="learning rate",default=0.000935)
parser.add_argument("-vi", "--validation_interval",type=int, help="how much batches to validate",default=50)
parser.add_argument("-bs", "--batch_size",type=int, help="GNN require batch sizes close to 1 due to memory constraints",default=4)
args = parser.parse_args()

################################
####### Comet Logging ##########
################################

file_buffer = open('meta.pickle', 'rb')
meta = pickle.load(file_buffer)
file_buffer.close()

if args.log_experiment:
    logger = experiment_logger("neptune", "gnn-optuna")
    logger.add_params(vars(args))
    logger.add_params(meta)
    neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

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
print(f"xtrain[0][0] = {xTrain[0][0]}")
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
#beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
#beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss

#\\\ Overall training options
nEpochs = args.epochs
batchSize = args.batch_size

validationInterval = args.validation_interval



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
    arch = nn.Sequential( archit.LocalGNN([1, 64], [5], True, nn.ReLU, [N], gml.NoPool, [1], [1], S), torch.nn.Linear(N,1, bias=True))

# GNN, 2 layers
elif args.model_name == 'gnn2':
    arch = nn.Sequential(archit.LocalGNN([1, 64, 32], [5, 5], True, nn.ReLU, [N, N], gml.NoPool, [1, 1], [1], S), torch.nn.Linear(N,1, bias=True))


#######################################
####### TRAINING DE VERDURA ###########
#######################################
    
#Train
trainingOptions = {}
trainingOptions['saveDir'] = ''
trainingOptions['printInterval'] = 1
trainingOptions['validationInterval'] = validationInterval
trainingOptions['doEarlyStopping'] = True
trainingOptions['doLearningRateDecay'] = True
trainingOptions['earlyStoppingLag'] = int(nEpochs//2)
def objective(trial):
################################
#### INIT OPTIM AND MODEL ######
################################
    learningRate = trial.suggest_loguniform('learning_rate', 1.5e-3, 1e-2)
    #doLearningRateDecay = True # Learning rate decay
    #learningRateDecayRate = 0.9 # Rate
    #learningRateDecayPeriod = 1 # How many epochs after which update the lr
    trainingOptions['learningRateDecayRate'] = trial.suggest_uniform('learningRateDecayRate', 0.8, 0.999)
    trainingOptions['learningRateDecayPeriod'] = 1#trial.suggest_int('learningRateDecayPeriod', 1, 10)
    beta1 = trial.suggest_uniform('beta1', 0.89, 0.91)
    beta2 = trial.suggest_uniform('beta2', 0.99, 0.9999)
    trainingOptions['trial'] = trial
    #\\\ Optimizer
    thisOptim = optim.Adam(arch.parameters(), lr = learningRate, betas = (beta1,beta2))

    #\\\ Model
    trainable_model = model.Model(arch.to(device), lossFunction(), thisOptim, trainer, evaluator, device, args.model_name, '')
    print("Training model %s..." % trainable_model, end = ' ', flush = True)

    
    thisTrainVars = trainable_model.train(data, nEpochs, batchSize, **trainingOptions)

    # Save the variables
    lossTrain = thisTrainVars['lossTrain']
    costTrain = thisTrainVars['costTrain']
    lossValid = thisTrainVars['lossValid']
    costValid = thisTrainVars['costValid']

    print("OK", flush = True)

    #######################################
    ####### EVALUATION       ##############
    #######################################

    thisEvalVars = trainable_model.evaluate(data)
    costBest = thisEvalVars['costBest']
    costLast = thisEvalVars['costLast']
    print("Last Evaluation \t%s: %6.2f%% [Best] %6.2f%% [Last]" % (trainable_model, costBest * 100, costLast * 100))
    return costBest

study = optuna.create_study(direction="minimize",
    pruner=optuna.pruners.HyperbandPruner(min_resource=3, max_resource=int(nEpochs*nTrain/batchSize),reduction_factor=3),
    sampler=TPESampler())
study.optimize(objective, n_trials=500, callbacks=[neptune_callback])
opt_utils.log_study(study)
print(study.best_params)

#r_train = pearsonr(yTrain.squeeze(), yHatTrain.squeeze())[0]
#r_test = pearsonr(yTest.squeeze(), yHatTest.squeeze())[0]





