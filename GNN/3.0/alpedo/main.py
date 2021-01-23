#Standard libraries
import os
import numpy as np
import pickle
import datetime
from copy import deepcopy
import torch; torch.set_default_dtype(torch.float64)
import torch.nn as nn
import torch.optim as optim

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

today = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
useGPU = True

################################
####### Argument Parsing #######
################################
parser = argparse.ArgumentParser()
parser.add_argument("model_name", help="linear fcn mfgf gnn gnn2")
parser.add_argument("-l", "--log_experiment", help="wether to log on comet", action="store_true")
parser.add_argument("-e", "--epochs",type=int, help="number of epochs",default=30)
parser.add_argument("-lr", "--learning_rate",type=float, help="learning rate",default=0.05)
parser.add_argument("-vi", "--validation_interval",type=int, help="how much batches to validate",default=10)
parser.add_argument("-bs", "--batch_size",type=int, help="GNN require batch sizes close to 1 due to memory constraints",default=10)
args = parser.parse_args()

################################
####### Comet Logging ##########
################################

if args.log_experiment:
    logger = experiment_logger("comet", "gnn-yeast")
    logger.add_params(vars(args))

################################
####### DATA GENERATION ########
################################

print("Loading Data...")
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

nNodes = S.shape[0] # number of nodes
nTrain = xTrain.shape[0]
nTest = xTest.shape[0]

xTrain = torch.Tensor(xTrain)
xTrain = xTrain.reshape([-1,1,N])
yTrain = torch.Tensor(yTrain)
yTrain = yTrain.reshape([-1,1,1])

xTest = torch.Tensor(xTest)
xTest = xTest.reshape([-1,1,N])
yTest = torch.Tensor(yTest)
yTest = yTest.reshape([-1,1,1])

##################
# TRAINING SETUP #
##################

#\\\ Individual model training options
optimAlg = 'ADAM' # Options: 'SGD', 'ADAM', 'RMSprop'
beta1 = 0.9 # beta1 if 'ADAM', alpha if 'RMSprop'
beta2 = 0.999 # ADAM option only

#\\\ Loss function choice
lossFunction = nn.MSELoss()

#\\\ Overall training options
nEpochs = args.epochs
batchSize = args.batch_size
learningRate = args.learning_rate
validationInterval = args.validation_interval

doLearningRateDecay = False # Learning rate decay
learningRateDecayRate = 0.9 # Rate
learningRateDecayPeriod = 1 # How many epochs after which update the lr

################################
######## ARCHITECTURES #########
################################

if args.model_name == 'linear':
# Linear parametrization    
    model = torch.nn.Linear(N,1, bias = True)  

elif args.model_name == 'fcn':
# Fully connected neural network   
    model = nn.Sequential(torch.nn.Linear(N,25, bias=True), nn.ReLU(), torch.nn.Linear(25,1, bias=True))

elif args.model_name == 'mfgf':
# Multi-feature graph filter
    model = nn.Sequential(archit.LocalGNN([1, 64], [5], True, gml.NoActivation, [N], gml.NoPool, [1], [1], S), torch.nn.Linear(N,1, bias = True))

# Graph filter
# LocalGNN(dimNodeSignals, nFilterTaps, bias, # Graph Filtering
#                 nonlinearity, # Nonlinearity
#                 nSelectedNodes, poolingFunction, poolingSize, # Pooling
#                 dimReadout, # Local readout layer
#                 GSO, order = None # Structure)

#elif args.model_name == 'gnn':
# GNN, 1 layer
#    model = nn.Sequential( archit.LocalGNN([1, 64], [5], True, nn.ReLU, [N], gml.NoPool, [1], [1], S), torch.nn.Linear(N,1, bias=True))

# GNN, 2 layers
#elif args.model_name == 'gnn2':
#    model = nn.Sequential(archit.LocalGNN([1, 64, 32], [5, 5], True, nn.ReLU, [N, N], gml.NoPool, [1, 1], [1], S), torch.nn.Linear(N,1, bias=True))


#\\\\\\\\\\\\\\\\\
#\\\ LOCAL GNN \\\
#\\\\\\\\\\\\\\\\\

elif args.model_name == 'gnn2':

    #\\\ Basic parameters for all the Local GNN architectures
    
    modelLclGNN = {} # Model parameters for the Local GNN (LclGNN)
    modelLclGNN['name'] = 'LclGNN'
    modelLclGNN['device'] = 'cuda:0' if (useGPU and torch.cuda.is_available()) else 'cpu'
    
    #\\\ ARCHITECTURE
    # Chosen architecture
    modelLclGNN['archit'] = archit.LocalGNN
    # Graph convolutional parameters
    modelLclGNN['dimNodeSignals'] = [1, 64, 32] # Features per layer
    modelLclGNN['nFilterTaps'] = [5, 5] # Number of filter taps per layer
    modelLclGNN['bias'] = True # Decide whether to include a bias term
    # Nonlinearity
    modelLclGNN['nonlinearity'] = nn.ReLU # Selected nonlinearity
    # Pooling
    modelLclGNN['poolingFunction'] = gml.NoPool # Summarizing function
    modelLclGNN['nSelectedNodes'] = None # To be determined later on
    modelLclGNN['poolingSize'] = [1, 1] # poolingSize-hop neighborhood that
        # is affected by the summary
    # Readout layer: local linear combination of features
    modelLclGNN['dimReadout'] = [1] # Dimension of the fully connected layers
        # after the GCN layers (map); this fully connected layer is applied only
        # at each node, without any further exchanges nor considering all nodes
        # at once, making the architecture entirely local.
    # Graph structure
    modelLclGNN['GSO'] = None # To be determined later on, based on data
    modelLclGNN['order'] = None # Not used because there is no pooling
    
    #\\\ TRAINER
    modelLclGNN['trainer'] = training.TrainerSingleNode

    #\\\ EVALUATOR
    modelLclGNN['evaluator'] = evaluation.evaluateSingleNode

elif args.model_name == 'gnn':
    args.model_name == 'gnn':
    modelLclGNN1Ly = deepcopy(modelLclGNN)

    modelLclGNN1Ly['name'] += '1Ly' # Name of the architecture
    
    modelLclGNN1Ly['dimNodeSignals'] = modelLclGNN['dimNodeSignals'][0:-1]
    modelLclGNN1Ly['nFilterTaps'] = modelLclGNN['nFilterTaps'][0:-1]
    modelLclGNN1Ly['poolingSize'] = modelLclGNN['poolingSize'][0:-1]

    
#%%##################################################################
#                                                                   #
#                    SETUP                                          #
#                                                                   #
#####################################################################

#\\\ Determine processing unit:
if useGPU and torch.cuda.is_available():
    torch.cuda.empty_cache()



#######################################
nValid = int(np.floor(0.1*nTrain))
xValid = xTrain[0:nValid,:,:]
yValid = yTrain[0:nValid,:,:]
xTrain = xTrain[nValid:,:,:]
yTrain = yTrain[nValid:,:,:]
nTrain = xTrain.shape[0]



