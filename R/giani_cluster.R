setwd("/mnt/R")
args <- commandArgs(trailingOnly = TRUE)
y_file = args[1]
n_it = args[2]
normalize_genotype=FALSE
normalize_phenotype=TRUE
list.of.packages <- c("doRNG", "doMC", "gbm", "randomForest" , "glmnet", "BGLR")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)>0){install.packages(new.packages, repos = "http://cran.us.r-project.org")}
library(doRNG)
library(doMC)
library(gbm)
library(randomForest)
library(glmnet)
require(BGLR)
registerDoMC(cores = 4)
#Loads
train<-foreach(i = 0 : n_it, .options.RNG = 100) %do% {
    t(read.csv(paste('../output/sim/',i,'_', 'idx_train.csv', sep=""), header=FALSE,  colClasses = 'integer'))+1
}
#print(dim(test[[1]]))
test<-foreach(i = 0 : n_it, .options.RNG = 100) %do% {
    t(read.csv(paste('../output/sim/',i,'_', 'idx_test.csv', sep=""), header=FALSE,  colClasses = 'integer'))+1
}
print("loading genotypes, this might take a while") 
X <- read.csv('../output/sim/snps_matrix.csv', header=FALSE, colClasses = rep('integer', 20138))
print("loading fenotypes") 
y<-t(read.csv(paste('../output/sim/',y_file, sep=""), header=FALSE, colClasses = 'numeric'))
#whichNa<-t(paste('../output/sim/',n_it,'_', idx_train.csv, sep=""), header=FALSE,  colClasses = 'integer'))
#y[whichNa]<-NA

###############################################
########        MAIN LOOP
##############################################
foreach(i = 1 : n_it, .options.RNG = 100) %dorng% {
    test_idx<-test[[i]]
    train_idx <- train[[i]]
    print("loaded idxs")
    if(normalize_phenotype){
        print("scaling phenotypes IDK")
        y<-(y-mean(y[train_idx]))/sd(y[train_idx])
    }
    print("done scaling")

    if(normalize_genotype){
        print("Normalizing Genotypes, A Murky Business ") 
        for(i in ncol(X))
        {
            std_xi<-sd(X[,i])
            if (std_xi==0) {
                X[,i]<-(X[,i]-mean(X[,i]))/std_xi
            } else {
                X[,i]<-(X[,i]-mean(X[,i]))
            }
            
        }
    }
    ################################################
    #######                RIDGE
    ################################################
    ytrain_m <- t(as.matrix(y[train_idx]))
    Xtrain_m <- as.matrix(X[train_idx, ])
    Xtest_m<- as.matrix(X[test_idx, ])
    ridge.mod <- cv.glmnet(Xtrain_m, ytrain_m, family = 'gaussian', alpha = 0, standardize = FALSE, nfolds = 10, parallel = F)
    ridge.pred <- predict(ridge.mod, Xtest_m)
    out_path = file.path('..','output', 'pred', 'ridge', paste(i,'_', y_file, sep=""))
    write.table(ridge.pred, out_path, row.names = FALSE, col.names = FALSE)

    ################################################
    #######                LASSO
    ################################################
    lasso.mod <- cv.glmnet(Xtrain_m, ytrain_m, family = 'gaussian', alpha = 1, standardize = FALSE, nfolds = 10, parallel = F)
    lasso.pred <- predict(lasso.mod, Xtest_m)
    out_path = file.path('..','output', 'pred', 'ridge', paste(i,'_', y_file, sep=""))
    write.table(lasso.pred, out_path, row.names = FALSE, col.names = FALSE)
    ################################################
    #######                GBM
    ################################################
    ytrain <- y[train_idx]
    nTrain_ <- 3/4 * length(ytrain)
    gbm.mod <- gbm.fit(X[train_idx,], ytrain, distribution = 'gaussian', n.trees = 1000, interaction.depth = 5, n.minobsinnode = 10, shrinkage = 0.01, bag.fraction = 0.5, nTrain = nTrain_, keep.data = FALSE, verbose = TRUE)
    gbm.pred<- predict(gbm.mod, X[test_idx, ])
    out_path = file.path('..','output', 'pred', 'gbm', paste(i,'_', y_file, sep=""))
    write.table(gbm.pred, out_path, row.names = FALSE, col.names = FALSE)


    ################################################
    #######                Rforest
    ################################################

    ytrain <- y[train_idx]
    ytest <- y[test_idx]
    mod <- randomForest(x = X[train_idx,], y = ytrain, importance = TRUE, xtest = X[test_idx,], ytest = ytest, ntree = 500, do.trace = 100)
    forest.pred <- mod$test$predicted
    out_path = file.path('..','output', 'pred', 'rforest', paste(i,'_', y_file, sep=""))
    write.table(forest.pred, out_path, row.names = FALSE, col.names = FALSE)
    print('tamos bien')

    ################################################
    #######                Bayes C
    ################################################
    y[test_idx]<-NA
    print("Starting Gibbs Sampling. Hold on to your hat!")
    print("Bayes C is my middle name")
    nIter=20000;
    burnIn=2500;
    thin=3;
    saveAt='';
    S0=NULL;
    weights=NULL;
    R2=0.5;
    ETA<-list(list(X=X,model='BayesB'))
      
    fit_BC=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)

    y_pred_BC<-fit_BC$yHat
    out_path = file.path('..','output', 'pred', 'bayes', 'b', paste(i,'_', y_file, sep=""))
    write.table(y_pred_BC[test_idx], out_path, row.names = FALSE, col.names = FALSE)
    print("Holy smokes! predicted phenotypes for Bayes C stored")

}
