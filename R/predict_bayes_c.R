args <- commandArgs(trailingOnly = TRUE)
y_file = args[1]
n_it = args[2]
normalize_genotype=FALSE
normalize_phenotype=FALSE
require(BGLR)
#Loads
print("loading genotypes, this might take a while") 
X <- read.csv('../output/sim/snps_matrix.csv', header=FALSE, colClasses = rep('integer', 20138))
print("loading fenotypes") 
y<-t(read.csv(paste('../output/sim/',y_file, sep=""), header=FALSE, colClasses = 'numeric'))
whichNa<-t(paste('../output/sim/',n_it,'_', idx_train.csv, sep=""), header=FALSE,  colClasses = 'integer'))
y[whichNa]<-NA
train_idx<-t(read.csv(paste('../output/sim/',n_it,'_', idx_train.csv, sep=""), header=FALSE,  colClasses = 'integer'))

if(normalize_phenotype){
    print("normalizing phenotypes IDK")
    y<-(y-mean(y[train_idx]))/sd(train_idx)
}


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
print("Starting Gibbs Sampling. Hold on to your hat!")
print("Bayes C is my middle name")
nIter=20000;
burnIn=2500;
thin=3;
saveAt='';
S0=NULL;
weights=NULL;
R2=0.5;
ETA<-list(list(X=X,model='BayesC'))
  
fit_BC=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)

y_pred_BC<-fit_BC$yHat
out_path = file.path('..','output', 'pred', 'bayes', 'b', paste(n_it,'_', y_file, sep=""))
write.table(y_pred_BC, out_path, row.names = FALSE, col.names = FALSE)
print("Holy smokes! predicted phenotypes for Bayes C stored")