import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#R Stuff
from rpy2.robjects import r
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()


class BayesRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, h2=0.5, prior = "BayesC", n_iter = 5000, std_feno=False, std_geno=False):
        self.h2 = h2
        self.n_iter = n_iter
        self.std_feno = std_feno
        self.std_geno = std_geno
        self.prior = prior
        self.check_r_dependencies()

    def fit(self, X, y):
        
        self.X_ = np.nan_to_num(X)
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X_test):

        # Check is fit had been called
        check_is_fitted(self)
        X_test = np.nan_to_num(X_test)
        X, y, idx_train, idx_test = self.split_to_idxs(self.X_, self.y_, X_test)
        self.check_r_dependencies()
        r.assign('X',X)
        r.assign('y',y)
        r.assign('whichNa',idx_test)
        r.assign('train_idx', idx_train)
        r.assign('test_idx', idx_test)
        r.assign('normalize_phenotype', self.std_feno)
        r.assign('normalize_genotype', self.std_geno)
        r.assign('n_it', self.n_iter)
        r.assign('R2', self.h2)
        r.assign('prior', self.prior)
        r('sink("/dev/null")')
        r('''
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
                y[test_idx]<-NA
        print("Starting Gibbs Sampling. Hold on to your hat!")
        print("Bayes C is my middle name")
        nIter=n_it;
        burnIn=10;
        thin=3;
        saveAt='';
        S0=NULL;
        weights=NULL;
        ETA<-list(list(X=X,model=prior))
          
        fit_BC=BGLR(y=y,ETA=ETA,nIter=nIter,burnIn=burnIn,thin=thin,saveAt=saveAt,df0=5,S0=S0,weights=weights,R2=R2)
    
        y_pred_BC<-fit_BC$yHat
        prior<-fit_BC$prior
        y_pred_train<-y_pred_BC[train_idx] 
        y_pred_test<-y_pred_BC[test_idx] 
        sink()
        ''')
        a = np.array(r['y_pred_test'])
        return np.nan_to_num(a)

    def get_params(self, deep=True):
        return {"h2": self.h2, "n_iter": self.n_iter, "std_geno": self.std_geno,"std_feno": self.std_feno, "prior":self.prior}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():                     
            setattr(self, parameter, value)     
        return self
    def check_r_dependencies(self):
        r('''
           list.of.packages <- c("doRNG", "doMC", "BGLR")
           new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
           if(length(new.packages)>0){install.packages(new.packages, repos = "http://cran.us.r-project.org")}
           library(doMC)
           require(BGLR)
            ''')
        return
    def split_to_idxs(self, X_train, Y_train, X_test):
        X = np.concatenate((X_train, X_test))
        y = np.concatenate((Y_train, np.empty(len(X_test))))
        idx_train = np.arange(len(Y_train))+1
        idx_test = len(Y_train)+np.arange(len(X_test))+1
        return X, y, idx_train, idx_test
    
