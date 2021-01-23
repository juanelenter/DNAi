import numpy as np

class split_data:

    def __init__(self, X,  y, idx_test, idx_train, std_geno = True, std_pheno = True):

        self.X_train = X[idx_train].astype(np.float32)
        self.Y_train = y[idx_train]
        self.X_test = X[idx_test].astype(np.float32)
        self.Y_test = y[idx_test]
        self.idx_test = idx_test
        self.idx_train = idx_train
        self.std_geno = std_geno
        self.std_pheno = std_pheno

    def standardize_geno(self):

        if self.std_geno:
            print('Standardizing genotypes... ')
            self.std_geno = 1
            x_train_mean = np.mean(self.X_train)
            x_train_std = np.std(self.X_train)
            self.X_train -= x_train_mean
            self.X_train /= x_train_std
            self.X_test -= x_train_mean
            self.X_test /= x_train_std

    def standardize_pheno(self):

        if self.std_pheno:
            print('Standardizing phenotypes... ')
            self.std_pheno = 1
            y_train_mean = np.mean(self.Y_train)
            y_train_std = np.std(self.Y_train)
            self.Y_train -= y_train_mean
            self.Y_train /= y_train_std
            self.Y_test -= y_train_mean
            self.Y_test /= y_train_std

    def debug_mode(self):

        self.X_train = self.X_train[::25]
        self.Y_train = self.Y_train[::25]
        self.X_test = self.X_test[::25]
        self.Y_test = self.Y_test[::25]
        print("Debug mode ON.")