# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 21:26:15 2020

Module that loads SNP data, eliminates IDs and columns with NaNs.

@author: Tecla
"""

import pandas as pd
import numpy as np
import config
    
def txt_to_np():
    '''
    Grabs SNP file from config.snps_path and returns SNP matrix idless.
    '''
    print(config.snps_path)
    with open(config.snps_path, "r") as snps:
        data = snps.readlines()
        data_idless = [ d[18:-1] for d in data ]
        data_idless_separated = [ [ snp for snp in line ] for line in data_idless ]
    return np.array(data_idless_separated)
    
def delete_3_4_5(matrix):
    '''
    Deletes columns that has 3, 4 or 5.
    '''
    columns_to_delete = []
    for j, column in enumerate(matrix.transpose()):
        if ("3" in column) or ("4" in column) or ("5" in column):
            columns_to_delete.append(j)
    matrix = np.delete(matrix, columns_to_delete, axis = 1)
    return matrix

if __name__ == "__main__":
    
    try:
        snps = txt_to_np()
        snps = delete_3_4_5(snps)
        snps = snps.astype(np.int)
        np.savetxt(config.snps_matrix_path, snps ,delimiter = "," , fmt='%d')
    
    except:
        print("Pone bien los paths en config.py gil de goma.")
    
