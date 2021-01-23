#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:24:10 2020

@author: ignaciohounie
"""

import sys
#import argparse
import numpy as np
import sklearn.metrics
import config
import os.path
from os import listdir
import pandas as pd

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ os.path.join(path_to_dir,filename) for filename in filenames if filename.endswith( suffix ) ]

results_stats = []

filenames = find_csv_filenames(config.eval_output_path, suffix=".csv")
for file in filenames:
    print(file)
    df = pd.read_csv(file)
    results_stats.append([os.path.split(file)[1], np.mean(df['mse']),  np.var(df['mse']), np.mean(df['r2']),  np.var(df['r2'])    ])
df_stats = pd.DataFrame(results_stats, columns =['Experiment', 'mean mse', 'var mse', 'mean r2', 'var r2'])
df_stats.to_csv(os.path.join(config.eval_output_path, "stats.csv"))
print('metrics stats saved at ', os.path.join(config.eval_output_path, "stats.csv"))    