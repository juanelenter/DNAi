#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 16:16:24 2020

@author: root
"""

import numpy as np

def sort_by_error(d, i, Y_test, test_preds, idx_test):
    '''
    Returns dictionnary with index array sorted by error, and error.
    '''
    error = Y_test-test_preds
    error_abs = np.abs(error)
    idx = np.argsort(error_abs)
    sorted_idx = idx_test[idx]
    sorted_error = error_abs[idx]
    d[f"sorted_idx{i}"] = sorted_idx
    d[f"sorted_error{i}"] = sorted_error
    d[f"raw_error{i}"] = error
    return d
