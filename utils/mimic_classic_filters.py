# -*- coding: utf-8 -*-
"""
Created on Sat May 13 13:34:00 2023

@author: wes_c
"""

import pickle
from datetime import datetime

from dfilters import FIR_test_Dataset, IIR_test_Dataset

with open('../data/dataset_mimic.pkl', 'rb') as input:
    Dataset = pickle.load(input)

cftet = []

# FIR
start_test = datetime.now()
[X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
end_test = datetime.now()
cftet.append(end_test - start_test)

test_results_FIR = [X_test_f, y_test_f, y_filter]

# Save FIR filter results
with open('mimic_test_results_FIR.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(test_results_FIR, output)
print('Results from experiment FIR filter saved')

# IIR
start_test = datetime.now()
[X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
end_test = datetime.now()
cftet.append(end_test - start_test)

test_results_IIR = [X_test_f, y_test_f, y_filter]

# Save IIR filter results
with open('mimic_test_results_IIR.pkl', 'wb') as output:  # Overwrites any existing file.
    pickle.dump(test_results_IIR, output)
print('Results from experiment IIR filter saved')

with open('mimic_timing_classic.pkl', 'wb') as output:
    pickle.dump([[0,0], cftet], output)
