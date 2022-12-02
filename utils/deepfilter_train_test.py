# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 21:07:12 2022

@author: wes_c
"""

import pickle
from datetime import datetime

from dfilters import FIR_test_Dataset, IIR_test_Dataset
from dl_pipeline import train_dl, test_dl

EXPERIMENTS = [
        'DRNN',
        'FCN-DAE'
        ]

def train_models(noise_version):

        # Load dataset
    with open('../data/dataset_nv' + str(noise_version) + '.pkl', 'rb') as input:
        Dataset = pickle.load(input)


    train_time_dict = {}
    test_time_dict = {}

    for experiment in range(len(EXPERIMENTS)):
        start_train = datetime.now()
        train_dl(Dataset, EXPERIMENTS[experiment])
        end_train = datetime.now()
        train_time_dict[EXPERIMENTS[experiment]] = end_train - start_train

        start_test = datetime.now()
        [X_test, y_test, y_pred] = test_dl(Dataset, EXPERIMENTS[experiment])
        end_test = datetime.now()
        test_time_dict[EXPERIMENTS[experiment]] = end_test - start_test

        test_results = [X_test, y_test, y_pred]

        # Save Results
        with open('test_results_' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
            pickle.dump(test_results, output)
        print('Results from experiment ' + EXPERIMENTS[experiment] + '_nv' + str(noise_version) + ' saved')

    # Classical Filters

    # FIR
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = FIR_test_Dataset(Dataset)
    end_test = datetime.now()
    train_time_dict['FIR Filter'] = 0
    test_time_dict['FIR Filter'] = end_test - start_test

    test_results_FIR = [X_test_f, y_test_f, y_filter]

    # Save FIR filter results
    with open('test_results_FIR_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_FIR, output)
    print('Results from experiment FIR filter nv ' + str(noise_version) + ' saved')

    # IIR
    start_test = datetime.now()
    [X_test_f, y_test_f, y_filter] = IIR_test_Dataset(Dataset)
    end_test = datetime.now()
    train_time_dict['IIR Filter'] = 0
    test_time_dict['IIR Filter'] = end_test - start_test

    test_results_IIR = [X_test_f, y_test_f, y_filter]

    # Save IIR filter results
    with open('test_results_IIR_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(test_results_IIR, output)
    print('Results from experiment IIR filter nv ' + str(noise_version) + ' saved')

    # Saving timing list
    timing = [train_time_dict, test_time_dict]
    with open('timing_nv' + str(noise_version) + '.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(timing, output)
    print('Timing nv ' + str(noise_version) + ' saved')
    return
