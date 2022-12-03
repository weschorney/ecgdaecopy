# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 21:51:23 2022

@author: wes_c
"""

import pickle
import data_preparation as dp

def prepare_data():
    noise_versions = [1, 2]
    for nv in noise_versions:
        Dataset = dp.Data_Preparation(noise_version=nv)
        # Save dataset
        with open('../data/dataset_nv' + str(nv) + '.pkl', 'wb') as output:
            pickle.dump(Dataset, output)
        print('Dataset saved')
    return
