# -*- coding: utf-8 -*-
"""
Created on Fri May 12 16:50:41 2023

@author: wes_c
"""

import pickle
import numpy as np

from tqdm import tqdm

def load_data():
    return pickle.load(open('../data/dataset_mimic.pkl', 'rb'))

def get_diffs_quantile(y_train, q=0.85):
    diffs = [ele.max() - ele.min() for ele in y_train]
    cutoff = np.quantile(diffs, q)
    return cutoff

def filter_data(X, y, cutoff):
    #only retain those vals below the cutoff
    new_X, new_y = [], []
    for x_ele, y_ele in tqdm(zip(X, y)):
        diff = y_ele.max() - y_ele.min()
        if diff < cutoff:
            new_X.append(x_ele)
            new_y.append(y_ele)
    return new_X, new_y

def postprocess():
    dataset = load_data()
    X_train, y_train, X_test, y_test = dataset
    cutoff = get_diffs_quantile(y_train)
    if cutoff < 2.5:
        print('Data already postprocessed')
        return
    X_train, y_train = filter_data(X_train, y_train, cutoff)
    X_test, y_test = filter_data(X_test, y_test, cutoff)
    with open('../data/dataset_mimic.pkl', 'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test], f)
    return

if __name__ == '__main__':
    postprocess()
    