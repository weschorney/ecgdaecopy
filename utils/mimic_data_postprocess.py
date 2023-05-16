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

def filter_data(X, y, cutoff, rnd_test=None, rnd_flag=None):
    #only retain those vals below the cutoff
    if rnd_flag:
        new_rnd = []
    new_X, new_y = [], []
    for idx, (x_ele, y_ele) in tqdm(enumerate(zip(X, y))):
        diff = y_ele.max() - y_ele.min()
        if diff < cutoff:
            new_X.append(x_ele)
            new_y.append(y_ele)
            if rnd_flag:
                new_rnd.append(rnd_test[idx])
    if rnd_flag:
        new_rnd = np.array(new_rnd)
        return new_X, new_y, new_rnd
    else:
        return new_X, new_y

def postprocess():
    dataset = load_data()
    rnd_test = np.load('./rnd_test_mimic.npy')
    X_train, y_train, X_test, y_test = dataset
    cutoff = get_diffs_quantile(y_train)
    if cutoff < 2.5:
        print('Data already postprocessed')
        return
    X_train, y_train = filter_data(X_train, y_train, cutoff)
    X_test, y_test, rnd_test = filter_data(X_test, y_test, cutoff,
                                           rnd_test=rnd_test, rnd_flag=True)
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    with open('../data/dataset_mimic.pkl', 'wb') as f:
        pickle.dump([X_train, y_train, X_test, y_test], f)
    np.save('rnd_test_mimic.npy', rnd_test)
    return

if __name__ == '__main__':
    postprocess()
    