# -*- coding: utf-8 -*-
"""
Created on Sun May  7 07:09:37 2023

@author: wes_c
"""

import re
import pickle
import numpy as np
import pandas as pd
import glob
import wfdb

from scipy import signal
from tqdm import tqdm
from cinc_data_preparation import sigs_to_parts, load_noise

#TODO: FIX DIV BY ZERO

def prepare_mimic():
    np.random.seed(777)
    nstdb = load_noise()
    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)
    #some ecgs only one lead so use noise from there
    noise_train = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    noise_test = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    noise_train2 = em_signals[0:int(em_signals.shape[0]/2), 0]
    noise_test2 = em_signals[int(em_signals.shape[0]/2):-1, 0]
    noise_train3 = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    noise_test3 = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    #load train and test
    signals = get_signal_paths()
    train, test = split_train_test(signals)
    train = make_dataset(train)
    test = make_dataset(test)
    train = [ele[:,0] if len(ele.shape) > 1 else ele for ele in train]
    test = [ele[:,0] if len(ele.shape) > 1 else ele for ele in test]
    beats_train = sigs_to_parts(train)
    beats_test = sigs_to_parts(test)
    sn_train = []
    sn_test = []
    train_labels = []
    test_labels = []
    samples = 512
    noise_index = 0
    # Adding noise to train
    #TODO: FIX MISSING BETWEEN TRAIN/LABEL AND SAME FOR TEST
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        noise2 = noise_train2[noise_index:noise_index + samples]
        noise3 = noise_train3[noise_index:noise_index + samples]
        noise = noise + noise2 + noise3
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        if beat_max_value == 0:
            continue
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        train_labels.append(beats_train[i])
        noise_index += samples
        if noise_index > (len(noise_train) - samples):
            noise_index = 0
    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    # Saving the random array so we can use it on the amplitude segmentation tables
    rnd_test2 = []
    #np.save('rnd_test_mimic.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        noise2 = noise_test2[noise_index:noise_index + samples]
        noise3 = noise_test3[noise_index:noise_index + samples]
        noise = noise + noise2 + noise3
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        if beat_max_value == 0:
            continue
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        sn_test.append(signal_noise)
        test_labels.append(beats_test[i])
        rnd_test2.append(rnd_test[i])
        noise_index += samples
        if noise_index > (len(noise_test) - samples):
            noise_index = 0
    X_train = np.array(sn_train)
    y_train = np.array(train_labels)
    X_test = np.array(sn_test)
    y_test = np.array(test_labels)
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)
    rnd_test2 = np.array(rnd_test2)
    np.save('rnd_test_mimic.npy', rnd_test2)
    Dataset = [X_train, y_train, X_test, y_test]
    with open('../data/dataset_mimic.pkl', 'wb') as f:
        pickle.dump(Dataset, f)
    return

def resample_signal(x, fs, fs_target):
    t = np.arange(x.shape[0]).astype("float64")
    if fs == fs_target:
        return x, t
    new_length = int(x.shape[0] * fs_target / fs)
    # Resample the array if NaN values are present
    if np.isnan(x).any():
        x = pd.Series(x.reshape((-1,))).interpolate().values
    resampled_x, resampled_t = signal.resample(x, num=new_length, t=t)
    assert (
        resampled_x.shape[0] == resampled_t.shape[0]
        and resampled_x.shape[0] == new_length
    )
    assert np.all(np.diff(resampled_t) > 0)
    return resampled_x, resampled_t

def load_signal(signal_name, fs=250):
    sig = wfdb.rdsamp(signal_name)
    sig, _ = resample_signal(sig[0], sig[1]['fs'], fs)
    return sig

def get_signal_paths():
    paths = glob.glob('../data/ADHD-ECG/ADHD/*')
    #now take only first recording from each patient
    signals = []
    for path in tqdm(paths):
        sigs = glob.glob(path + '/*.dat')
        sigs = [ele for ele in sigs if not re.search('.*[0-9]+n\.dat', ele)]
        #the following is inefficient since we will later load them again
        #however, this lets us inspect what we are working with and
        #we only need to do this once
        if sigs:
            for sig in sigs:
                my_sig = load_signal(sig[:-4])
                if not np.isnan(my_sig).any():
                    signals.append(sig)
                    break
    return signals

def split_train_test(signals, test_size=0.2):
    test_signals = np.random.choice(signals, size=int(len(signals)*test_size),
                                    replace=False)
    train_signals = [ele for ele in signals if ele not in test_signals]
    test_signals = list(test_signals)
    return train_signals, test_signals

def make_dataset(sig_list, fs=250):
    dataset = []
    for sig in sig_list:
        dataset.append(load_signal(sig[:-4]))
    return dataset

def sanity_check(arr):
    #check if any nans exist in the values in arr
    #can be used after preprocessing to ensure valid data
    sane = [not np.isnan(ele).any() for ele in arr]
    return all(sane)

if __name__ == '__main__':
    prepare_mimic()
