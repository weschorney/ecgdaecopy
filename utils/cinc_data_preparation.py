# -*- coding: utf-8 -*-
"""
Created on Fri May  5 20:06:17 2023

@author: wes_c
"""

import pickle
import numpy as np
import pandas as pd
import glob
import wfdb

from scipy import signal

def prepare_cinc(seed=777):
    np.random.seed(seed=seed)
    nstdb = load_noise()
    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)
    #cinc is lead1 equiv so use noise from there
    noise_train = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    noise_test = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    noise_train2 = em_signals[0:int(em_signals.shape[0]/2), 0]
    noise_test2 = em_signals[int(em_signals.shape[0]/2):-1, 0]
    noise_train3 = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    noise_test3 = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    #load train and test
    beats_train = load_train_data()
    beats_test = load_test_data()
    #split beats
    beats_train = sigs_to_parts(beats_train)
    beats_test = sigs_to_parts(beats_test)
    sn_train = []
    sn_test = []
    samples = 512
    noise_index = 0
    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(beats_train)) / 100
    for i in range(len(beats_train)):
        noise = noise_train[noise_index:noise_index + samples]
        noise2 = noise_train2[noise_index:noise_index + samples]
        noise3 = noise_train3[noise_index:noise_index + samples]
        noise = noise + noise2 + noise3
        beat_max_value = np.max(beats_train[i]) - np.min(beats_train[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_train[i] / Ase
        signal_noise = beats_train[i] + alpha * noise
        sn_train.append(signal_noise)
        noise_index += samples
        if noise_index > (len(noise_train) - samples):
            noise_index = 0
    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(beats_test)) / 100
    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save('rnd_test_cinc.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(beats_test)):
        noise = noise_test[noise_index:noise_index + samples]
        noise2 = noise_test2[noise_index:noise_index + samples]
        noise3 = noise_test3[noise_index:noise_index + samples]
        noise = noise + noise2 + noise3
        beat_max_value = np.max(beats_test[i]) - np.min(beats_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        Ase = noise_max_value / beat_max_value
        alpha = rnd_test[i] / Ase
        signal_noise = beats_test[i] + alpha * noise
        sn_test.append(signal_noise)
        noise_index += samples
        if noise_index > (len(noise_test) - samples):
            noise_index = 0
    X_train = np.array(sn_train)
    y_train = np.array(beats_train)
    X_test = np.array(sn_test)
    y_test = np.array(beats_test)
    X_train = np.expand_dims(X_train, axis=2)
    y_train = np.expand_dims(y_train, axis=2)
    X_test = np.expand_dims(X_test, axis=2)
    y_test = np.expand_dims(y_test, axis=2)
    Dataset = [X_train, y_train, X_test, y_test]
    with open('../data/dataset_cinc.pkl', 'wb') as f:
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

def load_train_data(my_fs=250):
    signals = []
    for folder_name in [f'A0{i}' for i in range(9)]:
        for sig in glob.glob(f'../data/cinc2017/training/{folder_name}/*.hea'):
            my_sig = wfdb.rdsamp(sig[:-4])
            my_sig, _ = resample_signal(my_sig[0], my_sig[1]['fs'], my_fs)
            my_sig = my_sig.reshape((-1,))
            signals.append(my_sig)
    return signals

def load_test_data(my_fs=250):
    signals = []
    for sig in glob.glob('../data/cinc2017/validation/*.hea'):
        my_sig = wfdb.rdsamp(sig[:-4])
        my_sig, _ = resample_signal(my_sig[0], my_sig[1]['fs'], my_fs)
        my_sig = my_sig.reshape((-1,))
        signals.append(my_sig)
    return signals

def load_noise():
    nstdb = pickle.load(open('../data/NoiseBWL.pkl', 'rb'))
    return nstdb

def sig_to_parts(sig, part_size=512):
    num_parts = sig.shape[0] // part_size
    parts = []
    for i in range(num_parts):
        parts.append(sig[i*part_size:(i+1)*part_size])
    return parts

def sigs_to_parts(sigs, part_size=512):
    parts = []
    for sig in sigs:
        parts += sig_to_parts(sig, part_size=part_size)
    return parts

if __name__ == '__main__':
    prepare_cinc()
