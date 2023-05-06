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
    nstdb = load_noise()
    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)


    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]

    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]

    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]
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

def load_data(my_fs=250):
    signals = []
    for folder_name in [f'A0{i}' for i in range(9)]:
        for sig in glob.glob(f'../data/cinc2017/training/{folder_name}/*.hea'):
            my_sig = wfdb.rdsamp(sig[:-4])
            my_sig = resample_signal(my_sig[0], my_sig[1]['fs'], my_fs)
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


