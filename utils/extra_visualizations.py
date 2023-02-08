# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 18:46:07 2023

@author: wes_c
"""

import os
import tensorflow as tf
import pickle
import random
import matplotlib.pyplot as plt
import dl_models

def load_model():
    #only implemented for one in paper
    model = dl_models.AttentionSkipDAE2()
    model.build(input_shape=(None, 512, 1))
    model.load_weights('Attention_Skip_DAE_weights.best.hdf5')
    return model

def load_data(dataset_name, idx=None):
    with open('../data/'+dataset_name, 'rb') as f:
        data = pickle.load(f)
    if idx is None:
        idx = random.randint(0,data[0].shape[0])
    return tf.expand_dims(data[0][idx,...], 0)

def input_output(model, dataset_name, idx=None):
    inp_ele = load_data(dataset_name, idx=idx)
    out_ele = model.encode(inp_ele)
    denoised = model.decode(out_ele)
    return inp_ele, out_ele, denoised

def plot_model_output(output, savename, alpha=0.4):
    plt.plot(output[0,...], alpha=alpha)
    plt.axis('off')
    plt.savefig(savename, dpi=500)
    plt.clf()
    return

def encode_layer_figures(model, dataset_name,
                         fig_prefix='encoder_output', idx=None):
    inp_ele = load_data(dataset_name, idx=idx)
    out_ele = model.b1(inp_ele)
    return out_ele

def full_plot(model, dataset_name, idx=None, efig_pref='encoder_output',
              dfig_pref='decoder_output'):
    inp_ele = load_data(dataset_name, idx=idx)
    efig_pref = '../data/figures/' + efig_pref
    dfig_pref = '../data/figures/' + dfig_pref
    if not os.path.exists('../data/figures/'):
        os.makedirs('../data/figures/')
    plot_model_output(inp_ele, '../data/figures/initial.png', alpha=1.0)
    #bruuuuuuuuuuuuuuuuuute force
    out_ele = model.b1(inp_ele)
    plot_model_output(out_ele, efig_pref+'0.png')
    out_ele = model.b2(out_ele)
    plot_model_output(out_ele, efig_pref+'1.png')
    out_ele = model.b3(out_ele)
    plot_model_output(out_ele, efig_pref+'2.png')
    out_ele = model.b4(out_ele)
    plot_model_output(out_ele, efig_pref+'3.png')
    out_ele = model.b5(out_ele)
    plot_model_output(out_ele, efig_pref+'4.png')
    out_ele = model.d5(out_ele)
    plot_model_output(out_ele, dfig_pref+'4.png')
    out_ele = model.d4(out_ele)
    plot_model_output(out_ele, dfig_pref+'3.png')
    out_ele = model.d3(out_ele)
    plot_model_output(out_ele, dfig_pref+'2.png')
    out_ele = model.d2(out_ele)
    plot_model_output(out_ele, dfig_pref+'1.png')
    out_ele = model.d1(out_ele)
    plot_model_output(out_ele, dfig_pref+'0.png')
    out_ele = model(inp_ele)
    plot_model_output(out_ele, '../data/figures/output.png', alpha=1.0)
    return

if __name__ == '__main__':
    full_plot(load_model(), 'dataset_nv1.pkl', idx=0)
