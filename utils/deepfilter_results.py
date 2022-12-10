# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 20:07:09 2022

@author: wes_c
"""

import pickle
import numpy as np
import visualization as vs

from metrics import MAD, SSD, PRD, COS_SIM


EXPERIMENTS = [
        'CNN-DAE',
        'Vanilla DAE',
        'DRNN',
        'FCN-DAE',
        'Multibranch LANLD',
        'ECA Skip DAE',
        'Attention Skip DAE'
        ]

def load_experiment(experiment_name):
    with open('test_results_' + experiment_name + '_nv1.pkl', 'rb') as input:
        test_nv1 = pickle.load(input)
    with open('test_results_' + experiment_name + '_nv2.pkl', 'rb') as input:
        test_nv2 = pickle.load(input)

    test_results = [np.concatenate((test_nv1[0], test_nv2[0])),
                    np.concatenate((test_nv1[1], test_nv2[1])),
                    np.concatenate((test_nv1[2], test_nv2[2]))]
    return test_results

def generate_results():
    
        #Load timing
    with open('timing_nv1.pkl', 'rb') as input:
        timing_nv1 = pickle.load(input)
        [train_time_list_nv1, test_time_list_nv1] = timing_nv1

    with open('timing_nv2.pkl', 'rb') as input:
        timing_nv2 = pickle.load(input)
        [train_time_list_nv2, test_time_list_nv2] = timing_nv2

    train_time_list = []
    test_time_list = []

    if isinstance(train_time_list_nv1, list):
        timing_names = None
        for i in range(len(train_time_list_nv1)):
            train_time_list.append(train_time_list_nv1[i] + train_time_list_nv2[i])
    
        for i in range(len(test_time_list_nv1)):
            test_time_list.append(test_time_list_nv1[i] + test_time_list_nv2[i])
    else:
        #assume it's a dict
        timing_names = []
        for key in train_time_list_nv1.keys():
            train_time_list.append(train_time_list_nv1[key] + train_time_list_nv2[key])
            test_time_list.append(test_time_list_nv1[key] + test_time_list_nv2[key])
            timing_names.append(key)

    timing = [train_time_list, test_time_list]

    test_CNN_DAE = load_experiment(EXPERIMENTS[0])
    test_Vanilla_DAE = load_experiment(EXPERIMENTS[1])
    test_DRNN = load_experiment(EXPERIMENTS[2])
    test_FCN_DAE = load_experiment(EXPERIMENTS[3])
    test_Multibranch_LANLD = load_experiment(EXPERIMENTS[4])
    test_ECADAE = load_experiment(EXPERIMENTS[5])
    test_ADAE = load_experiment(EXPERIMENTS[6])

    # Load Result FIR Filter
    with open('test_results_FIR_nv1.pkl', 'rb') as input:
        test_FIR_nv1 = pickle.load(input)
    with open('test_results_FIR_nv2.pkl', 'rb') as input:
        test_FIR_nv2 = pickle.load(input)

    test_FIR = [np.concatenate((test_FIR_nv1[0], test_FIR_nv2[0])),
                np.concatenate((test_FIR_nv1[1], test_FIR_nv2[1])),
                np.concatenate((test_FIR_nv1[2], test_FIR_nv2[2]))]

    # Load Result IIR Filter
    with open('test_results_IIR_nv1.pkl', 'rb') as input:
        test_IIR_nv1 = pickle.load(input)
    with open('test_results_IIR_nv2.pkl', 'rb') as input:
        test_IIR_nv2 = pickle.load(input)

    test_IIR = [np.concatenate((test_IIR_nv1[0], test_IIR_nv2[0])),
                np.concatenate((test_IIR_nv1[1], test_IIR_nv2[1])),
                np.concatenate((test_IIR_nv1[2], test_IIR_nv2[2]))]

    ####### Calculate Metrics #######

    print('Calculating metrics ...')

    # DL Metrics

    # Exp DRNN

    [X_test, y_test, y_pred] = test_DRNN

    SSD_values_DL_DRNN = SSD(y_test, y_pred)

    MAD_values_DL_DRNN = MAD(y_test, y_pred)

    PRD_values_DL_DRNN = PRD(y_test, y_pred)

    COS_SIM_values_DL_DRNN = COS_SIM(y_test, y_pred)


    # Exp FCN-DAE

    [X_test, y_test, y_pred] = test_FCN_DAE

    SSD_values_DL_FCN_DAE = SSD(y_test, y_pred)

    MAD_values_DL_FCN_DAE = MAD(y_test, y_pred)

    PRD_values_DL_FCN_DAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_FCN_DAE = COS_SIM(y_test, y_pred)

    # Exp CNN DAE

    [X_test, y_test, y_pred] = test_CNN_DAE
    y_pred = np.expand_dims(y_pred, -1)

    SSD_values_DL_CNN_DAE = SSD(y_test, y_pred)

    MAD_values_DL_CNN_DAE = MAD(y_test, y_pred)

    PRD_values_DL_CNN_DAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_CNN_DAE = COS_SIM(y_test, y_pred)

    # Exp Vanilla DAE

    [X_test, y_test, y_pred] = test_Vanilla_DAE
    y_pred = np.expand_dims(y_pred, -1)

    SSD_values_DL_Vanilla_DAE = SSD(y_test, y_pred)

    MAD_values_DL_Vanilla_DAE = MAD(y_test, y_pred)

    PRD_values_DL_Vanilla_DAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_Vanilla_DAE = COS_SIM(y_test, y_pred)

    # Exp Multibranch LANLD

    [X_test, y_test, y_pred] = test_Multibranch_LANLD

    SSD_values_DL_Multibranch_LANLD = SSD(y_test, y_pred)

    MAD_values_DL_Multibranch_LANLD = MAD(y_test, y_pred)

    PRD_values_DL_Multibranch_LANLD = PRD(y_test, y_pred)

    COS_SIM_values_DL_Multibranch_LANLD = COS_SIM(y_test, y_pred)

    # Exp ECA DAE

    [X_test, y_test, y_pred] = test_ECADAE

    SSD_values_DL_ECADAE = SSD(y_test, y_pred)

    MAD_values_DL_ECADAE = MAD(y_test, y_pred)

    PRD_values_DL_ECADAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_ECADAE = COS_SIM(y_test, y_pred)

    # Exp ADAE

    [X_test, y_test, y_pred] = test_ADAE

    SSD_values_DL_ADAE = SSD(y_test, y_pred)

    MAD_values_DL_ADAE = MAD(y_test, y_pred)

    PRD_values_DL_ADAE = PRD(y_test, y_pred)

    COS_SIM_values_DL_ADAE = COS_SIM(y_test, y_pred)

    # Digital Filtering

    # FIR Filtering Metrics
    [X_test, y_test, y_filter] = test_FIR

    SSD_values_FIR = SSD(y_test, y_filter)

    MAD_values_FIR = MAD(y_test, y_filter)

    PRD_values_FIR = PRD(y_test, y_filter)

    COS_SIM_values_FIR = COS_SIM(y_test, y_filter)


    # IIR Filtering Metrics (Best)
    [X_test, y_test, y_filter] = test_IIR

    SSD_values_IIR = SSD(y_test, y_filter)

    MAD_values_IIR = MAD(y_test, y_filter)

    PRD_values_IIR = PRD(y_test, y_filter)

    COS_SIM_values_IIR = COS_SIM(y_test, y_filter)


    ####### Results Visualization #######

    SSD_all = [
               SSD_values_DL_CNN_DAE,
               SSD_values_DL_Vanilla_DAE,
               SSD_values_DL_DRNN,
               SSD_values_DL_FCN_DAE,
               SSD_values_DL_Multibranch_LANLD,
               SSD_values_DL_ECADAE,
               SSD_values_DL_ADAE,
               SSD_values_FIR,
               SSD_values_IIR
               ]

    MAD_all = [
               MAD_values_DL_CNN_DAE,
               MAD_values_DL_Vanilla_DAE,
               MAD_values_DL_DRNN,
               MAD_values_DL_FCN_DAE,
               MAD_values_DL_Multibranch_LANLD,
               MAD_values_DL_ECADAE,
               MAD_values_DL_ADAE,
               MAD_values_FIR,
               MAD_values_IIR
               ]

    PRD_all = [
               PRD_values_DL_CNN_DAE,
               PRD_values_DL_Vanilla_DAE,
               PRD_values_DL_DRNN,
               PRD_values_DL_FCN_DAE,
               PRD_values_DL_Multibranch_LANLD,
               PRD_values_DL_ECADAE,
               PRD_values_DL_ADAE,
               PRD_values_FIR,
               PRD_values_IIR
               ]

    COS_SIM_all = [
                   COS_SIM_values_DL_CNN_DAE,
                   COS_SIM_values_DL_Vanilla_DAE,
                   COS_SIM_values_DL_DRNN,
                   COS_SIM_values_DL_FCN_DAE,
                   COS_SIM_values_DL_Multibranch_LANLD,
                   COS_SIM_values_DL_ECADAE,
                   COS_SIM_values_DL_ADAE,
                   COS_SIM_values_FIR,
                   COS_SIM_values_IIR,
                   ]

    if timing_names is not None:
        Exp_names = timing_names
    else:
        Exp_names = EXPERIMENTS + ['FIR Filter', 'IIR Filter']
    
    metrics = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    metric_values = [SSD_all, MAD_all, PRD_all, COS_SIM_all]

    # Metrics table
    vs.generate_table(metrics, metric_values, Exp_names)

    # Timing table
    timing_var = ['training', 'test']
    vs.generate_table_time(timing_var, timing, Exp_names, gpu=True)

    ################################################################################################################
    # Segmentation by noise amplitude
    rnd_test = np.load('rnd_test.npy')

    rnd_test = np.concatenate([rnd_test, rnd_test])

    segm = [0.2, 0.6, 1.0, 1.5, 2.0]  # real number of segmentations is len(segmentations) - 1
    SSD_seg_all = []
    MAD_seg_all = []
    PRD_seg_all = []
    COS_SIM_seg_all = []

    for idx_exp in range(len(Exp_names)):
        SSD_seg = [None] * (len(segm) - 1)
        MAD_seg = [None] * (len(segm) - 1)
        PRD_seg = [None] * (len(segm) - 1)
        COS_SIM_seg = [None] * (len(segm) - 1)
        for idx_seg in range(len(segm) - 1):
            SSD_seg[idx_seg] = []
            MAD_seg[idx_seg] = []
            PRD_seg[idx_seg] = []
            COS_SIM_seg[idx_seg] = []
            for idx in range(len(rnd_test)):
                # Object under analysis (oua)
                # SSD
                oua = SSD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    SSD_seg[idx_seg].append(oua)

                # MAD
                oua = MAD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    MAD_seg[idx_seg].append(oua)

                # PRD
                oua = PRD_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    PRD_seg[idx_seg].append(oua)

                # COS SIM
                oua = COS_SIM_all[idx_exp][idx]
                if rnd_test[idx] > segm[idx_seg] and rnd_test[idx] < segm[idx_seg + 1]:
                    COS_SIM_seg[idx_seg].append(oua)

        # Processing the last index
        # SSD
        SSD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = SSD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                SSD_seg[-1].append(oua)

        SSD_seg_all.append(SSD_seg)  # [exp][seg][item]

        # MAD
        MAD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = MAD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                MAD_seg[-1].append(oua)

        MAD_seg_all.append(MAD_seg)  # [exp][seg][item]

        # PRD
        PRD_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = PRD_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                PRD_seg[-1].append(oua)

        PRD_seg_all.append(PRD_seg)  # [exp][seg][item]

        # COS SIM
        COS_SIM_seg[-1] = []
        for idx in range(len(rnd_test)):
            # Object under analysis
            oua = COS_SIM_all[idx_exp][idx]
            if rnd_test[idx] > segm[-2]:
                COS_SIM_seg[-1].append(oua)

        COS_SIM_seg_all.append(COS_SIM_seg)  # [exp][seg][item]

    # Printing Tables
    seg_table_column_name = []
    for idx_seg in range(len(segm) - 1):
        column_name = str(segm[idx_seg]) + ' < noise < ' + str(segm[idx_seg + 1])
        seg_table_column_name.append(column_name)

    # SSD Table
    SSD_seg_all = np.array(SSD_seg_all)
    SSD_seg_all = np.swapaxes(SSD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the SSD metric')
    vs.generate_table(seg_table_column_name, SSD_seg_all, Exp_names)

    # MAD Table
    MAD_seg_all = np.array(MAD_seg_all)
    MAD_seg_all = np.swapaxes(MAD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the MAD metric')
    vs.generate_table(seg_table_column_name, MAD_seg_all, Exp_names)

    # PRD Table
    PRD_seg_all = np.array(PRD_seg_all)
    PRD_seg_all = np.swapaxes(PRD_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the PRD metric')
    vs.generate_table(seg_table_column_name, PRD_seg_all, Exp_names)

    # COS SIM Table
    COS_SIM_seg_all = np.array(COS_SIM_seg_all)
    COS_SIM_seg_all = np.swapaxes(COS_SIM_seg_all, 0, 1)

    print('\n')
    print('Printing Table for different noise values on the COS SIM metric')
    vs.generate_table(seg_table_column_name, COS_SIM_seg_all, Exp_names)

    ##############################################################################################################
    #TODO: FIX THIS!
    # Metrics graphs
    #vs.generate_hboxplot(SSD_all, Exp_names, 'SSD (au)', log=False, set_x_axis_size=(0, 100.1))
    #vs.generate_hboxplot(MAD_all, Exp_names, 'MAD (au)', log=False, set_x_axis_size=(0, 3.01))
    #vs.generate_hboxplot(PRD_all, Exp_names, 'PRD (au)', log=False, set_x_axis_size=(0, 100.1))
    #vs.generate_hboxplot(COS_SIM_all, Exp_names, 'Cosine Similarity (0-1)', log=False, set_x_axis_size=(0, 1))

    ################################################################################################################
    # Visualize signals

    signals_index = np.array([110, 210, 410, 810, 1610, 3210, 6410, 12810]) + 10

    ecg_signals2plot = []
    ecgbl_signals2plot = []
    dl_signals2plot = []
    fil_signals2plot = []

    signal_amount = 10

    [X_test, y_test, y_pred] = test_ADAE
    for id in signals_index:
        ecgbl_signals2plot.append(X_test[id])
        ecg_signals2plot.append(y_test[id])
        dl_signals2plot.append(y_pred[id])

    [X_test, y_test, y_filter] = test_IIR
    for id in signals_index:
        fil_signals2plot.append(y_filter[id])

    for i in range(len(signals_index)):
        vs.ecg_view(ecg=ecg_signals2plot[i],
                    ecg_blw=ecgbl_signals2plot[i],
                    ecg_dl=dl_signals2plot[i],
                    ecg_f=fil_signals2plot[i],
                    signal_name=None,
                    beat_no=None)

        vs.ecg_view_diff(ecg=ecg_signals2plot[i],
                         ecg_blw=ecgbl_signals2plot[i],
                         ecg_dl=dl_signals2plot[i],
                         ecg_f=fil_signals2plot[i],
                         signal_name=None,
                         beat_no=None)

    #make ecg plots
    test_outputs = [test_ADAE, test_DRNN, test_Multibranch_LANLD, test_IIR,
                    test_ECADAE, test_FCN_DAE]
    models_list = ['CBAM-DAE', 'DRNN', 'DeepFilter', 'IIR',
                   'ACDAE', 'FCN-DAE']
    #index 9582 is sel820
    generate_figs(test_outputs, models_list,
                  start_index=9582, n=8)
    return

def get_values_plot(values, start_index=1000, n=8):
    vals = values[start_index:(start_index+n),...]
    vals = vals.reshape(n*512)
    return vals

def make_figure(values, metrics_list, labels_list,
                label='', start_index=1000, n=8, **kwargs):
    #make the legend info
    values = get_values_plot(values, start_index=start_index, n=n)
    legend_info = []
    for metric, label in zip(metrics_list, labels_list):
        leg_inf = f'{label} = {metric}'
        legend_info.append(leg_inf)
    vs.ecg_plot(values, legend_info, label=label, **kwargs)
    return

def generate_figs(test_outputs, models_list,
                  start_index=1000, n=8, **kwargs):
    #hardcode label for now
    #first make one plot of baseline + noise added
    labels_list = ['SSD', 'MAD', 'PRD', 'COS_SIM']
    plot_base_noise = False
    for test_output, model in zip(test_outputs, models_list):
        if not plot_base_noise:
            #make baseline plots
            X_test, y_test, y_pred = test_output
            make_figure(X_test, [], [], label='',
                        start_index=start_index, n=n,
                        title='ECG + Noise', savename='ecg_noise.png')
            make_figure(y_test, [], [], label='', start_index=start_index,
                        n=n, title='Original ECG Signal', savename='ecg_orig.png')
            plot_base_noise = True
        _, y_test, y_pred = test_output
        y_test2 = y_test[start_index:(start_index+n),...]
        y_pred2 = y_pred[start_index:(start_index+n),...]
        ssd = round(SSD(y_test2, y_pred2).sum(), 3)
        mad = round(MAD(y_test2, y_pred2).max(), 3)
        prd = round(PRD(y_test2, y_pred2).mean(), 3)
        cs = round(COS_SIM(y_test2, y_pred2).mean(), 3)
        metric_list = [ssd, mad, prd, cs]
        make_figure(y_pred, metric_list, labels_list, start_index=start_index,
                    n=n, title=f'Filtered ECG using {model}',
                    savename=f'ecg_{model}.png')
    return

if __name__ == '__main__':
    generate_results()
