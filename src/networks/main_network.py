#! /usr/bin/env python

import pdb

from ftplib import all_errors
import pdb
import sys
sys.path.append("..")
import time
from collections import OrderedDict
from utilities import read_hdf5 #, write_hdf5, remove_points, normalize
import pylab as plt
import matplotlib
import numpy as np
import math as m

from utilities.parameters import *

import initializer
import data_manager
import network_manager

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  ### run code on CPU it is just fater for these size of networks!!!

def cart2sph(pos):
    pos_sph = np.array(np.squeeze(pos))
    if len(np.shape(pos_sph)) == 1:
        pos_sph = np.expand_dims(pos_sph, axis=0)
    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    ele = np.arcsin(np.divide(pos[:,2], r))
    azi = np.arcsin(np.divide(pos[:,1], np.multiply(r, np.cos(ele))))
    ele = ele/m.pi * 180.0
    azi = azi/m.pi * 180.0
    pos_sph[:,0] = np.squeeze(np.round(azi))
    pos_sph[:,1] = np.squeeze(np.round(ele))
    pos_sph[:,2] = np.squeeze(np.round(r,1))
    return np.expand_dims(pos_sph, axis=2)


def lsd(y, yhat, length=32):
    if (np.shape(yhat) == np.shape([1])):
        length = 1
    else:
        y_tmp = np.squeeze(y)
        yhat_tmp = np.squeeze(yhat)
        if length != 32:
            y = y_tmp[:length]
            yhat = yhat_tmp[:length]
        else:
            y = y_tmp
            yhat = yhat_tmp

    numer = 0
    denom = 0
    for i in range(length):
        numer = numer + (y[i] - yhat[i])**2.0
        denom += 1
    lsd = m.sqrt(numer/denom)
    return lsd
    

# def defunct_lsd(y, yhat, length=32):
#     return (lsd_lsd(y,yhat))
#     #return (lsd_lsd(y,yhat, length))
#     y_tmp = np.squeeze(y)
#     yhat_tmp = np.squeeze(yhat)
#     if length != 32:
#         y = y_tmp[:length]
#         yhat = yhat_tmp[:length]
#     else:
#         y = y_tmp
#         yhat = yhat_tmp
#     numer = 0
#     denom = 0
#     for i in range(length):
#         #numer = numer + (20*np.log10(np.divide(np.abs(y[i]),np.abs(yhat[i]))))**2.0
#         if y[i] < 0.0001:
#             y[i] = 0.0001
#         if yhat[i] < 0.0001:
#             yhat[i] = 0.0001
#         x = 20*np.log10(np.abs(y[i]))
#         xhat = 20*np.log10(np.abs(yhat[i]))
#         numer = numer + (x - xhat)**2.0
#         # numer = numer + (y[i] - yhat[i])**2.0
#         denom += 1
#     lsd = m.sqrt(numer/denom)
#     return lsd


def predict_all_lsd(all_models, inputs, all_outputs, fs=44.1, names=[], args=None, pos=None, test_idxs=None, lsd_user=0, left_right = [True, False], original=False):
    if args['db'] == 'scut':
        num_rows = 29
        num_cols = 170
        num_dists = 10
        num_cols_per_dist = int(num_cols/num_dists)
        frac = .03
        lsd_offset =  0
    if args['db'] == 'cipic':
        num_rows=50
        num_cols=25
        num_dists = 1
        num_cols_per_dist = int(num_cols/num_dists)
        lsd_offset =  num_rows *num_cols * lsd_user
        zero_idxs = np.concatenate([np.arange(lsd_offset + 8, lsd_offset + num_rows*num_cols, 50), np.arange(lsd_offset + num_rows*num_cols-10, lsd_offset + 39, -50)])
        frac = .05
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=7)
    pos_inputs = inputs['position']
    head_inputs = inputs['head']
    ear_inputs = inputs['ear']
    zeroazi_plot_pos = []
    zeroazi_plot_pos_cart = []
    lsd_0_azi = True
    for name in names:
        print ("Getting all LSD for " + name)
        if original == True:
            outputs = all_outputs['C_' + name]
        else: 
            outputs = all_outputs[name]
        lsds_l_11k_test = []
        lsds_r_11k_test = []
        lsds_l_test = []
        lsds_r_test = []
        lsds_l = []
        lsds_r = []
        lsds_l = np.zeros((num_rows, num_cols))
        lsds_r = np.zeros((num_rows, num_cols))
        lsds_l_11k = np.zeros((num_rows, num_cols))
        lsds_r_11k = np.zeros((num_rows, num_cols))
        high_idxs_l = {}
        high_idxs_r = {}
        thresh = 5.0
        for i in range(0, num_cols):
            for j in range(0, num_rows):
                idx = lsd_offset + i*num_rows + j
                curr_input_pos = np.array(pos_inputs[idx]).T
                curr_input_head = np.array(head_inputs[idx]).T
                curr_input_ear_l = np.expand_dims(np.array(ear_inputs[idx,:,0]), axis=0)
                curr_input_ear_r = np.expand_dims(np.array(ear_inputs[idx,:,1]), axis=0)
                length = 32
                if name  in ['magl']:
                    model = all_models[name]
                    pred_data = model.model.predict([curr_input_pos, curr_input_head, curr_input_ear_l], verbose=0)
                    curr_pred_data = pred_data[0]
                    left_right = [True, False]
                elif name  in ['maglmean']:
                    # The code for maglmean is not completed - as the input needs to be adjusted as well
                    model = all_models['magl']
                    pred_data = model.model.predict([curr_input_pos, curr_input_head, curr_input_ear_l], verbose=0)
                    curr_pred_data = pred_data[1]
                    left_right = [True, False]
                    lsd_0_azi = False
                elif name in ['magr']:
                    model = all_models[name]
                    pred_data = model.model.predict([curr_input_pos, curr_input_head, curr_input_ear_r], verbose=0)
                    curr_pred_data = pred_data[0]
                    left_right = [False, True]
                else:
                    model = all_models[name]
                    curr_pred_data = model.model.predict([curr_input_pos, curr_input_head, curr_input_ear_l, curr_input_ear_r], verbose=0)
                    left_right = [True, True]
                if left_right[0]:
                    lsds_l[j, i] = lsd(outputs[idx,:,0], curr_pred_data[0])
                    lsds_l_11k[j, i] = lsd(outputs[idx,:,0], curr_pred_data[0], 18)
                    if (lsds_l[j, i] > thresh):
                        high_idxs_l[idx] = lsds_l[j,i]
                    if left_right[1]:
                        lsds_r[j, i] = lsd(outputs[idx,:,1], curr_pred_data[1])
                        lsds_r_11k[j, i] = lsd(outputs[idx,:,1], curr_pred_data[1], 18)
                        if (lsds_r[j, i] > thresh):
                            high_idxs_r[idx] = lsds_r[j,i]
                elif left_right[1]:
                    lsds_r[j, i] = lsd(outputs[idx,:,1], curr_pred_data[0])
                    lsds_r_11k[j, i] = lsd(outputs[idx,:,1], curr_pred_data[0], 18)
                    if (lsds_r[j, i] > thresh):
                        high_idxs_r[idx] = lsds_r[j,i]
                if idx in test_idxs:
                    if left_right[0]:
                        lsds_l_test.append(lsd(outputs[idx,:,0], curr_pred_data[0]))
                        lsds_l_11k_test.append(lsd(outputs[idx,:,0], curr_pred_data[0], 18))
                    if left_right[1]:
                        lsds_r_test.append(lsd(outputs[idx,:,1], curr_pred_data[1]))
                        lsds_r_11k_test.append(lsd(outputs[idx,:,1], curr_pred_data[1], 18))
        #fig_lsdall_l = plt.figure()
        #fig_lsdall_l_11k = plt.figure()
        #fig_lsdall_r = plt.figure()
        #fig_lsdall_r_11k = plt.figure()
        if args['db'] == 'scut':
            fig_lsdall_l, axes_l = plt.subplots(nrows=2, ncols=1)
            fig_lsdall_r, axes_r = plt.subplots(nrows=2, ncols=1)
            fig_lsdall_l.subplots_adjust(hspace=0, wspace=0)
            fig_lsdall_r.subplots_adjust(hspace=0, wspace=0)
        if args['db'] == 'cipic':
            fig_lsdall_l, axes_l = plt.subplots(nrows=1, ncols=2)
            fig_lsdall_r, axes_r = plt.subplots(nrows=1, ncols=2)
            fig_lsd0l = plt.figure()
            fig_lsd0r= plt.figure()

        if args['db'] == 'cipic':
            curr_zero_azi = np.squeeze(cart2sph(np.array(pos_inputs[zero_idxs])))
            zeroazi_plot_pos = [int(pos[0]) for pos in curr_zero_azi]

            curr_input_pos = np.array(pos_inputs[zero_idxs])
            curr_input_head = np.array(head_inputs[zero_idxs])
            curr_input_ear_l = np.expand_dims(np.array(ear_inputs[zero_idxs,:,0]), axis=0)
            curr_input_ear_r = np.expand_dims(np.array(ear_inputs[zero_idxs,:,1]), axis=0)
            print ("left_right[0] = ", left_right[0])
            print ("left_right[1] = ", left_right[1])
            if (left_right[1] and left_right[0]):
                curr_pred_data = model.model.predict([np.squeeze(curr_input_pos), np.squeeze(curr_input_head), np.squeeze(curr_input_ear_l), np.squeeze(curr_input_ear_r)])
                pred_l = curr_pred_data[0]
                pred_r = curr_pred_data[1]
            elif left_right[0]:
                curr_pred_data = model.model.predict([np.squeeze(curr_input_pos), np.squeeze(curr_input_head), np.squeeze(curr_input_ear_l)])
                pred_l = curr_pred_data[0]
                np.set_printoptions(threshold=np.inf)
            elif left_right[1]:
                curr_pred_data = model.model.predict([np.squeeze(curr_input_pos), np.squeeze(curr_input_head), np.squeeze(curr_input_ear_r)])
                pred_r = curr_pred_data[1]
            else:
                print ("predict_all_lsd: No Ear was specified")
            lsds_l_1d = []
            lsds_l_1d_11k = []
            lsds_r_1d = []
            lsds_r_1d_11k = []

            if lsd_0_azi:
                if left_right[0]:
                    for (i, (idx, pred)) in enumerate(zip(zero_idxs, pred_l)):
                        lsds_l_1d.append(lsd(outputs[idx,:,0], pred))
                        lsds_l_1d_11k.append(lsd(outputs[idx,:,0], pred,18))

                if left_right[1]:
                    for (i, (idx, pred)) in enumerate(zip(zero_idxs, pred_r)):
                        lsds_r_1d.append(lsd(outputs[idx,:,1], pred))
                        lsds_r_1d_11k.append(lsd(outputs[idx,:,1], pred,18))
                ticks=[0,   3,  5,  7,  8,  9, 10, 11, 12, 13, 14, 15,16,17,  18,  19,  20,  21,  22,  23,  24,  25,  27,  29,  32,  33,  36,  38,  40,  41,  42,  43,  44,  45,  46,  47, 48,49,50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 62, 65]
                     #[80, 65, 55, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0, -5, -10, -15, -20, -25, -30, -35, -40, -45, -55, -65, -80, -80, -65, -55, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 55, 65, 80]

        if left_right[0]:
            axl = axes_l[0]
            caxl = axl.imshow(lsds_l, cmap='jet', norm=normalize)
            axl.set_title("Left Ear LSD Full")
            if args['db'] == 'scut':
                axl.set_xticks(np.arange(0,170,17))
                axl.set_xticklabels([1.0, .9, .8, .7, .6, .5, .4, .3, .25, .2])
                axl.set_xlabel("Radial Distance (m)")
                axl.axes.get_yaxis().set_ticklabels([])
            if args['db'] == 'cipic':
                axl.set_yticks([8, 25, 42])
                axl.set_yticklabels([0, 90, 180])
                axl.set_xticks([0, 12, 24])
                axl.set_xticklabels([80, 0, -80])
                axl.set_xlabel("Azimuthal Angle (deg)")
                axl.set_ylabel("Elevation Angle (deg)")

            axl_11k = axes_l[1]
            axl_11k.set_title("Left Ear LSD <11k")
            caxl_11k = axl_11k.imshow(lsds_l_11k, cmap='jet', norm=normalize)
            cbarl = fig_lsdall_l.colorbar(caxl_11k, fraction=frac, ax=axes_l.ravel().tolist())
            cbarl.ax.set_ylabel(' dB', rotation=360)
            if args['db'] == 'scut':
                axl_11k.set_xticks(np.arange(0,170,17))
                axl_11k.set_xticklabels([1.0, .9, .8, .7, .6, .5, .4, .3, .25, .2])
                axl_11k.set_xlabel("Radial Distance (m)")
                axl_11k.axes.get_yaxis().set_ticklabels([])
            if args['db'] == 'cipic':
                axl_11k.set_yticks([8, 25, 42])
                axl_11k.set_yticklabels([0, 90, 180])
                axl_11k.set_xticks([0, 12, 24])
                axl_11k.set_xticklabels([80, 0, -80])
                axl_11k.set_xlabel("Azimuthal Angle (deg) <11K")
                #axl_11k.set_ylabel("Elevation Angle (deg)")
            fig_lsdall_l.savefig("./figures/lsdall_l" + args['db'] + ".eps",bbox_inches='tight')
            #lsds_l_avg = np.mean(np.mean(lsds_l))
            #lsds_l_11k_avg = np.mean(np.mean(lsds_l_11k))
            lsds_l_avg = []
            lsds_l_11k_avg = []
            lsds_l_test_avg = []
            lsds_l_11k_test_avg = []
            for i in range(num_dists):
                lsds_l_avg.append(np.mean(np.mean(lsds_l[:,i*num_cols_per_dist:(i+1)*num_cols_per_dist])))
                lsds_l_11k_avg.append(np.mean(np.mean(lsds_l_11k[:,i*num_cols_per_dist:(i+1)*num_cols_per_dist]))) #            axl.set_title('Full Bandwidth LSD')
                lsds_l_test_avg.append(np.mean(lsds_l_test[:])) #            axl.set_title('Full Bandwidth LSD')
                lsds_l_11k_test_avg.append(np.mean(lsds_l_11k_test[:])) #            axl.set_title('Full Bandwidth LSD')
#            axl_11k.set_title('<11k LSD')
            print ("Left " + name + " [full, <11k]: [" + str(lsds_l_avg) + ", "+ str(lsds_l_11k_avg) + "]")
            if test_idxs is not None:
                print ("Left " + name + " test [full, <11k]: [" + str(lsds_l_test_avg) + ", "+ str(lsds_l_11k_test_avg) + "]")

            if lsd_0_azi:
                if args['db'] == 'cipic':
                    axl0 = fig_lsd0l.add_subplot(111)
                    axl0.plot(ticks, lsds_l_1d, 'bo')
                    axl0.plot(ticks, lsds_l_1d, 'b', label="Full Bandwidth")
                    axl0.plot(ticks, lsds_l_1d_11k, 'go')
                    axl0.plot(ticks, lsds_l_1d_11k, 'g--', label="< 11 kHz")
                    axl0.set_xticks(ticks)
                    axl0.set_xticklabels(zeroazi_plot_pos)
                    axl0.autoscale(enable=True, axis='x', tight=True)
                    axl0.legend()
                    axl0.grid(True)
                    axl0.set_xlabel("Azimuthal Angle (deg)")
                    axl0.set_ylabel("Log Spectral Distortion (dB)")
                    axl0.set_title("Left Ear Spectral Distortion")
            
        if left_right[1]:
            axr = axes_r[0]
            axr.set_title("Right Ear LSD Full")
            caxr = axr.imshow(lsds_r, cmap='jet', norm=normalize)
            if args['db'] == 'scut':
                axr.set_xticks(np.arange(0,170,17))
                axr.set_xticklabels([1.0, .9, .8, .7, .6, .5, .4, .3, .25, .2])
                axr.set_xlabel("Radial Distance (m)")
                axr.axes.get_yaxis().set_ticklabels([])
            if args['db'] == 'cipic':
                axr.set_yticks([8, 25, 42])
                axr.set_yticklabels([0, 90, 180])
                axr.set_xticks([0, 12, 24])
                axr.set_xticklabels([80, 0, -80])
                axr.set_xlabel("Azimuthal Angle (deg)")
                axr.set_ylabel("Elevation Angle (deg)")

            axr_11k = axes_r[1]
            axr_11k.set_title("Right Ear LSD < 11 kHz")
            caxr_11k = axr_11k.imshow(lsds_r_11k, cmap='jet', norm=normalize)
            cbarr = fig_lsdall_r.colorbar(caxr_11k, fraction=frac, ax=axes_r.ravel().tolist())
            cbarr.ax.set_ylabel(' dB', rotation=360)
            if args['db'] == 'scut':
                axr_11k.set_xticks(np.arange(0,170,17))
                axr_11k.set_xticklabels([1.0, .9, .8, .7, .6, .5, .4, .3, .25, .2])
                axr_11k.set_xlabel("Radial Distance (m)")
                axr_11k.axes.get_yaxis().set_ticklabels([])
            if args['db'] == 'cipic':
                axr_11k.set_yticks([8, 25, 42])
                axr_11k.set_yticklabels([0, 90, 180])
                axr_11k.set_xticks([0, 12, 24])
                axr_11k.set_xticklabels([80, 0, -80])
                axr_11k.set_xlabel("Azimuthal Angle (deg)")
#                axr_11k.set_ylabel("Elevation Angle (deg)")
            fig_lsdall_r.savefig("./figures/lsdall_r" + args['db'] + ".eps",bbox_inches='tight')
#            axr.set_title(name + ' Right Ear Full LSD')
#            axr_11k.set_title(name + ' Right Ear <11k LSD')
            #lsds_r_avg = np.mean(np.mean(lsds_r))
            #lsds_r_11k_avg = np.mean(np.mean(lsds_r_11k))
            lsds_r_avg = []
            lsds_r_11k_avg = []
            lsds_r_test_avg = []
            lsds_r_11k_test_avg = []
            for i in range(num_dists):
                lsds_r_avg.append(np.mean(np.mean(lsds_r[:,i*num_cols_per_dist:(i+1)*num_cols_per_dist])))
                lsds_r_11k_avg.append(np.mean(np.mean(lsds_r_11k[:,i*num_cols_per_dist:(i+1)*num_cols_per_dist])))
                lsds_r_test_avg.append(np.mean(lsds_r_test[:])) #            axl.set_title('Full Bandwidth LSD')
                lsds_r_11k_test_avg.append(np.mean(lsds_r_11k_test[:])) #            axl.set_title('Full Bandwidth LSD')
            print ("Right " + name + " [full, <11k]: [" + str(lsds_r_avg) + ", " +  str(lsds_r_11k_avg) + "]")
            if test_idxs is not None:
                print ("Right " + name + " test [full, <11k]: [" + str(lsds_r_test_avg) + ", "+ str(lsds_r_11k_test_avg) + "]")

            if lsd_0_azi:
                if args['db'] == 'cipic':
                    axr0 = fig_lsd0r.add_subplot(111)
                    axr0.plot(ticks, lsds_r_1d, 'bo')
                    axr0.plot(ticks, lsds_r_1d, 'b', label="Full Bandwidth")
                    axr0.plot(ticks, lsds_r_1d_11k, 'go')
                    axr0.plot(ticks, lsds_r_1d_11k, 'g--', label="< 11 kHz")
                    axr0.legend()
                    axr0.grid(True)
                    axr0.set_xticks(ticks)
                    axr0.set_xticklabels(zeroazi_plot_pos)
                    axr0.autoscale(enable=True, axis='x', tight=True)
                    axr0.set_xlabel("Azimuthal Angle (deg)")
                    axr0.set_ylabel("Log Spectral Distortion (dB)")
                    axr0.set_title("Right Ear Spectral Distortion")
    plt.show()


def predict(models, curr_pred_data_list, inputs, outputs, idx, axsl, axsr, fs=44.1, lsd_only=False, C_hrir=None):
    #Bar graph settings for mean and std
    width = 0.35
    #Get the position and anthro data for prediction
    if idx > np.shape(list(outputs.values())[0])[0]:
#         print ("Index to large for this data. Must less than %d" % np.shape(outputs.values())[1])
        print ("Index to large for this data. Must less than %d" % np.shape(list(outputs.values())[0])[1])
        return
    pos_inputs = inputs['position']
    head_inputs = inputs['head']
    ear_inputs = inputs['ear']
    curr_input_pos = np.array(pos_inputs[idx]).T
    curr_input_head = np.array(head_inputs[idx]).T
    curr_input_ear_l = np.expand_dims(np.array(ear_inputs[idx,:,0]), axis=0)
    curr_input_ear_r = np.expand_dims(np.array(ear_inputs[idx,:,1]), axis=0)
    nfft = np.shape(outputs[list(curr_pred_data_list.keys())[0]])[1]-1
    if nfft == 0:
        nfft = 64
    freqs = np.fft.rfftfreq(nfft*2) * fs
    mag_idx = []
    #Get the predictions
    for (i, k) in enumerate(curr_pred_data_list.keys()):
        #Input to real and imag models are different 
#        if k in ['real', 'imag']:
#            curr_pred_data = models[k].model.predict([curr_input_pos])
        if k  in ['magl', 'maglmean', 'maglstd']:
            pred_data = models['magl'].model.predict([curr_input_pos, curr_input_head, curr_input_ear_l])
            curr_pred_data = {}
            curr_pred_data['magl'] = pred_data[0]
            curr_pred_data['maglmean'] = pred_data[1]
            curr_pred_data['maglstd'] = pred_data[2]
        elif k in ['magr', 'magrmean', 'magrstd']:
            pred_data = models['magr'].model.predict([curr_input_pos, curr_input_head, curr_input_ear_r])
            curr_pred_data = {}
            curr_pred_data['magr'] = pred_data[0]
            curr_pred_data['magrmean'] = pred_data[1]
            curr_pred_data['magrstd'] = pred_data[2]
        elif k in ['magtotal', 'magtotalmean', 'magtotalstd']:
            pred_data = models[k].model.predict([curr_input_pos, curr_input_head, curr_input_ear_l, curr_input_ear_r])
            curr_pred_data={}
            curr_pred_data['magtotal'] = pred_data[0:2]
            curr_pred_data['magtotalmean'] = pred_data[2:4]
            curr_pred_data['magtotalstd'] = pred_data[4:5]
        else:
            curr_pred_data = models[k].model.predict([curr_input_pos, curr_input_head, curr_input_ear_l, curr_input_ear_r])
        if isinstance(curr_pred_data, dict):
            curr_pred_data_list[k] = curr_pred_data[k]
        else:
            curr_pred_data_list[k] = curr_pred_data
    for (i, (k, v)) in enumerate(curr_pred_data_list.items()):
        v = np.array(v)
        #Print the mean and std values
        #Add the difference to real and imag
#        if (k in ['real', 'imag']) and (k+'diff' in models_to_predict):
#            v = (v+curr_pred_data_list[k+'diff'])
        #Renormalize the predictions: (predictions + difference) * std + mean
        if (k in models_to_renormalize):
            if 'mag' in k:
                if ('magl' in models.keys()) and ('magr' in models.keys()):
                    v[0] = v[0]*curr_pred_data_list['maglstd']+curr_pred_data_list['maglmean']
                    v[1] = v[1]*curr_pred_data_list['magrstd']+curr_pred_data_list['magrmean']
            elif (k+'std' in models.keys()) and (k+'mean' in models.keys()):
                v = v*curr_pred_data_list[k+'std']+curr_pred_data_list[k+'mean']
        #Plot the predictions
        if ('mean' in k) or ('std' in k):
#            if ('magmeanl' in k):
#                d = np.mean(curr_pred_data_list['magl'][0])
#                axsl[i].bar(0, d, width, color='blue', label=k+' Prediction')
#                axsr[i].bar(0, d, width, color='blue', label=k+' Prediction')
#            elif ('magstdl' in k):
#                d = np.std(curr_pred_data_list['magl'][0])
#                axsl[i].bar(0, d, width, color='blue', label=k+' Prediction')
#                axsr[i].bar(0, d, width, color='blue', label=k+' Prediction')
            if ('magmeanl' in k) or ('magstdl' in k) or ('maglmean' in k) or ('maglstd' in k):
                axsl[i].bar(0, v[0], width, color='blue', label=k+' Prediction')
            elif ('magmeanr' in k) or ('magstdr' in k) or ('magrmean' in k) or ('magrstd' in k):
                axsr[i].bar(0, v[0], width, color='blue', label=k+' Prediction')
            else:
                v = np.squeeze(v)
                print (k, np.shape(v))
                print("")
                print ('Predicted %s (l, r): (%f, %f)' % (k, round(v[0], 5), round(v[1],5)))
                print ('Actual %s (l, r):    (%f, %f)'% (k, round(outputs[k][idx,:,0][0],5), round(outputs[k][idx,:,1][0], 5)))
                print("")
                print("")
                axsl[i].bar(0, v[0], width, color='blue', label=k+' Prediction')
                axsr[i].bar(0, v[1], width, color='blue', label=k+' Prediction')
        elif ('magl' == k) or ('magreconl' == k):
            axsl[i].plot(freqs, v[0].T, 'b', label= k+' Prediction')
            print ("lsd = ", lsd(v[0].T, outputs[k][idx,:,0]))
        elif ('magr'== k) or ('magreconr' == k):
            axsr[i].plot(freqs, v[0].T, 'b', label= k+' Prediction')
            print ("lsd = ", lsd(v[0].T, outputs[k][idx,:,1]))
        else:
            axsl[i].plot(freqs[:np.shape(v[0].T)[0]], v[0].T, 'b', label= k+' Prediction')
            axsr[i].plot(freqs[:np.shape(v[1].T)[0]], v[1].T, 'b', label= k+' Prediction')
        #Plot the actual outputs over the predictions
#        if ('diff' in k):
#            axsl[i].plot(outputs['diff'][idx,:,0], 'r', label='Diff')
#            axsr[i].plot(outputs['diff'][idx,:,1], 'r', label='Diff')
        if ('mean' in k) or ('std' in k):
            if ('meanl' in k) or ('stdl' in k) or ('maglmean' in k) or ('maglstd' in k):
                axsl[i].bar(width, outputs[k][idx,:,0], width, color='red', label=k+' Actual')
            elif ('meanr' in k) or ('stdr' in k) or ('magrmean' in k) or ('magrstd' in k):
                axsr[i].bar(width, outputs[k][idx,:,1], width, color='red', label=k+' Actual')
            else:
                axsl[i].bar(width, outputs[k][idx,:,0], width, color='red', label=k+' Actual')
                axsr[i].bar(width, outputs[k][idx,:,1], width, color='red', label=k+' Actual')
        elif ('magl' == k) or ('magreconl'==k):
            axsl[i].plot(freqs, outputs[k][idx,:,0], 'r', label= k+' Actual')
#            print "LSD Left " + k + ': ' + str(lsd(outputs[k][idx,:,0], v[0]))
            mag_idx.append(i)
        elif ('magr' == k) or ('magreconr' == k):
            axsr[i].plot(freqs, outputs[k][idx,:,1], 'r', label= k+' Actual')
            if i not in mag_idx:
                mag_idx.append(i)
        else:
            # print ("outputs[k][:,:,0] = ", outputs[k][:,:,0])
            axsl[i].plot(freqs, outputs[k][idx,:,0], 'r', label= k+' Actual')
            if C_hrir is not None:
                axsl[i].plot(freqs, outputs['C_' + k][idx,:,0], 'g', label= k+' original')
            # axsl[i].set_ylim(-60,0)
            axsr[i].plot(freqs, outputs[k][idx,:,1], 'r', label= k+' Actual')
            if C_hrir is not None:
                axsr[i].plot(freqs, outputs['C_' + k][idx,:,1], 'g', label= k+' Actual')
            # axsr[i].set_ylim(-60,0)
            print ("LSD Left " + k + ': ' + str(lsd(outputs[k][idx,:,0], v[0], 31)))
            print ("LSD Right " + k + ': ' + str(lsd(outputs[k][idx,:,1], v[1], 31)))
            print("")
            mag_idx.append(i)
#        axsl[i].legend()
#        axsr[i].legend()
        for m in mag_idx:
            axsl[m].set_xlim(0, fs/2)
            axsr[m].set_xlim(0, fs/2)
    axsl[len(mag_idx)-1].set_xlabel('Frequency (kHz)')
    axsr[len(mag_idx)-1].set_xlabel('Frequency (kHz)')
    if lsd_only:
        return lsd_dict
    return curr_input_pos.T




def main():



    initializer.parseargs()
    initializer.init()
    
    for key, value in initializer.args.items():
        print(key, ' : ', value)

    train_anthro = initializer.train_anthro
    args = initializer.args
    subjects = initializer.subjects
    model_details = initializer.model_details
    model_details_prev = initializer.model_details_prev

    ##### this is got creating local variables so we can use them lately otherwise this doen't work due to py2 to py3 conversion. 
    models_to_train_1_loc = models_to_train_1
    models_to_train_2_loc = models_to_train_2
    models_to_train_3_loc = models_to_train_3
    models_to_train_4_loc = models_to_train_4
    models_to_predict_loc = models_to_predict
    models_to_eval_loc = models_to_eval
    finals_loc = finals

    if args['train_only'] is not None:
        models_to_train_1_loc = [ args['train_only']]
        models_to_train_2_loc = []
        models_to_train_3_loc = []
        models_to_train_4_loc = []
        models_to_predict_loc = models_to_train_1_loc
        models_to_eval_loc = models_to_predict_loc
        finals_loc = models_to_predict_loc

    #Read and format the data
    C_hrir= None
    if args['db'] == 'scut':
        if 'predict' in args['action']:
            hrir, pos, fs, nn = read_hdf5.getData(args['db'], subjects, db_filepath=args['db_path'], ring=args['ring'], ear=args['ear'], hrir_type=args['hrir_type'], radius=None)
        else:
            hrir, pos, fs, nn = read_hdf5.getData(args['db'], subjects, db_filepath=args['db_path'], ring=args['ring'], ear=args['ear'], hrir_type=args['hrir_type'], radius=SCUT_RADII)
    else:
        hrir, pos, fs, nn = read_hdf5.getData(args['db'], subjects, db_filepath=args['db_path'], ring=args['ring'], ear=args['ear'], hrir_type=args['hrir_type'], radius=None)
        t64_hrir, pos, fs, nn = read_hdf5.getData(args['db'], subjects, db_filepath=args['db_path'], ring=args['ring'], ear=args['ear'], hrir_type='trunc_64', radius=None)
        if args['C_hrir_type'] != args['hrir_type']:
            C_hrir, pos, fs, nn = read_hdf5.getData(args['db'], subjects, db_filepath=args['db_path'], ring=args['ring'], ear=args['ear'], hrir_type=args['C_hrir_type'], radius=None)

        
    data_manager.format_inputs_outputs(pos, hrir, nn, C_hrir=C_hrir)
    position, head, ear, magnitude, magnitude_raw, real, imaginary , C_magnitude, C_real, C_imaginary = data_manager.get_data()

    models = OrderedDict()
    #If we're training
    if('train' in args['action']):
        models = network_manager.make_all_models(models_to_train_1_loc, models)
        print("")
        print("")
        print("**********************************Loaded all ", models.keys()," **********************************")
        
#        for key, mod in models.iteritems():
        for mod in models_to_train_1_loc:
            #if it's a difference model, generate new training data from the last prediction of real or imag
            print("")
            print("")
            print("**********************************Training ", mod," **********************************")
            print("")
            
            # print(models[mod].model.summary()) 
            models[mod].train()
            
        models = network_manager.make_all_models(models_to_train_2_loc, models)
        if len(models_to_train_2_loc) != 0:
            print("**********************************Loaded all ", models.keys()," **********************************")
        for mod in models_to_train_2_loc:
            #if it's a difference model, generate new training data from the last prediction of real or imag
            print("")
            print("")
            print("**********************************Training ", mod," **********************************")
            print("")
            
            # print(models[mod].model.summary())
            models[mod].train()
        
        models = network_manager.make_all_models(models_to_train_3_loc, models)
        if len(models_to_train_3_loc) != 0:
            print("**********************************Loaded all ", models.keys()," **********************************")
        for mod in models_to_train_3:
            #if it's a difference model, generate new training data from the last prediction of real or imag
            print("")
            print("")
            print("**********************************Training ", mod," **********************************")
            print("")
            
            # print(models[mod].model.summary())
            models[mod].train()
        
        models = network_manager.make_all_models(models_to_train_4_loc, models)
        if len(models_to_train_4_loc) != 0:
            print("**********************************Loaded all ", models.keys()," **********************************")
        for mod in models_to_train_4_loc:
            #if it's a difference model, generate new training data from the last prediction of real or imag
            print("")
            print("")
            print("**********************************Training ", mod," **********************************")
            print("")
            
            # print(models[mod].model.summary())
            models[mod].train()    
    
    if ('train+' in args['action']):
        models_to_train = models_to_train_1_loc + models_to_train_2_loc + models_to_train_3_loc + models_to_train_4_loc
        models = network_manager.make_all_models(models_to_train, models, run_type='train+')
        for mod in models_to_train:
            #if it's a difference model, generate new training data from the last prediction of real or imag
            print("")
            print("")
            print("**********************************Training ", mod," **********************************")
            print("")
            
            # print(models[mod].model.summary())
            models[mod].train()    
    
    
    if ('train' in args['action']) or ('compile' in args['action']):
        #Which models/outputs to use for final model
        
        models = network_manager.make_all_models(finals_loc, models, run_type='compile')
        models_final = OrderedDict()
        for final_mod in finals_loc:
            models_final[final_mod] = models[final_mod]
        model = network_manager.get_model(models_final)
        model.save('./kmodels/'+model_details+'.h5');
        model.save('./kmodels/most_recent.h5');   
    
    
    #If we're eval
    if ('eval' in args['action']):
        all_models = network_manager.make_all_models(models_to_eval_loc, models, run_type='eval')
        for name, mod in all_models.items():
            
            if name in models_to_eval_loc:
                mod.evaluate()
        plt.show()
        
        
    if ('predict' in args['action']):   
        all_models = network_manager.make_all_models(models_to_predict_loc, models, run_type='predict')
        for name, mod in all_models.items():
            if name in models_to_predict_loc:
                models[name] = mod
        diff_inputs = OrderedDict([('position', position.getRawData())])
        #setup the inputs and outputs
        inputs_train = {}
        inputs_train['position'] = position.getRawData()
        pos_sph = data_manager.cart2sph(inputs_train['position'])
        inputs_train['head'] = head.getRawData()
        inputs_train['ear'] = ear.getRawData()
        outputs_train = {}
        if 'real' in models_to_renormalize:
            outputs_train['real'] = real.getRawData()
            if C_real is not None:
                outputs_train['C_real'] = C_real.getRawData()
        else:
            outputs_train['real'] = real.getNormalizedData()
            if C_real is not None:
                outputs_train['C_real'] = C_real.getNormalizedData()
        if 'imag' in models_to_renormalize:
            outputs_train['imag'] = imaginary.getRawData()
            if C_imaginary is not None:
                outputs_train['C_imag'] = C_imaginary.getRawData()
        else:
            outputs_train['imag'] = imaginary.getNormalizedData()
            if C_imaginary is not None:
                outputs_train['C_imag'] = C_imaginary.getNormalizedData()
        if 'magri' in models_to_renormalize:
            outputs_train['magri'] = magnitude.getRawData()
            if C_magnitude is not None:
                outputs_train['C_magri'] = C_magnitude.getRawData()
        else:
            outputs_train['magri'] = magnitude.getNormalizedData()
            if C_magnitude is not None:
                outputs_train['C_magri'] = C_magnitude.getNormalizedData()
        if 'mag' in models_to_renormalize:
            outputs_train['mag'] = magnitude.getRawData()
            if C_magnitude is not None:
                outputs_train['C_mag'] = C_magnitude.getRawData()
        else:
            outputs_train['mag'] = magnitude.getNormalizedData()
            if C_magnitude is not None:
                outputs_train['C_mag'] = C_magnitude.getNormalizedData()
        if 'magfinal' in models_to_renormalize:
            outputs_train['magfinal'] = magnitude.getRawData()
            if C_magnitude is not None:
                outputs_train['C_magfinal'] = C_magnitude.getRawData()
        else:
            outputs_train['magfinal'] = magnitude.getNormalizedData()
            if C_magnitude is not None:
                outputs_train['C_magfinal'] = C_magnitude.getNormalizedData()
        outputs_train['magtotal'] = magnitude_raw.getRawData()
        outputs_train['realmean'] = real.getMean()
        outputs_train['realstd'] = real.getStd()
        outputs_train['magreconl'] = magnitude.getRawData()
        outputs_train['magreconr'] = magnitude.getRawData()

        if C_magnitude is not None:
            outputs_train['C_magtotal'] = C_magnitude.getRawData()
            outputs_train['C_realmean'] = C_real.getMean()
            outputs_train['C_realstd'] = C_real.getStd()
            outputs_train['C_imagmean'] = C_imaginary.getMean()
            outputs_train['C_imagstd'] = C_imaginary.getStd()
            outputs_train['C_magmean'] = C_magnitude.getMean()
            outputs_train['C_magstd'] = C_magnitude.getStd()
            outputs_train['C_magl'] = C_magnitude.getRawData()
            outputs_train['C_magmeanl'] = C_magnitude.getMean()
            outputs_train['C_magstdl'] = C_magnitude.getStd()
            outputs_train['C_maglmean'] = C_magnitude.getMean()
            outputs_train['C_maglstd'] = C_magnitude.getStd()
            outputs_train['C_magr'] = C_magnitude.getRawData()
            outputs_train['C_magmeanr'] = C_magnitude.getMean()
            outputs_train['C_magstdr'] = C_magnitude.getStd()
            outputs_train['C_magrmean'] = C_magnitude.getMean()
            outputs_train['C_magrstd'] = C_magnitude.getStd()
            outputs_train['C_magrecon'] = C_magnitude.getRawData()
            outputs_train['C_magreconl'] = C_magnitude.getRawData()
            outputs_train['C_magreconr'] = C_magnitude.getRawData()

        inputs = inputs_train
        outputs = outputs_train
        curr_pred_data_list = OrderedDict()

        #setup the figure
        fig = plt.figure()
        axsl = []
        axsr = []
        plt.ion()
        num_plot_rows = len(models_to_predict_loc)
        num_plot_cols = 2 #2 ears
        tmp_list = []
        for i, model_name in enumerate(models_to_predict_loc):
            axsl.append(fig.add_subplot(num_plot_rows,num_plot_cols,(2*i)+1))
            axsr.append(fig.add_subplot(num_plot_rows,num_plot_cols,(2*i)+2))
            curr_pred_data_list[model_name] = None
        fig.canvas.draw()
        plt.show()

        curr_idx = 0
        #print test_inds_sorted
        while(True):
           
            #take in prediction index 1 and 2
            pred_nums = input("Prediction Indices: ")
            # TODO: calculate all the indexes of ts and iterate in them somehow (???)
            # if (test_available) and ('ts' in pred_nums):  # this won't work anymore in this new setiing
            #     print ("Using test subjects")
            #     print("")
            #     inputs = inputs_test
            #     outputs = outputs_test
            if 'ti' in pred_nums:
                print ("Test indices are: ")
                print (magnitude.getTestIdx())
            elif 'tr' in pred_nums:
                print ("Using training subjects")
                print ("")
                inputs = inputs_train
                outputs = outputs_train
            if 'lsd' in pred_nums:
                lsd_list = pred_nums.split()
                for x in lsd_list:
                    if x == 'lsd':
                        continue
                    else:
                        print ("x = ", x)
                        predict_all_lsd(models, inputs, outputs, names=['magtotal'], args=args, test_idxs=magnitude.getTestIdx(), lsd_user=int(x))#, 'magl', 'magr'])
                        # predict_all_lsd(models, inputs, outputs, names=['magl'], args=args, test_idxs=magnitude.getTestIdx(), lsd_user=int(x))#, 'magl', 'magr'])
                        if (C_hrir is not None):
                            predict_all_lsd(models, inputs, outputs, names=['magtotal'], args=args, test_idxs=magnitude.getTestIdx(), lsd_user=int(x), original=True)
                        # predict_all_lsd(models, inputs, outputs, names=['maglmean'], args=args, test_idxs=magnitude.getTestIdx(), lsd_user=int(x), left_right = [True, False])#, 'magl', 'magr'])
                        print ("after predict_all")
                        # predict_all_lsd(models, inputs, outputs, names=['magrmean'], args=args, test_idxs=magnitude.getTestIdx(), lsd_user=int(x), left_right = [True, True])#, 'magl', 'magr'])
                # predict_all_lsd(models, inputs, outputs, names=['magtotal'], args=args, test_idxs=magnitude.getTestIdx())#, 'magl', 'magr'], lsd_user=lsd_user)
                continue
            if 'meanl' in pred_nums:
                lsd_list = pred_nums.split()
                for x in lsd_list:
                    if x == 'meanl':
                        continue
                    else:
                        print ("x = ", x)
                        predict_all_lsd(models, inputs, outputs, names=['maglmean'], args=args, test_idxs=magnitude.getTestIdx(), lsd_user=int(x))#, 'magl', 'magr'])
                        print ("after predict_all")

                continue
            for ax in axsl:
                ax.clear()
            for ax in axsr:
                ax.clear()
            if (":" in pred_nums):
                print(pred_nums)
                rng = [int(s) for s in pred_nums.split(":") if s.isdigit()]
                print(rng)
                indices = range(rng[0], rng[1])
                print(indices)
            elif (";" in pred_nums):
                print (pred_nums)
                rng = [int(s) for s in pred_nums.split(";") if s.isdigit()]
                print(rng)
                if (len(rng) == 1):
                    indices = range(rng[0], rng[0] + 32 * 1250, 1250)
                else:
                    indices = range(rng[0], rng[0] + (rng[1]-1) * 1250, 1250)
                    print(indices)
            else:
                indices = [int(s) for s in pred_nums.split() if s.isdigit()]

            if '-' in pred_nums:
                curr_idx = curr_idx - 1
            if '' == pred_nums:
                curr_idx = curr_idx + 1

            if indices:
                for idx in indices:
                    print ("going to predict")
                    curr_input_pos = predict(models, curr_pred_data_list, inputs, outputs, idx, axsl, axsr,  C_hrir=C_hrir)
                    print ("predicted")
                curr_idx = idx
            else:   
                curr_input_pos = predict(models, curr_pred_data_list, inputs, outputs, curr_idx, axsl, axsr,  C_hrir=C_hrir)


            for i, model_name in enumerate(models_to_predict_loc):
                if i==0:
                    axsl[i].set_title("Left Index: %d\nPosition: (%.2f, %.2f, %.2f)\n(%.2f, %.2f, %.2f)\n%s" %(curr_idx, curr_input_pos[0], curr_input_pos[1], curr_input_pos[2], pos_sph[curr_idx,0], pos_sph[curr_idx,1], pos_sph[curr_idx,2], model_name))    
                    axsr[i].set_title("Right Index: %d\nPosition: (%.2f, %.2f, %.2f)\n(%.2f, %.2f, %.2f)\n%s" %(curr_idx, curr_input_pos[0], curr_input_pos[1], curr_input_pos[2], pos_sph[curr_idx,0], pos_sph[curr_idx,1], pos_sph[curr_idx,2], model_name))    
                else:
                    axsl[i].set_title(model_name)
                    axsr[i].set_title(model_name)
                if (model_name == 'magtotal'):
                    axsl[i].set_ylim(-70, 0)
                    axsr[i].set_ylim(-70, 0)

#                axsl[i].legend(loc='lower center')
                axsl[i].grid(True)
#                axsr[i].legend(loc='lower center')
                axsr[i].grid(True)

            plt.show()
            #plt.close()
        

if __name__ == '__main__':
    main()

