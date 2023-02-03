#! /usr/bin/env python

import pdb

# from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate, Lambda
# from keras.models import Sequential, Model, model_from_json, load_model
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects

import sys, os, shutil, argparse, h5py, time
import numpy as np
# import scipy.io as sio
# import pylab as plt
import math as m
# from collections import OrderedDict
from utilities.network_data import Data
from utilities.parameters import *
from utilities import read_hdf5
import initializer

import matplotlib.pyplot as plt

def sph2cart(pos):
    "pos should be (#subjs, #positions, [azi, ele, r])"
    pos_cart = np.array(pos)
    pos_cart[:,:,0] = np.multiply(pos[:,:,2], np.multiply(np.cos(pos[:,:,1]/180 * m.pi), np.cos(pos[:,:,0]/180 * m.pi)))
    pos_cart[:,:,1] = np.multiply(pos[:,:,2], np.multiply(np.cos(pos[:,:,1]/180 * m.pi), np.sin(pos[:,:,0]/180 * m.pi)))
    pos_cart[:,:,2] = np.multiply(pos[:,:,2], np.sin(pos[:,:,1]/180 * m.pi))
    return pos_cart

def cart2sph(pos):
    pos_sph = np.array(np.squeeze(pos))
    r = np.sqrt(pos[:,0]**2 + pos[:,1]**2 + pos[:,2]**2)
    
    ele = np.arcsin(np.divide(pos[:,2], r))
    
    azi = np.arcsin(np.clip(np.divide(pos[:,1], np.multiply(r, np.cos(ele))),a_min = -1, a_max = 1))
    ele = ele/m.pi * 180.0
    azi = azi/m.pi * 180.0
    pos_sph[:,0] = np.squeeze(np.round(azi))
    pos_sph[:,1] = np.squeeze(np.round(ele))
    pos_sph[:,2] = np.squeeze(np.round(r,1))
    return np.expand_dims(pos_sph, axis=2)

def set_valid_training(pers=False):
    global position, head, ear, magnitude, phase, real, imaginary
    position.setValidData(percent_valid_points, seed=validation_seed, pers=pers)
    magnitude.setValidData(percent_valid_points, seed=validation_seed, pers=pers)
    real.setValidData(percent_valid_points, seed=validation_seed, pers=pers)
    imaginary.setValidData(percent_valid_points, seed=validation_seed, pers=pers) 
    head.setValidData(percent_valid_points, seed=validation_seed, pers=pers)
    ear.setValidData(percent_valid_points, seed=validation_seed, pers=pers)

def format_inputs_outputs(pos, hrir, nn, ret_subjs=False):
    '''
    format_inputs_outputs(): (formats all training, validation, test data sets)
    this function received the raw hrir's and creates the different
    classes of globals such as (position, head, ear, magnitude, etc.
    within these classes, training, validation, and tests will be separated
    currenlty test subject values are separated into their own class but this will change
 
    parameters:
        pos:list (list of positions of hrirs)
        hrir:list (hrirs)
        nn:list (nearest neigbours  -- experimental, not used)
        ret_subjs:boolean (return test subjects indices or not)

    return:
        nothing unless ret_subjs is true
    '''

    global position, head, ear, magnitude, magnitude_raw, real, imaginary
    global position_test, head_test, ear_test, magnitude_test, magnitude_raw_test, real_test, imaginary_test
    global subj_removed
    args = initializer.args
    subjects = initializer.subjects
    #Include the anthropometric inputs
    hrir_local = np.asarray(hrir, dtype=np.float32)
    pos_local = np.asarray(pos, dtype=np.float32)
    nn_local = np.asarray(nn, dtype=np.float32)
    pos_local = sph2cart(pos_local)
    if initializer.train_anthro:
        head_inputs, ear_inputs, subj_with_nan = read_hdf5.getAnthroData(args['db'], subjects, db_filepath=args['db_path'], hrir_type=args['hrir_type'])
        head_local = np.repeat(np.expand_dims(head_inputs, axis=1), np.shape(pos)[1], axis=1)
        ear_local = np.repeat(np.expand_dims(ear_inputs, axis=1), np.shape(pos)[1], axis=1)
        head_local = np.delete(head_local, subj_with_nan, axis=0)
        ear_local = np.delete(ear_local, subj_with_nan, axis=0)
    else:
        head_local = np.zeros((np.shape(hrir_local)[0], np.shape(hrir_local)[1], num_head_params))
        ear_local = np.zeros((np.shape(hrir_local)[0], np.shape(hrir_local)[1], num_ear_params, 2))

#    ear_norm_local = np.zeros(np.shape(ear_norm))
#    pos_norm_local = np.divide(pos_local, Pos_div)
#    ear_norm_local[:,:,:,0] = np.divide(ear_local[:,:,:,0], LeftEar_div)
#    ear_norm_local[:,:,:,1] = np.divide(ear_local[:,:,:,1], LeftEar_div)
#    head_norm_local = np.divide(head_local, Head_div)
    
    num_subj = np.shape(hrir_local)[0]
    test_subj_num = int(m.floor(num_subj*.1))
    subj_removed = False
    if test_subj_num > 0:
        subj_removed = True
    np.random.seed(12345)
    test_subj_idx = np.random.randint(num_subj, size=test_subj_num)

    if subj_removed:
        pos_local_test = pos_local[test_subj_idx]
        position_test = Data(pos_local_test, nn_local, test_percent=0, normalize=False)
        head_test = Data(head_local[test_subj_idx], nn_local, test_percent=0, normalize=False)
        ear_test = Data(ear_local[test_subj_idx], nn_local, test_percent=0, normalize=False)

    pos_local = np.delete(pos_local, test_subj_idx, axis=0)
    position = Data(pos_local, nn_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)

    head_local = np.delete(head_local, test_subj_idx, axis=0)
    ear_local = np.delete(ear_local, test_subj_idx, axis=0)
    head = Data(head_local, nn_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)
    ear = Data(ear_local, nn_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)

    #Magnitude formatting
    scale = min(args['nfft'], np.shape(hrir_local)[2])
    #Get the HRTF
    outputs_fft = np.fft.rfft(hrir_local, args['nfft'], axis=2)
    outputs_complex = np.zeros(np.shape(outputs_fft), dtype=outputs_fft.dtype)
    for (s, h) in enumerate(outputs_fft):
        outputs_complex[s,:,:,:] = outputs_fft[s,:,:,:]/np.max(np.abs(outputs_fft[s,:,:,:]))
    outputs_mag = abs(outputs_complex) 
    outputs_mag = 20.0*np.log10(outputs_mag)

    if subj_removed:
        magnitude_test = Data(outputs_mag[test_subj_idx], nn_local, pos=pos_local_test, test_percent=0, normalize=False)
        magnitude_raw_test = Data(outputs_mag[test_subj_idx], nn_local, pos=pos_local_test, test_percent=0, normalize=False)
    outputs_mag = np.delete(outputs_mag, test_subj_idx, axis=0)
    # TODO - It seems that we no longer need magnitude raw as we no longer normalize the data
    magnitude = Data(outputs_mag, nn_local, pos=pos_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)
    magnitude_raw = Data(outputs_mag, nn_local, pos=pos_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)
    #Real formatting
    outputs_real = np.real(outputs_complex)
    if subj_removed:
        real_test = Data(outputs_real[test_subj_idx], nn_local, pos=pos_local_test, test_percent=0, normalize=False)
    outputs_real = np.delete(outputs_real, test_subj_idx, axis=0)
    real = Data(outputs_real, nn_local, pos=pos_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)
    #Imaginary formatting
    outputs_imag = np.imag(outputs_complex)
    if subj_removed:
        imaginary_test = Data(outputs_imag[test_subj_idx], nn_local, pos=pos_local_test, test_percent=0, normalize=False)
    outputs_imag = np.delete(outputs_imag, test_subj_idx, axis=0)
    imaginary = Data(outputs_imag, nn_local, pos=pos_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)
    set_valid_training(pers=subj_removed)
    if ret_subjs:
        return test_subj_idx

def get_data():
    global position, head, ear, magnitude, magnitude_raw, real, imaginary
    return position, head, ear, magnitude, magnitude_raw, real, imaginary

def get_test_subj_data():
    global position_test, head_test, ear_test, magnitude_test, magnitude_raw_test, real_test, imaginary_test
    return position_test, head_test, ear_test, magnitude_test, magnitude_raw_test, real_test, imaginary_test
