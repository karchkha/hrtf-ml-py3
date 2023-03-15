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

##python
#model reduction algorithm
import scipy
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.signal import ss2tf
from scipy.linalg import hankel

def IIR_app(hrir,k):
    #k=1; #order is one less than matlab
    num=1;

    # x=hrir[5,zz,:,0];
    L=len(hrir) #check if same with mat
    A=np.zeros((L-1,L-1))
    for column in range(L-2):
        A[column+1,column]=1

    B=np.zeros(L-1) #indeces start from 0
    B[0]=1
    B.shape = (L-1,1)
    C=hrir[1:] #need for transpose?
    D=np.zeros(1)
    D[0]=hrir[0]


    hankmat=hankel(hrir[1:])
    v, s, vt = np.linalg.svd(hankmat)
    vtrans=np.transpose(v)
    vpart = np.transpose(vtrans[:][0:k])
    spart= s[0:k]
    Ak= np.matmul(np.matmul(np.transpose(vpart),A),vpart)
    Bk= np.matmul(np.transpose(vpart),B);

    Ck=np.matmul(C,vpart);
    Dk=hrir[0];
    num,den=scipy.signal.ss2tf(Ak,Bk,Ck,Dk)
    z,p,gain=scipy.signal.tf2zpk(num,den)
    fig1, ax1 = plt.subplots()
    ax1.scatter(p.real,p.imag)
    ax1.scatter(z.real,z.imag)
    print ("p.real = ", p.real)
    print ("p.imag = ", p.imag)
    print ("z.real = ", z.real)
    print ("z.imag = ", z.imag)
    t=np.linspace(1, 360.0, num=360)
    ax1.plot(np.cos(t), np.sin(t), linewidth=0.05)

    fig2, ax2 = plt.subplots() #can put subplots in here
    print ("num = ", num)
    print ("den = ", den)
    w,h=scipy.signal.freqz(np.transpose(num),den, worN=32)
    print ("XXXXXw = ", np.shape(w))
    print ("w = ", w)
    print ("h = ", abs(h))
    IIR=ax2.plot(w, 20 * np.log10(abs(h)), 'r')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')

    w,h=scipy.signal.freqz(hrir)
    FIR=ax2.plot(w, 20 * np.log10(abs(h)), 'b')
    #plt.legend((IIR, FIR), ('IIR Mag response', 'FIR Mag response'))
    plt.show()



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

def set_valid_training():
    global position, head, ear, magnitude, real, imaginary, C_magnitude, C_real, C_imaginary
    position.setValidData(percent_valid_points, seed=validation_seed)
    magnitude.setValidData(percent_valid_points, seed=validation_seed)
    real.setValidData(percent_valid_points, seed=validation_seed)
    imaginary.setValidData(percent_valid_points, seed=validation_seed)
    head.setValidData(percent_valid_points, seed=validation_seed)
    ear.setValidData(percent_valid_points, seed=validation_seed)
    if C_magnitude is not None:
        C_magnitude.setValidData(percent_valid_points, seed=validation_seed)
        C_real.setValidData(percent_valid_points, seed=validation_seed)
        C_imaginary.setValidData(percent_valid_points, seed=validation_seed)


def format_inputs_outputs(pos, hrir, nn, ret_subjs=False, C_hrir=None):
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

    global position, head, ear, magnitude, magnitude_raw, real, imaginary, C_magnitude, C_real, C_imaginary
    global subj_removed

    # IIR_app(hrir[0,0,:,0],8)
    args = initializer.args
    subjects = initializer.subjects
    #Include the anthropometric inputs
    hrir_local = np.asarray(hrir, dtype=np.float32)
    print(C_hrir)
    if C_hrir is not None:
        C_hrir_local = np.asarray(C_hrir, dtype=np.float32)
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
    print ("printing the removed subjects")
    print(subj_removed)
    np.random.seed(12345)
    test_subj_idx = np.random.randint(num_subj, size=test_subj_num)

    # Scale Anthro data
    head_div = [16.20, 23.84, 23.06, 3.77, 1.90, 14.04, 9.47, 12.53, 36.84, 17.56, 30.25, 54.10, 8.41, 200.66, 110.00, 65.00, 131.00] 
    left_ear_div = [2.24, 1.05, 2.10, 2.24, 7.61, 3.53, 0.86, 1.31, 0.78, 0.68]
    right_ear_div = [2.29, 0.98, 1.99, 2.20, 7.95, 3.52, 0.91, 1.26, 0.72, 0.87]
    ear_div = [2.24, 1.05, 2.10, 2.24, 7.61, 3.53, 0.86, 1.31, 0.78, 0.68, 2.29, 0.98, 1.99, 2.20, 7.95, 3.52, 0.91, 1.26, 0.72, 0.87]
    head_local = head_local / head_div
    ear_local[:,:,:,0] = ear_local[:,:,:,0] / left_ear_div
    ear_local[:,:,:,1] = ear_local[:,:,:,1] / right_ear_div

    # Pick a slice
    #
    nn = 0
    mm = 1250
    head_new = np.zeros((num_subj, mm, np.shape(head_local)[2]))
    ear_new = np.zeros((num_subj, mm, np.shape(ear_local)[2], 2))
    pos_new = np.zeros((num_subj, mm, np.shape(pos_local)[2]))
    hrir_new = np.zeros((num_subj, mm, np.shape(hrir_local)[2], 2))
    if C_hrir is not None:
        C_hrir_new = np.zeros((num_subj, mm, np.shape(C_hrir_local)[2], 2))
    for i in range (num_subj):
        head_new[i] = head_local[i,nn:mm,:]
        ear_new[i] = ear_local[i,nn:mm,:,:]
        pos_new[i] = pos_local[i,nn:mm,:]
        hrir_new[i] = hrir_local[i,nn:mm,:,:]
        if C_hrir is not None:
            C_hrir_new[i] = C_hrir_local[i,nn:mm,:,:]

    head_local = head_new
    ear_local = ear_new
    pos_local = pos_new
    hrir_local = hrir_new
    if C_hrir is not None:
        C_hrir_local = C_hrir_new

    # experimental pole/zero search
    # hzeroes, hpoles, zp_h = utils.U_IIR_app(hrir_local,4)

    p_t_r = percent_test_points
    position = Data(pos_local, nn_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)

    head = Data(head_local, nn_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
    ear = Data(ear_local, nn_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)


    #Magnitude formatting
    scale = min(args['nfft'], np.shape(hrir_local)[2])
    #Get the HRTF
    outputs_fft = np.fft.rfft(hrir_local, args['nfft'], axis=2)
    outputs_complex = np.zeros(np.shape(outputs_fft), dtype=outputs_fft.dtype)
    for (s, h) in enumerate(outputs_fft):
        outputs_complex[s,:,:,:] = outputs_fft[s,:,:,:]/np.max(np.abs(outputs_fft[s,:,:,:]))
    outputs_mag = abs(outputs_complex) 
    outputs_mag = 20.0*np.log10(outputs_mag)

    # TODO - It seems that we no longer need magnitude raw as we no longer normalize the data
    magnitude = Data(outputs_mag, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
    magnitude_raw = Data(outputs_mag, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
    #Real formatting
    outputs_real = np.real(outputs_complex)
    real = Data(outputs_real, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
    #Imaginary formatting
    outputs_imag = np.imag(outputs_complex)
    imaginary = Data(outputs_imag, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
    if C_hrir is not None:
        #Get the HRTF
        outputs_fft = np.fft.rfft(C_hrir_local, args['nfft'], axis=2)
        outputs_complex = np.zeros(np.shape(outputs_fft), dtype=outputs_fft.dtype)
        for (s, h) in enumerate(outputs_fft):
            outputs_complex[s,:,:,:] = outputs_fft[s,:,:,:]/np.max(np.abs(outputs_fft[s,:,:,:]))
        outputs_mag = abs(outputs_complex)
        outputs_mag = 20.0*np.log10(outputs_mag)

        # TODO - It seems that we no longer need magnitude raw as we no longer normalize the data
        C_magnitude = Data(outputs_mag, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
        C_magnitude_raw = Data(outputs_mag, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
        #Real formatting
        outputs_real = np.real(outputs_complex)
        C_real = Data(outputs_real, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
        #Imaginary formatting
        outputs_imag = np.imag(outputs_complex)
        C_imaginary = Data(outputs_imag, nn_local, pos=pos_local, test_percent=p_t_r, test_seed=test_seed, normalize=False, pers=subj_removed)
    else:
        C_magnitude = None
        C_magnitude_raw = None
        C_real = None
        C_imaginary = None
     
    set_valid_training()

    if ret_subjs:
        return test_subj_idx

def get_data():
    global position, head, ear, magnitude, magnitude_raw, real, imaginary, C_magnitude, C_real, C_imaginary
    return position, head, ear, magnitude, magnitude_raw, real, imaginary, C_magnitude, C_real, C_imaginary
