#! /usr/bin/python

import pdb

import numpy as np
import sys
import os
import scipy.io as sio
import pylab as plt
from tempfile import TemporaryFile
import math as m
import argparse
import h5py
import random
import time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def remove_idx(data, percent_to_remove=0, num_subj=1, seed=None, pers=False):   
    '''Returns a certain percentage of data points. Will be random unless seed is provided
    Data: (#subjs, #positions, #points, 2)
    Output: (subjects, aziumth_idxs, ele_idx) '''
    if percent_to_remove == 0:
        return [], []
    if seed is not None:
        random.seed(seed)
    else:
        random.seed()

    data_local = np.array(data)
    num_subj = np.shape(data_local)[0]
    num_pos = np.shape(data_local)[1]
    num_points = np.shape(data_local)[2]
    try: num_ear = np.shape(data_local)[3]
    except: num_ear = 1
    if pers:
        num_to_remove = m.ceil(num_subj*percent_to_remove)
        idx_to_remove = random.sample(range(num_subj),  int(num_to_remove))
    else:
        num_to_remove = m.ceil(num_subj*num_pos*percent_to_remove)
        idx_to_remove = random.sample(range(num_subj*num_pos),  int(num_to_remove))

#    removed_idxs = []
#    nn_local_remv = []
#
#
#    for s in range(num_subj):
#        removed_idxs_subj = []
#        for i in range(int(num_to_remove)):
#            nns = []
#            while len(nns) == 0:
#                if np.all(nn_local == None):
#                    break;
#                idx_to_remove = np.random.randint(np.shape(nn_local)[1]) 
#                nns = np.squeeze(nn_local[s,idx_to_remove,:])
#                nns = nns[~np.isnan(nns)]
#                nns = [int(n) for n in nns]
#            nn_local[s,idx_to_remove,:] = None
#            removed_idxs_subj.append(idx_to_remove)
#        nn_local_remv.append(np.delete(nn_local[s], np.squeeze(removed_idxs_subj), axis=0))
#        removed_idxs.append(removed_idxs_subj)
#    removed_idxs = np.array(removed_idxs)
    return idx_to_remove

def remove_points(data, removed_idxs, pers=False):
    '''Remove the indices from data specified by removed_idxs
    Inputs:
        data: (#subj, #position, ...)
    Ouptuts:
        removed_data: The data that was removed by removed_idxs
        manipulated_data: The original data after deleting the indices'''
    if len(removed_idxs) == 0:
        return data, []
    #data_local = np.array(data) 
    #removed_data = []
    #manipulated_data = []
    #for (s, rem_idx) in enumerate(removed_idxs):
    #    manipulated_data.append(np.delete(data_local[s], rem_idx, axis=0))
    #    removed_data.append(data_local[s,rem_idx,:])
    #manipulated_data = np.array(manipulated_data)
    #removed_data = np.array(removed_data)
    data_local = np.array(data)
    num_subj = np.shape(data_local)[0]
    num_pos = np.shape(data_local)[1]
    num_points = np.shape(data_local)[2]
    try: num_ear = np.shape(data_local)[3]
    except: num_ear = 1

    if pers:
        data_removed = data_local[removed_idxs, :]
        data_manipulated = np.delete(data_local, removed_idxs, axis=0)
    else:

        data_local = np.reshape(data_local, (num_subj*num_pos, num_points, -1))
        data_removed = data_local[removed_idxs, :]
        data_manipulated = np.delete(data_local, removed_idxs, axis=0)
        
        # print("removed_idxs",len(removed_idxs))
        # print(data_local.shape)
        # print(data_removed.shape)
        # print(data_manipulated.shape,"\n")
        
        data_removed = np.reshape(data_removed, (num_subj, -1, num_points, num_ear))
        data_manipulated = np.reshape(data_manipulated, (num_subj, -1, num_points, num_ear))

    return data_removed, data_manipulated


def sph2cart(pos):
    "pos should be (#subjs, #positions, [azi, ele, r])"
    pos_cart = np.array(pos)
    pos_cart[:,:,0] = np.multiply(pos[:,:,2], np.multiply(np.cos(pos[:,:,1]/180 * m.pi), np.cos(pos[:,:,0]/180 * m.pi)))
    pos_cart[:,:,1] = np.multiply(pos[:,:,2], np.multiply(np.cos(pos[:,:,1]/180 * m.pi), np.sin(pos[:,:,0]/180 * m.pi)))
    pos_cart[:,:,2] = np.multiply(pos[:,:,2], np.sin(pos[:,:,1]/180 * m.pi))
    return pos_cart

def get_neighborhood(data, pos, azi_n=None, ele_n=None):
    '''Shape of data_out is
        (# subjects, # samples, neighborhood size, length of data, # ears (if different data per ear))'''
    data_local=np.array(data)
    pos_local = np.expand_dims(np.array(pos[0]), axis=0)
    if azi_n == None:
        azi_n = 2
    if ele_n == None:
        ele_n = 2
    reshape_rows = 25
    reshape_cols = (np.shape(data_local)[1]/reshape_rows)
    data_local = np.reshape(data_local, (np.shape(data)[0], reshape_rows, reshape_cols, np.shape(data)[2], -1))
    pos_local = np.reshape(pos_local, (1, reshape_rows, reshape_cols, np.shape(pos)[2], -1))
    #data_out shape = (#subjs, neighborhood size, number of points, 
    data_out = np.zeros((np.shape(data)[0], np.shape(data)[1], (azi_n+ele_n+1), np.shape(data)[2], 1)) 
    if len(np.shape(data))>3:
        data_out = np.zeros((np.shape(data)[0], np.shape(data)[1], (azi_n+ele_n+1), np.shape(data)[2], np.shape(data)[3])) 
    for s, subj in enumerate(data_local):
        print ('Subject ', s)
        for a, azi in enumerate(subj):
            for e, ele in enumerate(azi):
                idx = a*(reshape_cols)+e
                data_out[s, idx, 0] = data_local[s, a, e]
                for i in range(1, int(m.floor(ele_n/2))+1):
                    data_out[s, idx, i, :] = data_local[s, a, (e-(int(m.floor(ele_n/2))-(i-1)))]
                for j in range(1, int(m.floor(ele_n/2))+1):
                    data_out[s, idx, i+j, :] = data_local[s, a, (e+j)%reshape_cols]
                end_idx = i+j

                #flip x and z to find point on edge circle
                new_pos = np.array([-pos_local[0, a, e, :]])
                dist = np.sum(np.multiply(pos_local-new_pos, pos_local-new_pos), axis=3)
                new_idx = np.unravel_index(np.argmin(dist), np.shape(dist))
                new_idx = new_idx[2]
                last_z = False
                for i in range(1, int(m.floor(azi_n/2)+1)):
                    i1 = i
                    if last_z:
                        i = i-1
                    if (a-(int(m.floor(azi_n/2))-(i-1))) == 0:
                        if not last_z:
                            data_out[s, idx, end_idx+i1, :] = data_local[s, abs(a-(int(m.floor(azi_n/2))-(i-1))), e]
                            last_z = True
                        else:
                            data_out[s, idx, end_idx+i1, :] = data_local[s, abs(a-(int(m.floor(azi_n/2))-(i-1))), new_idx]
                    elif (a-(int(m.floor(azi_n/2))-(i-1))) < 0:
                        data_out[s, idx, end_idx+i1, :] = data_local[s, abs(a-(int(m.floor(azi_n/2))-(i))), new_idx]
                    else:
                        data_out[s, idx, end_idx+i1, :] = data_local[s, (a-(int(m.floor(azi_n/2))-(i-1))), e]
                last_z = False
                for j in range(1, int(m.floor(azi_n/2)+1)):
                    j1 = j
                    if last_z:
                        j = j-1
                    if (a+j) == reshape_rows-1:
                        if not last_z:
                            data_out[s, idx, end_idx+j1+i1, :] = data_local[s, (a+j), e]
                            last_z = True
                        else:
                            data_out[s, idx, end_idx+j1+i1, :] = data_local[s, (a+j), new_idx]
                    elif (a+j) > reshape_rows-1:
                        data_out[s, idx, end_idx+j1+i1, :] = data_local[s, reshape_rows-j, new_idx]
                    else:
                        data_out[s, idx, end_idx+j1+i1, :] = data_local[s, (a+j), e]
    plot = False
    if plot:
        idx = 3
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(data_local[0, :, :, 0], data_local[0, :, :, 1], data_local[0, :, :, 2], c=[1.0, 0.0, 0.0, 0.1])
        ax1.scatter(data_out[0, idx, :,0], data_out[0, idx, :, 1], data_out[0, idx, :, 2], c=[0.0, 0.0, 1.0, 1])
        plt.show()
        exit()

    return data_out

def get_mean_and_power(data):
    '''data shape is (#subjs, #pos, data_len)
    Calculates the average and the average power at each position for normalization
    Returns:
        unbiased (#subjs, #pos, unbiased)
        mean (#subjs, #pos, mean)
        std (#subjs, #pos, std)'''
    mean = np.mean(data, axis=2)
    h_unbiased = np.zeros(np.shape(data))
    for i, h_subj in enumerate(data):
        h_unbiased[i] = [(h-mean[i][j]) for j, h in enumerate(h_subj)]
    var = np.mean(np.power(h_unbiased, 2), axis=2)  
    std = np.sqrt(var)
    for i, h_subj in enumerate(h_unbiased):
        for j, h in enumerate(h_subj):
            if len(np.shape(h)) < 2:
                h = np.expand_dims(h, axis=1)
            if not isinstance(std[i][j], np.ndarray):
                st = np.expand_dims(std[i][j], axis=0)
            else:
                st = std[i][j]
            h_unbiased_new = []
            for (k, _) in enumerate(st):
                if st[k] == 0:
                    h_unbiased_new.append(h[:,k])
                else:
                    h_unbiased_new.append(h[:,k]/st[k])
            h_unbiased_new = np.swapaxes(h_unbiased_new, 0, 1)
            h_unbiased[i][j] = np.squeeze(h_unbiased_new)
#    for i, h_subj in enumerate(h_unbiased):
#        h_unbiased[i] = [h/std[i][j] for j, h in enumerate(h_subj)]
    max_h = np.max(np.abs(h_unbiased))
    var = np.mean(np.power(h_unbiased, 2), axis=2)
    mean = np.expand_dims(mean, axis=2)
    std = np.expand_dims(std, axis=2)
    return h_unbiased, mean, std

class Data:
    '''
    Data is data class which initialized the various fields
    navg is used for nearest neighbout avg and it is generally not used, 
    --- TODO delete all navg references
    --- TODO delete all sng_usr references

    All getters are reshaping the appropriate data (data, mean, std) as (#subjects * #pos, #samples, #ears)
    '''
    def __init__(self, data, nn, pos=None, test_percent=.1, test_seed=None, normalize=False, navg=False, navg_normalize=False, pers=False):
        self.raw_data = data
        self.nn = nn
        self.normalize= normalize
        self.num_subj = np.shape(data)[0]
        self.normalized_data, self.mean, self.std = get_mean_and_power(self.raw_data)
        self.pos = None
        if pos is not None:
            self.pos = pos;
        if self.normalize:
            self.data = self.normalized_data
        else:
            self.data = self.raw_data
        if navg:
            self.setNAvgData(normalize=navg_normalize)
        self.setTestData(test_percent, seed=test_seed, navg=navg)
        self.max = np.max(self.data)
        self.min = np.min(self.data)

    def setTestData(self, percent, seed=None, navg=False):
        self.test_idx = remove_idx(self.data, num_subj=self.num_subj, percent_to_remove=percent, seed=seed)
        self.test_data, self.training_valid_data = remove_points(self.data, self.test_idx)
        self.test_mean, self.training_valid_mean = remove_points(self.mean, self.test_idx)
        self.test_std, self.training_valid_std = remove_points(self.std, self.test_idx)
        if self.pos is not None:
            # print("pos???")
            self.test_pos, self.training_valid_pos = remove_points(self.pos, self.test_idx)
        if navg:
            navg_test_idx = remove_idx(self.navg_data, percent_to_remove=percent, seed=seed)
            self.navg_test_data, self.navg_training_valid_data = remove_points(self.navg_data, navg_test_idx)
            self.navg_test_mean, self.navg_training_valid_mean = remove_points(self.navg_mean, navg_test_idx)
            self.navg_test_std, self.navg_training_valid_std = remove_points(self.navg_std, navg_test_idx)

    def setValidData(self, percent, seed=None, navg=False, pers=False):
        self.valid_idx = remove_idx(self.training_valid_data, num_subj=self.num_subj, percent_to_remove=percent, seed=seed, pers=pers)
        self.valid_data, self.training_data = remove_points(self.training_valid_data, self.valid_idx, pers=pers)
        self.valid_mean, self.training_mean = remove_points(self.training_valid_mean, self.valid_idx, pers=pers)
        self.valid_std, self.training_std = remove_points(self.training_valid_std, self.valid_idx, pers=pers)
        if self.pos is not None:
            self.valid_pos, self.training_pos = remove_points(self.training_valid_pos, self.valid_idx, pers=pers)

    def plotRemovedData(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        ax1.scatter(self.training_pos[0,:,0], self.training_pos[0,:,1], self.training_pos[0,:,2], c=[1.0, 0.0, 0.0, 1.0], label='Training')
        ax1.scatter(self.valid_pos[0,:,0], self.valid_pos[0,:,1], self.valid_pos[0,:,2], c=[0.0, 1.0, 0.0, 1.0], label='Valid')
        ax1.scatter(self.test_pos[0,:,0], self.test_pos[0,:,1], self.test_pos[0,:,2], c=[0.0, 0.0, 1.0, 1.0], label='Test')
        ax1.legend()
        plt.show()

    def setNAvgData(self, normalize=False):
        self.navg_raw_data = get_neighborhood(self.data, self.pos)
        #Average over subjects (axis 0) and neighborhood (axis 2)
        for a in [2, 0]:
            self.navg_raw_data = np.mean(self.navg_raw_data, axis=a)    
        self.navg_raw_data = np.expand_dims(self.navg_raw_data, axis=0)
        self.navg_normalized_data, self.navg_mean, self.navg_std = get_mean_and_power(self.navg_raw_data)
        if normalize:
            self.navg_data = self.navg_normalized_data
        else:
            self.navg_data = self.navg_raw_data

    def plotNAvgData(self):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(self.navg_data[0,0,:,0])
        plt.show()
        exit()

    def getTestData(self, navg_diff=False, navg=False, sng_usr=False):
        if navg:
            ret_data = self.navg_test_data
        elif navg_diff:
            ret_data = self.navg_test_data - self.test_data 
        else:
            ret_data = self.test_data
        if sng_usr:
            ret_data = np.expand_dims(ret_data[0,:,:], axis=0)
        return np.reshape(ret_data, (np.shape(ret_data)[0]*np.shape(ret_data)[1], np.shape(ret_data)[2], -1))

    def getTestMean(self, navg=False):
        if navg:
            ret_mean = self.navg_test_mean
        else:
            ret_mean = self.test_mean
        return np.reshape(ret_mean, (np.shape(ret_mean)[0]*np.shape(ret_mean)[1], np.shape(ret_mean)[2], -1))

    def getTestStd(self, navg=False):
        if navg:
            ret_std = self.navg_test_std
        else:
            ret_std = self.test_std
        return np.reshape(ret_std, (np.shape(ret_std)[0]*np.shape(ret_std)[1], np.shape(ret_std)[2], -1))   

    def getTestIdx(self):
        idx = np.squeeze(self.test_idx)
        return np.sort(idx)

    def getValidData(self, navg_diff=False, navg=False, sng_usr=False):
        if navg:
            ret_data = self.navg_valid_data
        elif navg_diff:
            ret_data = self.navg_valid_data - self.valid_data 
        else:
            ret_data = self.valid_data
        if sng_usr:
            ret_data = np.expand_dims(ret_data[0,:,:], axis=0)
        return np.reshape(ret_data, (np.shape(ret_data)[0]*np.shape(ret_data)[1], np.shape(ret_data)[2], -1))   

    def getValidMean(self, navg=False):
        if navg:
            ret_mean = self.navg_valid_mean
        else:
            ret_mean = self.valid_mean
        return np.reshape(ret_mean, (np.shape(ret_mean)[0]*np.shape(ret_mean)[1], np.shape(ret_mean)[2], -1))   

    def getValidStd(self, navg=False):
        if navg:
            ret_std = self.navg_valid_std
        else:
            ret_std = self.valid_std
        return np.reshape(ret_std, (np.shape(ret_std)[0]*np.shape(ret_std)[1], np.shape(ret_std)[2], -1))   

    def getTrainingData(self, navg_diff=False, navg=False, sng_usr=False):
        if navg:
            ret_data = self.navg_training_data
        elif navg_diff:
            ret_data = self.navg_training_data - self.training_data 
        else:
            ret_data = self.training_data
        if sng_usr:
            ret_data = np.expand_dims(ret_data[0,:,:], axis=0)
        return np.reshape(ret_data, (np.shape(ret_data)[0]*np.shape(ret_data)[1], np.shape(ret_data)[2], -1))   

    def getTrainingMean(self, navg=False):
        if navg:
            ret_mean = self.navg_training_mean
        else:
            ret_mean = self.training_mean
        return np.reshape(ret_mean, (np.shape(ret_mean)[0]*np.shape(ret_mean)[1], np.shape(ret_mean)[2], -1))   

    def getTrainingStd(self, navg=False):
        if navg:
            ret_std = self.navg_training_std
        else:
            ret_std = self.training_std
        return np.reshape(ret_std, (np.shape(ret_std)[0]*np.shape(ret_std)[1], np.shape(ret_std)[2], -1))   

    def getNAvgData(self, navg_diff=False):
        if navg_diff:
            ret_data = self.navg_data - self.data 
        else:
            ret_data = self.navg_data
        return np.reshape(ret_data, (np.shape(ret_data)[0]*np.shape(ret_data)[1], np.shape(ret_data)[2], -1))   

    def getPosition(self, data_type=None):
        if data_type == 'training':
            ret_data = self.training_pos
        elif data_type == 'valid':
            ret_data = self.valid_pos
        elif data_type == 'test':
            ret_data = self.test_pos
        else:
            ret_data = self.pos
        return np.reshape(ret_data, (np.shape(ret_data)[0]*np.shape(ret_data)[1], np.shape(ret_data)[2], -1))   

    def getRawData(self):
        return np.reshape(self.raw_data, (np.shape(self.raw_data)[0]*np.shape(self.raw_data)[1], np.shape(self.raw_data)[2], -1))    

    def getNormalizedData(self):
        return np.reshape(self.normalized_data, (np.shape(self.normalized_data)[0]*np.shape(self.normalized_data)[1], np.shape(self.normalized_data)[2], -1))    

    def getMean(self):
        return np.reshape(self.mean, (np.shape(self.mean)[0]*np.shape(self.mean)[1], np.shape(self.mean)[2], -1))    

    def getStd(self):
        return np.reshape(self.std, (np.shape(self.std)[0]*np.shape(self.std)[1], np.shape(self.std)[2], -1))    
