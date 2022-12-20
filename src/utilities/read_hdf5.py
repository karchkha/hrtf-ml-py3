#! /usr/bin/python

'''
	Script to read the cipic dataset hdf5 for all subjects with the following organization
	
	subject
		hrir_l, hrir_r, azimuth, elevation, OnL, OnR, ITD
		attrs
			D, age, sex, WeightKilograms, X, theta, id 


	subject: subject_###
	hrir_l: Impulse Response of Left Ear
	hrir_r: Impulse Response of Right Ear
	azimuth: List of azimuth angles
	elevation: List of elevation angles
	OnL: Onset to the left ear (ms)
	OnR: Onset to the right ear (ms)
	ITD: Interaural Time Difference (|OnL-OnR|) (ms)

	For anthropometric data
	X: See README
	D: See README. Indices [0:8] are Left Ear. Indices [8:16] are Right Ear
	theta: See README. Indices [0:2] are Left Ear. Indices [2:4] are Right Ear
'''

import os
import h5py
import glob
import math as m
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import time
import sys
import argparse

def printname(name):
	print(name)

def getData(db, subjects, db_filepath='../datasets/', hrir_type='trunc_64', ear=['l','r'], ring=None, list_subjects=False, ret_subjs=False, radius=None):
    '''
    getData()
    Parameters:
        db:str [cipic, scut] (database name)
        subject:list (subject names)
        db_filepath:str (dababase directory)
        hrir_type:str ["raw" or "trunc_64"] (hrir type)
        ear:list ['l', 'r'] (ear)
        ring:float (elevation or azimuthal angle)
        list_subjects:boolean (print sublect list or not)
        ret_subjs:boolean (return the subjets'; name)
        radius:list (list of radia)
    
    returns:
            return hrir, srcpos, fs, subjs, nn
            return hrir, srcpos, fs, nn
    
    '''
    
    #Load the dataset
    f = h5py.File(os.path.expanduser(db_filepath+'/'+db+'.hdf5'), 'r')
    
    if list_subjects:
    	f.visit(printname)
    
    #Get the subjects	
    fs = []
    hrir = []
    srcpos = []
    subjs = []
    nn = []
    
    pos_ds_loc = 'srcpos/' + hrir_type
    nn_ds_loc = 'nn/' + hrir_type
    
    idx = 0
    
    for name in subjects:
        try:
            subj = f['subject_'+name]
            cur_fs = list(subj.attrs['fs'])[0]
            cur_p = list(subj[pos_ds_loc][:])
            cur_nn = list(subj[nn_ds_loc][:])
            fs.append(cur_fs)
            srcpos.append(cur_p)
            nn.append(cur_nn)
            hear = []
            for e in ear:
                hrir_ds_loc = 'hrir_'+e+'/'+hrir_type
                cur_h = list(-subj[hrir_ds_loc][:])
                hear.append(cur_h)
            hrir.append(hear)
            subjs.append(name)
        except Exception as err:
            # raise ValueError('Subject Not Found %s', name)
            continue
        	
        hrir = np.swapaxes(hrir, 1, 2)
        hrir = np.swapaxes(hrir, 2, 3)
    #Get appropriate ring positions based on azimuth or elevation	
    if (ring != None) or (radius != None):	
        hrir, srcpos = getRing(np.array(hrir), np.array(srcpos),ear=ear, ring_type=ring, db=db, radius=radius)
    else:
        print ("No ring specified, returning all hrir")
        
    if ret_subjs:
        #return hrir, srcpos, fs, subjs
        return hrir, srcpos, fs, subjs, nn
    else:
        return hrir, srcpos, fs, nn

def getAnthroData(db, subjects, db_filepath='../datasets/', hrir_type='raw', ring=None):
    '''
        Returns
            headinfo: Head info of the subjects. Dimensions are 
                (#subjects, 17
            earinfo: Ear info for all subjects. Dimensions are
                (#subjects, 10, 2)
                Of the 10, indices 0-7 are 'D' data and 8-9 are 'theta'
                The 2 is for both ears 0 is left, 1 is right
            subj_with_nan: A list of all the subjects that did not have anthro data
    '''
    #Load the dataset
    f = h5py.File(os.path.expanduser(db_filepath+'/'+db+'.hdf5'), 'r')
    
    headinfo = []
    earinfo = []
    subj_with_nan = []
    
    #Get the subjects
    idx = 0
    for name in subjects:
        try:
            subj = f['subject_'+name]
            
            cur_h = list(subj.attrs['X'])
    		
            #[0:7] are left ear
            #[8:16] are right ear
            cur_e_l = list(subj.attrs['D'][0:8])
            cur_e_r = list(subj.attrs['D'][8:16])
            #[0:2] are left ear
            #[3:4] are right ear
            cur_e_l = np.hstack([cur_e_l, list(subj.attrs['theta'][0:2])])
            cur_e_r = np.hstack([cur_e_r, list(subj.attrs['theta'][2:4])])
    
            cur_e = np.stack([cur_e_l, cur_e_r], axis=1)

        except Exception as err:
            print (err)
            continue

        if 'X' in subj.attrs.keys():
            headinfo.append(cur_h)
        if 'D' in subj.attrs.keys():
            earinfo.append(cur_e)
        
        if np.isnan(cur_h).any() or np.isnan(cur_e).any():
            subj_with_nan.append(idx)
        
        idx += 1

    return headinfo, earinfo, subj_with_nan






def getRing(hrir, srcpos, ring=0., radius=[1.0], ear=['l', 'r'], ring_type='azimuth', db=None):
    '''
	getRing(): choose a set of hrir based on the elevation ring, for scut only for now

	parameters:
	    hrir:list (hrir set)
	    srcpos:list (correspoinding source positions)
	    ring:float (elevation angle)
	    ear:list ['l', 'r'] (ear)
	    ring_type"string ('azimuth', 'elevation'k

	returns:
	    hrir, srcpos
	'''
    if db == 'scut':
        hrir_new = []
        srcpos_new = []
        for n, e in enumerate(ear):
            hrir_tmp = []
            srcpos_tmp = []
            hrir_e = hrir[:,:,:,n]
            for i, (s, h) in enumerate(zip(np.asarray(srcpos), np.asarray(hrir_e))):
                for r in radius:
                    #print r
                    dist_ring = np.where(np.round(s[:,2], 2) == r)
                    dist_s = s[dist_ring[0],:]
                    dist_h = h[dist_ring[0],:]
                    if ring_type == 'azimuth':
                        indices = np.argwhere(np.round(dist_s[:,1], 2)==0.)
                    elif ring_type == 'elevation':
                        indices = np.vstack([np.argwhere(np.round(dist_s[:,0], 2)==0.), np.argwhere(np.round(dist_s[:,0], 2) == 180.)])
                    else:
                        indices = np.arange(np.shape(dist_s)[0])
                    dist_h = np.squeeze(dist_h[indices,:])
                    dist_s = np.squeeze(np.round(dist_s[indices,:], 2))
                    #Sort the azimuth points and elevation points
                    if ring_type == 'azimuth':
                        sorted_inds = np.argsort(dist_s, axis=0)[:,0]
                    elif ring_type == 'elevation':
                        src = np.array(dist_s)
                        src[np.argwhere(dist_s==180)[:,0], 1] = -src[np.argwhere(dist_s==180)[:,0], 1]
                        src[:,1] = src[:,1]+90 + dist_s[:,0]
                        sorted_inds = np.argsort(src, axis=0)[:,1]
                    else:
                        # sorted_inds = np.argsort(dist_s, axis=0)[:,0]
                        sorted_inds = indices
                    hrir_tmp.append(dist_h[sorted_inds])
                    srcpos_tmp.append(dist_s[sorted_inds])
                hrir_tmp = np.reshape(hrir_tmp, (1, np.shape(hrir_tmp)[0]*np.shape(hrir_tmp)[1], -1))
                hrir_new.append(hrir_tmp)
                srcpos_new = np.reshape(srcpos_tmp, (1, np.shape(srcpos_tmp)[0]*np.shape(srcpos_tmp)[1], -1))
        hrir_new = np.array(hrir_new)
        srcpos_new = np.array(srcpos_new)
    else:
        hrir_new = []
        srcpos_new = []
        for n, e in enumerate(ear):
            hrir_tmp = []
            hrir_e = hrir[:,:,:,n]
            for i, (s, h) in enumerate(zip(np.asarray(srcpos), np.asarray(hrir_e))):
                if ring_type == 'azimuth':
                    indices = np.argwhere(np.round(s[:,1], 2)==0.)
                if ring_type == 'elevation':
                    indices = np.vstack([np.argwhere(np.round(s[:,0], 2)==0.), np.argwhere(np.round(s[:,0], 2) == 180.)])
                h = np.squeeze(h[indices,:])
                s = np.squeeze(np.round(s[indices,:], 1))
                #Sort the azimuth points and elevation points
                if ring_type == 'azimuth':
                    sorted_inds = np.argsort(s, axis=0)[:,0]
                if ring_type == 'elevation':
                    src = np.array(s)
                    src[np.argwhere(s==180)[:,0], 1] = -src[np.argwhere(s==180)[:,0], 1]
                    src[:,1] = src[:,1]+90 + s[:,0]
                    sorted_inds = np.argsort(src, axis=0)[:,1]
                hrir_tmp.append(h[sorted_inds])
            hrir_new.append(np.array(hrir_tmp))
        srcpos_new.append(s[sorted_inds])
        hrir = np.array(hrir_new)
        srcpos = np.array(srcpos_new)
        hrir = np.swapaxes(hrir, 0, 1)
        hrir = np.swapaxes(hrir, 1, 2)
        hrir = np.swapaxes(hrir, 2, 3)
    return hrir, srcpos