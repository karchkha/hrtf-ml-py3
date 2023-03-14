#! /usr/bin/env python

'''
'''
import sys
sys.path.append("..")
import os
import h5py
import glob
import math as m
import scipy.io as sio
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import matplotlib.colors as col
import time
import sys
import argparse

from utilities import calc_phase, read_hdf5, write_hdf5
from utilities.network_data import Data
from utilities.parameters import *

def parseargs():
    parser = argparse.ArgumentParser(description='Plot the data from a selected dataset and subject')
    parser.add_argument('db',type=str, help='URL to downnload from')    
    parser.add_argument('subjects', type=str, nargs='+', help='Subject number')
    parser.add_argument('-nt', '--network_type', type=str, dest='network_type', default='mag_phase', help='network_type')
    parser.add_argument('-t', dest='hrir_type', type=str, default='raw', help='Type of the database (default: raw. options: trunc_64)')
    parser.add_argument('-anthro', dest='anthro', action='store_true', help='Do you want to include anthropometric data?')
    parser.add_argument('-a', '--action', type=str, default='train', nargs='+', help='(train|predict|eval)')
    parser.add_argument('-nn', '--network_number', type=int, default=0, help='Iterate the network number if you changed the network.')
    parser.add_argument('-e', '--ear',  nargs='+', dest='ear', type=str, default=['l', 'r'], help='Ear [l,r] (default: l)')   
    parser.add_argument('-n', '--nfft', dest='nfft', type=int, default=64, help='Length of the fft')
    parser.add_argument('-d', '--db_path', dest='db_path', type=str, default='../../datasets/', help='Directory of datasets')
    parser.add_argument('-r', '--ring', dest='ring', type=str, default=None, help='Which ring to view animation over [azimuth, elevation] (default: azimuth)')
    parser.add_argument('-clean', action='store_true', help='Clean saved files for corresponding model')
    parser.add_argument('-cleanall', action='store_true', help='Clean saved files for corresponding model')
    args = vars(parser.parse_args())
    return args

def sph2cart(pos):
    "pos should be (#subjs, #positions, [azi, ele, r])"
    pos_cart = np.array(pos)
    pos_cart[:,:,0] = np.multiply(pos[:,:,2], np.multiply(np.cos(pos[:,:,1]/180 * m.pi), np.cos(pos[:,:,0]/180 * m.pi)))
    pos_cart[:,:,1] = np.multiply(pos[:,:,2], np.multiply(np.cos(pos[:,:,1]/180 * m.pi), np.sin(pos[:,:,0]/180 * m.pi)))
    pos_cart[:,:,2] = np.multiply(pos[:,:,2], np.sin(pos[:,:,1]/180 * m.pi))
    return pos_cart

def plot():
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(511)
    ax2 = fig1.add_subplot(512)
    ax3 = fig1.add_subplot(513)
    ax4 = fig1.add_subplot(514)
    ax5 = fig1.add_subplot(515)
    axs = [ax1, ax2, ax3, ax4, ax5]
    for (i, ax) in enumerate(axs):
        ax.plot(means[i*7,:, :,0].T, '.-')
        ax.grid(True)
        ax.set_ylim([0, .35])
        ax.set_ylabel('Spectrum Mean')
    ax5.set_xlabel('Index Around a ring (0 -> Front Below Head, 50->Back Below Head)')

    fig5 = plt.figure()
    ax1 = fig5.add_subplot(211)
    ax2 = fig5.add_subplot(212)
    ax1.plot((means[:,24,:,0]/.18).T, '.-')
    ax2.plot((means[:,0,:,0]).T, '.-')
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylim([0, .35])
    ax2.set_ylim([0, .35])
    ax1.set_ylabel('Spectrum Mean')
    ax2.set_ylabel('Spectrum Mean')

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(means_avg[:,:,0].T, '.-')
    ax2.grid(True)
    ax2.set_ylim([0, .35])
    ax2.set_ylabel('Spectrum Mean')
    ax2.set_xlabel('Index Around a ring (0 -> Front Below Head, 50->Back Below Head)')

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(511)
    ax2 = fig3.add_subplot(512)
    ax3 = fig3.add_subplot(513)
    ax4 = fig3.add_subplot(514)
    ax5 = fig3.add_subplot(515)
    axs = [ax1, ax2, ax3, ax4, ax5]
    for (i, ax) in enumerate(axs):
        ax.plot(means[i*7,:, :,0], '.-')
        ax.grid(True)
        ax.set_ylim([0, .35])
        ax.set_ylabel('Spectrum Mean')
    ax5.set_xlabel('Index of Ring (0-> Left Side, 25->Right Side')

    fig4 = plt.figure()
    ax2 = fig4.add_subplot(111)
    ax2.plot(means_avg[:,:,0], '.-')
    ax2.grid(True)
    ax2.set_ylim([0, .35])
    ax2.set_ylabel('Spectrum Mean')
    ax2.set_xlabel('Index of Ring (0-> Left Side, 25->Right Side')
    plt.show()

def main():
    #Parse Arguments
    args = parseargs()

    if 'all' in args['subjects']:
        subj_list = []
        for i in range(200):
            num_0 = 3-len(str(i))
            a = []
            for j in range(num_0):
                a.append('0')
            a.append(str(i))
            a = ''.join(a)
            subj_list.append(a)
    else:
        subj_list = args['subjects']        

    subjects = subj_list

    hrir, pos, fs, _ = read_hdf5.getData(args['db'], subjects, db_filepath=args['db_path'], ring=args['ring'], ear=args['ear'], hrir_type=args['hrir_type'])

    hrir_local = np.asarray(hrir, dtype=np.float32)
    pos_local = np.asarray(pos, dtype=np.float32)
    pos_local_deg = pos_local
    pos_local = sph2cart(pos_local)
    head_inputs, ear_inputs, subj_with_nan = read_hdf5.getAnthroData(args['db'], subjects, db_filepath=args['db_path'], hrir_type=args['hrir_type'])
    head_local = np.repeat(np.expand_dims(head_inputs, axis=1), np.shape(pos)[1], axis=1)
    ear_local = np.repeat(np.expand_dims(ear_inputs, axis=1), np.shape(pos)[1], axis=1)
    head_local = np.delete(head_local, subj_with_nan, axis=0)
    ear_local = np.delete(ear_local, subj_with_nan, axis=0)
    num_subj = np.shape(ear_local)[0]
    print()
    print ('Removed %d subjects for nan in head or ear inputs' % len(subj_with_nan))
    print ()
    print ('Are there any leftover nans (anthro, pos, hrir)?')
    print (np.isnan(head_local).any(), np.isnan(pos_local).any(), np.isnan(hrir_local).any())
    position = Data(pos_local, nn=None, test_percent=percent_test_points, test_seed=0, normalize=False)
    position_deg = Data(pos_local_deg,  nn=None, test_percent=percent_test_points, test_seed=0, normalize=False)
    head = Data(head_local,  nn=None, test_percent=percent_test_points, test_seed=0, normalize=False)
    ear = Data(ear_local, nn=None, test_percent=percent_test_points, test_seed=0, normalize=False)

    #Magnitude formatting
    outputs_complex = np.fft.fft(hrir_local, n=args['nfft'], axis=2, norm="ortho")
    outputs_mag = abs(outputs_complex[:,:,:(args['nfft']//2)])   
    magnitude = Data(outputs_mag,  nn=None, pos=pos_local, test_percent=percent_test_points, test_seed=0, normalize=True)
    #Real formatting
    outputs_real = np.real(outputs_complex[:,:,:(args['nfft']//2)])
    real = Data(outputs_real,  nn=None, pos=pos_local, test_percent=percent_test_points, test_seed=0, normalize=True)
    #Imaginary formatting
    outputs_imag = np.imag(outputs_complex[:,:,:(args['nfft']//2)])
    imaginary = Data(outputs_imag,  nn=None, pos=pos_local, test_percent=percent_test_points, test_seed=0, normalize=True)

    means = magnitude.getMean()
    means = np.reshape(means, (num_subj, 25, 50, 2))

    pmeans = np.swapaxes(means, 0, 1)
#    print np.shape(pmeans)
    #Fit a polynomial to the means
    polys = []
    for (i, m) in enumerate(pmeans[0,:]):
        x = np.linspace(0, 1, np.shape(m)[0])
        p_l = np.polyfit(x, m[:,0], 8)
        p_data_l = np.polyval(p_l, x)
        p_r = np.polyfit(x, m[:,1], 8)
        p_data_r = np.polyval(p_l, x)
        p = np.vstack([p_data_l, p_data_r])
        polys.append(p)
    polys = np.swapaxes(polys, 1,2)
#    means_avg = np.mean(means, axis=0)

    ear_data = ear.getRawData()
    ear_data = np.reshape(ear_data, (num_subj, 25, 50, 10, 2))
    ear_data = np.squeeze(ear_data[:,0,0,:,:])
    if num_subj == 1:
        ear_data = np.expand_dims(ear_data, axis=0)
    theta_2_left = ear_data[:,-1,0]
    theta_1_left = ear_data[:,-2,0]
    pinna_height_left = ear_data[:,4,0]
    pinna_width_left = ear_data[:,5,0]
    pinna_area = np.multiply(pinna_height_left, pinna_width_left)

    head_data = head.getRawData()
    head_data = np.reshape(head_data, (num_subj, 25, 50, 17))
    head_data = np.squeeze(head_data[:,0,0,:])
    if num_subj == 1:
        head_data = np.expand_dims(head_data, axis=0)
    head_size = head_data[:,0]/100.
    ear_offset = head_data[:,4]

    pos_data = position.getRawData()
    pos_data = np.reshape(pos_data, (num_subj, 25, 50, 3))

    pos_data_deg = position_deg.getRawData()
    pos_data_deg = np.reshape(pos_data_deg, (num_subj, 1250, 3))

    #vec_ear = np.array([(np.cos(theta_1_left)+np.cos(theta_2_left)), np.sin(theta_2_left), np.sin(theta_1_left)]).T
    x = np.multiply(np.cos(theta_2_left), np.cos(theta_1_left))
    y = np.multiply(np.sin(theta_2_left), np.cos(theta_1_left))
    z = np.sin(theta_1_left)
    vec_ear = np.array([np.zeros(np.shape(theta_2_left)), 
        np.zeros(np.shape(theta_2_left)),
        np.ones(np.shape(theta_2_left))]).T
    vec_ear = np.array([x, y, z]).T  + np.array([0, 2, 0])

    for i, ve in enumerate(vec_ear):
        vec_ear[i] = vec_ear[i]/norm(vec_ear[i]) # + np.array([0,1,0])

    dot_prod = []
    for (i, p) in enumerate(pos_data):
        dot_prod.append(np.dot(p, vec_ear[i]))

    dot_prod = np.reshape(dot_prod, (num_subj, 1250))
    for subj in range(num_subj):
        dot_prod[subj][dot_prod[subj]<(0)] = dot_prod[subj][dot_prod[subj]<(0)]/pinna_width_left[subj]
#        dot_prod[subj][dot_prod[subj]<(0)] = dot_prod[subj][dot_prod[subj]<(0)]/2
    dot_prod = np.reshape(dot_prod, (num_subj, 25, 50))

#    for i, dp in enumerate(dot_prod):
#        #dot_prod[i] = (dot_prod[i]+np.abs(np.min(dp)))/2
#        dot_prod[i] = (dot_prod[i]+1)/2

    origin = [0, 0, 0]
    l_ear = origin
    X1, Y1, Z1 = zip(l_ear)
    U1, V1, W1 = zip(vec_ear[0])
    perc_thresh = .85
    color_means = np.reshape(means[0,:,:,0], (1250,))
    quiv_dot_prod = np.reshape(dot_prod[0,:,:], (1250,))
    quiv_thresh = perc_thresh*np.max(quiv_dot_prod)
    mean_thresh = perc_thresh*np.max(color_means)

    quiv_idx_to_use = np.where(quiv_dot_prod>quiv_thresh)
    quiv_pos = np.reshape(pos_data[0,:,:], (1250, 3))
    quiv_pos = quiv_pos[quiv_idx_to_use]
    X2, Y2, Z2 = [quiv_pos[:,0], quiv_pos[:,1], quiv_pos[:,2]]
    quiv_dir = (-1*quiv_pos)
    U2, V2, W2 = [quiv_dir[:,0], quiv_dir[:,1], quiv_dir[:,2]]
    quiv_dot_prod = quiv_dot_prod[quiv_idx_to_use]
    color = np.expand_dims(quiv_dot_prod, axis=1)
    zeros = np.zeros((len(quiv_idx_to_use[0]), 1))
    alphas = np.ones((len(quiv_idx_to_use[0]), 1))*.4
    color = np.concatenate((color, zeros, zeros), axis=1)
    color = np.divide(color, np.max(color))
    color = np.concatenate((color, alphas), axis=1)

    mean_idx_to_use = np.where(color_means>mean_thresh)
    mean_pos = np.reshape(pos_data[0,:,:], (1250, 3))
    mean_pos = mean_pos[mean_idx_to_use]
    color_means = color_means[mean_idx_to_use]
    color_means = np.expand_dims(color_means,axis=1)
    zeros = np.zeros((len(mean_idx_to_use[0]), 1))
    color_means = np.concatenate((zeros, color_means, zeros), axis=1)
    color_means = np.divide(color_means, np.max(color_means))

    freqs = np.zeros((0))
    dists = np.zeros((num_subj, 1250))

    for subj in range(num_subj):
        for e in ['l']:
            if e=='l':	
                listener = np.array([90+ear_offset[subj], 0, head_size[subj]/2.], dtype=float)
            elif e =='r':
                listener = np.array([90+ear_offset[subj], 0, -head_size[subj]/2.], dtype=float)

            for j, source in enumerate(pos_data_deg[subj]):
                _, _, dists[subj, j] = calc_phase.get_phase(listener, source, freqs, 44100, phase_type='ideal', quantize=False) 

    dists = np.reshape(dists, (num_subj, 25, 50))

    est = np.zeros(np.shape(dists))
#    for subj in range(num_subj):
#        est[subj, :, :] = np.multiply(np.add(dot_prod[subj],1)/(pinna_width_left[subj]+pinna_height_left[subj]), 1/dists[subj]**(pinna_height_left[subj]+pinna_width_left[subj]))
    est = np.multiply(dot_prod, (1/dists**2.))
#    est = dot_prod

    #Plot 3d sphere of points for
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot([origin[0]], [origin[1]], [origin[2]], 'r.')
    ax1.plot([l_ear[0]], [l_ear[1]], [l_ear[2]], 'b.')
    ax1.scatter([mean_pos[:,0]], [mean_pos[:,1]], [mean_pos[:,2]], 'o', c=color_means, depthshade=True, label='mean')
    ax1.scatter([quiv_pos[:,0]], [quiv_pos[:,1]], [quiv_pos[:,2]], 'o', c=color, depthshade=True, label='dot prod with ear vec')
    ax1.quiver(X1,Y1,Z1,U1,V1,W1,pivot='tail', color='b',arrow_length_ratio=.1) 
#    ax1.quiver(X2,Y2,Z2,U2,V2,W2,pivot='tail', color=color, arrow_length_ratio=0) 
    ax1.legend(loc='lower left')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_xlim([-1,1])
    ax1.set_ylim([-1,1])
    ax1.set_zlim([-1,1])
    ax1.set_title('Comparing highest %d perc of <loc, ear> and mean' % int(perc_thresh * 100))

    #Plot variance per ring per subject
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(np.sort(means[:,0,:,0], axis=0), '.-')
    ax1.set_title('Variance of far left ring for all subjects')
    ax1.set_xlabel('Subject number')
    ax1.set_ylabel('Mean power')
    ax1.set_xlim([-1, np.shape(means)[0]])
    ax1.grid(True)
    ax2.plot(np.sort(means[:,:,0,0], axis=1), '.-')
    ax2.set_title('Variance of ring of points in front and below (azimuthally) for all subjects')
    ax2.set_xlabel('Subject number')
    ax2.set_ylabel('Mean power')
    ax2.set_xlim([-1, np.shape(means)[0]])
    ax2.grid(True)

    #Plot means for single subject
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(means[0,:,:,0].T, '.-')
    ax1.set_title('Single Subject. Mean power around each ring. Each color is a different ring.')
    ax1.set_xlabel('Index around ring (0=front and below, 24=above, 49=behind and below)')
    ax1.set_ylabel('Mean power of spectrum')
    ax1.set_xlim([-1, np.shape(means)[2]])
    ax1.grid(True)
    ax2.plot(means[0,:,:,0], '.-')
    ax2.set_title('Single Subject. Mean power from left to right. Each color is a differen azimuthal slice.')
    ax2.set_xlabel('Index of ring (0=far left, 12=center, 24=far right')
    ax2.set_ylabel('Mean power of spectrum')
    ax2.set_xlim([-1, np.shape(means)[1]])
    ax2.grid(True)

    #Plot estimates for single subject
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(est[0].T, '.-')
    ax1.set_title('Single Subject Estimates. Mean power around each ring. Each color is a different ring.')
    ax1.set_xlabel('Index around ring (0=front and below, 24=above, 49=behind and below)')
    ax1.set_ylabel('Mean power of spectrum')
    ax1.set_xlim([-1, np.shape(means)[2]])
    ax1.grid(True)
    ax2.plot(est[0], '.-')
    ax2.set_title('Single Subject Estimates. Mean power from left to right. Each color is a differen azimuthal slice.')
    ax2.set_xlabel('Index of ring (0=far left, 12=center, 24=far right')
    ax2.set_ylabel('Mean power of spectrum')
    ax2.set_xlim([-1, np.shape(means)[1]])
    ax2.grid(True)

    plt.show()



if __name__ == '__main__':
    main()
