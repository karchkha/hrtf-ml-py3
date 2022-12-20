#! /usr/bin/env python

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

import matplotlib.animation as animation

from utilities import calc_phase, read_hdf5, write_hdf5

def parseargs():
    parser = argparse.ArgumentParser(description='Plot the data from a selected dataset and subject')
    parser.add_argument('db',type=str, help='URL to downnload from')	
    parser.add_argument('subjects', type=str, nargs='+',help='Subject number')
    parser.add_argument('-t', '--type', type=str, default='raw', help='Type of dataset to load (default:raw)')
    parser.add_argument('-e', '--ear',  dest='ear', nargs='+', type=str, default=['l'], help='Ear [l,r] (default: l)')	
    parser.add_argument('-n', '--nfft', dest='nfft', type=int, default=4096, help='Length of the fft')
    parser.add_argument('-d', '--dir', dest='directory', type=str, default='../datasets/', help='Directory of datasets')
    parser.add_argument('-r', '--ring', dest='ring', type=str,default='azimuth', help='Which ring to view animation over [azimuth, elevation] (default: azimuth)')
    parser.add_argument('-x', '--xaxis', dest='xaxis', type=str, default='freq', help='X-Axis for the plots [freq, angle, time] (default: freq)')
    parser.add_argument('-p', '--plot', dest='plottype', type=str, default='mag', help='What to plot [mag, mag_db, wrap, uwrap, grpdel] (default: mag)')
    parser.add_argument('--time', dest='deltime', type=float, default=0.5, help='Time delay (in seconds) between frames of animation (default: .5)')
    parser.add_argument('-phase_plots', type=str, default=None, help='Whether to additionally plot ideal and euclidean phase [ideal, euclidean]')
    parser.add_argument('-list', action='store_true', help="List the subject numbers for the given database")
    parser.add_argument('-fn', '--fname', dest='fname', type=str, default='myvideo', help='Filename of video')
    parser.add_argument('-clean', action='store_true', default=None, help="Erase the animation with that name")

    args = vars(parser.parse_args())
    return args


def estimate_phase_slope(phase, freqs):
	slope_estimate = np.array(phase)
	for i, s, in enumerate(phase):
		for j, p in enumerate(s):
			numer = np.sum(np.multiply(p, freqs))
			denom = np.sum(np.power(freqs, 2))
			slope = numer/denom
			slope_estimate[i,j] = np.multiply(slope,freqs)
	return slope_estimate


def main():
    #Parse Arguments
    args = parseargs()
    
    db = args['db']
    db_filepath = args['directory']+db
    subjects = args['subjects']
    ear = args['ear']
    nfft = args['nfft'] 
    ring = args['ring']
    plot_type = args['plottype']
    xaxis = args['xaxis']
    del_time = args['deltime']
    db_type = args['type']
    list_subjects = args['list']
    
    #Read data in	
    hrir = []
    fs = []
    srcpos = []
    subject_names = subjects
	
    try:
        if db == 'scut':
            radius = [.2, .5, 1.0]
        else:
            radius = [1.0]
        
        hrir, srcpos, fs, nn = read_hdf5.getData(db, subjects,  ring=ring, ear=ear, hrir_type=db_type, list_subjects=list_subjects, radius=radius)
    
    except: # ValueError as err:	
        print ("error getting hrirs on this subject") #, err)
        exit()

    scale = min(nfft, np.shape(hrir)[2])
    #Get the HRTF
    HRTF = np.fft.rfft(hrir, nfft, axis=2)
    HRTF = HRTF/np.max(np.abs(HRTF))
    nfft = np.shape(HRTF)[2]
    
    #Get the mag, mag_db, uwrap, and wrap
    mag = np.abs(HRTF)
    #mag = mag/np.max(mag)
    mag_db = 20*np.log10(mag)
    phase_w = np.angle(HRTF)
    phase_uw = np.unwrap(phase_w, axis=2)
    grpdel = np.gradient(phase_uw, axis=2)
#	phase_uw_estimate = estimate_phase_slope(phase_uw, freqs)
#	phase_uw_gradient = np.gradient(phase_uw, axis=1)
#	phase_uw_sub = phase_uw - phase_uw_estimate

    #Set the appropriate labels
    if xaxis != 'time':
        if plot_type == 'mag':
            plot_data = mag
            ylabel = 'Magnitude'
        elif plot_type == 'mag_db':
    	    plot_data = mag_db
    	    ylabel = 'Magnitude (dB)'
        elif plot_type == 'wrap':
            plot_data = phase_w
            ylabel = 'Angle'
        elif plot_type == 'uwrap':
            # plot_data = np.concatenate((phase_uw, phase_uw_estimate, phase_uw_sub, phase_uw_gradient), axis=0)
            plot_data = phase_uw
            ylabel = 'Angle'
        elif plot_type == 'grpdel':
            plot_data = grpdel
            ylabel = 'Angle'


    	#If we're dealing with phases, check the ideal and euc flags
        if plot_type == 'wrap' or plot_type == 'uwrap' or plot_type == 'mag_phase':
            if ear=='l':	
                listener = np.array([90, 0, 0.09], dtype=float)
            elif ear =='r':
                listener = np.array([270, 0, 0.09], dtype=float)
    
    
            ideal_phases = np.array(plot_data[0], dtype=float)
            euclidean_phases = np.array(plot_data[0], dtype=float)
    
    
            freqs = np.linspace(0, fs[0]/2, nfft)
            #for k, f in enumerate(fs):
            for j, source in enumerate(srcpos[0]):
                if args['phase_plots'] == 'ideal':
                    ideal_phases[j], tmp = calc_phase.get_phase(listener, source, freqs, fs[0], phase_type=args['phase_plots']) 
                if args['phase_plots'] == 'euclidean':
                    euclidean_phases[j],tmp  = -1*calc_phase.get_phase(listener, source, freqs, fs[0], phase_type=args['phase_plots'])
            #ideal_phases = ( ideal_phases + np.pi) % (2 * np.pi ) - np.pi
            
            ideal_phases=-1*ideal_phases
            euclidean_phases=-1*euclidean_phases
    elif xaxis == 'time':
        plot_data = hrir*-1	
        ylabel = 'Amplitude'
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_ylabel(ylabel)
    if plot_type != 'uwrap' or plot_type!= 'mag_phase':
	    ax.set_ylim([np.min(plot_data), np.max(plot_data)])
    ax.grid(True)
    pls = []
    Writer = animation.writers['ffmpeg']
    interval = args['deltime']*1000.
    fps = 1/args['deltime']
    animations_dir = '../figures/data/animations/'
    if not os.path.exists(animations_dir):
        os.makedirs(animations_dir)
	
    animation_fname = animations_dir + args['fname'] + '.mp4'
    if args['clean']:
        os.remove(animation_fname)
        exit()
    writer = Writer(fps=fps,metadata=dict(artist='Me'))

	#If we're plotting with xaxis as frequency, animation over angle
    with writer.saving(fig, animation_fname, 250):
        if xaxis == 'freq':
            for i, pd in enumerate(plot_data):	
                for j, e in enumerate(ear):
                        pl, = ax.plot(range(nfft), pd[0,:,j].reshape((-1,)), label=subject_names[i]+'_'+e)
                        pls.append(pl)

            if args['phase_plots'] == 'ideal':
                pl_ideal, = ax.plot(range(nfft), ideal_phases[0,:].reshape((-1)), label='ideal phase')

            if args['phase_plots'] == 'euclidean':
                pl_euc, = ax.plot(range(nfft), euclidean_phases[0,:].reshape((-1)), label='euclidean phase')		

            ax.legend()
            ax.set_xlabel('Frequency')
            ax.set_xticks(np.linspace(0, nfft, 5))
            ax.set_xticklabels(np.linspace(0,nfft,5)/nfft * fs[0])
            ax.autoscale(enable=True, axis='x', tight=True)

            for idx, pos in enumerate(srcpos[0]):
                for i, pd in enumerate(plot_data):
                    i = i*len(ear)
                    for j, e in enumerate(ear):
                        pls[i+j].set_data(range(nfft),pd[idx,:,j].reshape((-1,)))
                    ax.set_title(''.join(ear).upper()+' ear. pos: (%.2f, %.2f, %.2f)' % (pos[0], pos[1], pos[2]))

                if args['phase_plots'] == 'ideal':
                    pl_ideal.set_data(range(nfft), ideal_phases[idx,:].reshape((-1)))

                if args['phase_plots'] == 'euclidean':
                    pl_euc.set_data(range(nfft), euclidean_phases[idx,:].reshape((-1)))		
                writer.grab_frame()

            # fig.canvas.draw()
            # plt.pause(del_time)

        #If we're plotting xaxis as angle, animation over frequency
        if xaxis == 'angle':
            for i, pd in enumerate(plot_data):	
                    for j, e in enumerate(ear):
                        pl, = ax.plot(range(len(srcpos[i])), pd[:,0, j].reshape((-1,)), label=subject_names[i]+'_'+e)
                        pls.append(pl)
                    if ring == 'azimuth':
                        xlabels = srcpos[i][0::5,0]
                    if ring == 'elevation':
                        xlabels = srcpos[i][0::5,1]

            if args['phase_plots'] == 'ideal':
                pl_ideal, = ax.plot(range(len(srcpos[i])), ideal_phases[:,0].reshape((-1)), label='ideal phase')

            if args['phase_plots'] == 'euclidean':
                pl_euc, = ax.plot(range(len(srcpos[i])), euclidean_phases[:,0].reshape((-1)),label ='eculidean phase')		

            ax.legend()
            ax.set_xlabel('Angle')
            ax.set_xticks(range(0,len(srcpos[i]),5))
            ax.set_xticklabels(xlabels)
            ax.autoscale(enable=True, axis='x', tight=True)

            freqs = np.concatenate((np.linspace(0,1000,200), np.logspace(3,4.3,500)))
            for freq in freqs:

                for i, pd in enumerate(plot_data):
                    i = i*len(ear)
                    freq_samp = int(nfft*(float(freq)/float(fs[i])))
                    for j, e in enumerate(ear):
                        pls[i+j].set_data(range(len(srcpos[i])),pd[:,freq_samp,j].reshape((-1,)))

                if args['phase_plots'] == 'ideal':
                    pl_ideal.set_data(range(len(srcpos[i])), ideal_phases[:,freq_samp].reshape((-1)))

                if args['phase_plots'] == 'euclidean':
                    pl_euc.set_data(range(len(srcpos[i])), euclidean_phases[:,freq_samp].reshape((-1)))		

                ax.set_title(''.join(ear).upper()+' ear. '+ring+". Frequency: %s" % freq)
                writer.grab_frame()
#			fig.canvas.draw()
#			
#			plt.pause(del_time)
#
        #If we're plotting xaxis as time, animation over position
        if xaxis == 'time':
            for i, pd in enumerate(plot_data):
                for j, e in enumerate(ear):
                    pl, = ax.plot(range(np.shape(pd)[1]), pd[0,:,j].reshape((-1,)), label=subject_names[i]+'_'+e)
                    pls.append(pl)

            ax.legend()
            ax.set_xlabel('Time')
            ax.set_xticks(np.linspace(0, np.shape(plot_data)[2], 5))
            ax.autoscale(enable=True, axis='x', tight=True)

            for idx, pos in enumerate(srcpos[0]):
                for i, pd in enumerate(plot_data):
                    i = i*len(ear)
                    for j, e in enumerate(ear):
                        pls[i+j].set_data(range(np.shape(pd)[1]),pd[idx,:,j].reshape((-1,)))
                    ax.set_title(''.join(ear).upper()+' ear. pos: (%.2f, %.2f, %.2f)' % (pos[0], pos[1], pos[2]))
                    writer.grab_frame()

#			fig.canvas.draw()
#			plt.pause(del_time)
    # plt.savefig("plot_hd5.png")

if __name__ == '__main__':
	main()
