#! /usr/bin/env python

import os
import sys
sys.path.append("..")
import h5py

import glob
import math as m
import scipy.io as sio
import scipy.io.wavfile as wav
import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import time

import argparse

from utilities import calc_phase, read_hdf5, write_hdf5

num_head_params = 17
num_ear_params = 10

def parseargs():
  parser = argparse.ArgumentParser(description='Delete samples before based on David\'s ideal distance algorithm.\nOnly keeps the first N samples (as secified by -n). Also scales by d (in meters) from the source to the ear in order to counteract the 1/d scaling in space3d.')
  parser.add_argument('db', type=str,help='URL to downnload from')	
  parser.add_argument('subjects', type=str, nargs='+', help='Subject number')
  parser.add_argument('-t', '--type', dest='type', type=str, default=None, help='Name of the dataset in the hdf5')
  parser.add_argument('-n', '--num', dest='num', type=int, default=(-1), help='Number of samples to keep (default: all)')
  parser.add_argument('-d', '--dir', dest='directory', type=str, default='../../datasets/', help='Directory of datasets')
  parser.add_argument('-r', '--ring', dest='ring', type=str, default=None, help='Which ring to view animation over [azimuth, elevation] (default: None)')
  parser.add_argument('-p', '--phasetype', dest='phase_type', type=str, default='ideal', help='Which type of phase to reconstruct with')
  parser.add_argument('-e', '--ears', dest='ears', nargs='+', type=str, default=['l', 'r'], help="Which ears to view (default ['l', 'r'])")

  parser.add_argument('-list', action='store_true', help="List the subject numbers for the given database")	
  parser.add_argument('-quantize', action='store_true', help='Quantize the reconstruction or not. (default: off)')
  parser.add_argument('-f', '--force', dest='force', action='store_true', help='Force overwrite')

  args = vars(parser.parse_args())
  return args
	
def printname(name):
	print (name)

def main():
  #Parse Arguments
  args = parseargs()

  db = args['db']
  db_filepath = args['directory']+db
  subjects = args['subjects']
  ear = args['ears']
  ring = args['ring']
  db_type = args['type']
  force = args['force']
  list_subjects = args['list']
  num_samples = args['num']
  
  if db_type is None:
    print ("Specify a name for the new truncated database: -t <NEW-NAME>")
    exit()

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
	
	#Number of samples to remove
  rem_samps = 0

	#Read in the data
  hrirs, srcpos, fs, subj_list, nn = read_hdf5.getData(db, subj_list, db_filepath=args['directory'], ring=ring, ear=ear, hrir_type='raw', list_subjects=list_subjects, ret_subjs=True)

  headinfo, _, nan_subjs = read_hdf5.getAnthroData(db, subj_list, db_filepath=args['directory'], ring=ring, hrir_type='raw')

  if np.shape(nan_subjs)[0] != 0:
      hrirs = np.delete(hrirs, nan_subjs, axis=0)
      srcpos = np.delete(srcpos, nan_subjs, axis=0)
      fs = np.delete(fs, nan_subjs, axis=0)
      headinfo = np.delete(headinfo, nan_subjs, axis=0)
      subj_list = np.delete(subj_list, nan_subjs, axis=0)
  else:
      headinfo = np.zeros((np.shape(subj_list)[0], num_head_params))

	#Initialize the output array
  out = []
  hrirs_end = []

	#Pad the correct number of zeros for each dataset
  num_zeros=0

  if args['db'] == 'cipic':	
    calc_phase.SoundSpeed = 345
    num_zeros = 105
  elif args['db'] == 'listen':
    num_zeros = 250
  elif args['db'] == 'sadie':
    num_zeros = 200
  elif args['db'] == 'scut':
    calc_phase.SoundSpeed = 345
    num_zeros = 87
  elif args['db'] == 'ari':
    num_zeros = 50

	#Generate continuous window for decay
  win_fadeinlen = 4;
  win_len = num_samples;
  if win_len == -1:
      win_fadeoutlen = 0;
      window = 1
  else:
      win_fadeoutlen = 8;
      window = np.ones((win_len))
      window[0:win_fadeinlen] = (np.sin(np.linspace(-m.pi/2,m.pi/2, win_fadeinlen))+1)/2
      window[win_len-win_fadeoutlen::] = (np.cos(np.linspace(0,m.pi, win_fadeoutlen))+1)/2

  for (subj_idx, hrir) in enumerate(hrirs):
    for n, e in enumerate(ear):
      #Pad the hrir
      hrir_raw= hrir[rem_samps::1,:,n]
      #  print np.shape(hrir_raw)
      hrir_inloop = np.zeros((np.shape(hrir_raw)[0], np.shape(hrir_raw)[1]+num_zeros))	
      for i, h in enumerate(hrir_raw):
              hrir_inloop[i] = np.pad(h, (num_zeros,0), 'constant', constant_values=(0,0))	

      #Set the listener position for the ideal and euclidean phase calculations
      if not headinfo[subj_idx][0]:
          headsize = 0.18;
      else:
          headsize = headinfo[subj_idx][0]/100.

      if e=='l':	
              listener = np.array([90, 0, headsize/2.], dtype=float)
      elif e =='r':
              listener = np.array([270, 0, headsize/2.], dtype=float)	

      # Initialize arrays
      if win_len == -1:
          hrir_deleted = np.zeros((np.shape(hrir_inloop)[0], np.shape(hrir_inloop)[1]-1))
      else:
          hrir_deleted = np.zeros((np.shape(hrir_inloop)[0], win_len))
      dsamp = np.zeros((np.shape(hrir_inloop)[0]), dtype = np.int64)
      freqs = np.zeros((0))
      dists = np.zeros((np.shape(hrir_inloop)[0]))

      for j, source in enumerate(srcpos[subj_idx]):
          _, dsamp[j], dists[j] = calc_phase.get_phase(listener, source, freqs, fs[subj_idx], phase_type=args['phase_type'], quantize=args['quantize']) 

      for i, nsamp in enumerate(dsamp):
          new_hrir = np.delete(hrir_inloop[i], np.arange(0,nsamp))*dists[i]
          hrir_del = np.pad(new_hrir, (0,np.shape(hrir_inloop[i])[0]-np.shape(new_hrir)[0]), 'constant', constant_values=(0,0))
          hrir_deleted[i] = np.multiply(hrir_del[0:win_len], window)
              
      pos = list(srcpos[subj_idx][:][:])
      nn_e = list(nn[subj_idx][:][:])
      #                        print 'subject: ' + subj_list[subj_idx]
      #                        print ('hrir shape: ',e, np.shape(hrir_deleted))
      write_hdf5.writeData(db=db, subject=subj_list[subj_idx], db_filepath=args['directory'],hrir_data=hrir_deleted, pos_data=pos, nn_data=nn_e, hrir_name=args['type'], ear=e, force=force)

      '''
      #Save the original and reconstructed noise bursts
      for i, out_type in enumerate(out):

        filepath = args['filepath']+'/'+args['db']+'/subject_'+args['subject']+'/'

        if  args['type'][i] == 'ori':
          filename = 'subject_'+args['subject']+'_nb_'+args['type'][i]+'.wav'
        else:
          filename = 'subject_'+args['subject']+'_nb_'+args['type'][i]+'_ph'+str(args['phase_type'])+'_q'+str(args['quantize'])+'_lp'+str(args['cutoff'])+'_p'+str(args['padto'])+'.wav'
        if not os.path.exists(filepath):
          os.makedirs(filepath)	
        wav.write(filepath+filename, int(fs[0]), np.asarray(out_type).T)
      '''

	
if __name__ == '__main__':
	main()
