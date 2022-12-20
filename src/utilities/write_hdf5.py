import os
import h5py
import glob
import argparse
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

def writeData(db, subject, hrir_data, hrir_name, pos_data=None, nn_data=None, db_filepath='../datasets/', ear='l', force=False):
	db_filepath = db_filepath+'/'+db
	subject_name = 'subject_'+subject

	if hrir_name == 'raw':
		print ("Cannot overwrite raw data")
		return
	else:
		hrir_ds_loc = 'hrir_'+ear+'/'+hrir_name
		if pos_data != None:
			pos_ds_loc = 'srcpos/'+hrir_name
		if nn_data != None:
			nn_ds_loc = 'nn/'+hrir_name
	
	f = h5py.File(os.path.expanduser(db_filepath+'.hdf5'), 'a')

	try:
		f.create_dataset(subject_name+'/'+hrir_ds_loc, data=hrir_data, maxshape=(None, None))
		if pos_data != None:
			f.create_dataset(subject_name+'/'+pos_ds_loc, data=pos_data, maxshape=(None, None))
		if nn_data != None:
			f.create_dataset(subject_name+'/'+nn_ds_loc, data=nn_data, maxshape=(None, None))
	except:
		if not force:
		    answer = raw_input('Overwrite %s/%s [y/n]? ' % (subject_name, hrir_ds_loc))
		else:
		    answer = "y"
		if 'n' in answer.lower():
			print ('Not overwriting %s' % hrir_ds_loc)
		elif 'y' in answer.lower():
			print ('Overwriting %s' % hrir_ds_loc)
			dset = f[subject_name+'/'+hrir_ds_loc]	
			if np.shape(dset) != np.shape(hrir_data):
				f[subject_name+'/'+hrir_ds_loc].resize(np.shape(hrir_data))

			f[subject_name+'/'+hrir_ds_loc][:] = hrir_data
			if pos_data != None:
				f[subject_name+'/'+pos_ds_loc][:] = pos_data
			if nn_data != None:
				f[subject_name+'/'+nn_ds_loc][:] = nn_data
		else:
			print ('Not a valid response. Not overwriting')

