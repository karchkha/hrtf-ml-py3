#! /usr/bin/env python

# from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate, Lambda
# from keras.models import Sequential, Model, model_from_json, load_model
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects

import sys, os, shutil, argparse, h5py, time
import numpy as np
# import scipy.io as sio
# import pylab as plt
# import math as m
# from collections import OrderedDict
# from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.parameters import *
import time

def parseargs():
    parser = argparse.ArgumentParser(description='Plot the data from a selected dataset and subject')

    # parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
    # parser.add_argument('-db',type=str, default='cipic', help='URL to downnload from')    
    # parser.add_argument('-subjects', type=str, nargs='+', default=['all'], help='Subject number')

    parser.add_argument('db',type=str, help='URL to downnload from')    
    parser.add_argument('subjects', type=str, nargs='+', help='Subject number')
    parser.add_argument('-nt', '--network_type', type=str, dest='network_type', default='mag_phase', help='network_type')
    parser.add_argument('-t', dest='hrir_type', type=str, default='trunc_64', help='Type of the database (default: trunc_64. options: raw,trunc_64,smooth_1,smooth_2,smooth_3)')
    parser.add_argument('-C', dest='C_hrir_type', type=str, default='trunc_64', help='Type of the database to compare the prediction with (default: trunc_64. options: raw,trunc_64,smooth_1,smooth_2,smooth_3)')
    parser.add_argument('-O', dest='train_only', type=str, default=None, help='Only train the listed models')
    parser.add_argument('--noanthro', dest='noanthro', action='store_true', help='Flag to not train with anthro. By default, the network uses anthro data')
    parser.add_argument('-a', '--action', type=str, default='train', nargs='+', help='(train|predict|eval|compile)')
    parser.add_argument('-nn', '--network_number', type=int, default=0, help='Iterate the network number if you changed the network.')
    parser.add_argument('--tag', type=str, default=None, help='Add a tag to the network name')
    parser.add_argument('-e', '--ear',  nargs='+', dest='ear', type=str, default=['l', 'r'], help='Ear [l,r] (default: l)')   
    parser.add_argument('-n', '--nfft', dest='nfft', type=int, default=64, help='Length of the fft')
    parser.add_argument('-d', '--db_path', dest='db_path', type=str, default='../../datasets/', help='Directory of datasets')
    parser.add_argument('-r', '--ring', dest='ring', type=str, default=None, help='Which ring to view animation over [azimuth, elevation] (default: azimuth)')
    parser.add_argument('-clean', action='store_true', help='Clean saved files for corresponding model')
    parser.add_argument('-cleanall', action='store_true', help='Clean saved files for corresponding model')
    parser.add_argument('-cleanmodel', nargs='+', default=None, help='Clean saved files for corresponding model')
    args = vars(parser.parse_args())
    return args

def init():
    global model_details, model_details_prev
    global subjects, train_anthro
    global args

    args = parseargs()
    train_anthro=True

    if (args['db'] in databases_with_no_anthro) or (args['noanthro']):
        train_anthro=False

    if args['tag'] is not None:
        tag = list(args['tag'].replace(' ', ''))
        tag[0] = tag[0].upper()
        tag = ''.join(tag)
    else:
        tag = ''

    #Generate model detail filenames
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
        if args['network_number'] > 0:
            model_details = str(args['network_number'])+'_'+args['db']+'_all'+'_t'+args['hrir_type']+'_r'+str(args['ring'])+'_a'+str(train_anthro)+'_e'+str(''.join(args['ear']))+'_n'+str(args['nfft'])+'_0'
        else:
            model_details = args['db']+'_all'+'_t'+args['hrir_type']+'_r'+str(args['ring'])+'_a'+str(train_anthro)+'_e'+str(''.join(args['ear']))+'_n'+str(args['nfft'])+'_0'

    else:
        subj_list = args['subjects']        
        if args['network_number'] > 0:
            model_details = str(args['network_number'])+'_'+args['db']+'_'+''.join(args['subjects'])+'_t'+args['hrir_type']+'_r'+str(args['ring'])+'_a'+str(train_anthro)+'_e'+str(''.join(args['ear']))+'_n'+str(args['nfft'])+'_0'
        else:
            model_details = args['db']+'_'+''.join(args['subjects'])+'_t'+args['hrir_type']+'_r'+str(args['ring'])+'_a'+str(train_anthro)+'_e'+str(''.join(args['ear']))+'_n'+str(args['nfft'])+'_0'

    #Model trained with main_network-0.1
    model_details = branch + "_" + model_details + '_tag'+tag

    #Setup directories to write checkpoints and finished models.
    dirs = ['./kmodels/', './weights/', './mse/', './tfmodels/', './diff/', './val_loss/', './graphs/', './jsonmodels/']
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)  
        #Remove previous files if clean
        if args['cleanall']:
            shutil.rmtree(d)
        if args['clean']:
            list_files = os.listdir(d)
            for f in list_files:
                if model_details[:-2] in f:
                    os.remove(d+f)
        if args['cleanmodel'] is not None:
            for mod in args['cleanmodel']:
                list_files = os.listdir(d)
                for f in list_files:
                    if mod == os.path.splitext(f)[0].split('_')[-1]:
                        print ("Removing model: " + d + f + " before run.")
                        time.sleep(.5)
                        os.remove(d+f)

    #Exit on clean
    if args['clean'] or args['cleanall']:
        sys.exit()

    model_details_prev = model_details
    subjects = subj_list
