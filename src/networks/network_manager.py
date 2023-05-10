#! /usr/bin/env python

from tensorflow.keras.layers import Input, Lambda #, Dense, Activation, Dropout, Flatten, Reshape, concatenate, 
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
# from keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
import tensorflow as tf

# import sys, os, shutil, argparse, h5py, time
import numpy as np
# import scipy.io as sio
# import pylab as plt
# import math as m
from collections import OrderedDict
# from utilities import read_hdf5, write_hdf5, remove_points, normalize
# from utilities.network_data import Data
from utilities.parameters import *
from network_classes import *
import initializer
import data_manager


def load_model(model_name, models):
    if model_name not in models.keys():
        print("")
        print("")
        print("**********************************Loading ",model_name," **********************************")
        #update: updates the dictionary with the given key:value pairs in the argument
        models.update(make_models(model_name, models))

def load_deps(model_name, models):
    # model_deps defined in parameters.py
    if model_name not in model_deps.keys():
        load_model(model_name, models)
        return 
    else:
        for dep in model_deps[model_name]:
            load_deps(dep, models)
        load_model(model_name, models)

def get_deps(model_name, deps):
    if model_name not in model_deps.keys():
        return
    else:
        for dep in model_deps[model_name]:
            if dep not in deps:
                deps.append(dep)
            get_deps(dep, deps)

def make_models(model_names, models_list, created_by = "", run_type = 'train'):
    '''
    inputs:
        model_names : list (list of strings)
        models_list : dict (keys : model names; values : network classes; list of already loaded models)

    Function modifies the models_list dictionary 

    Example:
        Loading the 'mag' network
            models_list['mag'] = NetworkMag (NetworkMag is a Network class from the directory network_classes)
    '''
    if isinstance(model_names, str):
        model_names = [model_names]

    if ('real' in model_names) and ('real' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        # Define Keras's input layers for this network Input() is a Keras function
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_real')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_real')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_real_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],),name='ear_inputs_real_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l), ('ear_right',ear_input_r)])
        #Input data to the network, must match the number and names of input layers
        nw_real = NetworkReal(OrderedDict([('real',real)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['real'] = nw_real

    if ('realmean' in model_names) and ('realmean' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_realmean')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_realmean')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_realmean_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_realmean_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l), ('ear_right',ear_input_r)])
        #Input data to the network, must match the number and names of input layers
        nw_realmean = NetworkRealmean(OrderedDict([('realmean',real)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['realmean'] = nw_realmean

    if ('realstd' in model_names) and ('realstd' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_realstd')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_realstd')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_realstd_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_realstd_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_realstd = NetworkRealstd(OrderedDict([('realstd',real)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['realstd'] = nw_realstd

    if ('imag' in model_names) and ('imag' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_imag')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_imag')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_imag_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_imag_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l), ('ear_right',ear_input_r)])
        nw_imag = NetworkImag(OrderedDict([('imag',imaginary)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations,epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['imag'] = nw_imag

    if ('imagmean' in model_names) and ('imagmean' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_imagmean')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_imagmean')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_imagmean_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_imagmean_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_imagmean = NetworkImagmean(OrderedDict([('imagmean',imaginary)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['imagmean'] = nw_imagmean

    if ('imagstd' in model_names) and ('imagstd' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_imagstd')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_imagstd')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_imagstd_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_imagstd_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_imagstd = NetworkImagstd(OrderedDict([('imagstd',imaginary)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['imagstd'] = nw_imagstd

    if ('mag' in model_names) and ('mag' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_mag')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_mag')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_mag_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_mag_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l), ('ear_right',ear_input_r)])
        nw_mag = NetworkMag(OrderedDict([('mag',magnitude)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations,epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['mag'] = nw_mag


    if ('magmean' in model_names) and ('magmean' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magmean')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magmean')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magmean_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magmean_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_magmean = NetworkMagmean(OrderedDict([('magmean',magnitude)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magmean'] = nw_magmean

    if ('magstd' in model_names) and ('magstd' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magstd')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magstd')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magstd_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magstd_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_magstd = NetworkMagstd(OrderedDict([('magstd',magnitude)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magstd'] = nw_magstd

    if ('magraw' in model_names) and ('magraw' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magraw')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magraw')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magraw_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magraw_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l), ('ear_right',ear_input_r)])
        nw_magraw = NetworkMagraw(OrderedDict([('magraw',magnitude_raw)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations,epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magraw'] = nw_magraw

    if ('magl' in model_names) and ('magl' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magl')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magl')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magl_left')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l)])
        nw_magl = NetworkMagL(OrderedDict([('magl',magnitude_raw), ('maglmean', magnitude_raw), ('maglstd', magnitude_raw)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=maglr_iterations,epochs=maglr_epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magl'] = nw_magl

    if ('magmeanl' in model_names) and ('magmeanl' not in models_list):
        in_networks = model_deps['magmeanl']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magmeanl')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magmeanl')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magmeanl_left')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l)])
        nw_magmeanl = NetworkMagmeanL(OrderedDict([('magmeanl',magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=maglr_iterations,epochs=maglr_epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magmeanl'] = nw_magmeanl

    if ('magstdl' in model_names) and ('magstdl' not in models_list):
        in_networks = model_deps['magstdl']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magstdl')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magstdl')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magstdl_left')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_left',ear_input_l)])
        nw_magstdl = NetworkMagstdL(OrderedDict([('magstdl',magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=maglr_iterations,epochs=maglr_epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magstdl'] = nw_magstdl

    if ('magr' in model_names) and ('magr' not in models_list):
        input_data = OrderedDict([('position',position), ('head', head), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magr')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magr')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magr_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_right',ear_input_l)])
        nw_magr = NetworkMagR(OrderedDict([('magr',magnitude_raw), ('magrmean', magnitude_raw), ('magrstd', magnitude_raw)]), input_data, input_layers, model_details=model_details, model_details_prev=model_details_prev, iterations=maglr_iterations,epochs=maglr_epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magr'] = nw_magr

    if ('magmeanr' in model_names) and ('magmeanr' not in models_list):
        in_networks = model_deps['magmeanr']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magmeanr')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magmeanr')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magmeanr_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_right',ear_input_l)])
        nw_magmeanr = NetworkMagmeanR(OrderedDict([('magmeanr',magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=maglr_iterations,epochs=maglr_epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magmeanr'] = nw_magmeanr

    if ('magstdr' in model_names) and ('magstdr' not in models_list):
        in_networks = model_deps['magstdr']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_right', ear)])
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magstdr')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magstdr')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magstdr_right')
        input_layers = OrderedDict([('position',pos_input), ('head',head_input), ('ear_right',ear_input_l)])
        nw_magstdr = NetworkMagstdR(OrderedDict([('magstdr',magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=maglr_iterations,epochs=maglr_epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magstdr'] = nw_magstdr

    if ('magri' in model_names) and ('magri' not in models_list):
        in_networks = model_deps['magri']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)]) 
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magri')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magri')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magri_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magri_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        # TODO - As we do not normalize the data any more (but in cost functions) magnitude and magnitude_raw are the same
        nw_magri = NetworkMagRI(OrderedDict([('magri', magnitude_raw), ('magrimean', magnitude), ('magristd', magnitude), ('magrinorm', magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magri'] = nw_magri

    if ('magfinal' in model_names) and ('magfinal' not in models_list):
        in_networks = model_deps['magfinal']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)]) 
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magfinal')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magfinal')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magfinal_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magfinal_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_magfinal = NetworkMagFinal(OrderedDict([('magfinal', magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed,  created_by = created_by, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magfinal'] = nw_magfinal

    if ('magrecon' in model_names) and ('magrecon' not in models_list):
        in_networks = model_deps['magrecon']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)]) 
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magrecon')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magrecon')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magrecon_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magrecon_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_magrecon = NetworkMagRecon(OrderedDict([('magrecon', magnitude_raw)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magrecon'] = nw_magrecon

    if ('magreconl' in model_names) and ('magreconl' not in models_list):
        in_networks = model_deps['magreconl']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)]) 
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magreconl')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magreconl')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magreconl_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magreconl_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_magreconl = NetworkMagReconL(OrderedDict([('magreconl', magnitude_raw)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magreconl'] = nw_magreconl

    if ('magreconr' in model_names) and ('magreconr' not in models_list):
        in_networks = model_deps['magreconr']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)]) 
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magreconr')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magreconr')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magreconr_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magreconr_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        nw_magreconr = NetworkMagReconR(OrderedDict([('magreconr', magnitude_raw)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magreconr'] = nw_magreconr

    if ('magtotal' in model_names) and ('magtotal' not in models_list):
        in_networks = model_deps['magtotal']
        input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)]) 
        pos_input = Input(shape=(np.shape(position.getTrainingData())[NUMPOINTS_DIM],), name='pos_inputs_magtotal')
        head_input = Input(shape=(np.shape(head.getTrainingData())[NUMPOINTS_DIM],), name='head_inputs_magtotal')
        ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magtotal_left')
        ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[NUMPOINTS_DIM],), name='ear_inputs_magtotal_right')
        input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
        #nw_magtotal = NetworkMagTotal(OrderedDict([('magtotal', magnitude_raw), ('magtotalmean', magnitude_raw), ('magtotalstd', magnitude_raw), ('magtotalnorm', magnitude)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed)
        nw_magtotal = NetworkMagTotal(OrderedDict([('magtotal', magnitude_raw), ('magtotalmean', magnitude_raw), ('magtotalstd', magnitude_raw)]), input_data, input_layers, input_networks=in_networks, model_details=model_details, model_details_prev=model_details_prev, iterations=iterations, epochs=epochs, init_valid_seed=validation_seed, run_type = run_type, batch_size = batch_size, dropout = dropout)
        models_list['magtotal'] = nw_magtotal
    return models_list

def get_diff_data(model_name):
    diff_data = []
    if 'train' in args['action']:
        return diff_data
    elif os.path.isfile('./diff/'+model_details+'_'+model_name+'_diff.npy'):
        diff_data = np.load('./diff/'+model_details+'_'+model_name+'_diff.npy')
        return diff_data
    else:
        print ("Need to train %s first" % model_name)
        exit()

def get_data_from_model(model, model_name, input_data, subtract_data=None):
    model_output = get_diff_data(model_name)
    pos_local = input_data['position']
    if len(model_output) == 0:
        print ("Generating the difference data")
        for (i, d) in enumerate(input_data.values()[0]):
            pred_in_data = []
            for v in input_data.values():
                pred_in_data.append(v[i].T)
            model_output.append(model.predict(pred_in_data))
        model_output = np.squeeze(np.array(model_output))
        model_output = np.swapaxes(model_output, 1,2)
        #We want to learn the difference between this model output and the actual real data
        if subtract_data is not None:
            model_output = subtract_data.getNormalizedData() - model_output
        num_subj = np.shape(model_output)[0]/1250
        pos_local = np.squeeze(np.reshape(pos_local, (num_subj, 1250, np.shape(pos_local)[1], np.shape(pos_local)[2])))
        model_output = np.reshape(model_output, (num_subj, 1250, np.shape(model_output)[1], np.shape(model_output)[2]))
        np.save('./diff/'+model_details+'_'+model_name+'_diff.npy', model_output) 
    else:
        print ("Loading the last difference data used to train")
    #This is our new training/valid/test data for the second layer of networks
    new_data = Data(model_output, pos=pos_local, test_percent=percent_test_points, test_seed=test_seed, normalize=False)
    new_data.setValidData(percent_valid_points, seed=validation_seed)
    return new_data

def get_left_layer(layer):
    return tf.identity(layer[0])

def get_right_layer(layer):
    return tf.identity(layer[1])

def get_layer(layer):
    return tf.identity(layer)

def get_model(models):
    input_data = OrderedDict([('position',position), ('head', head), ('ear_left', ear), ('ear_right', ear)])
    pos_input = Input(shape=(np.shape(position.getTrainingData())[1],), name='pos_inputs')
    head_input = Input(shape=(np.shape(head.getTrainingData())[1],), name='head_inputs')
    ear_input_l = Input(shape=(np.shape(ear.getTrainingData())[1],), name='ear_inputs_left')
    ear_input_r = Input(shape=(np.shape(ear.getTrainingData())[1],), name='ear_inputs_right')
    input_layers = OrderedDict([('position', pos_input), ('head', head_input), ('ear_left', ear_input_l), ('ear_right', ear_input_r)])
    model_outputs = []
    i = 0
    for mod in models.values():
        model_name = mod.model.name.split('_')[0]
        if model_name == 'magl':
            connection = mod.model([pos_input, head_input, ear_input_l])
        elif model_name == 'magr':
            connection = mod.model([pos_input, head_input, ear_input_r])
        else:
            connection = mod.model([pos_input, head_input, ear_input_l, ear_input_r])
        for (k, outputs) in enumerate(mod.model.outputs):
            layer_name = outputs.name.split('/')[0]
            layer = Lambda(get_layer, name=layer_name)(connection[k])
            print ("Adding " + layer_name + " to model_outputs at index " + str(i))
            i = i+1
            model_outputs.append(layer)
    model = Model(inputs=input_layers.values(), outputs=model_outputs)
    return model

def make_all_models(models_list, models, run_type='train'):
    global position, head, ear, magnitude, magnitude_raw, real, imaginary
    global model_details, model_details_prev, args, subjects
    args = initializer.args
    subjects = initializer.subjects
    model_details = initializer.model_details
    model_details_prev = initializer.model_details_prev
    position, head, ear, magnitude, magnitude_raw, real, imaginary , C_magnitude, C_real, C_imaginary, mean_data = data_manager.get_data()

    # if run_type == 'train':
    #     for model_name in models_list:
    #         load_deps(model_name, models)
    # else:
    #     models = make_models(models_list, models)
    if run_type == 'train':
        models = make_models(models_list, models)
    else:
        models = make_models(models_list, models, run_type = run_type)
        
        
    
    return models
