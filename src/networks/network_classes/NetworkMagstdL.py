from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, concatenate
import keras.layers as kl
from keras.models import Sequential, Model, model_from_json, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model
from keras.engine.topology import Layer

import sys, os, shutil, argparse, h5py, time
import scipy.io as sio
import pylab as plt
import math as m
import numpy as np
from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.network_data import Data
from Network import Network
import globalvars
 
class NetworkMagstdL(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, input_networks=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None): 
        if type(data) == str:
            self.model_name = data
        else:
            self.model_name = data.keys()[0]
        self.model_details = model_details
        try:
            Network.__init__(self, 
                data, inputs, input_layers,
                percent_validation_data=percent_validation_data,
                model_details=model_details, 
                model_details_prev=model_details_prev, 
                epochs=epochs, 
                iterations=iterations, 
                batch_size=batch_size,
                init_valid_seed=init_valid_seed,
                both_ears=['l'])
        except Exception as err:
            if isinstance(input_networks, dict):
                self.input_networks = input_networks
            elif isinstance(input_networks, list):
                self.input_networks = {}
                for network in input_networks:
                    self.load_external_model(network)
            Network.__init__(self, 
                data, inputs, input_layers,
                percent_validation_data=percent_validation_data,
                model_details=model_details, 
                model_details_prev=model_details_prev, 
                epochs=epochs, 
                iterations=iterations, 
                batch_size=batch_size,
                init_valid_seed=init_valid_seed,
                both_ears=['l'])

    def load_external_model(self, network_name):
        try:
            self.input_networks[network_name] = load_model('./kmodels/'+self.model_details+'_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
            print ('./kmodels/'+self.model_details+'_'+network_name +'.h5')
        except Exception as err:
            print (err)

    def make_model(self):
        init_seed = 100
        magl = self.input_networks['magl']
        magl.name = 'magl_model'
        mag = Lambda(globalvars.identity, name=self.model_name+'_lambda_magl')(magl(self.input_layers.values()))
        magstd = Lambda(globalvars.std, name=self.output_names[0])(mag)
        self.model = Model(inputs=self.input_layers.values(), outputs=[magstd])
        self.model.name = "magstdl_model"

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingStd()[:,:,0], 'valid': d.getValidStd()[:,:,0], 'test': d.getTestStd()[:,:,0]}
#            self.output_dict[k+'_r'] = {'training': d.getTrainingMean()[:,:,1], 'valid': d.getValidMean()[:,:,1], 'test': d.getTestMean()[:,:,1]}

    def train(self):
        Network.train(self)

    def evaluate_local(self):
        Network.evaluate_local(self)
