from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, multiply, add, concatenate
import keras.layers as kl
from keras.models import Sequential, Model, model_from_json, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.engine.topology import Layer
import keras.initializers as ki

import sys, os, shutil, argparse, h5py, time
import scipy.io as sio
import pylab as plt
import math as m
import numpy as np
from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.network_data import Data
from Network import Network
from NetworkMag import NetworkMag
from NetworkMagmean import NetworkMagmean
from NetworkMagRI import NetworkMagRI
from NetworkMagstd import NetworkMagstd
from NetworkMagFinal import NetworkMagFinal
import globalvars

class NetworkMagReconL(Network):
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
                both_ears=['l'], 
                loss_function = globalvars.custom_loss_renormalize)
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
                both_ears=['l'], 
                loss_function = globalvars.custom_loss_renormalize)

    def load_external_model(self, network_name):
        print ('./kmodels/'+self.model_details+'_'+network_name +'.h5')
        try:
            self.input_networks[network_name] = load_model('./kmodels/'+self.model_details+'_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
        except Exception as err:
            print (err)

    def make_model(self):
        init_seed = 100
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        #Load previous input models
        mag = self.input_networks['mag']
        magmeanl = self.input_networks['magmeanl']
#        magmeanr = self.input_networks['magmeanr']
        magstdl = self.input_networks['magstdl']
#        magstdr = self.input_networks['magstdr']
        #Get new magri, magrimean, magristd 
        mag_out = Lambda(globalvars.identity, name=self.model_name+'_lambda_mag')(mag(self.input_layers.values()))
        mag_out_l = Lambda(globalvars.get_left, name=self.model_name+'_lambda_mag_l')(mag_out)
#        mag_out_r = Lambda(globalvars.get_right, name=self.model_name+'_lambda_mag_r')(mag_out)
        magmean_out_l = Lambda(globalvars.identity, name=self.model_name+'_lambda_magmean_l')(magmeanl([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']]))
#        magmean_out_r = Lambda(globalvars.identity, name=self.model_name+'_lambda_magmean_r')(magmeanr([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']]))
        magstd_out_l = Lambda(globalvars.identity, name=self.model_name+'_lambda_magstd_l')(magstdl([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']]))
#        magstd_out_r = Lambda(globalvars.identity, name=self.model_name+'_lambda_magstd_r')(magstdr([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']]))
        mag_l = Lambda(globalvars.recalc, name=self.model_name+'_lambda_mag_recalc_l')([mag_out_l, magmean_out_l, magstd_out_l])
#        mag_r = Lambda(globalvars.recalc, name=self.model_name+'_lambda_mag_recalc_r')([mag_out_r, magmean_out_r, magstd_out_r])
        num_in_neurons = int(mag_l.shape[1])
        input_l = concatenate([mag_l, self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']], axis=1)
        layert_l = Dense(num_out_neurons, kernel_initializer=globalvars.custom_init, activation=globalvars.custom_activation_sig, name=self.model_name+'input_l')(input_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden1_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden2_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden3_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden4_l')(layert_l)
#        layert_l = Dense(3*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden2_l')(layert_l)
#        layert_l = Dense(6*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden3_l')(layert_l)
#        layert_l = Dense(9*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden4_l')(layert_l)
#        layert_l = Dense(12*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden5_l')(layert_l)
#        layert_l = Dense(9*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden6_l')(layert_l)
#        layert_l = Dense(6*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden7_l')(layert_l)
#        layert_l = Dense(3*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden8_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden5_l')(layert_l)
        output_magrecon_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.output_names[0])(layert_l)
        #Split into right
#        num_in_neurons = int(mag_r.shape[1])
#        input_r = concatenate([mag_r, self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']], axis=1)
#        layert_r = Dense(num_out_neurons, kernel_initializer=globalvars.custom_init, activation=globalvars.custom_activation_sig, name=self.model_name+'input_r')(input_r)
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden1_r')(mag_r)
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden2_r')(layert_r)
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden3_r')(layert_r)
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden4_r')(layert_r)
##        layert_r = Dense(3*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden2_r')(layert_r)
##        layert_r = Dense(6*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden3_r')(layert_r)
##        layert_r = Dense(9*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden4_r')(layert_r)
##        layert_r = Dense(12*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden5_r')(layert_r)
##        layert_r = Dense(9*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden6_r')(layert_r)
##        layert_r = Dense(6*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden7_r')(layert_r)
##        layert_r = Dense(3*num_in_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation, name=self.model_name+'_hidden8_r')(layert_r)
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.model_name+'_hidden5_r')(layert_r)
#        output_magrecon_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0),  activation=globalvars.custom_activation_sig, name=self.output_names[1])(layert_r)
        self.model = Model(inputs=self.input_layers.values(), outputs=[output_magrecon_l])
        self.model.name = 'magreconl_model'

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
        #    self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    def train(self):
        Network.train(self)

    def evaluate_local(self):
        Network.evaluate_local(self)


