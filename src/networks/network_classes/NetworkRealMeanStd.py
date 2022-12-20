from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, concatenate
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
from NetworkReal import NetworkReal
from NetworkRealmean import NetworkRealmean
from NetworkRealstd import NetworkRealstd
import globalvars

class NetworkRealMeanStd(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, input_networks=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None): 
        self.run_type = run_type
        if type(data) == str:
            self.model_name = data
        else:
            self.model_name = data.keys()[0]
        self.model_details = model_details
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
            init_valid_seed=init_valid_seed)
       
    def load_external_model(self, network_name):
        print ('./kmodels/'+self.model_details+'_'+network_name +'.h5')
        try:
            self.input_networks[network_name] = load_model('./kmodels/'+self.model_details+'_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
        except Exception as err:
            print (err)

    def make_model(self):
#        self.output_names.append('magrimean')
#        self.output_names.append('magristd')
        init_seed = 100
        #Load previous input models
        real = self.input_networks['real']
        real.name = 'real_model'
        #Get outputs of previous models 
        layerreal = Lambda(globalvars.identity, name=self.model_name+'_lambda_real')(real(self.input_layers.values()))
        real_l = Lambda(globalvars.get_left, name=self.model_name+'_lambda_get_left')(layerreal)
        real_r = Lambda(globalvars.get_right, name=self.model_name+'_lambda_get_right')(layerreal)
        #Get new magri, magrimean, magristd 
        main_input = concatenate([self.input_layers['position'], self.input_layers['head']])
        num_in_neurons = int(main_input.shape[1])
        layert = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden1')(main_input)
        layert = Dense(3*num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden2')(layert)
        layert = Dense(6*num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden3')(layert)
        layert = Dense(12*num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden4')(layert)
        layert = Dense(6*num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden5')(layert)
        layert = Dense(3*num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden6')(layert)
        layert = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_pos_head_hidden7')(main_input)

#        mean_input_l = concatenate([layert, self.input_layers['ear_left']])
#        mean_input_r = concatenate([layert, self.input_layers['ear_right']])
#        num_in_neurons = int(mean_input_l.shape[1])
#        layertl = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'_l_mean_hidden1')(mean_input_l)
#        layertl = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+'_l_mean_hidden2')(layertl)
#        layertl = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+'_l_mean_hidden3')(layertl)
#        layertl = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+'_l_mean_hidden4')(layertl)
#        output_mean_l = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.output_names[2])(layertl)
#        num_in_neurons = int(mean_input_r.shape[1])
#        layertr = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'_r_mean_hidden1')(mean_input_r)
#        layertr = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+'_r_mean_hidden2')(layertr)
#        layertr = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+'_r_mean_hidden3')(layertr)
#        layertr = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+'_r_mean_hidden4')(layertr)
#        output_mean_r = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.output_names[3])(layertr)

        std_input_l = concatenate([layert, self.input_layers['ear_left']])
        std_input_r = concatenate([layert, self.input_layers['ear_right']])
        num_in_neurons = int(std_input_l.shape[1])
        layertl = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'_l_std_hidden1')(std_input_l)
        layertl = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+'_l_std_hidden2')(layertl)
        layertl = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+'_l_std_hidden3')(layertl)
        layertl = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+'_l_std_hidden4')(layertl)
        output_std_l = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.output_names[0])(layertl)
        num_in_neurons = int(std_input_r.shape[1])
        layertr = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'_r_std_hidden1')(std_input_r)
        layertr = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+'_r_std_hidden2')(layertr)
        layertr = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+'_r_std_hidden3')(layertr)
        layertr = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+'_r_std_hidden4')(layertr)
        output_std_r = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.output_names[1])(layertr)

        output_l = Lambda(globalvars.renormalize, name=self.output_names[2])([real_l, output_std_l])
        output_r = Lambda(globalvars.renormalize, name=self.output_names[3])([real_r, output_std_r])
        self.model = Model(inputs=self.input_layers.values(), outputs=[output_std_l, output_std_r, output_l, output_r])
        self.model.name = 'realms_model'

    def get_inputs_for_model(self, models):
        model_inputs = []
        for mod in models:
            for i, m in enumerate(mod.inputs):
                model_inputs.append(m)
        for i, m in enumerate(self.input_layers):
            model_inputs.append(m)
        return model_inputs

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            if 'mean' in k:
                self.output_dict[k+'_l'] = {'training': d.getTrainingMean()[:,:,0], 'valid': d.getValidMean()[:,:,0], 'test': d.getTestMean()[:,:,0]}
                self.output_dict[k+'_r'] = {'training': d.getTrainingMean()[:,:,1], 'valid': d.getValidMean()[:,:,1], 'test': d.getTestMean()[:,:,1]}
            elif 'std' in k:
                self.output_dict[k+'_l'] = {'training': d.getTrainingStd()[:,:,0], 'valid': d.getValidStd()[:,:,0], 'test': d.getTestStd()[:,:,0]}
                self.output_dict[k+'_r'] = {'training': d.getTrainingStd()[:,:,1], 'valid': d.getValidStd()[:,:,1], 'test': d.getTestStd()[:,:,1]}
            else:
                self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
                self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}
        print (self.output_dict.keys())

    def train(self):
        Network.train(self)

    def evaluate(self):
        Network.evaluate(self)


