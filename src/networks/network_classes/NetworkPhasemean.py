from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate
from keras.models import Sequential, Model, model_from_json, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import keras.initializers as initializers

import sys, os, shutil, argparse, h5py, time
import numpy as np
import scipy.io as sio
import pylab as plt
import math as m
from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.network_data import Data
from Network import Network
 
class NetworkPhasemean(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2): 
        self.custom_objects={'custom_activation_phasemean': NetworkPhasemean.custom_activation_phasemean}
        Network.__init__(self, 
            data, inputs, input_layers,
            percent_validation_data=percent_validation_data,
            model_details=model_details, 
            model_details_prev=model_details_prev, 
            epochs=epochs, 
            iterations=iterations, 
            batch_size=batch_size)

    def make_model(self):
        init_seed = 100
        main_input = concatenate(self.input_layers)
        num_in_neurons = 0
        for iput in self.inputs.values():
            num_in_neurons = num_in_neurons + np.shape(iput.getTrainingData())[1]
        layert = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=NetworkPhasemean.custom_activation_phasemean, name=self.model_name+'_hidden1')(main_input)
        layert = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed),  activation=NetworkPhasemean.custom_activation_phasemean, name=self.model_name+'_hidden2')(layert)
        layert = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed),  activation=NetworkPhasemean.custom_activation_phasemean, name=self.model_name+'_hidden3')(layert)
        layert = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed),  activation=NetworkPhasemean.custom_activation_phasemean, name=self.model_name+'_hidden4')(layert)
        output = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed),  activation=NetworkPhasemean.custom_activation_phasemean, name=self.output_names[0])(layert)
        self.model = Model(inputs=self.input_layers, outputs=output)

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
            self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    @staticmethod
    def custom_activation_phasemean(x):
        return 4*(K.tanh(x/4))

    def train(self):
        Network.train(self)

    def evaluate(self):
        Network.evaluate(self)
