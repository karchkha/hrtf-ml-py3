from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, multiply, add, concatenate
from keras.models import Sequential, Model, model_from_json, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.engine.topology import Layer

import sys, os, shutil, argparse, h5py, time
import scipy.io as sio
import pylab as plt
import math as m
import numpy as np
from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.network_data import Data
from Network import Network
from NetworkReal import NetworkReal
from NetworkImag import NetworkImag
from NetworkPhase import NetworkPhase
 
class NetworkPhaseRI(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, input_networks=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2): 
        self.custom_objects={'custom_activation_real': NetworkReal.custom_activation_real,
                            'custom_activation_imag': NetworkImag.custom_activation_imag,
                            'custom_activation_phase': NetworkPhase.custom_activation_phase,
                            'custom_activation_phaseri': NetworkPhaseRI.custom_activation_phaseri}
        self.input_networks = input_networks
        Network.__init__(self, 
            data, inputs, input_layers,
            percent_validation_data=percent_validation_data,
            model_details=model_details, 
            model_details_prev=model_details_prev, 
            epochs=epochs, 
            iterations=iterations, 
            batch_size=batch_size)
       
    def make_model(self):
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        real = self.input_networks['real']
        imag = self.input_networks['imag']
        phase = self.input_networks['phase']
        r = real.model(self.input_layers)
        i = imag.model(self.input_layers)
        p = phase.model(self.input_layers)
        #phaseri_out = Lambda(self.ri_to_phase)([r, i])
        #phase_out = phase.model(self.input_layers)
        phaseri = concatenate([p, r, i], axis=1)
        layert = Dense(num_out_neurons, activation=NetworkPhaseRI.custom_activation_phaseri, name=self.model_name+'_hidden1')(phaseri)
        layert = Dense(3*num_out_neurons, activation=NetworkPhaseRI.custom_activation_phaseri, name=self.model_name+'_hidden2')(layert)
        layert = Dense(6*num_out_neurons, activation=NetworkPhaseRI.custom_activation_phaseri, name=self.model_name+'_hidden3')(layert)
        layert = Dense(12*num_out_neurons, activation=NetworkPhaseRI.custom_activation_phaseri, name=self.model_name+'_hidden4')(layert)
        layert = Dense(6*num_out_neurons, activation=NetworkPhaseRI.custom_activation_phaseri, name=self.model_name+'_hidden5')(layert)
        output = Dense(num_out_neurons, activation=NetworkPhaseRI.custom_activation_phaseri, name=self.output_names[0])(layert)
        self.model = Model(inputs=self.input_layers, outputs=[output, real.model.outputs[0], imag.model.outputs[0], phase.model.outputs[0]])

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
            self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    def ri_to_phase(self, ri):
        ri = K.pow(ri, 2)
        ri_2 = K.sum(ri, axis=0)
        magri_out = K.sqrt(ri_2)
        return magri_out

    @staticmethod
    def custom_activation_phaseri(x):
        return 4*(K.tanh(x/4))
    def custom_activation_phase(self, x):
        return self.max*(K.tanh(x/self.max))

    def train(self):
        Network.train(self)

    def evaluate_local(self):
        Network.evaluate_local(self)
