from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate
from keras.models import Sequential, Model, model_from_json, load_model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.utils import plot_model
import keras.initializers as ki

import sys, os, shutil, argparse, h5py, time
import scipy.io as sio
import pylab as plt
import math as m
import numpy as np
from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.network_data import Data
from Network import Network
import globalvars
 
class NetworkMagraw(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None): 
        Network.__init__(self, 
            data, inputs, input_layers,
            percent_validation_data=percent_validation_data,
            model_details=model_details, 
            model_details_prev=model_details_prev, 
            epochs=epochs, 
            iterations=iterations, 
            batch_size=batch_size,
            init_valid_seed=init_valid_seed,
            loss_function=globalvars.custom_loss_renormalize)

    def make_model(self):
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        
        init_seed = 100
        main_input_l = concatenate([self.input_layers['position'], self.input_layers['head']], axis=1)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+'_hidden1_l')(main_input_l)
        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+'_hidden2_l')(layert_l)
        layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+'_hidden3_l')(layert_l)
        layert_l = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+'_hidden4_l')(layert_l)
        layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.model_name+'_hidden5_l')(layert_l)
        layert_l = concatenate([self.input_layers['ear_left'], layert_l], axis=1)
        layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+'hidden6_l')(layert_l)
        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'hidden7_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'hidden8_l')(layert_l)
        output_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.output_names[0])(layert_l)

        init_seed = 100
        main_input_r = concatenate([self.input_layers['position'], self.input_layers['head']], axis=1)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+'_hidden1_r')(main_input_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+'_hidden2_r')(layert_r)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+'_hidden3_r')(layert_r)
        layert_r = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+'_hidden4_r')(layert_r)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.model_name+'_hidden5_r')(layert_r)
        layert_r = concatenate([self.input_layers['ear_right'], layert_r], axis=1)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+'hidden6_r')(layert_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'hidden7_r')(layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'hidden8_r')(layert_r)
        output_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.output_names[1])(layert_r)

#        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
#        main_input_l = concatenate([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']], axis=1)
#        main_input_l = concatenate([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']], axis=1)
##        layert = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+'_hidden1')(main_input)
##        layert = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+'_hidden2')(layert)
##        layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+'_hidden3')(layert)
##        layert = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+'_hidden4')(layert)
##        layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.model_name+'_hidden5')(layert)
#        #Split into left
#        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_hidden7_l')(layert_l)
#        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_hidden7_l')(layert_l)
#        layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+'_hidden6_l')(layert_l)
#        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_hidden7_l')(layert_l)
#        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'_hidden8_l')(layert_l)
#        #Split into right
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_hidden7_r')(layert_r)
#        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_hidden7_r')(layert_r)
#        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+'_hidden6_r')(layert_r)
#        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+'_hidden7_r')(layert_r)
#        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+'_hidden8_r')(layert_r)
#        output_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.output_names[0])(layert_l)
#        output_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.output_names[1])(layert_r)
        self.model = Model(inputs=self.input_layers.values(), outputs=[output_l, output_r])
        self.model.name = 'magl_model'

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
            self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    def train(self):
        Network.train(self)

    def evaluate_local(self):
        Network.evaluate_local(self)
