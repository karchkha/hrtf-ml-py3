from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate, Lambda
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.initializers as ki

# import sys, os, shutil, argparse, h5py, time
# import scipy.io as sio
# import pylab as plt
# import math as m
import numpy as np
# # from utilities import read_hdf5, write_hdf5, remove_points, normalize
# from utilities.network_data import Data
from .Network import Network
import network_classes.globalvars as globalvars
 
class NetworkReal(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None, created_by="", run_type = "train" , dropout = 0.0):
        self.run_type = run_type
        self.created_by = created_by
        self.dropout = dropout
        super().__init__( #Network.__init__(self, 
            data, inputs, input_layers,
            percent_validation_data=percent_validation_data,
            model_details=model_details, 
            model_details_prev=model_details_prev, 
            epochs=epochs, 
            iterations=iterations, 
            batch_size=batch_size,
            init_valid_seed=init_valid_seed,
            loss_function=globalvars.custom_loss_normalized)

    def make_model(self):
        init_seed = 100
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        
        #main_input = concatenate([self.input_layers['position'], self.input_layers['head']], axis=1)
        #layert = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed),  activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden1')(main_input)
        #layert = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden2')(layert)
        #layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden3')(layert)
        #layert = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden4')(layert)
        #layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden5')(layert)
        ##Split into left
        #layert_l = concatenate([self.input_layers['ear_left'], layert], axis=1)
        #layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'hidden6_l')(layert_l)
        #layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'hidden7_l')(layert_l)
        #layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'hidden8_l')(layert_l)
        ##Split into right
        #layert_r = concatenate([self.input_layers['ear_right'], layert], axis=1)
        #layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'hidden6_r')(layert_r)
        #layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'hidden7_r')(layert_r)
        #layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'hidden8_r')(layert_r)

        anthro_num = globalvars.anthro_num
        # Left anthro
        head_input_l = concatenate([self.input_layers['ear_left'], self.input_layers['head']], axis=1)
        layer_hl = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl1')(head_input_l)
        layer_hl = Dropout(self.dropout) (layer_hl)
        layer_hl = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl2')(layer_hl)
        layer_hl = Dropout(self.dropout) (layer_hl)
        layer_hl = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl3')(layer_hl)
        layer_hl = Dropout(self.dropout) (layer_hl)
        layer_hl = Dense(12*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl4')(layer_hl)
        layer_hl = Dropout(self.dropout) (layer_hl)
        layer_hl = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl5')(layer_hl)
        layer_hl = Dropout(self.dropout) (layer_hl)
        layer_hl = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl6')(layer_hl)
        layer_hl = Dropout(self.dropout) (layer_hl)
        layer_hl = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hl7')(layer_hl)
        layer_hl = Dropout(self.dropout) (layer_hl)


        main_input_l = concatenate([self.input_layers['position'], layer_hl], axis=1)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden1l')(main_input_l)
        layert_l = Dropout(self.dropout) (layert_l)
        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden2l')(layert_l)
        layert_l = Dropout(self.dropout) (layert_l)
        layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden3l')(layert_l)
        layert_l = Dropout(self.dropout) (layert_l)
        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden6l')(layert_l)
        layert_l = Dropout(self.dropout) (layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden7l')(layert_l)
        layert_l = Dropout(self.dropout) (layert_l)

        # right  anthro
        head_input_r = concatenate([self.input_layers['ear_right'], self.input_layers['head']], axis=1)
        layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr1')(head_input_r)
        layer_hr = Dropout(self.dropout) (layer_hr)
        layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr2')(layer_hr)
        layer_hr = Dropout(self.dropout) (layer_hr)
        layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr3')(layer_hr)
        layer_hr = Dropout(self.dropout) (layer_hr)
        layer_hr = Dense(12*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr4')(layer_hr)
        layer_hr = Dropout(self.dropout) (layer_hr)
        layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr5')(layer_hr)
        layer_hr = Dropout(self.dropout) (layer_hr)
        layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr6')(layer_hr)
        layer_hr = Dropout(self.dropout) (layer_hr)
        layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hr7')(layer_hr)
        layer_hr = Dropout(self.dropout) (layer_hr)

        main_input_r = concatenate([self.input_layers['position'], layer_hr], axis=1)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden1r')(main_input_r)
        layert_r = Dropout(self.dropout) (layert_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden2r')(layert_r)
        layert_r = Dropout(self.dropout) (layert_r)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden3r')(layert_r)
        layert_r = Dropout(self.dropout) (layert_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden6r')(layert_r)
        layert_r = Dropout(self.dropout) (layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name + self.created_by+'_hidden7r')(layert_r)  
        layert_r = Dropout(self.dropout) (layert_r)     
        
        #Outputs
        output_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.output_names[0])(layert_l)
        output_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.output_names[1])(layert_r)
        self.model = Model(inputs=list(self.input_layers.values()), outputs=[output_l, output_r])
        self.model._name = "real_model"

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
            self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    def train(self):
        super().train() # Network.train(self)

    def evaluate(self):
        super().evaluate() # Network.evaluate(self)
