from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate, Lambda
from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
import tensorflow.keras.initializers as ki

# import sys, os, shutil, argparse, h5py, time
# import scipy.io as sio
# import pylab as plt
import math as m
import numpy as np
# # from utilities import read_hdf5, write_hdf5, remove_points, normalize
# from utilities.network_data import Data
from .Network import Network
import network_classes.globalvars as globalvars

 
class NetworkImagmean(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None, created_by ="", run_type = "train", dropout = 0.0):   
        self.run_type = run_type
        self.created_by = created_by
        self.dropout = dropout
        super().__init__(# Network.__init__(self, 
            data, inputs, input_layers,
            percent_validation_data=percent_validation_data,
            model_details=model_details, 
            model_details_prev=model_details_prev, 
            epochs=epochs, 
            iterations=iterations, 
            batch_size=batch_size,
            init_valid_seed=init_valid_seed,
            # loss_function=globalvars.custom_loss_MSE,
            loss_function=globalvars.custom_loss_meannetworks,
            )

    def make_model(self):
        init_seed = 100
        main_input_l = concatenate([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']], axis=1)
        main_input_r = concatenate([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']], axis=1)
        num_in_neurons = 0
        for iput in self.inputs.values():
            num_in_neurons = num_in_neurons + np.shape(iput.getTrainingData())[1]
        layertl = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_l_hidden1')(main_input_l)
        layertl = Dropout(self.dropout) (layertl)
        layertl = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_l_hidden2')(layertl)
        layertl = Dropout(self.dropout) (layertl)
        layertl = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_l_hidden3')(layertl)
        layertl = Dropout(self.dropout) (layertl)
        layertl = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_l_hidden4')(layertl)
        layertl = Dropout(self.dropout) (layertl)
        layertr = Dense(num_in_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_r_hidden1')(main_input_r)
        layertr = Dropout(self.dropout) (layertr)
        layertr = Dense(int(m.ceil(num_in_neurons/2)), kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_r_hidden2')(layertr)
        layertr = Dropout(self.dropout) (layertr)
        layertr = Dense(int(m.ceil(num_in_neurons/4)), kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_r_hidden3')(layertr)
        layertr = Dropout(self.dropout) (layertr)
        layertr = Dense(int(m.ceil(num_in_neurons/8)), kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_r_hidden4')(layertr)
        layertr = Dropout(self.dropout) (layertr)
        output_l = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.output_names[0])(layertl)
        output_r = Dense(1, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation, name=self.output_names[1])(layertr)
        self.model = Model(inputs=list(self.input_layers.values()), outputs=[output_l, output_r])
        self.model._name = 'imagmean_model'

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingMean()[:,:,0], 'valid': d.getValidMean()[:,:,0], 'test': d.getTestMean()[:,:,0]}
            self.output_dict[k+'_r'] = {'training': d.getTrainingMean()[:,:,1], 'valid': d.getValidMean()[:,:,1], 'test': d.getTestMean()[:,:,1]}

    # def compile_model(self):
    #     checkpoint = ModelCheckpoint('./weights/'+self.model_details+'_'+self.model_name, verbose=1, save_best_only=True)
    #     self.callbacks_list = [checkpoint]
    #     self.model.compile(optimizer='adam',
    #             loss=globalvars.custom_loss_meannetworks)
                
                
    def train(self):
        super().train() # Network.train(self)

    def evaluate_local(self):
        super().evaluate_local() # Network.evaluate_local(self)
