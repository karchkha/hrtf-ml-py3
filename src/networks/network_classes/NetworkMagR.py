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
 
class NetworkMagR(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None, run_type = "train"):
        self.run_type = run_type
        if type(data) == str:
            self.model_name = data
        else:
            self.data = data
            self.model_name = list(self.data.keys())[0]
        self.output_names = []
        self.output_names.append(self.model_name)
        self.output_names.append(self.model_name+'mean')
        self.output_names.append(self.model_name+'std')
        loss_functions = {}

        loss_functions[self.output_names[0]] = globalvars.custom_loss_MSE
        #loss_functions[self.output_names[0]] = globalvars.custom_PZ_Dist_MSE
        # loss_functions[self.output_names[0]] = globalvars.custom_loss_normalized
        # loss_functions[self.output_names[0]] = globalvars.custom_loss_LSD_MSE
        # loss function for mean
#        loss_functions[self.output_names[1]] = globalvars.custom_loss_MSEOverY
        loss_functions[self.output_names[1]] = globalvars.custom_loss_MSE
        # loss function for std
#        loss_functions[self.output_names[2]] = globalvars.custom_loss_MSEOverY2
        loss_functions[self.output_names[2]] = globalvars.custom_loss_MSE
        self.loss_weights = {}
        self.loss_weights[self.output_names[0]] = 0.1
        self.loss_weights[self.output_names[1]] = 1.0
        self.loss_weights[self.output_names[2]] = 1.0

        super().__init__( #Network.__init__(self, 
            data, inputs, input_layers,
            percent_validation_data=percent_validation_data,
            model_details=model_details, 
            model_details_prev=model_details_prev, 
            epochs=epochs, 
            iterations=iterations, 
            batch_size=batch_size,
            init_valid_seed=init_valid_seed,
            both_ears=['r'],
            loss_function=loss_functions)

    def make_model(self):
        init_seed = init_seed = np.random.randint(0,200,1)[0] #100
        anthro_num = globalvars.anthro_num
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]

        #main_input = concatenate([self.input_layers['position'], self.input_layers['head']], axis=1)
        #layert = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden1_r')(main_input)
        #layert = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden2_r')(layert)
        #layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden3_r')(layert)
        #layert = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden4_r')(layert)
        #layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden5_r')(layert)
        #layert = concatenate([self.input_layers['ear_right'], layert], axis=1)
        #layert = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden6_r')(layert)
        #layert = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden7_r')(layert)
        #layert = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation_maglr, name=self.model_name+'_hidden8_r')(layert)
        #output_mag = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation_maglr, name=self.output_names[0])(layert)
        
        
        ActivationFunc = globalvars.custom_activation_maglr_LSD
        head_input_r = concatenate([self.input_layers['ear_right'], self.input_layers['head']], axis=1)
        layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed), activation=ActivationFunc,  name=self.model_name+'_hr1')(head_input_r)
        layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=ActivationFunc, name=self.model_name+'_hr2')(layer_hr)
        layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=ActivationFunc, name=self.model_name+'_hr3')(layer_hr)
        layer_hr = Dense(12*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=ActivationFunc, name=self.model_name+'_hr4')(layer_hr)
        layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=ActivationFunc, name=self.model_name+'_hr5')(layer_hr)
        layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=ActivationFunc, name=self.model_name+'_hr6')(layer_hr)
        layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=ActivationFunc, name=self.model_name+'_hr7')(layer_hr)
        # layer_hr = Dense(10, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=ActivationFunc, name=self.model_name+'_hr8')(layer_hr)

        main_input_r = concatenate([self.input_layers['position'], layer_hr], axis=1)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=ActivationFunc, name=self.model_name+'_hidden1r')(main_input_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=ActivationFunc, name=self.model_name+'_hidden2r')(layert_r)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=ActivationFunc, name=self.model_name+'_hidden3r')(layert_r)
        layert_r = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=ActivationFunc, name=self.model_name+'_hidden4r')(layert_r)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=ActivationFunc, name=self.model_name+'_hidden5r')(layert_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=ActivationFunc, name=self.model_name+'_hidden6r')(layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+4), activation=ActivationFunc, name=self.model_name+'_hidden7r')(layert_r)

        # layert_r = Lambda(globalvars.positive,  name=self.model_name+'_lambda_pos_magr')(layert_r)
        # output_mag = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation_maglr_final, name=self.output_names[0])(layert_r)
        output_mag = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=ActivationFunc, name=self.output_names[0])(layert_r)
        
        
        
        
        
        
        
        
        
        
        output_magmean = Lambda(globalvars.mean, name=self.output_names[1])(output_mag)
        output_magstd = Lambda(globalvars.std, name=self.output_names[2])(output_mag)
        self.model = Model(inputs=list(self.input_layers.values()), outputs=[output_mag, output_magmean, output_magstd])
        self.model._name = 'magr_model'

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            if 'mean' in k:
                self.output_dict[k] = {'training': d.getTrainingMean()[:,:,1], 'valid': d.getValidMean()[:,:,1], 'test': d.getTestMean()[:,:,1]}
            elif 'std' in k:
                self.output_dict[k] = {'training': d.getTrainingStd()[:,:,1], 'valid': d.getValidStd()[:,:,1], 'test': d.getTestStd()[:,:,1]}
            else:
                self.output_dict[k] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    def train(self):
        super().train() # Network.train(self)

    def evaluate(self):
        super().evaluate() # Network.evaluate(self)
