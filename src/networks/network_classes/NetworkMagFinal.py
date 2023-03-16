from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, multiply, add, concatenate


import tensorflow.keras.layers as kl

# import tensorflow.keras.engine as ke
# import tensorflow.keras.utils.layer_utils as ke
import tensorflow.python.keras.engine as ke

from tensorflow.keras.models import Sequential, Model, model_from_json, load_model
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
# from keras.engine.topology import Layer
import tensorflow.keras.initializers as ki

import sys, os, shutil, argparse, h5py, time
# import scipy.io as sio
# import pylab as plt
# import math as m
import numpy as np
# from utilities import read_hdf5, write_hdf5, remove_points, normalize
# from utilities.network_data import Data
# from Network import Network
# from NetworkReal import NetworkReal
# from NetworkRealmean import NetworkRealmean
# from NetworkRealstd import NetworkRealstd
# from NetworkImag import NetworkImag
# from NetworkImagmean import NetworkImagmean
# from NetworkImagstd import NetworkImagstd
# from NetworkMag import NetworkMag
# from NetworkMagmean import NetworkMagmean
# from NetworkMagRI import NetworkMagRI
# from NetworkMagstd import NetworkMagstd
from .Network import Network
import network_classes.globalvars as globalvars

import network_manager
from collections import OrderedDict

class NetworkMagFinal(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, input_networks=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None, created_by ="", run_type = "train"): 
        self.model_details = model_details
        #self.loss_function = globalvars.custom_loss_normalized
        self.loss_function = globalvars.custom_loss_MSE
        self.created_by = created_by
        self.run_type = run_type
        try:
            super().__init__(#Network.__init__(self, 
                data, inputs, input_layers,
                percent_validation_data=percent_validation_data,
                model_details=model_details, 
                model_details_prev=model_details_prev, 
                epochs=epochs, 
                iterations=iterations, 
                batch_size=batch_size,
                init_valid_seed=init_valid_seed,
                loss_function = globalvars.custom_loss_normalized)
        except Exception as err:
            if isinstance(input_networks, dict):
                self.input_networks = input_networks
            elif isinstance(input_networks, list):
                self.input_networks = {}
                for network in input_networks:
                    self.load_external_model(network)
            super().__init__(#Network.__init__(self, 
                data, inputs, input_layers,
                percent_validation_data=percent_validation_data,
                model_details=model_details, 
                model_details_prev=model_details_prev, 
                epochs=epochs, 
                iterations=iterations, 
                batch_size=batch_size,
                init_valid_seed=init_valid_seed,
                loss_function = globalvars.custom_loss_normalized)

    def load_external_model(self, network_name):
        
        
        print ('Loading dependency ' + network_name)
        
        models = OrderedDict()
        models = network_manager.make_models(network_name, models, created_by = self.created_by + "_magfinal")
        self.input_networks[network_name] = models[network_name]

        # print ('Loading dependency ./kmodels/'+self.model_details+'_'+network_name +'.h5')
        # try:
        #     self.input_networks[network_name] = load_model('./kmodels/'+self.model_details+'_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
        # except Exception as err:
        #     print (err)

    def make_model(self):
        init_seed = 100
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        #Load previous input models
        mag = self.input_networks['mag'].model
        magri = self.input_networks['magri'].model
        mag._name = 'mag_model'
        magri._name = 'magri_model'
        #Get new magri, magrimean, magristd 
        mag_out = Lambda(globalvars.identity, trainable=False,name=self.model_name + self.created_by+'_lambda_mag')(mag(self.input_layers.values()))
        print (mag_out.shape)
        mag_out_l = Lambda(globalvars.get_left,trainable=False, name=self.model_name + self.created_by+'_lambda_mag_l')(mag_out)
        mag_out_r = Lambda(globalvars.get_right,trainable=False, name=self.model_name + self.created_by+'_lambda_mag_r')(mag_out)
        magri_out = Lambda(globalvars.identity,trainable=False, name=self.model_name + self.created_by+'_lambda_magri')(magri(self.input_layers.values()))
        magri_out_l = Lambda(globalvars.get_left,trainable=False, name=self.model_name + self.created_by+'_lambda_magri_l')(magri_out)
        magri_out_r = Lambda(globalvars.get_right,trainable=False, name=self.model_name + self.created_by+'_lambda_magri_r')(magri_out)
        
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        anthro_num = globalvars.anthro_num
        # Left anthro
        mag_magfinal_l  = concatenate([self.input_layers['ear_left'], self.input_layers['head']], axis=1)
        layer_hl = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl1')(mag_magfinal_l)
        layer_hl = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl2')(layer_hl)
        layer_hl = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl3')(layer_hl)
        layer_hl = Dense(12*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl4')(layer_hl)
        layer_hl = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl5')(layer_hl)
        layer_hl = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl6')(layer_hl)
        layer_hl = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hl7')(layer_hl)

        n=anthro_num+3
        layer_hpl = concatenate([self.input_layers['position'], layer_hl], axis=1)
        layer_hpl = Dense(n, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden1pl')(layer_hpl)
        layer_hpl = Dense(n*3, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden2pl')(layer_hpl)
        layer_hpl = Dense(n*6, kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden3pl')(layer_hpl)
        layer_hpl = Dense(n*3, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden4pl')(layer_hpl)
        layer_hpl = Dense(n, kernel_initializer=ki.glorot_uniform(init_seed+12), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden5pl')(layer_hpl)

        main_input_l = concatenate([layer_hpl, mag_out_l, magri_out_l], axis=1)
        layert_l = Dense(12 * num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden1l')(main_input_l)
        layert_l = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+13), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden5l')(layert_l)
        layert_l = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+14), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden6l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+15), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden7l')(layert_l)

        # right  anthro
        # mag_magri_r  = concatenate([self.input_layers['ear_right'], self.input_layers['head'], mag_out_r, magri_out_r], axis=1)
        # layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr1')(mag_magri_r)
        # layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr2')(layer_hr)
        # layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr3')(layer_hr)
        # layer_hr = Dense(12*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr4')(layer_hr)
        # layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr5')(layer_hr)
        # layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr6')(layer_hr)
        # layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr7')(layer_hr)

        # main_input_r = concatenate([self.input_layers['position'], layer_hr], axis=1)
        # layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden1r')(main_input_r)
        # layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden2r')(layert_r)
        # layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden3r')(layert_r)
        # layert_r = Dense(12*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden4r')(layert_r)
        # layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden5r')(layert_r)
        # layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden6r')(layert_r)
        # layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+12), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden7r')(layert_r)
        mag_magfinal_r  = concatenate([self.input_layers['ear_right'], self.input_layers['head']], axis=1)
        layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr1')(mag_magfinal_r)
        layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+1), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr2')(layer_hr)
        layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+2), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr3')(layer_hr)
        layer_hr = Dense(12*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+3), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr4')(layer_hr)
        layer_hr = Dense(6*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+5), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr5')(layer_hr)
        layer_hr = Dense(3*anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+6), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr6')(layer_hr)
        layer_hr = Dense(anthro_num, kernel_initializer=ki.glorot_uniform(init_seed+7), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hr7')(layer_hr)

        n=anthro_num+3
        layer_hpr = concatenate([self.input_layers['position'], layer_hr], axis=1)
        layer_hpr = Dense(n, kernel_initializer=ki.glorot_uniform(init_seed+8), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden1pr')(layer_hpr)
        layer_hpr = Dense(n*3, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden2pr')(layer_hpr)
        layer_hpr = Dense(n*6, kernel_initializer=ki.glorot_uniform(init_seed+10), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden3pr')(layer_hpr)
        layer_hpr = Dense(n*3, kernel_initializer=ki.glorot_uniform(init_seed+11), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden4pr')(layer_hpr)
        layer_hpr = Dense(n, kernel_initializer=ki.glorot_uniform(init_seed+12), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden5pr')(layer_hpr)

        main_input_r = concatenate([layer_hpr, mag_out_r, magri_out_r], axis=1)
        layert_r = Dense(12 * num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden1r')(main_input_r)
        layert_r = Dense(6*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+13), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden5r')(layert_r)
        layert_r = Dense(3*num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+14), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden6r')(layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+15), activation=globalvars.custom_activation, name=self.model_name+self.created_by+'_hidden7r')(layert_r)
       
        
        
        
        
        mean_l = Lambda(globalvars.mag_to_magmean, name=self.model_name + self.created_by+'hidden10_l')(layert_l)
        mean_r = Lambda(globalvars.mag_to_magmean, name=self.model_name + self.created_by+'hidden10_r')(layert_r)
        std_l = Lambda(globalvars.mag_to_magstd, name=self.model_name + self.created_by+'hidden11_l')(layert_l)
        std_r = Lambda(globalvars.mag_to_magstd, name=self.model_name + self.created_by+'hidden11_r')(layert_r)
        output_magfinal_l = Lambda(globalvars.mag_to_magnorm, name=self.output_names[0])([layert_l, mean_l, std_l])
        output_magfinal_r = Lambda(globalvars.mag_to_magnorm, name=self.output_names[1])([layert_r, mean_r, std_r])
        #output_magfinal_l = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.output_names[0])(layert_l)
        #output_magfinal_r = Dense(num_out_neurons, kernel_initializer=ki.glorot_uniform(init_seed+9), activation=globalvars.custom_activation, name=self.output_names[1])(layert_r)
        self.model = Model(inputs=list(self.input_layers.values()), outputs=[output_magfinal_l, output_magfinal_r])
        self.model._name = 'magfinal_model'
        
        #if not os.path.isfile('./kmodels/'+self.model_details+'_'+self.model_name +'.h5'):
        #    print ("Initializing weights to same as mag")
        #    i = 0
        #    for l in reversed(mag.layers):
        #        if isinstance(l, Dense): #if isinstance(l, kl.core.Dense):
        #            if (i < 14):
        #                layer = list(reversed(self.model.layers))[i]
        #                print ("magfinal layer:", layer, "weight are set to mag layer:", l, "weights")

        #                layer.set_weights(l.get_weights())
        #        elif isinstance(l, kl.InputLayer): #elif isinstance(l, ke.topology.InputLayer):
        #            i = i-1
        #        i = i + 1

    def set_output_dict(self):
        self.output_dict={}
        for k, d in self.data.items():
            self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
            self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}

    def train(self):
        super().train() # Network.train(self)

    def evaluate(self):
        super().evaluate() # Network.evaluate(self)


