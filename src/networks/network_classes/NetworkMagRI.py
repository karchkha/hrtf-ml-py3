from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, concatenate
# import keras.layers as kl
from tensorflow.keras.models import load_model, Sequential, Model, model_from_json
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
# from keras.engine.topology import Layer

# import sys, os, shutil, argparse, h5py, time
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
# from NetworkMagstd import NetworkMagstd
# import globalvars
from .Network import Network
import network_classes.globalvars as globalvars

import network_manager
from collections import OrderedDict

class NetworkMagRI(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, input_networks=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None, created_by ="", run_type = "train"): 
        self.run_type = run_type
        self.model_details = model_details
        self.created_by = created_by
        try:
            super().__init__(# Network.__init__(self, 
                data, inputs, input_layers,
                percent_validation_data=percent_validation_data,
                model_details=model_details, 
                model_details_prev=model_details_prev, 
                epochs=epochs, 
                iterations=iterations, 
                batch_size=batch_size,
                init_valid_seed=init_valid_seed,
                loss_function=globalvars.custom_loss_normalized)
        except Exception as err:
            if isinstance(input_networks, dict):
                self.input_networks = input_networks
            elif isinstance(input_networks, list):
                self.input_networks = {}
                for network in input_networks:
                    self.load_external_model(network)
            super().__init__(# Network.__init__(self, 
                data, inputs, input_layers,
                percent_validation_data=percent_validation_data,
                model_details=model_details, 
                model_details_prev=model_details_prev, 
                epochs=epochs, 
                iterations=iterations, 
                batch_size=batch_size,
                init_valid_seed=init_valid_seed,
                loss_function=globalvars.custom_loss_normalized)
       

        
    def load_external_model(self, network_name):
        
        print ('Loading dependency ' + network_name)
        
        models = OrderedDict()
        models = network_manager.make_models(network_name, models, created_by = self.created_by + "_magri")
        self.input_networks[network_name] = models[network_name]

        # print ('./kmodels/'+self.model_details+'_'+network_name +'.h5')
        # try:
        #     self.input_networks[network_name] = load_model('./kmodels/'+self.model_details+'_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
        # except Exception as err:
        #     print (err)

    def make_model(self):
#        self.output_names.append('magrimean')
#        self.output_names.append('magristd')
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        #Load previous input models
        real = self.input_networks['real'].model
        realmean = self.input_networks['realmean'].model
        realstd = self.input_networks['realstd'].model
        imag = self.input_networks['imag'].model
        imagmean = self.input_networks['imagmean'].model
        imagstd = self.input_networks['imagstd'].model
        real._name = 'real_model'
        realmean._name = 'realmean_model'
        realstd._name = 'realstd_model'
        imag._name = 'imag_model'
        imagmean._name = 'imagmean_model'
        imagstd._name = 'imagstd_model'
        #Get outputs of previous models 
        
        # print(real)
        # print(self.input_layers.values())
        
        r = Lambda(globalvars.identity,  name=self.model_name + self.created_by+'_lambda_real')(real(self.input_layers.values()))
        rmean = Lambda(globalvars.identity,  name=self.model_name + self.created_by+'_lambda_realmean')(realmean(self.input_layers.values()))
        rstd = Lambda(globalvars.identity,  name=self.model_name + self.created_by+'_lambda_realstd')(realstd(self.input_layers.values()))
        i = Lambda(globalvars.identity,  name=self.model_name + self.created_by+'_lambda_imag')(imag(self.input_layers.values()))
        imean = Lambda(globalvars.identity, name=self.model_name + self.created_by+'_lambda_imagmean')(imagmean(self.input_layers.values()))
        istd = Lambda(globalvars.identity, name=self.model_name + self.created_by+'_lambda_imagstd')(imagstd(self.input_layers.values()))
        #Get new magri, magrimean, magristd 
        magri = Lambda(globalvars.ri_to_mag, name=self.model_name + self.created_by+'_lambda_ri_to_mag')([r, rmean, rstd, i, imean, istd])
        magri_db = Lambda(globalvars.mag_to_db, name=self.model_name + self.created_by+'_lambda_mag_to_db_r')(magri)
        magri_l = Lambda(globalvars.get_left, name=self.model_name + self.created_by+'_lambda_get_left')(magri_db)
        output_magrimean_l = Lambda(globalvars.mag_to_magmean, name=self.model_name + self.created_by+'_lambda_mag_to_magmean_l')(magri_l)
        output_magristd_l = Lambda(globalvars.mag_to_magstd, name=self.model_name + self.created_by+'_lambda_mag_to_magstd_l')(magri_l)
        output_magri_l = Lambda(globalvars.mag_to_magnorm, name=self.output_names[0])([magri_l, output_magrimean_l, output_magristd_l])
        magri_r = Lambda(globalvars.get_right, name=self.model_name + self.created_by+'_lambda_get_right')(magri_db)
        output_magrimean_r = Lambda(globalvars.mag_to_magmean, name=self.model_name + self.created_by+'_lambda_mag_to_magmean_r')(magri_r)
        output_magristd_r = Lambda(globalvars.mag_to_magstd, name=self.model_name + self.created_by+'_lambda_mag_to_magstd_r')(magri_r)
        output_magri_r = Lambda(globalvars.mag_to_magnorm, name=self.output_names[1])([magri_r, output_magrimean_r, output_magristd_r])
        self.model = Model(inputs=list(self.input_layers.values()), outputs=[output_magri_l, output_magri_r])
        self.model._name = 'magri_model'

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
            
            # print("k", k)
            # print ("d", d)
            # if 'mean' in k:
            #     self.output_dict[k+'_l'] = {'training': d.getTrainingMean()[:,:,0], 'valid': d.getValidMean()[:,:,0], 'test': d.getTestMean()[:,:,0]}
            #     self.output_dict[k+'_r'] = {'training': d.getTrainingMean()[:,:,1], 'valid': d.getValidMean()[:,:,1], 'test': d.getTestMean()[:,:,1]}
            # elif 'std' in k:
            #     self.output_dict[k+'_l'] = {'training': d.getTrainingStd()[:,:,0], 'valid': d.getValidStd()[:,:,0], 'test': d.getTestStd()[:,:,0]}
            #     self.output_dict[k+'_r'] = {'training': d.getTrainingStd()[:,:,1], 'valid': d.getValidStd()[:,:,1], 'test': d.getTestStd()[:,:,1]}
            # else:
            #     self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
            #     self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}
        
            if k == "magri":
                self.output_dict[k+'_l'] = {'training': d.getTrainingData()[:,:,0], 'valid': d.getValidData()[:,:,0], 'test': d.getTestData()[:,:,0]}
                self.output_dict[k+'_r'] = {'training': d.getTrainingData()[:,:,1], 'valid': d.getValidData()[:,:,1], 'test': d.getTestData()[:,:,1]}
        # print(self.output_dict)

    def train(self):
        super().train() # Network.train(self)

    def evaluate(self):
        super().evaluate() # Network.evaluate(self)



