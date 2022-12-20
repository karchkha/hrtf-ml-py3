from tensorflow.keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, Lambda, Concatenate, multiply, add, concatenate
# import keras.layers as kl
from tensorflow.keras.models import  load_model, Sequential, Model, model_from_json
# from keras.callbacks import ModelCheckpoint
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
# from keras.engine.topology import Layer
import tensorflow.keras.initializers as ki
import tensorflow as tf

# import sys, os, shutil, argparse, h5py, time
# import scipy.io as sio
# import pylab as plt
# import math as m
import numpy as np
# from utilities import read_hdf5, write_hdf5, remove_points, normalize
# from utilities.network_data import Data
# from Network import Network
# from NetworkMag import NetworkMag
# from NetworkMagmean import NetworkMagmean
# from NetworkMagRI import NetworkMagRI
# from NetworkMagstd import NetworkMagstd
# from NetworkMagFinal import NetworkMagFinal

import network_manager
from collections import OrderedDict

from .Network import Network
import network_classes.globalvars as globalvars

class NetworkMagTotal(Network):
    def __init__(self, data=None, inputs=None, input_layers=None, input_networks=None, model_details=None, model_details_prev=None, iterations=10, epochs=20, batch_size=32, percent_validation_data=.2, init_valid_seed=None, run_type = "train"): 
        
        self.run_type = run_type
        if type(data) == str:
            self.model_name = data
        else:
            self.data = data
            self.model_name = list(self.data.keys())[0]
        self.model_details = model_details
        self.output_names = []
        self.output_names.append(self.model_name+'_l')
        self.output_names.append(self.model_name+'_r')
        self.output_names.append(self.model_name+'mean_l')
        self.output_names.append(self.model_name+'mean_r')
        self.output_names.append(self.model_name+'std_l')
        self.output_names.append(self.model_name+'std_r')
#        self.output_names.append(self.model_name+'norm_l')
#        self.output_names.append(self.model_name+'norm_r')
        loss_functions = {}
#        loss_functions[self.output_names[0]] = globalvars.custom_loss_renormalize
#        loss_functions[self.output_names[1]] = globalvars.custom_loss_renormalize
#        loss_functions[self.output_names[2]] = globalvars.custom_loss_MSEOverY
#        loss_functions[self.output_names[3]] = globalvars.custom_loss_MSEOverY
#        loss_functions[self.output_names[4]] = globalvars.custom_loss_MSEOverY2
#        loss_functions[self.output_names[5]] = globalvars.custom_loss_MSEOverY2
#        loss_functions[self.output_names[6]] = globalvars.custom_loss_normalized
#        loss_functions[self.output_names[7]] = globalvars.custom_loss_normalized
        loss_functions[self.output_names[0]] = globalvars.custom_loss_MSE
        loss_functions[self.output_names[1]] = globalvars.custom_loss_MSE
        loss_functions[self.output_names[2]] = globalvars.custom_loss_MSE
        loss_functions[self.output_names[3]] = globalvars.custom_loss_MSE
        loss_functions[self.output_names[4]] = globalvars.custom_loss_MSE
        loss_functions[self.output_names[5]] = globalvars.custom_loss_MSE
#        loss_functions[self.output_names[6]] = globalvars.custom_loss_normalized
#        loss_functions[self.output_names[7]] = globalvars.custom_loss_normalized
        self.loss_weights = {}
        self.loss_weights[self.output_names[0]] = 1.0
        self.loss_weights[self.output_names[1]] = 1.0
        self.loss_weights[self.output_names[2]] = 1.0
        self.loss_weights[self.output_names[3]] = 1.0
        self.loss_weights[self.output_names[4]] = 1.0
        self.loss_weights[self.output_names[5]] = 1.0
#        self.loss_weights[self.output_names[6]] = 3.0
#        self.loss_weights[self.output_names[7]] = 3.0
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
                loss_function=globalvars.custom_loss_magtotal)
        except Exception as err:
            print ("Failed loading previously trained model. Creating now.")
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
                loss_function=loss_functions)

    def load_external_model(self, network_name):
        
        print ('Loading dependency ' + network_name)
        
        models = OrderedDict()
        models = network_manager.make_models(network_name, models, created_by = "_magtotal")
        self.input_networks[network_name] = models[network_name]

        # print ('Loading dependency ./kmodels/'+self.model_details+'_'+network_name +'.h5')
        # try:
        #     self.input_networks[network_name] = load_model('./kmodels/'+self.model_details+'_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
        # except Exception as err:
        #     self.input_networks[network_name] = load_model('./kmodels/most_recent_'+network_name+'.h5', custom_objects=globalvars.custom_objects)
        
        # print(self.input_networks)

    def make_model(self):
        print ("Making model")
        init_seed = 100
        num_out_neurons = np.shape(self.data[self.model_name].getTrainingData())[1]
        #Load previous input models
        mag = self.input_networks['mag'].model
        magri = self.input_networks['magri'].model
        magfinal = self.input_networks['magfinal'].model
        magl = self.input_networks['magl'].model
        magr = self.input_networks['magr'].model
        mag._name = 'mag_model'
        magri._name = 'magri_model'
        magfinal._name = 'magfinal_model'
        magl._name = 'magl_model'
        magr._name = 'magr_model'
        
        
        #Change vriables names so it won't repeat and model works properly
        # for l in mag.weights:
        #     l = tf.Variable(l, name="0_"+l.name)
        # for l in magri.weights:
        #     l = tf.Variable(l, name="0_"+l.name)
            
        # mag.save("temp")
        # mag = load_model("temp", custom_objects=globalvars.custom_objects)
        
        
        #Get new magri, magrimean, magristd 
#        mag.trainable=False
#        magri.trainable=False
#        magfinal.trainable = False
#        magl.trainable = False
#        magr.trainable = False
#        mag_out = Lambda(globalvars.identity, trainable=False, name=self.model_name+'_lambda_mag')(mag(self.input_layers.values()))
#        mag_out_l = Lambda(globalvars.get_left,trainable=False, name=self.model_name+'_lambda_mag_l')(mag_out)
#        mag_out_r = Lambda(globalvars.get_right,trainable=False, name=self.model_name+'_lambda_mag_r')(mag_out)
#        magri_out = Lambda(globalvars.identity, trainable=False, name=self.model_name+'_lambda_magri')(magri(self.input_layers.values()))
#        magri_out_l = Lambda(globalvars.get_left, trainable=False, name=self.model_name+'_lambda_magri_l')(magri_out)
#        magri_out_r = Lambda(globalvars.get_right, trainable=False, name=self.model_name+'_lambda_magri_r')(magri_out)
#        magfinal_out = Lambda(globalvars.identity,trainable=False, name=self.model_name+'_lambda_magfinal')(magfinal(self.input_layers.values()))
#        magfinal_out_l = Lambda(globalvars.get_left, trainable=False, name=self.model_name+'_lambda_magfinal_l')(magfinal_out)
#        magfinal_out_r = Lambda(globalvars.get_right, trainable=False,  name=self.model_name+'_lambda_magfinal_r')(magfinal_out)
#        magmean_out_l = Lambda(globalvars.get_first, trainable=False, name=self.model_name+'_lambda_magmean_l')(magl([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']]))
#        magmean_out_r = Lambda(globalvars.get_first, trainable=False, name=self.model_name+'_lambda_magmean_r')(magr([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']]))
#        magstd_out_l = Lambda(globalvars.get_second, trainable=False, name=self.model_name+'_lambda_magstd_l')(magl([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']]))
#        magstd_out_r = Lambda(globalvars.get_second, trainable=False, name=self.model_name+'_lambda_magstd_r')(magr([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']]))
#        mag_l = Lambda(globalvars.recalc, trainable=False, name=self.model_name+'_lambda_mag_recalc_l')([mag_out_l, magmean_out_l, magstd_out_l])
#        mag_r = Lambda(globalvars.recalc, trainable=False, name=self.model_name+'_lambda_mag_recalc_r')([mag_out_r, magmean_out_r, magstd_out_r])
#        magri_l = Lambda(globalvars.recalc, trainable=False, name=self.model_name+'_lambda_magri_recalc_l')([magri_out_l, magmean_out_l, magstd_out_l])
#        magri_r = Lambda(globalvars.recalc, trainable=False, name=self.model_name+'_lambda_magri_recalc_r')([magri_out_r, magmean_out_r, magstd_out_r])
#        magfinal_l = Lambda(globalvars.recalc, trainable=False, name=self.model_name+'_lambda_magfinal_recalc_l')([magfinal_out_l, magmean_out_l, magstd_out_l])
#        magfinal_r = Lambda(globalvars.recalc, trainable=False, name=self.model_name+'_lambda_magfinal_recalc_r')([magfinal_out_r, magmean_out_r, magstd_out_r])
        mag_out = Lambda(globalvars.identity,  name=self.model_name+'_lambda_mag')(mag(self.input_layers.values()))
        mag_out_l = Lambda(globalvars.get_left, name=self.model_name+'_lambda_mag_l')(mag_out)
        mag_out_r = Lambda(globalvars.get_right, name=self.model_name+'_lambda_mag_r')(mag_out)
        magri_out = Lambda(globalvars.identity,  name=self.model_name+'_lambda_magri')(magri(self.input_layers.values()))
        magri_out_l = Lambda(globalvars.get_left,  name=self.model_name+'_lambda_magri_l')(magri_out)
        magri_out_r = Lambda(globalvars.get_right,  name=self.model_name+'_lambda_magri_r')(magri_out)
        magfinal_out = Lambda(globalvars.identity, name=self.model_name+'_lambda_magfinal')(magfinal(self.input_layers.values()))
        magfinal_out_l = Lambda(globalvars.get_left,  name=self.model_name+'_lambda_magfinal_l')(magfinal_out)
        magfinal_out_r = Lambda(globalvars.get_right,   name=self.model_name+'_lambda_magfinal_r')(magfinal_out)
        magmean_out_l = Lambda(globalvars.get_first,  name=self.model_name+'_lambda_magmean_l')(magl([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']]))
        magmean_out_r = Lambda(globalvars.get_first,  name=self.model_name+'_lambda_magmean_r')(magr([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']]))
        magstd_out_l = Lambda(globalvars.get_second,  name=self.model_name+'_lambda_magstd_l')(magl([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_left']]))
        magstd_out_r = Lambda(globalvars.get_second,  name=self.model_name+'_lambda_magstd_r')(magr([self.input_layers['position'], self.input_layers['head'], self.input_layers['ear_right']]))
        mag_l = Lambda(globalvars.recalc,  name=self.model_name+'_lambda_mag_recalc_l')([mag_out_l, magmean_out_l, magstd_out_l])
        mag_r = Lambda(globalvars.recalc,  name=self.model_name+'_lambda_mag_recalc_r')([mag_out_r, magmean_out_r, magstd_out_r])
        magri_l = Lambda(globalvars.recalc,  name=self.model_name+'_lambda_magri_recalc_l')([magri_out_l, magmean_out_l, magstd_out_l])
        magri_r = Lambda(globalvars.recalc,  name=self.model_name+'_lambda_magri_recalc_r')([magri_out_r, magmean_out_r, magstd_out_r])
        magfinal_l = Lambda(globalvars.recalc,  name=self.model_name+'_lambda_magfinal_recalc_l')([magfinal_out_l, magmean_out_l, magstd_out_l])
        magfinal_r = Lambda(globalvars.recalc,  name=self.model_name+'_lambda_magfinal_recalc_r')([magfinal_out_r, magmean_out_r, magstd_out_r])
#        input_mag_l = concatenate([mag_l, Input(shape=(num_out_neurons,), tensor=K.variable(np.ones((num_out_neurons,1))))], axis=1)
#        input_magri_l = concatenate([magri_l, Input(shape=(num_out_neurons,), tensor=K.variable(np.ones((num_out_neurons,1))))], axis=1)
#        input_magfinal_l = concatenate([magfinal_l, Input(shape=(num_out_neurons,), tensor=K.variable(np.ones((num_out_neurons,1))))], axis=1)
#       #input_mag_l = concatenate([mag_l, K.constant(1.0, shape=(num_out_neurons,))], axis=1)
#        const_lam = Lambda(globalvars.identity, name=self.model_name+'_lambda_const_input')(const)
#        input_magri_l = concatenate([magri_l, const], axis=1)
#        input_magfinal_l = concatenate([magfinal_l, const], axis=1)
        layert_mag_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=(1.0/3.0)), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_mag_input_l')(mag_l)
        layert_magri_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=(1.0/3.0)), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_magri_input_l')(magri_l)
        layert_magfinal_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=(1.0/3.0)), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_magfinal_input_l')(magfinal_l)
        layert_magavg_l = add([layert_mag_l, layert_magri_l, layert_magfinal_l], name=self.model_name+'_magavg_l')
#        layert_magavg_db_l = Lambda(globalvars.mag_to_db, trainable=False, name=self.model_name+'_lambda_mag_to_db_l')(layert_magavg_l)
#        input_mag_r = concatenate([mag_r, const], axis=1)
#        input_magri_r = concatenate([magri_r, const], axis=1)
#        input_magfinal_r = concatenate([magfinal_r, const], axis=1)
        layert_mag_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=(1.0/3.0)), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_mag_input_r')(mag_r)
        layert_magri_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=(1.0/3.0)), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_magri_input_r')(magri_r)
        layert_magfinal_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=(1.0/3.0)), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_magfinal_input_r')(magfinal_r)
        layert_magavg_r = add([layert_mag_r, layert_magri_r, layert_magfinal_r], name=self.model_name+'_magavg_r')
#        layert_magavg_db_r = Lambda(globalvars.mag_to_db, trainable=False, name=self.model_name+'_lambda_mag_to_db_r')(layert_magavg_r)
        pos_norm = Lambda(globalvars.data_normalize, trainable=False, name=self.model_name+'_lambda_pos_normalize_l', arguments={'div': globalvars.pos_div, 'scale': globalvars.input_scale})(self.input_layers['position'])
        head_norm = Lambda(globalvars.data_normalize, trainable=False, name=self.model_name+'_lambda_head_normalize_l', arguments={'div': globalvars.head_div, 'scale': globalvars.input_scale})(self.input_layers['head'])
        ear_l_norm = Lambda(globalvars.data_normalize, trainable=False, name=self.model_name+'_lambda_ear_normalize_l', arguments={'div': globalvars.left_ear_div, 'scale': globalvars.input_scale})(self.input_layers['ear_left'])
        ear_r_norm = Lambda(globalvars.data_normalize, trainable=False, name=self.model_name+'_lambda_ear_normalize_r', arguments={'div': globalvars.right_ear_div, 'scale': globalvars.input_scale})(self.input_layers['ear_right'])
        input_l = concatenate([layert_magavg_l, pos_norm, head_norm, ear_l_norm], axis=1)
        input_r = concatenate([layert_magavg_r, pos_norm, head_norm, ear_r_norm], axis=1)
        num_in_neurons = int(layert_magavg_l.shape[1])
        layert_l = Dense(num_out_neurons, kernel_initializer=globalvars.custom_init_zeros_ident(), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_magavg_db_input_l')(layert_magavg_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden1_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden2_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden3_l')(layert_l)
        layert_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden4_l')(layert_l)
        output_magtotal_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.output_names[0])(layert_l)
        #output_magtotal_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal, name=self.output_names[0])(layert_l)
#        output_magtotal_norm_l = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal, name=self.output_names[6])(output_magtotal_l)
#        output_magtotal_l = Lambda(globalvars.positive, name=self.output_names[0])(layert_l)
        output_magmean_l = Lambda(globalvars.mean, name=self.output_names[2])(output_magtotal_l)
        output_magstd_l = Lambda(globalvars.std, name=self.output_names[4])(output_magtotal_l)
#        output_magtotal_l = Lambda(globalvars.positive, name=self.output_names[0])(layert_l)
        #Split into right
#        num_in_neurons = int(input_r.shape[1])
        num_in_neurons = int(layert_magavg_r.shape[1])
        layert_r = Dense(num_out_neurons, kernel_initializer=globalvars.custom_init_zeros_ident(), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_magavg_db_input_r')(layert_magavg_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden1_r')(layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden2_r')(layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden3_r')(layert_r)
        layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.model_name+'_hidden4_r')(layert_r)
        output_magtotal_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal_relu, name=self.output_names[1])(layert_r)
        #output_magtotal_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal, name=self.output_names[1])(layert_r)
        #layert_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal, name=self.model_name+'_hidden5_r')(layert_r)
#        output_magtotal_norm_r = Dense(num_out_neurons, kernel_initializer=ki.identity(gain=1.0), activation=globalvars.custom_activation_magtotal, name=self.output_names[7])(output_magtotal_r)
        output_magmean_r = Lambda(globalvars.mean, name=self.output_names[3])(output_magtotal_r)
        output_magstd_r = Lambda(globalvars.std, name=self.output_names[5])(output_magtotal_r)
        #output_magtotal_r = Lambda(globalvars.positive, name=self.output_names[1])(layert_r)
        self.model = Model(inputs=list(self.input_layers.values()), outputs=[output_magtotal_l, output_magtotal_r, output_magmean_l, output_magmean_r, output_magstd_l, output_magstd_r])
        self.model._name = 'magtotal_model'

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

    def train(self):
        super().train() # Network.train(self)

    def evaluate(self):
        super().evaluate() # Network.evaluate(self)


