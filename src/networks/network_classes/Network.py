# from keras.layers import Input, Dense, Activation, Dropout, Flatten, Reshape, concatenate
from tensorflow.keras.models import load_model #Sequential, Model, model_from_json, 
# from keras.callbacks import ModelCheckpoint
# import keras.optimizers as ko
# from keras import backend as K
# from keras.utils.generic_utils import get_custom_objects
from tensorflow.keras.utils import plot_model, CustomObjectScope
import tensorflow as tf

from tensorflow import keras
import sys, os, shutil, argparse, h5py, time
import scipy.io as sio
import numpy as np
# import matplotlib.pyplot as plt 
import pylab as plt
import math as m
# from utilities import read_hdf5, write_hdf5, remove_points, normalize
from utilities.network_data import Data
from time import sleep
# import globalvars
import network_classes.globalvars as globalvars


class Network(object):
    def __init__(self, 
                data, 
                inputs=None, 
                input_layers=None, 
                percent_validation_data=None, 
                model_details='most_recent', 
                model_details_prev=None, 
                epochs=None, 
                iterations=None, 
                batch_size=None, 
                init_valid_seed=100,
                both_ears=['l', 'r'],
                loss_function=None, 
                ):
        try:
            self.model_name
        except:
            if type(data) == str:
                self.model_name = data

            else:
                self.data = data
                self.model_name = list(self.data.keys())[0]
                
        try:
            self.created_by 
        except:
            self.created_by = ""
        
        try:
            self.run_type
        except:
            self.run_type = "train"            
            
        if percent_validation_data != None:
            self.percent_validation_data=percent_validation_data
        if model_details != None:
            self.model_details = model_details
        else:
            self.model_details = 'most_recent'
        if model_details_prev != None:
            self.model_details_prev = model_details_prev
        
        self.modelpath = './kmodels/'+self.model_details+'_'+self.model_name +'.h5'
        self.jsonpath = './jsonmodels/'+self.model_details+'_'+self.model_name +'.json'
        self.val_losspath ='./val_loss/'+self.model_details+'_'+self.model_name+'.npy'
        self.weightspath = './weights/'+self.model_details+'_'+self.model_name+'.h5'
        self.msepath = './mse/'+self.model_details+'_'+self.model_name+'.npy'
        self.graphpath = './graphs/'+self.model_details+'_'+self.model_name+'.png'
        

        if epochs != None:
            self.epochs = epochs
        if iterations != None:
            self.iterations = iterations
        if batch_size != None:
            self.batch_size = batch_size
        if inputs != None:
            self.inputs = inputs
        if input_layers != None:
            self.input_layers = input_layers
        self.seed = init_valid_seed
        self.both_ears = both_ears
        if loss_function is None:
            self.loss_function = globalvars.custom_loss_MSE
        else:
            self.loss_function = loss_function
        
        print("\nStarting:", self.model_name, "in", self.run_type, "mode!")       
        
        if type(data) != str:
            try:
                self.output_names
  
            except:
                self.output_names = [self.model_name+self.created_by+'_l']
                self.output_names.append(self.model_name+self.created_by+'_r')
                
            self.set_valid_training_inputs_outputs()
            self.set_test_inputs_outputs()
            self.mse_shape=(3,len(self.output_names))
            self.load_mse()
            self.load_val_loss()

        self.load_model()
        self.compile_model()
        self.load_weights()
        

    def load_model(self):
        # if os.path.isfile(self.modelpath):
        #     print ("Loading " + self.model_name + ": " + self.modelpath)
        #     self.model = load_model(self.modelpath, custom_objects=globalvars.custom_objects)
        #     self.trained = True
        # else:
        #     self.make_model()
        #     plot_model(self.model, to_file=self.graphpath)
        #     self.trained = False
        
        if os.path.isfile(self.modelpath):
            if self.run_type =="train":
                self.make_model()
                #self.load_weights()
                self.trained = False
            else:
                print ("Loading " + self.model_name + ": " + self.modelpath)
                self.model = load_model(self.modelpath, custom_objects=globalvars.custom_objects)
                self.trained = True
        else:
            self.make_model()
            plot_model(self.model, to_file=self.graphpath)
            self.trained = False
            print("Starting model ", self.model_name, " weights from scratch")


    def compile_model(self):
        if not self.trained:
            print ("Compiling model")
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)     # default was 0.001
            #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)     # default was 0.001
            try:
                self.loss_weights
                self.model.compile(optimizer=optimizer, loss=self.loss_function, loss_weights=self.loss_weights)
            except:
                self.model.compile(optimizer=optimizer, loss=self.loss_function)

    def load_weights(self):
        if os.path.isfile(self.weightspath):
            print ("Loading weights: " + self.weightspath)
            self.model.load_weights(self.weightspath)
        else:
            print("Starting model ", self.model_name, " weights from scratch");

    def load_mse(self):
        if os.path.exists(self.msepath):
            print ("Loading previous mse")
            self.mse = np.load(self.msepath)
        else:
            self.mse = []

    def load_val_loss(self):
        if os.path.exists(self.val_losspath):
            print ("Loading previous losses")
            self.val_loss = np.load(self.val_losspath)
        else:
            self.val_loss = []

    def write_mse(self):
        print ("Saving mse")
        np.save(self.msepath, self.mse) 

    def write_val_loss(self):
        print ("Saving losses")
        np.save(self.val_losspath, self.val_loss) 

    def write_model(self):
        print ("Saving model")
        self.model.save(self.modelpath);
        self.model.save('./kmodels/most_recent_'+self.model_name+'.h5');
        print("Saved "+self.model_name+"  model to disk")

    def write_weights(self):
        print ("Saving weights")
        self.model.save_weights(self.weightspath)

    def set_data(self, key, data):
        print ("Initializing data")
        self.data[key] = data
        self.set_output_dict()
        self.set_valid_training_inputs_outputs()
        self.set_test_inputs_outputs()

    def set_valid_training_inputs_outputs(self):
        print ("Setting validation and training data")
        for d in self.data.values():
            d.setValidData(self.percent_validation_data, seed=self.seed)
        self.set_output_dict()
        in_train_dict, in_valid_dict = {}, {}
        for i, iput in enumerate(self.inputs.values()):
            iput.setValidData(self.percent_validation_data, seed=self.seed)
            
            # print(self.input_layers.values())
            
            if 'left' in list(self.input_layers.values())[i].name.split(':')[0]:
                if np.shape(iput.getTrainingData())[2] == 2:
                    in_train_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTrainingData()[:,:,0])
                    in_valid_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getValidData()[:,:,0])
                else:
                    in_train_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTrainingData())
                    in_valid_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getValidData())
                    
            elif 'right' in list(self.input_layers.values())[i].name.split(':')[0]:
                if np.shape(iput.getTrainingData())[2] == 2:
                    in_train_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTrainingData()[:,:,1])
                    in_valid_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getValidData()[:,:,1])
                else:
                    in_train_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTrainingData())
                    in_valid_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getValidData())
            else:
                in_train_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTrainingData())
                in_valid_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getValidData())
        out_train_dict, out_valid_dict = {}, {}
        for koput, voput in self.output_dict.items():
            out_train_dict[koput] = voput['training']
            out_valid_dict[koput] = voput['valid']
        self.training = (in_train_dict, out_train_dict)
        self.validation = (in_valid_dict, out_valid_dict)

    def set_test_inputs_outputs(self):
        print ("Setting test data")
        in_test_dict = {}
        for i, iput in enumerate(self.inputs.values()):
            if 'left' in list(self.input_layers.values())[i].name.split(':')[0]:
                if np.shape(iput.getTestData())[2] == 2:
                    in_test_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTestData()[:,:,0])
                else:
                    in_test_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTestData())
            elif 'right' in list(self.input_layers.values())[i].name.split(':')[0]:
                if np.shape(iput.getTestData())[2] == 2:
                    in_test_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTestData()[:,:,1])
                else:
                    in_test_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTestData())
            else:
                in_test_dict[list(self.input_layers.values())[i].name.split(':')[0]] = np.squeeze(iput.getTestData())
        out_test_dict = {}
        for koput, voput in self.output_dict.items():
            out_test_dict[koput] = voput['test']
        self.test = (in_test_dict, out_test_dict)

    def get_loss(self, in_dict, out_dict):
        outputs = self.model.predict(in_dict)
        if len(self.output_names) == 1:
            outputs = np.expand_dims(outputs, axis=0)
        losses = []
        try:
            self.loss_weights
            weights = self.loss_weights
        except:
            weights = {}
            for on in self.output_names:
                weights[on] = 1

        for (i, on) in enumerate(self.output_names):
            actual = np.array(out_dict[on])
            if callable(self.loss_function):
                func = self.loss_function
            elif isinstance(self.loss_function, dict):
                func = self.loss_function[on]

            loss = weights[on] * func(actual, outputs[i])

            loss = np.mean(loss, axis=1)
            loss = np.mean(loss, axis=0)
            losses.append(loss)

        return losses

    def get_mse(self):
        training_mse = self.get_loss(self.training[0], self.training[1])
        validation_mse = self.get_loss(self.validation[0], self.validation[1])
        test_mse = self.get_loss(self.test[0], self.test[1])
        return training_mse, validation_mse, test_mse

    def checkpoint(self , validation_loss):
        # validation_loss = self.get_loss(self.validation[0], self.validation[1])
        # validation_loss = np.sum(validation_loss, axis=0)

        if len(self.val_loss) == 0:
            self.val_loss.append(validation_loss)
            print ("\nval_loss started at %f. Saving weights" % (validation_loss))
            self.write_weights()
            self.write_val_loss()
        elif validation_loss < self.val_loss[0]:
            print ("\nval_loss improved from %f to %f. Saving weights" % (self.val_loss[0], validation_loss))
            self.val_loss[0] = validation_loss
            self.write_weights()
            self.write_val_loss()
        
    def train(self):
        print ("Begin training")
        
        Model_ckpt = Custom_ModelCheckpoint(self)   #Here I call intance of custom checkpoint that is defined below
        # Early_Stop =  keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        mse = np.zeros((self.iterations, self.mse_shape[0], self.mse_shape[1]))
        for n in range(self.iterations): 
            self.seed = self.seed + 1
            
            # print("inputs",list(self.training[0].keys()))
            # print("outputs",list(self.training[1].keys()))
            
            self.model.fit( self.training[0],     
                            self.training[1],    
                            epochs= self.epochs, 
                            batch_size = self.batch_size, 
                            validation_data = self.validation,        
                            shuffle = True,
                            callbacks = [Model_ckpt], #Early_Stop],
                            verbose=1) 
            self.load_weights()
            training_mse, valid_mse, test_mse = self.get_mse()
            mse[n, 0, :] = training_mse
            mse[n, 1, :] = valid_mse
            mse[n, 2, :] = test_mse
            self.set_valid_training_inputs_outputs()
        print ("End training")
        if 0 in np.shape(self.mse):
            self.mse = mse
        else:
            self.mse = np.vstack([self.mse, mse])
        self.write_mse()
        self.write_model()

    def evaluate(self):
        fig = plt.figure()
        axl = fig.add_subplot(2,1,1)
        axr = fig.add_subplot(2,1,2)
        axs = [axl, axr]
        self.load_mse()
        axl.plot(self.mse[:,0,0], label='training left')
        axl.plot(self.mse[:,1,0], label='validation left')
        axl.plot(self.mse[:,2,0], label='test left')
        axl.set_title('MSE for '+self.model_name+'Left Ear Data')
        axr.plot(self.mse[:,0,1], label='training right')
        axr.plot(self.mse[:,1,1], label='validation right')
        axr.plot(self.mse[:,2,1], label='test right')
        axr.set_title('MSE for '+self.model_name+'Right Ear Data')
        for ax in axs:
            ax.set_ylabel('MSE')
            ax.legend(loc='best')
            ax.grid(markevery=1)
            ax.set_xlabel('Iteration')
        return fig



#Custom callback to save beest checkpoint after avery epoch

class Custom_ModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, Network):
        super().__init__()
        self.Network = Network

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        Network.checkpoint(self.Network, val_loss)
        # print("\nEnd epoch {} of training; got val_loss: {}".format(epoch, logs["val_loss"]))
