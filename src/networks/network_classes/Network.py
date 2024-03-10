from tensorflow.keras.models import load_model #Sequential, Model, model_from_json, 
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
from utilities.parameters import *

# tf.config.experimental_run_functions_eagerly(True)

class Network(tf.keras.Model):
# class Network(object):
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
                lr = initial_lr,
                output_names = None, 
                loss_weights = None, 
                mask_type = mask_type,               
                ):
        super(Network, self).__init__()
        self.output_names = output_names
        self.loss_weights = loss_weights
        self.lr = lr
        self.mask_type = mask_type
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

            if self.output_names is None:
                self.output_names = [self.model_name+self.created_by+'_l']
                self.output_names.append(self.model_name+self.created_by+'_r')  
            else:
                pass
                # self.output_names = [self.model_name+self.created_by+'_l']
                # self.output_names.append(self.model_name+self.created_by+'_r')
                
            self.set_valid_training_inputs_outputs()
            self.set_test_inputs_outputs()
            self.mse_shape=(3,len(self.output_names))
            self.load_mse()
            self.load_val_loss()


       # Initialize a loss tracker for each output
        loss_names = ["loss"] + self.output_names
        self.loss_trackers = {loss_name: tf.keras.metrics.Mean(name=f"{loss_name}") for loss_name in loss_names}

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
            # plot_model(self.model, to_file=self.graphpath)
            self.trained = False
            print("Starting model ", self.model_name, " weights from scratch")


    def compile_model(self):
        if not self.trained:
            print ("Compiling model")
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr) # it worked well with 0.0005
            #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # orginal

            # if self.loss_weights is not None:

            #     self.model.compile(optimizer=optimizer, loss=self.loss_function, loss_weights=self.loss_weights)
            # else:
            #     self.model.compile(optimizer=optimizer, loss=self.loss_function)
            self.compile(optimizer=optimizer, loss=None)
    def load_weights(self):
        if os.path.isfile(self.weightspath):
            print ("Loading weights: " + self.weightspath)
            self.model.load_weights(self.weightspath)
        else:
            print("Starting model ", self.model_name, " weights from scratch")

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
        self.model.save(self.modelpath)
        self.model.save('./kmodels/most_recent_'+self.model_name+'.h5')
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

        if self.loss_weights is not None:
            weights = self.loss_weights
        else:
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
            loss = self.mask_loss(loss, in_dict[f'pos_inputs_{self.model_name}'], on, mask_type = self.mask_type)
            loss = np.mean(loss, axis=1)
            loss = np.mean(loss, axis=0)
            losses.append(loss)

        return losses

    def get_mse(self):
        training_mse = self.get_loss(self.training[0], self.training[1])
        validation_mse = self.get_loss(self.validation[0], self.validation[1])
        test_mse = self.get_loss(self.test[0], self.test[1])
        return training_mse, validation_mse, test_mse

    def checkpoint_np(self , validation_loss):              #### TODO this is from old version it is not used anymore might delete later

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


    def checkpoint(self, validation_loss):
        # SDY Jury is stil out on which validation will be better
        # score = self.evaluate(self.validation[0], self.validation[1], verbose=1) 

        # #score = self.model.evaluate_local(self.training[0], self.training[1], verbose=1) 
        # validation_loss = score[0]

        print ("\nvalidation score = ", validation_loss)
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

    def call(self, inputs, training=True):
        return self.model(inputs, training)

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass

            # Your custom loss calculation goes here
            if len(self.output_names) == 1:
                y_pred = [y_pred]
            losses_dict = {}
            losses = []
            if self.loss_weights is not None:

                weights = self.loss_weights
            else:
                weights = {}
                for on in self.output_names:
                    weights[on] = 1

            for (i, on) in enumerate(self.output_names):
                actual = tf.cast(y[on], dtype=tf.float32)
                if callable(self.loss_function):
                    func = self.loss_function
                elif isinstance(self.loss_function, dict):
                    func = self.loss_function[on]

                loss = weights[on] * func(actual, y_pred[i])
                loss = self.mask_loss(loss, x[f'pos_inputs_{self.model_name}'], on, mask_type = self.mask_type)
                loss = tf.reduce_mean(loss)
                losses_dict[on]= loss
                losses.append(loss)


            sum_loss = tf.reduce_sum(losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(sum_loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    
        # Add total loss to the dictionary
        losses_dict_combined = {'loss': sum_loss, **losses_dict}

        # return losses_dict_combined

        # Update each loss tracker
        for name, loss in losses_dict_combined.items():
            self.loss_trackers[name].update_state(loss)

        # Prepare return value: aggregate current state of each loss tracker
        return {name: tracker.result() for name, tracker in self.loss_trackers.items()}

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Forward pass
        y_pred = self(x, training=False)  # Ensure the model is in inference mode

        # Similar to train_step, handle the case of a single output
        if len(self.output_names) == 1:
            y_pred = [y_pred]
        losses_dict = {}
        losses = []
        if self.loss_weights is not None:
            weights = self.loss_weights
        else:
            weights = {on: 1 for on in self.output_names}

        # Calculate losses for each output
        for (i, on) in enumerate(self.output_names):
            actual = tf.cast(y[on], dtype=tf.float32)
            if callable(self.loss_function):
                func = self.loss_function
            elif isinstance(self.loss_function, dict):
                func = self.loss_function[on]

            # Compute the loss without the gradient tape
            loss = weights[on] * func(actual, y_pred[i])    
            loss = self.mask_loss(loss, x[f'pos_inputs_{self.model_name}'], on, mask_type = self.mask_type)
            loss = tf.reduce_mean(loss)
            losses_dict[on] = loss
            losses.append(loss)

        # Sum up the total loss
        total_loss = tf.reduce_sum(losses)

        # Update metrics here if needed
        # For example, if you have self.custom_accuracy as a metric
        # self.custom_accuracy.update_state(y, y_pred)

        # Include total_loss first in the dictionary
        losses_dict_combined = {'loss': total_loss, **losses_dict}

        # return losses_dict_combined

        # Update each loss tracker
        for name, loss in losses_dict_combined.items():
            self.loss_trackers[name].update_state(loss)

        # Prepare return value: aggregate current state of each loss tracker
        return {name: tracker.result() for name, tracker in self.loss_trackers.items()}

    def mask_loss(self, loss, pos, on, mask_type=None):

        if mask_type=="lateral_masked":
            ### masked 
            mask = tf.abs(pos[:, 2]) < 0.01
            mask = tf.cast(mask, tf.float32)
            retval = loss * tf.expand_dims(mask, axis=-1)  # Ensure mask is applied across the correct dimension

        elif mask_type=="lateral_weighted":
            # weighted
            weights = 1 - tf.abs(pos[:, 2])  # Compute the weight as 1 minus the absolute value of the last element of pos
            weights = tf.expand_dims(weights, axis=-1)  # Make sure weights shape is compatible with squared_diff
            weights = tf.cast(weights, tf.float32)
            retval = loss * weights

        elif mask_type=="left_right_masked":

            # Create masks for left (including center) and right (including center)
            if on in ['real_l', 'realmean_l','realstd_l', 'imag_l','imagmean_l','imagstd_l','mag_l', 'magl', 'maglmean', 'maglstd', 'magri_l','magfinal_l', 'magtotal_l', 'magtotalmean_l', 'magtotalstd_l']:
                mask = tf.cast(pos[:, 1] >= 0, dtype=tf.float32)  # Include center in left
            elif on in ['real_r', 'realmean_r', 'realstd_r', 'imag_r', 'imagmean_r', 'imagstd_r', 'mag_r', 'magr', 'magrmean', 'magrstd', 'magri_r', 'magfinal_r', 'magtotal_r',  'magtotalmean_r',  'magtotalstd_r']:
                mask =  tf.cast(pos[:, 1] <= 0, dtype=tf.float32)  # Include center in right

            retval = loss * tf.expand_dims(mask, axis=-1)

        elif mask_type=="left_right_weighted":

            # Create masks for left (including center) and right (including center)
            if on in ['real_l', 'realmean_l','realstd_l', 'imag_l','imagmean_l','imagstd_l','mag_l', 'magl', 'maglmean', 'maglstd', 'magri_l','magfinal_l', 'magtotal_l', 'magtotalmean_l', 'magtotalstd_l']:
                mask = tf.cast(tf.clip_by_value((pos[:, 1] + 1) / 2, 0, 1), dtype=tf.float32)  # linear decline
            elif on in ['real_r', 'realmean_r', 'realstd_r', 'imag_r', 'imagmean_r', 'imagstd_r', 'mag_r', 'magr', 'magrmean', 'magrstd', 'magri_r', 'magfinal_r', 'magtotal_r',  'magtotalmean_r',  'magtotalstd_r']:
                mask =   tf.cast(tf.clip_by_value((1 - pos[:, 1]) / 2, 0, 1), dtype=tf.float32)  # linear decline

            retval = loss * tf.expand_dims(mask, axis=-1)

        elif mask_type=="left_right_weighted_reverted":

            # Create masks for left (including center) and right (including center)
            if on in ['real_l', 'realmean_l','realstd_l', 'imag_l','imagmean_l','imagstd_l','mag_l', 'magl', 'maglmean', 'maglstd', 'magri_l','magfinal_l', 'magtotal_l', 'magtotalmean_l', 'magtotalstd_l']:
                mask = tf.cast(tf.clip_by_value((1 - pos[:, 1]) / 2, 0, 1), dtype=tf.float32)  # linear decline
            elif on in ['real_r', 'realmean_r', 'realstd_r', 'imag_r', 'imagmean_r', 'imagstd_r', 'mag_r', 'magr', 'magrmean', 'magrstd', 'magri_r', 'magfinal_r', 'magtotal_r',  'magtotalmean_r',  'magtotalstd_r']:
                mask =   tf.cast(tf.clip_by_value((pos[:, 1] + 1) / 2, 0, 1), dtype=tf.float32)  # linear decline

            retval = loss * tf.expand_dims(mask, axis=-1)

        elif mask_type=="combined_weighted":
            # First, compute the lateral weight
            lateral_weights = 1 - tf.abs(pos[:, 2])
            lateral_weights = tf.expand_dims(lateral_weights, axis=-1)
            lateral_weights = tf.cast(lateral_weights, tf.float32)
            
            # Next, compute the left/right weight based on the `on` parameter
            if on in ['real_l', 'realmean_l','realstd_l', 'imag_l','imagmean_l','imagstd_l','mag_l', 'magl', 'maglmean', 'maglstd', 'magri_l','magfinal_l', 'magtotal_l', 'magtotalmean_l', 'magtotalstd_l']:
                left_right_weights = tf.cast(tf.clip_by_value((pos[:, 1] + 1) / 2, 0, 1), dtype=tf.float32)
            elif on in ['real_r', 'realmean_r', 'realstd_r', 'imag_r', 'imagmean_r', 'imagstd_r', 'mag_r', 'magr', 'magrmean', 'magrstd', 'magri_r', 'magfinal_r', 'magtotal_r',  'magtotalmean_r',  'magtotalstd_r']:
                left_right_weights = tf.cast(tf.clip_by_value((1 - pos[:, 1]) / 2, 0, 1), dtype=tf.float32)
            
            # Combine the lateral and left/right weights
            combined_weights = lateral_weights * tf.expand_dims(left_right_weights, axis=-1) 
            
            # Apply the combined weights to the loss
            retval = loss * combined_weights

        elif mask_type=="single_point":
            # Target point to match
            target_point = tf.constant([0.54660094, 0.70710683, 0.44858381], dtype=tf.float32)
            # Tolerance for floating-point comparison
            tolerance = 1e-5

            # Compute the absolute difference between each pos and the target point
            abs_diff = tf.abs(pos - target_point)

            # Check if the absolute difference is within the tolerance for all three dimensions
            mask = tf.reduce_all(abs_diff < tolerance, axis=1)

            # Convert the boolean mask to float32 to use in multiplication
            mask = tf.cast(mask, tf.float32)

            # Apply the mask to the loss, ensuring it's broadcasted correctly
            retval = loss * tf.expand_dims(mask, axis=-1)

        else:
            retval = loss

        return retval

    def train(self):
        print ("Begin training")
        
        Model_ckpt = Custom_ModelCheckpoint(self)   #Here I call intance of custom checkpoint that is defined below
        # Early_Stop =  keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        
        mse = np.zeros((self.iterations, self.mse_shape[0], self.mse_shape[1]))
        for n in range(self.iterations): 
            print ("\nInteration: ", n)
            self.seed = self.seed + 1
            self.fit(self.training[0],     
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
        

    def evaluate_local(self):
       fig = plt.figure()
       self.load_mse()
    #    print(np.shape(self.mse))
       if np.shape(self.mse)[2] == 2 or np.shape(self.mse)[2] == 6:
           axl = fig.add_subplot(2,1,1)
           axr = fig.add_subplot(2,1,2)
           axs = [axl, axr]
        #    print ("shape of self.mse = ", np.shape(self.mse))
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
       elif np.shape(self.mse)[2] == 3:
           axshape = fig.add_subplot(3,1,1)
           axmean = fig.add_subplot(3,1,2)
           axstd = fig.add_subplot(3,1,3)
           axs = [axshape, axmean, axstd]
           axshape.plot(self.mse[:,0,0], label='training left')
           axshape.plot(self.mse[:,1,0], label='validation left')
           axshape.plot(self.mse[:,2,0], label='test left')
           axshape.set_title('MSE for '+self.model_name+'Shape Data')
           axmean.plot(self.mse[:,0,1], label='training right')
           axmean.plot(self.mse[:,1,1], label='validation right')
           axmean.plot(self.mse[:,2,1], label='test right')
           axmean.set_title('MSE for '+self.model_name+'Mean Data')
           axstd.plot(self.mse[:,0,2], label='training right')
           axstd.plot(self.mse[:,1,2], label='validation right')
           axstd.plot(self.mse[:,2,2], label='test right')
           axstd.set_title('MSE for '+self.model_name+'Std Data')
           for ax in axs:
               ax.set_ylabel('MSE')
               ax.legend(loc='best')
               ax.grid(markevery=1)
               ax.set_xlabel('Iteration')
           return fig


    # def evaluate_local(self):
    #     fig = plt.figure()
    #     axl = fig.add_subplot(2,1,1)
    #     axr = fig.add_subplot(2,1,2)
    #     axs = [axl, axr]
    #     self.load_mse()
    #     axl.plot(self.mse[:,0,0], label='training left')
    #     axl.plot(self.mse[:,1,0], label='validation left')
    #     axl.plot(self.mse[:,2,0], label='test left')
    #     axl.set_title('MSE for '+self.model_name+'Left Ear Data')
    #     axr.plot(self.mse[:,0,1], label='training right')
    #     axr.plot(self.mse[:,1,1], label='validation right')
    #     axr.plot(self.mse[:,2,1], label='test right')
    #     axr.set_title('MSE for '+self.model_name+'Right Ear Data')
    #     for ax in axs:
    #         ax.set_ylabel('MSE')
    #         ax.legend(loc='best')
    #         ax.grid(markevery=1)
    #         ax.set_xlabel('Iteration')
    #     return fig




#Custom callback to save beest checkpoint after avery epoch

class Custom_ModelCheckpoint(keras.callbacks.Callback):

    def __init__(self, Network):
        super().__init__()
        self.Network = Network

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get("val_loss")
        Network.checkpoint(self.Network, val_loss)
        # print("\nEnd epoch {} of training; got val_loss: {}".format(epoch, logs["val_loss"]))


# class PerBatchValidationMetrics(tf.keras.callbacks.Callback):
#     def __init__(self, validation_data):
#         super().__init__()
#         self.validation_data = validation_data

#     def convert_to_float32(self, inputs, outputs):
#         # Convert input tensors in the dictionary to float32
#         inputs_converted = {key: tf.cast(value, tf.float32) for key, value in inputs.items()}
        
#         # Convert output tensors in the dictionary to float32
#         outputs_converted = {key: tf.cast(value, tf.float32) for key, value in outputs.items()}
        
#         return inputs_converted, outputs_converted

#     def on_epoch_end(self, epoch, logs=None):
#         per_batch_losses = []
#         per_batch_metrics = []

#         val_dataset = tf.data.Dataset.from_tensor_slices(self.validation_data)
        
#         # Apply the conversion function to each element in the dataset
#         val_dataset = val_dataset.map(self.convert_to_float32)

#         # Batch the dataset
#         val_dataset = val_dataset.batch(32)
        
#         # Manually iterate over the validation dataset
#         for x_batch_val, y_batch_val in val_dataset:
#             # Compute per-batch metrics
#             loss, *metrics= self.model.test_on_batch(x_batch_val, y_batch_val)
#             per_batch_losses.append(loss)
#             per_batch_metrics.append(metrics)
        
#         # Here, you can do something with the collected per-batch metrics,
#         # like printing, logging, or saving them.
#         print(f"Epoch {epoch+1}:")
#         print(f"Per-batch Losses: {per_batch_losses}")
#         print(f"Per-batch Metrics: {per_batch_metrics}")