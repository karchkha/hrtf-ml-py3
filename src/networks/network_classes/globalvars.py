import pdb

import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

'''Scaling parameters for magtotal'''
head_div = [16.20, 23.84, 23.06, 3.77, 1.90, 14.04, 9.47, 12.53, 36.84, 17.56, 30.25, 54.10, 8.41, 200.66, 110.00, 65.00, 131.00] 
left_ear_div = [2.24, 1.05, 2.10, 2.24, 7.61, 3.53, 0.86, 1.31, 0.78, 0.68]
right_ear_div = [2.29, 0.98, 1.99, 2.20, 7.95, 3.52, 0.91, 1.26, 0.72, 0.87]
pos_div = [1.0, 1.0, 1.0]
input_scale = 0

def custom_activation(x):
    return 4.0*(K.tanh(x/4.0))

def custom_activation_maglr(x):
#    return 4.0*K.tanh(x/4.0) + 0.001
    return 300.0 * K.tanh(x/300.0)
#    return 1.0*K.tanh(x/2.5)

def custom_activation_magtotal_relu(x):
#    return K.abs(1.0*K.tanh(x/1.0))+0.0001
#    return K.abs(1.0 * K.tanh(10.0*x))
#    return K.abs(4.0 * K.tanh(x/4.0)) + 0.0005
#    return 20.0*log10(K.abs(4.0*K.tanh(x/4.0)) + 0.0005)
    return 300.0 * K.tanh(x/300.0)

def custom_activation_magtotal(x):
#    return 1.0*K.tanh(x/1.0)
    return 4.0*K.tanh(x/4.0)

def custom_activation_magri(x):
    return K.relu(x, alpha=.001)

def custom_activation_softsign(x):
    return K.relu(x, alpha=-.001)

def custom_activation_sig(x):
    return 2.0*K.tanh(x/2.0)

def identity(inp):
    return tf.identity(inp) #K.identity(inp)

def get_left(inp):
    return tf.identity(inp[0]) #K.identity(inp[0])

def get_right(inp):
    return tf.identity(inp[1]) #K.identity(inp[1])

def get_first(inp):
    return tf.identity(inp[1]) #K.identity(inp[1])

def get_second(inp):
    return tf.identity(inp[2]) #K.identity(inp[2])

def ri_to_mag(ri):
    r = kl.add([kl.multiply([ri[0], ri[2]]), ri[1]])
    i = kl.add([kl.multiply([ri[3], ri[5]]), ri[4]])
    ri = [r, i]
    ri = K.pow(ri, 2)
    ri_2 = K.sum(ri, axis=0)
    magri_out = K.sqrt(ri_2)
    return magri_out

# def ri_to_mag(ri):
#     r = kl.add([kl.multiply([ri[0], ri[2]]), ri[1]])
#     i = kl.add([kl.multiply([ri[3], ri[5]]), ri[4]])
#     ri = [r, i]
#     ri = K.pow(ri, 2)
#     ri_2 = K.sum(ri, axis=0)
#     magri_out = K.sqrt(ri_2)
#     return magri_out


def mag_to_magmean(x):
    magri_mean = K.mean(x, axis=1, keepdims=True)
    return magri_mean

def mag_to_magstd(x):
    magri_std = K.std(x, axis=1, keepdims=True)
    return magri_std

def recalc(x):
    mag = (x[0]*x[2])+x[1]
    return mag

def mag_to_db(x):
    numerator = K.log(K.abs(x))
    denominator = K.log(10.0)
    return 20.0*(numerator/denominator)

def mag_to_magnorm(x):
    magri = (x[0]-x[1])/x[2]
    return magri

def positive(x):
    return K.abs(x)

def mean(x):
    out_mean = K.mean(x, axis=1, keepdims=True)
    return out_mean

def std(x):
    out_std = K.std(x, axis=1, keepdims=True)
    return out_std

def norm(x):
    out_norm = (x[0]-x[1])/x[2]
    #magri = Multiply()([Add()([x[0], Multiply()([-1,x[1]])]), 1/x[2]])
    return out_norm

def data_normalize(x, div=None, scale=None):
    return x * np.divide(1,div) * scale

#def head_normalize(x):
#    head_div = [16.20, 23.84, 23.06, 3.77, 1.90, 14.04, 9.47, 12.53, 36.84, 17.56, 30.25, 54.10, 8.41, 200.66, 110.00, 65.00, 131.00] 
#    return x * np.divide(1,head_div) * input_scale
#
#def ear_l_normalize(x):
#    left_ear_div = [2.24, 1.05, 2.10, 2.24, 7.61, 3.53, 0.86, 1.31, 0.78, 0.68]
#    return x * np.divide(1,left_ear_div) * input_scale
#
#def ear_r_normalize(x):
#    right_ear_div = [2.29, 0.98, 1.99, 2.20, 7.95, 3.52, 0.91, 1.26, 0.72, 0.87]
#    return x * np.divide(1, right_ear_div) * input_scale

def custom_loss_MSE(y, yhat):
    return (yhat-y)**2

def custom_loss_MSEOverY(y, yhat):
    return ((yhat-y)**2.0)/((y*y)**(1.0/2.0))

def custom_loss_MSEOverY2(y, yhat):
    return ((yhat-y)**2)/(y*y)

def custom_loss_meannetworks(y, yhat):
    return (yhat - y)**2 / ((y*y)**(1/2))

#def custom_init(shape, dtype=None):
#    ident =  np.identity(shape[1])
#    zer =  np.zeros((np.abs(shape[0]-shape[1]), shape[1]))
#    weights = np.concatenate([ident, zer], axis=0)
#    return weights

def custom_init_magtotal(shape, dtype=None):
    ident = np.identity(shape[1]) * .33
    return ident

class custom_init_zeros_ident(ki.Initializer):
    def __call__(self, shape, dtype=None):
        ident =  np.identity(shape[1])
        zer =  np.zeros((np.abs(shape[0]-shape[1]), shape[1]))
        weights = np.concatenate([ident, zer], axis=0)
        return weights
        
        
# class custom_loss_normalized(tf.keras.losses.Loss):
#     def __init__(self):
#         super().__init__()
#     def call(self, y, yhat):        
#         if isinstance(y, tf.Tensor):
            
#             y_mean = K.mean(y, axis=1, keepdims=True)
#             y_zeromean = y - y_mean
#             y_std = K.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
            
#              # y_norm = np.divide(y_zeromean, y_std)           #################?????
#             y_norm = tf.math.divide(y_zeromean, y_std) 
#             # yhat_norm = np.divide(yhat - y_mean, y_std)
#             yhat_norm = tf.math.divide(yhat - y_mean, y_std)
            
#         elif isinstance(y, np.ndarray):
#             y_mean = np.mean(y, axis=1, keepdims=True)
#             y_zeromean = y - y_mean
#             y_std = np.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
#             y_norm = np.divide(y_zeromean, y_std)
            
#             yhat_norm = np.divide(yhat - y_mean, y_std)
        

#         retval = (yhat - y_norm)**2.0 
        
#         print(retval)

#         return retval
        

def custom_loss_normalized(y, yhat):
    
    if isinstance(y, tf.Tensor):
        
        y_mean = K.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = K.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        
         # y_norm = np.divide(y_zeromean, y_std)           #################?????
        y_norm = tf.math.divide(y_zeromean, y_std) 
        # yhat_norm = np.divide(yhat - y_mean, y_std)
        yhat_norm = tf.math.divide(yhat - y_mean, y_std)
        
    elif isinstance(y, np.ndarray):
        y_mean = np.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = np.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        y_norm = np.divide(y_zeromean, y_std)
        
        yhat_norm = np.divide(yhat - y_mean, y_std)

    #retval = (yhat - y_norm)**2.0 
    retval = (yhat - y_norm)**2.0 
    # print (retval)
    return retval


    
def custom_loss_renormalize(y, yhat):
    if isinstance(y, tf.Tensor):
        y_mean = K.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = K.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        y_pow = K.var(y, axis=1, keepdims=True)
        y_norm = np.divide(y_zeromean, y_std)
        
        yhat_mean = K.mean(yhat, axis=1, keepdims=True)
        yhat_zeromean = yhat - yhat_mean
        yhat_std = K.var(yhat_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        yhat_pow = K.var(yhat, axis=1, keepdims=True)
        yhat_norm = np.divide(yhat_zeromean, yhat_std)
    elif isinstance(y, np.ndarray):
        y_mean = np.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = np.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        y_pow = np.var(y, axis=1, keepdims=True)
        y_norm = np.divide(y_zeromean, y_std)
        
        yhat_mean = np.mean(yhat, axis=1, keepdims=True)
        yhat_zeromean = yhat - yhat_mean
        yhat_std = np.var(yhat_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        yhat_pow = np.var(yhat, axis=1, keepdims=True)
        yhat_norm = np.divide(yhat_zeromean, yhat_std)

    y_recalc = (y_norm * yhat_std) + yhat_mean
    y_renorm = (y-yhat_mean) / yhat_std

#    retval = ((y-yhat)**2.0 + (yhat_norm - y_norm)**2.0 + (y_recalc - y)**2.0 + (y_renorm - y_norm)**2.0 + (y_mean - yhat_mean)**2.0 + (y_std - yhat_std)**2.0) * (yhat_std/y_std)
    retval = ((y-yhat)**2.0 + (y_recalc - y)**2.0 + (y_renorm - y_norm)**2.0 + (y_mean - yhat_mean)**2.0 + (y_std - yhat_std)**2.0) * (yhat_std/y_std)
    return retval

def custom_loss_magtotal(y, yhat):
    if isinstance(y, tf.Tensor):
        
        pdb.set_trace()
        
        y_mean = K.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = K.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        y_pow = K.var(y, axis=1, keepdims=True)
        # y_norm = tf.math.divide(y_zeromean, y_std)
        y_norm = np.divide(y_zeromean, y_std)
        
        yhat_mean = K.mean(yhat, axis=1, keepdims=True)
        yhat_zeromean = yhat - yhat_mean
        yhat_std = K.var(yhat_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        yhat_pow = K.var(yhat, axis=1, keepdims=True)
        # yhat_norm = tf.math.divide(yhat_zeromean, yhat_std)
        yhat_norm = np.divide(yhat_zeromean, yhat_std)
    
    elif isinstance(y, np.ndarray):
        y_mean = np.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = np.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        y_pow = np.var(y, axis=1, keepdims=True)
        y_norm = np.divide(y_zeromean, y_std)
        
        yhat_mean = np.mean(yhat, axis=1, keepdims=True)
        yhat_zeromean = yhat - yhat_mean
        yhat_std = np.var(yhat_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        yhat_pow = np.var(yhat, axis=1, keepdims=True)
        yhat_norm = np.divide(yhat_zeromean, yhat_std)

    y_recalc = (y_norm * yhat_std) + yhat_mean
    y_renorm = (y-yhat_mean) / yhat_std
#    retval = ((y-yhat)**2.0 + (yhat_norm - y_norm)**2.0 + (y_recalc - y)**2.0 + (y_renorm - y_norm)**2.0 + (y_mean - yhat_mean)**2.0 + (y_std - yhat_std)**2.0) * (yhat_std/y_std)
    retval = ((y-yhat)**2.0  + (y_recalc - y)**2.0 + (y_renorm - y_norm)**2.0 + (y_mean - yhat_mean)**2.0 + (y_std - yhat_std)**2.0) * (yhat_std/y_std)
    return retval

###Definitions for all custom functions used in all networks.
custom_objects = {'custom_activation': custom_activation,
        'custom_loss_MSEOverY': custom_loss_MSEOverY,
        'custom_loss_MSEOverY2': custom_loss_MSEOverY2,
        'custom_loss_MSE': custom_loss_MSE,
        'custom_loss_meannetworks':custom_loss_meannetworks,
        'custom_loss_renormalize':custom_loss_renormalize,
        'custom_loss_normalized':custom_loss_normalized,
        'custom_loss_magtotal':custom_loss_magtotal,
        'custom_activation_magtotal': custom_activation_magtotal,
        'custom_activation_magtotal_relu': custom_activation_magtotal_relu,
        'custom_activation_magri': custom_activation_magri,
        'custom_activation_maglr': custom_activation_maglr,
        'custom_activation_sig': custom_activation_sig,
        'custom_activation_softsign': custom_activation_softsign,
        'custom_init_zeros_ident': custom_init_zeros_ident,
        'normalize': data_normalize,
        'positive': positive,
        'mean':mean,
        'std':std,
        'norm':norm,
        'ri_to_mag':ri_to_mag,
        'mag_to_db':mag_to_db,
        'get_left':get_left,
        'get_right':get_right,
        'identity':identity,
        'kl':kl}
