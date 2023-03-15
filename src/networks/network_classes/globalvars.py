import pdb

import tensorflow.keras.layers as kl
import tensorflow.keras.initializers as ki
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

from utilities import utils
from scipy import signal

import matplotlib.pyplot as plt


'''Scaling parameters for magtotal'''
head_div = [16.20, 23.84, 23.06, 3.77, 1.90, 14.04, 9.47, 12.53, 36.84, 17.56, 30.25, 54.10, 8.41, 200.66, 110.00, 65.00, 131.00] 
left_ear_div = [2.24, 1.05, 2.10, 2.24, 7.61, 3.53, 0.86, 1.31, 0.78, 0.68]
right_ear_div = [2.29, 0.98, 1.99, 2.20, 7.95, 3.52, 0.91, 1.26, 0.72, 0.87]
pos_div = [1.0, 1.0, 1.0]
input_scale = 0

anthro_num=37

def custom_activation(x):
    return 4.0*(K.tanh(x/4.0))


def custom_activation_maglr_final(x):
    return K.exp((x-1)*5)/(1+K.exp((x-1)*5))
    return K.exp((x-0.5)*10)/(1+K.exp((x-0.5)*10))

def custom_activation_maglr_LSD(x):
    return 100.0 * K.softsign(x/100.0)
    return 1.0 * K.tanh(x/1.0)
    return (K.tanh(2.0 * x) + 1.000001) / 2.0
    return 300.0 * K.tanh(x/300.0)
    return 1.0 * K.softsign(x/1.0) + 0.001
    return 1.0 * K.sigmoid(x/1.0)
    return 4.0*K.tanh(x/4.0)
    # return 300.0 * K.tanh(x/300.0)
#    return 1.0*K.tanh(x/2.5)

def custom_activation_maglr(x):
    return 1.0 * K.tanh(x/1.0)
    return (K.tanh(2.0 * x) + 1.000001) / 2.0
    return 300.0 * K.tanh(x/300.0)
    return 1.0 * K.softsign(x/1.0) + 0.001
    return 1.0 * K.sigmoid(x/1.0)
    return 100.0 * K.softsign(x/100.0)
    return 4.0*K.tanh(x/4.0)
    # return 300.0 * K.tanh(x/300.0)
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

    #mag = tf.math.add(tf.math.multiply(x[0],x[2]),x[1])
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
    # y = tf.cond(x < 0.0001, lambda: tf.constant(0.0001), lambda: x)
    y  = tf.cond(x < x, lambda: tf.add(x, x), lambda: tf.square(x))
    return (y)

def mean(x):
    #out_mean = K.mean(x, axis=1, keepdims=True)
    out_mean = tf.math.reduce_mean(x, axis=1, keepdims=True)
    return out_mean

def std(x):
    #out_std = K.std(x, axis=1, keepdims=True)
    out_std = tf.math.reduce_std(x, axis=1, keepdims=True)
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

def Klog10(x):
    numerator = K.log(x)
    denominator = K.log(K.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def custom_loss_LSD_MSE(y, yhat):
    if isinstance(y, tf.Tensor):
        return Klog10(abs((yhat+.00001)/(y+.00001)))**2 # * (1 - x - x1)
        return Klog10(abs((yhat+1.001)/(y+1.001)))**2 # * (1 - x - x1)
        return K.sqrt(Klog10(abs((yhat)/(y)))**2) # * (1 - x - x1)
        #  x = Klog10(100.0)
        # x = tf.Print(x, [x])
        y = y+1
        y = tf.Print(y, [y], "y = ")
        yhat = yhat + 1
        yhat = tf.Print(yhat, [yhat], "yhat = ")
        w = Klog10(abs(y/yhat))**2 # * (1 - x - x1)
        w = tf.Print(w, [w], "K.sq(Klog10) = = ")
        return w
        return z * w
        x = Klog10(y)
        y = K.pow(y, 
        Klog10(abs(y/yhat))**2) # * (1 - x - x1)
        return K.sqrt(Klog10(abs(yhat/y))**2) # * (1 - x - x1)
    else:
        return np.sqrt(np.log(abs((yhat+1.0)/(y+1.0)))**2)
        return (yhat-y)**2
    x = tf.Print(y, [y])
    x3 = tf.Print(K.log(abs(y)), [K.log(abs(y))])
    x1 = tf.Print(yhat, [yhat])
    x2 = tf.Print(K.log(yhat), [K.log(yhat)])
    return x + x1 + x2 + x3 + (20.0*K.log(abs(y/yhat)))**2
    return (yhat-y)**2

def custom_loss_ZERO(y, yhat):
    return (yhat-y)**2 * 0

def custom_loss_MSE(y, yhat):
    return (yhat-y)**2






def findpeaks(y, psize):
    pidx = signal.find_peaks_cwt(y, psize)
    if len(pidx) == 0:
        return pidx
    # fix the erros of find_peaks_cwt as it could be off by one index
    print (y)
    print ("pidx before = ", pidx)
    for i in range (len(pidx)):
        if pidx[i] == 0:
            if y[1] > y[0]:
                pidx[i] = 1
        elif y[pidx[i]-1] > y[pidx[i]]:
            print (y[pidx[i]-1], y[pidx[i]])
            pidx[i] = pidx[i] - 1
    print ("pidx after = ", pidx)
    return pidx

def PZ_error(y, yhat):
    vsize = np.shape(y)[1]
    print ("shape = ", np.shape(y)[0])
    pz_err = np.zeros(np.shape(y)[0])
    order = 1
    psize = np.arange(8,10)

    for i in range (np.shape(y)):
        # ypeakind = findpeaks(y[i], psize)
        # yzeroind = findpeaks(y[i] * -1, psize)
        # yhatpeakind = findpeaks(yhat[i], psize)
        # yhatzeroind = findpeaks(yhat[i] * -1, psize)
        ypeakind, yzeroind = utils.peakdet(y[i], 1.5)
        yhatpeakind, yhatzeroind = utils.peakdet(yhat[i], 1.5)
        if len(ypeakind) == 0:
            ypeakind = np.array([0])
        if len(yhatpeakind) == 0:
            yhatpeakind = np.array([0])
        if len(yzeroind) == 0:
            yzeroind = np.array([0])
        if len(yhatzeroind) == 0:
            yhatzeroind = np.array([0])
        ypeakind.resize(order)
        yzeroind.resize(order)
        yhatpeakind.resize(order)
        yhatzeroind.resize(order)

        pz_err[i] = (np.sum((ypeakind - yhatpeakind)**2) +  np.sum((yzeroind - yhatzeroind)**2)) * 10
        x1 = np.linspace(0, 1, vsize)

        print (pz_err[i])
        if 0:
            plt.plot(x1, y[i], label='Original')
            plt.plot(x1, yhat[i], label='prediction')

            plt.xlabel('x label')
            plt.ylabel('y label')
            plt.title("Simple Plot")
            plt.legend()
            if len(ypeakind) != 0:
                plt.plot(x1[ypeakind], y[i][ypeakind], "x")
            if len(yzeroind) != 0:
                plt.plot(x1[yzeroind], yhat[i][yzeroind], "o")
            if len(yhatpeakind) != 0:
                plt.plot(x1[yhatpeakind], yhat[i][yhatpeakind], "x")
            if len(yhatzeroind) != 0:
                plt.plot(x1[yhatzeroind], yhat[i][yhatzeroind], "o")
            plt.show()
    return pz_err




def X_PZ_error(y, yhat):
    vsize = np.shape(y)[1]
    print ("shape = ", np.shape(y)[0])
    pz_err = np.zeros(np.shape(y)[0])
    order = 1
    psize = np.arange(8,10)

    for i in range (np.shape(y)):
        # ypeakind = findpeaks(y[i], psize)
        # yzeroind = findpeaks(y[i] * -1, psize)
        # yhatpeakind = findpeaks(yhat[i], psize)
        # yhatzeroind = findpeaks(yhat[i] * -1, psize)
        ypeakind, yzeroind = utils.peakdet(y[i], 1.5)
        yhatpeakind, yhatzeroind = utils.peakdet(yhat[i], 1.5)
        if len(ypeakind) == 0:
            ypeakind = np.array([0])
        if len(yhatpeakind) == 0:
            yhatpeakind = np.array([0])
        if len(yzeroind) == 0:
            yzeroind = np.array([0])
        if len(yhatzeroind) == 0:
            yhatzeroind = np.array([0])
        ypeakind.resize(order)
        yzeroind.resize(order)
        yhatpeakind.resize(order)
        yhatzeroind.resize(order)

        pz_err[i] = (np.sum((ypeakind - yhatpeakind)**2) +  np.sum((yzeroind - yhatzeroind)**2)) * 10
        x1 = np.linspace(0, 1, vsize)

        print (pz_err[i])
        if 0:
            plt.plot(x1, y[i], label='Original')
            plt.plot(x1, yhat[i], label='prediction')

            plt.xlabel('x label')
            plt.ylabel('y label')
            plt.title("Simple Plot")
            plt.legend()
            if len(ypeakind) != 0:
                plt.plot(x1[ypeakind], y[i][ypeakind], "x")
            if len(yzeroind) != 0:
                plt.plot(x1[yzeroind], yhat[i][yzeroind], "o")
            if len(yhatpeakind) != 0:
                plt.plot(x1[yhatpeakind], yhat[i][yhatpeakind], "x")
            if len(yhatzeroind) != 0:
                plt.plot(x1[yhatzeroind], yhat[i][yhatzeroind], "o")
            plt.show()
    return pz_err

def zero_descent(prev, cur):
    """reduces all descent steps to zero"""
    return tf.cond(prev[0] < cur, lambda: (cur, cur), lambda: (cur, 0.0))

def skeletonize_1d(tens):
    """reduces all point other than local maxima to zero"""
    # initializer element values don't matter, just the type.
    initializer = (np.array(0, dtype=np.float32), np.array(0, dtype=np.float32))
    # First, zero out the trailing side
    trail = tf.scan(zero_descent, tens, initializer)
    # Next, let's make the leading side the trailing side
    trail_rev = tf.reverse(trail[1], [0])
    # Now zero out the leading (now trailing) side
    lead = tf.scan(zero_descent, trail_rev, initializer)
    # Finally, undo the reversal for the result
    return tf.reverse(lead[1], [0])

def find_local_maxima(tens):
    return tf.where(skeletonize_1d >0)
    
def XXcustom_PZ_Dist_MSE(y, yhat):
    if isinstance(y, tf.Tensor):
        y1 = find_local_maxima(y)
        yhat1 = find_local_maxima(yhat)
        # if y1 != 0:
            # y1 = 1
        # if yhat1 != 0:
            # yhat = 1
        print ("y = ", np.shape(y))
        print ("y1 = ", np.shape(y1))
        print ("yhat1 = ", np.shape(yhat1))
        mult = tf.matmul(y1, yhat1)
        print ("mult = ", np.shape(mult))

        return (1.0-mult)
        return (yhat-y)**2+(1.0-mult)   ###?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    else:
        # err = PZ_error(y, yhat)
        # print err
        return  (yhat-y)**2
 
def custom_PZ_Dist_MSE(y, yhat):
    # tf.Print(y, [y])
    # tf.Print(yhat, [yhat])
    # print "hello world")
    if isinstance(y, tf.Tensor):
        # z = K.shape(y)
        # w = tf.Print(z, [z], "shape (y) = ")
        # yhat = tf.Print(yhat, [yhat], "yhat = ", summarize=512)
        yhat1 = K.argmax(K.transpose(yhat), axis=0)
        yhat1 = K.transpose(yhat1)
        yhat1 = K.argmax(K.transpose(yhat), axis=0)
        # yhat1 = tf.Print (yhat1, [yhat1], "yhat1 ", summarize=512)
        yhat1 = K.cast(yhat1, 'float32')
        z = K.shape(yhat1)
        # x = tf.Print(z, [z], "shape (yhat1) = ")
        # y = tf.Print(y, [y], "y = ", summarize=512)
        y1 = K.argmax(K.transpose(y), axis=0)
        # y1 = tf.Print (y1, [y1], "y1 ", summarize=512)
        y1 = K.cast(y1, 'float32')
        err = ((y1-yhat1)**2)
        # err = tf.Print (err, [err], "err ", summarize=512)
        err = K.transpose(err)
        err = tf.expand_dims(err, axis=1)
        # err = tf.Print (err, [err], "err massaged = ", summarize=512)
        mse_val = ((yhat-y)**2)
        # mse_val = tf.Print (mse_val, [mse_val], "mse_val = ", summarize=512)
        # err_out = ((yhat-y)**2)+(err*100)
        err_out = (yhat-y)**2+0*err
        # err_out = tf.Print (err_out, [err_out], "err_out = ", summarize=512)
        return  err_out
        return  (yhat-y)**2+err  ###?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    else:
        #  err = PZ_error(y, yhat)
        # print err
        # return  (yhat-y)**2+err.reshape(-1,1)
        # z = K.shape(y)
        # w = tf.Print(z, [z], "shape (y) = ")
        return y*0+1
        yhat1 = K.argmax(K.transpose(yhat), axis=0)
        yhat1 = K.transpose(yhat1)
        yhat1 = K.argmax(K.transpose(yhat), axis=0)
        yhat1 = tf.Print (yhat1, [yhat1], "yhat1 ", summarize=512)
        yhat1 = K.cast(yhat1, 'float32')
        z = K.shape(yhat1)
        x = tf.Print(z, [z], "shape (yhat1) = ")
        y1 = K.argmax(K.transpose(y), axis=0)
        y1 = tf.Print (y1, [y1], "y1 ", summarize=512)
        y1 = K.cast(y1, 'float32')
        err = ((y1-yhat1)**2)
        err = tf.Print (err, [err], "err ", summarize=512)
        err = K.transpose(err)
        err = tf.expand_dims(err, axis=1)
        return  (yhat-y)**2+err*1000  ###?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    return (yhat-y)**2   ###?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    # return ((yhat-y)**4)**0.2

def X3custom_PZ_Dist_MSE(y, yhat):
    # tf.Print(y, [y])
    # tf.Print(yhat, [yhat])
    # print "hello world")
    if isinstance(y, tf.Tensor):
        err = K_PZ_error(y, yhat)
        return (yhat-y)**2+err.reshapte(-1,1)
        return (yhat-y)**2
    else:
        err = PZ_error(y, yhat)
        print (err)
        return  (yhat-y)**2+err.reshape(-1,1)
    # return (yhat-y)**2
    # return ((yhat-y)**4)**0.2




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
        y_norm = tf.math.divide(y_zeromean, y_std)
        
        yhat_mean = K.mean(yhat, axis=1, keepdims=True)
        yhat_zeromean = yhat - yhat_mean
        yhat_std = K.var(yhat_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        yhat_norm = tf.math.divide(yhat - yhat_mean, yhat_std)
    elif isinstance(y, np.ndarray):
        y_mean = np.mean(y, axis=1, keepdims=True)
        y_zeromean = y - y_mean
        y_std = np.var(y_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        y_norm = np.divide(y_zeromean, y_std)
        
        yhat_mean = np.mean(yhat, axis=1, keepdims=True)
        yhat_zeromean = yhat - yhat_mean
        yhat_std = np.var(yhat_zeromean, axis=1, keepdims=True)**(1.0/2.0)
        yhat_norm = np.divide(yhat - yhat_mean, yhat_std)

    #retval = (yhat - y_norm)**2.0 
    retval = (yhat_norm - y_norm)**2.0 
    return retval


def Xcustom_loss_normalized(y, yhat):

    
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
    retval = (yhat_norm - y_norm)**2.0 
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
    #retval = ((y-yhat)**2.0 + (y_recalc - y)**2.0 + (y_renorm - y_norm)**2.0 + (y_mean - yhat_mean)**2.0 + (y_std - yhat_std)**2.0) * (yhat_std/y_std)
    retval = ((y-yhat)**2.0) * (yhat_std/y_std)
    return retval

def custom_loss_magtotal(y, yhat):
    if isinstance(y, tf.Tensor):
        
        # pdb.set_trace()
        
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
        'custom_loss_ZERO': custom_loss_ZERO,
        'custom_loss_LSD_MSE': custom_loss_LSD_MSE,
        'custom_loss_meannetworks':custom_loss_meannetworks,
        'custom_loss_renormalize':custom_loss_renormalize,
        'custom_loss_normalized':custom_loss_normalized,
        'custom_loss_magtotal':custom_loss_magtotal,
        'custom_activation_magtotal': custom_activation_magtotal,
        'custom_activation_magtotal_relu': custom_activation_magtotal_relu,
        'custom_activation_magri': custom_activation_magri,
        'custom_activation_maglr': custom_activation_maglr,
        'custom_activation_maglr_LSD': custom_activation_maglr_LSD,
        'custom_activation_maglr_final': custom_activation_maglr_final,
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
