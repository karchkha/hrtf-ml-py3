import scipy
import numpy as np
import matplotlib.pyplot as plt
import cmath
from scipy.signal import ss2tf
from scipy.linalg import hankel
from tensorflow.keras import backend as K

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def U_IIR_app(hrir,k, hsize=33):
    lenhrir=np.shape(hrir)
    # print(lenhrir)
    # print(hrir)
    zeros=np.zeros([lenhrir[0],lenhrir[1],k-1,lenhrir[3]],dtype=complex)
    poles=np.zeros([lenhrir[0],lenhrir[1],k,lenhrir[3]],dtype=complex)
    hrirs=np.zeros([lenhrir[0],lenhrir[1],hsize,lenhrir[3]],dtype=complex)
    for person in xrange(lenhrir[0]):
        for position in xrange(lenhrir[1]):
            for chan in xrange(lenhrir[3]):
                x=hrir[person,position,:, chan]
                #k=1; #order is one less than matlab
                #num=1;
                #x=hrir[5,zz,:,0];
                L=len(x) #check if same with mat
                A=np.zeros((L-1,L-1))
                for column in range(L-2):
                     A[column+1,column]=1

                B=np.zeros(L-1) #indeces start from 0
                B[0]=1
                B.shape = (L-1,1)
                C=x[1:] #need for transpose?
                D=np.zeros(1)
                D[0]=x[0]


                hankmat=hankel(x[1:])
                v, s, vt = np.linalg.svd(hankmat)
                vtrans=np.transpose(v)
                vpart = np.transpose(vtrans[:][0:k])
                spart= s[0:k]
                Ak= np.matmul(np.matmul(np.transpose(vpart),A),vpart)
                Bk= np.matmul(np.transpose(vpart),B);

                Ck=np.matmul(C,vpart);
                Dk=x[0];
                num,den=scipy.signal.ss2tf(Ak,Bk,Ck,Dk)
                z,p,gain=scipy.signal.tf2zpk(num,den)
                w,h=scipy.signal.freqz(np.transpose(num),den, worN=(hsize-1))
                j = abs(np.resize(h, hsize))
                j[hsize-1] = 0.001 # set nyquist to 0 
                hrirs[person,position,:,chan] = j
                #fig1, ax1 = plt.subplots()
                #ax1.scatter(p.real,p.imag)
                #ax1.scatter(z.real,z.imag)
                #t=np.linspace(1, 360.0, num=360)
                #ax1.plot(np.cos(t), np.sin(t), linewidth=0.05)

                #fig2, ax2 = plt.subplots() #can put subplots in here
                #w,h=scipy.signal.freqz(np.transpose(num),den)
                #IIR=ax2.plot(w, 20 * np.log10(abs(h)), 'r')
                #plt.ylabel('Amplitude [dB]', color='b')
                #plt.xlabel('Frequency [rad/sample]')

                #w,h=scipy.signal.freqz(x)
                #FIR=ax2.plot(w, 20 * np.log10(abs(h)), 'b')
                #plt.legend((IIR, FIR), ('IIR Mag response', 'FIR Mag response'))
                #plt.show()
                zeros[person,position,:,chan]=z
                poles[person,position,:,chan]=p
    return zeros,poles, hrirs

def K_IIR_app(hrir,k, hsize=33):
    lenhrir=K.shape(hrir)
    print("1 - ", lenhrir)
    print("2 - ", hrir)
    zeros=K.zeros([lenhrir[0],lenhrir[1],k-1])
    poles=K.zeros([lenhrir[0],lenhrir[1],k])
    hrirs=K.zeros([lenhrir[0],lenhrir[1],33])
    for person in xrange(lenhrir[0]):
        for position in xrange(lenhrir[1]):
            for chan in xrange(lenhrir[3]):
                x=hrir[person,position,:, chan]
                #k=1; #order is one less than matlab
                #num=1;
                #x=hrir[5,zz,:,0];
                L=len(x) #check if same with mat
                A=np.zeros((L-1,L-1))
                for column in range(L-2):
                     A[column+1,column]=1

                B=np.zeros(L-1) #indeces start from 0
                B[0]=1
                B.shape = (L-1,1)
                C=x[1:] #need for transpose?
                D=np.zeros(1)
                D[0]=x[0]


                hankmat=hankel(x[1:])
                v, s, vt = np.linalg.svd(hankmat)
                vtrans=np.transpose(v)
                vpart = np.transpose(vtrans[:][0:k])
                spart= s[0:k]
                Ak= np.matmul(np.matmul(np.transpose(vpart),A),vpart)
                Bk= np.matmul(np.transpose(vpart),B);

                Ck=np.matmul(C,vpart);
                Dk=x[0];
                num,den=scipy.signal.ss2tf(Ak,Bk,Ck,Dk)
                z,p,gain=scipy.signal.tf2zpk(num,den)
                w,h=scipy.signal.freqz(np.transpose(num),den, worN=(hsize-1))
                j = abs(np.resize(h, hsize))
                j[hsize-1] = 0.001 # set nyquist to 0 
                hrirs[person,position,:,chan] = j
                #fig1, ax1 = plt.subplots()
                #ax1.scatter(p.real,p.imag)
                #ax1.scatter(z.real,z.imag)
                #t=np.linspace(1, 360.0, num=360)
                #ax1.plot(np.cos(t), np.sin(t), linewidth=0.05)

                #fig2, ax2 = plt.subplots() #can put subplots in here
                #w,h=scipy.signal.freqz(np.transpose(num),den)
                #IIR=ax2.plot(w, 20 * np.log10(abs(h)), 'r')
                #plt.ylabel('Amplitude [dB]', color='b')
                #plt.xlabel('Frequency [rad/sample]')

                #w,h=scipy.signal.freqz(x)
                #FIR=ax2.plot(w, 20 * np.log10(abs(h)), 'b')
                #plt.legend((IIR, FIR), ('IIR Mag response', 'FIR Mag response'))
                #plt.show()
                print (np.shape(z))
                print (np.shape(p))
                print (np.shape(zeros))
                zeros[person,position,:,chan]=z
                poles[person,position,:,chan]=p
    return zeros,poles, hrirs


import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array

def K_peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab), array(mintab)

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append(mxpos)
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append(mnpos)
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

