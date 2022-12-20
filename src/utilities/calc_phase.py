#! /usr/bin/python

import numpy as np
import math as m
import sys

SoundSpeed = 340

def get_phase(listener, source, freqs, fs, phase_type=None, quantize=False):
    '''
    Implementation of David's ideal phase algorithm. 
    Calculates the ideal phase around the head for the listener and source 
    
    Parameters:
    	listener: (azimuth, elevation, r) for the ear
    	source: (azimuth, elevation, r) for the source
    	freqs: a list of frequencies to calculate phase for, if None 
    	fs: sampling frequency
    	phase_type: [ideal, euclidean]
    	quantized: whether to quantize the phase based on nyquist (nyquist can only take on values of pi and 0)
    '''
    
    listener_rad = np.array(listener)
    source_rad = np.array(source)
    
    # Convert from degrees to radians
    for i in range(0,2):
        listener_rad[i] = listener[i]/180. * m.pi
        source_rad[i] = source[i]/180. * m.pi
        
        #Force source to unit distance
    source_rad[2] = 1.0
    
    # Calculate alpha
    a = m.acos(m.cos(source_rad[1])*m.cos(listener_rad[1])*m.cos(source_rad[0]-listener_rad[0])+ m.sin(source_rad[1])*m.sin(listener_rad[1]))-m.acos(listener_rad[2]/source_rad[2])
    
    # Convert listener to cartesian
    listener_xyz = np.array(listener)
    listener_xyz[0] = listener_rad[2]*m.cos(listener_rad[0])*m.cos(listener_rad[1])
    listener_xyz[1] = listener_rad[2]*m.sin(listener_rad[0])*m.cos(listener_rad[1])
    listener_xyz[2] = listener_rad[2]*m.sin(listener_rad[1])
    
    # Convert source to cartesian
    source_xyz = np.array(source)
    source_xyz[0] = source_rad[2]*m.cos(source_rad[0])*m.cos(source_rad[1])
    source_xyz[1] = source_rad[2]*m.sin(source_rad[0])*m.cos(source_rad[1])
    source_xyz[2] = source_rad[2]*m.sin(source_rad[1])
    
    
    # Get max distance
    max_dist = m.sqrt(source_rad[2]**2 - listener_rad[2]**2)
    
    # Get direct distance
    direct_dist = 0
    for i in range(3):
    	direct_dist += (listener_xyz[i] - source_xyz[i])**2
    direct_dist = m.sqrt(direct_dist)
    
    if phase_type == 'ideal':
    	# Select the appropriate distance to use
    	if direct_dist<=max_dist:
    		d = direct_dist
    	else:
    		d = m.sqrt(source_rad[2]**2 - listener_rad[2]**2) + listener_rad[2]*a
    elif phase_type =='euclidean':
    	d = direct_dist
    else:
    	return
    
    # Speed of sound
    c = SoundSpeed
    
    if quantize:
    	nyquist = float(fs)/2.
    	lamn = float(c)/float(nyquist)
    	d = d-m.fmod(d,lamn)
    
    #Distance in number of samples
    dsamp = int(d/float(c)*float(fs))
    
    phase = np.zeros(np.shape(freqs))
    
    for i, freq in enumerate(freqs):
    	#Return 0 if frequency is 0
    	if freq == 0:
    		phase[i] = 0
    
    	else:
    		# Calculate lambda for specific frequency
    		lam = float(c)/float(freq)
    
    		# Calculate phase	
    		phase[i] = (d/lam) * (2*m.pi)
    
    return phase, dsamp, d


def get_euclidean_phase(listener, source, freq):	
	'''
	Calculates the euclidean phase through the head for the listener and source 
	
	Parameters:
		listener: (azimuth, elevation, r) for the ear
		source: (azimuth, elevation, r) for the source
		freq: frequency band to calculate phase for
	'''
	
	if freq == 0:
		return 0
	
	listener_rad = np.array(listener)
	source_rad = np.array(source)

	# Convert from degrees to radians
	for i in range(0,2):
		listener_rad[i] = listener[i]/180. * m.pi
		source_rad[i] = source[i]/180. * m.pi

	# Convert listener to cartesian
	listener_xyz = np.array(listener)
	listener_xyz[0] = listener_rad[2]*m.cos(listener_rad[0])*m.cos(listener_rad[1])
	listener_xyz[1] = listener_rad[2]*m.sin(listener_rad[0])*m.cos(listener_rad[1])
	listener_xyz[2] = listener_rad[2]*m.sin(listener_rad[1])

	# Convert source to cartesian
	source_xyz = np.array(source)
	source_xyz[0] = source_rad[2]*m.cos(source_rad[0])*m.cos(source_rad[1])
	source_xyz[1] = source_rad[2]*m.sin(source_rad[0])*m.cos(source_rad[1])
	source_xyz[2] = source_rad[2]*m.sin(source_rad[1])


	# Get direct distance
	direct_dist = 0
	for i in range(3):
		direct_dist += (listener_xyz[i] - source_xyz[i])**2

	d = m.sqrt(direct_dist)

	# Speed of sound
	c = SoundSpeed

	lam = float(c)/float(freq)
	phase = (d/lam) * (2*m.pi)
	
	return phase

def main():
    listener = np.array([90, 0, 0.09], dtype=float)
    source = np.array(sys.argv[1:4], dtype=float)	
    print (get_ideal_phase(listener, source, 1000))
    print (get_euclidean_phase(listener, source, 1000))


if __name__ == '__main__':
	main()
