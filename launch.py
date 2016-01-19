"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code creates a hebbian neural network object and trains it on the MNIST dataset. The learning rule is a hebbian learning rule augmented with a learning mechanism inspired from dopamine signalling in animal cortex.
"""

import os
import matplotlib
if 'Documents' in os.getcwd():
	matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers

import numpy as np
import pickle
import time
import hebbian_net
import helper.external as ex

hebbian_net = reload(hebbian_net)
ex = reload(ex)

""" create Hebbian neural network """
net = hebbian_net.Network(	dopa_values		= {	'dHigh' 	: 4.5,
												'dMid' 		: 0.02,
												'dNeut' 	: -0.1,
												'dLow' 		: -2.0,
												},
							protocol		= 'gabor',
							name 			= 'gabor_0',
							n_runs 			= 1,		
							n_epi_crit		= 5,				
							n_epi_dopa		= 5,				
							t				= 0.1, 							
							A 				= 1.2,
							lr				= 0.01,
							batch_size 		= 20,
							n_hid_neurons	= 49,
							init_file		= None,	
							lim_weights		= False,
							noise_std		= 0.2, #digit: 4.0; gabor: 0.2 (?)
							exploration		= True,
							pdf_method 		= 'fit',
							classifier		= 'neural',
							test_each_epi	= True,
							verbose			= True,
							seed 			= 995 #np.random.randint(1000)
							)

""" load and pre-process training and testing images """
images_dict, labels_dict, images_params = ex.load_images(	protocol 		= net.protocol,
												A 				= net.A,
												verbose 		= net.verbose,
												digit_params 	= {	'classes' 		: np.array([ 4, 7, 9 ], dtype=int),
																	'dataset_train'	: 'test',
																	'dataset_path' 	: '/Users/raphaelholca/Documents/data-sets/MNIST',
																	},
												gabor_params 	= {	'n_train' 		: 10000,
																	'n_test' 		: 1000,
																	'target_ori' 	: 85.,
																	'excentricity' 	: 3.,
																	'noise_crit'	: 0.,
																	'noise_train'	: 0.,
																	'noise_test'	: 0.2,
																	'im_size'		: 28,
																	}
												)

tic = time.time()

net.train(images_dict, labels_dict, images_params)

toc = time.time()

perf_dict = net.test(images_dict, labels_dict)

net.assess(	images_dict['train'], 
			labels_dict['train'],
			save_data	= True, 
			show_W_act	= True, 
			sort		= None, 
			target 		= None
			)

print '\nrun name:\t' + net.name
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))





















