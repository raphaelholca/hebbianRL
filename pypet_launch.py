"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code uses PyPet to explore the parameters of the hebbian neural network object.
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
import helper.assess_network as an

hebbian_net = reload(hebbian_net)
ex = reload(ex)
an = reload(an)

""" create Hebbian neural network """
net = hebbian_net.Network(	dopa_values		= {	'dHigh' 	: 4.5,
												'dMid' 		: 0.02,
												'dNeut' 	: -0.1,
												'dLow' 		: -2.0,
												},
							protocol		= 'digit',
							name 			= 'digit_long_highExplr',
							n_runs 			= 3,		
							n_epi_crit		= 10,				
							n_epi_dopa		= 20,				
							t				= 0.1, 							
							A 				= 1.2,
							lr				= 0.01,				#0.005
							batch_size 		= 20,
							n_hid_neurons	= 49,
							init_file		= None,	
							lim_weights		= False,
							noise_std		= 0.2
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

an.assess(	net,
			images_dict['train'], 
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




















