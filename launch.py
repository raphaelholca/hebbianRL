"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code creates a hebbian neural network object and trains it on the MNIST dataset. The learning rule is a hebbian learning rule augmented with a learning mechanism inspired from dopamine signalling in animal cortex.
"""

import os
import matplotlib
if 'Documents' in os.getcwd(): matplotlib.use('Agg')
import numpy as np
import time
import hebbian_net
import helper.external as ex
import helper.assess_network as an

hebbian_net = reload(hebbian_net)
ex = reload(ex)
an = reload(an)

""" create Hebbian neural network """
net = hebbian_net.Network(	dHigh 			= 0.0,
							dMid 			= 0.20,
							dNeut 			= 0.0,
							dLow 			= -4.0,
							protocol		= 'digit',
							name 			= 'compare_launch',
							n_runs 			= 1,		
							n_epi_crit		= 0,				
							n_epi_dopa		= 15,				
							t				= 0.1, 							
							A 				= 1.2,
							lr				= 0.01,				#0.01
							batch_size 		= 20,
							n_hid_neurons	= 300,
							init_file		= 'digit_pretrained_large',	
							lim_weights		= False,
							noise_std		= 0.2,
							exploration		= False,
							pdf_method 		= 'fit',
							classifier		= 'neural',
							test_each_epi	= True,
							verbose			= True,
							seed 			= 997 #np.random.randint(1000)
							)

""" load and pre-process training and testing images """
images_dict, labels_dict, images_params = ex.load_images(	protocol 		= net.protocol,
															A 				= net.A,
															verbose 		= net.verbose,
															digit_params 	= {	'classes' 		: np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype=int),
																				'dataset_train'	: 'train',
																				'dataset_path' 	: '/Users/raphaelholca/Documents/data-sets/MNIST',
																				},
															gabor_params 	= {	'n_train' 		: 10000,
																				'n_test' 		: 1000,
																				'target_ori' 	: 85.,
																				'excentricity' 	: 3.,
																				'noise_crit'	: 0.,
																				'noise_train'	: 0.,
																				'noise_test'	: 0.,
																				'im_size'		: 28,
																				}
															)

tic = time.time()

net.train(images_dict, labels_dict, images_params)

toc = time.time()

perf_dict = net.test(images_dict, labels_dict)

ex.save_net(net)

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





















