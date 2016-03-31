"""
Author: Raphael Holca-Lamarre
Date: 23/10/2014

This code creates a hebbian neural network object and trains it on the MNIST dataset. The learning rule is a hebbian learning rule augmented with a learning mechanism inspired from dopamine signalling in animal cortex.
"""

import os
import matplotlib
if 'mnt' in os.getcwd(): matplotlib.use('Agg')
import numpy as np
import time
import hebbian_net
import helper.external as ex
import helper.assess_network as an

hebbian_net = reload(hebbian_net)
ex = reload(ex)
an = reload(an)

""" create Hebbian neural network """
net = hebbian_net.Network(	dHigh 			= 1.0,#0.8,#1.6,#
							dMid 			= 0.00,#0.001,#0.0,#
							dNeut 			= -0.1,#-0.04,#-0.08,#
							dLow 			= -1.0,#-0.2,#-0.4,#
							dopa_out_same 	= False,
							dHigh_out		= 8.0,#0.0
							dMid_out		= 0.00,#0.2
							dNeut_out		= -0.1,#-0.3
							dLow_out		= -1.0,#-0.5
							protocol		= 'gabor',#'digit',#
							name 			= 'gabor_t_1-0_nhid_8_noise_0-3_DA_lowdHigh',
							n_runs 			= 1,		
							n_epi_crit		= 20,				
							n_epi_dopa		= 30,				
							t				= 1.0,#0.1,
							A 				= 1.2,
							lr_hid			= 5e-3,
							lr_out			= 5e-7,
							batch_size 		= 50,
							block_feedback 	= False,
							n_hid_neurons	= 8,#49,#
							init_file		= '',
							lim_weights		= False,
							noise_std		= 0.2,
							exploration		= True,
							pdf_method 		= 'fit',
							classifier		= 'neural',
							test_each_epi	= True,
							early_stop 		= False,
							verbose			= True,
							seed 			= 981 #np.random.randint(1000)
							)

""" load and pre-process training and testing images """
images_dict, labels_dict, ori_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																	A 				= net.A,
																	verbose 		= net.verbose,
																	digit_params 	= {	'dataset_train'		: 'train',
																						# 'classes' 			: np.array([ 4, 7, 9 ], dtype=int),
																						'classes' 			: np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype=int),
																						'dataset_path' 		: '/Users/raphaelholca/Documents/data-sets/MNIST',
																						'shuffle'			: False
																						},
																	gabor_params 	= {	'n_train' 			: 10000,
																						'n_test' 			: 10000,
																						'renew_trainset'	: True,
																						'target_ori' 		: 165.,
																						'excentricity' 		: 90.,#3.0,#1.5,
																						'noise'				: 0.3,
																						'im_size'			: 50#28,
																						}
																	)

tic = time.time()

net.train(images_dict, labels_dict, images_params)

toc = time.time()

perf_dict = net.test(images_dict, labels_dict)

ex.save_net(net)

an.assess(	net,
			show_W_act		= True, 
			sort			= None, 
			target 			= None,
			test_all_ori 	= False
			)

print '\nrun name:\t' + net.name
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(toc))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(toc-tic))





















