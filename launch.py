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
net = hebbian_net.Network(	dHigh 			= 0.0,#1.6,#
							dMid 			= 0.01,#0.0,#
							dNeut 			= -0.5,#-0.08,#
							dLow 			= -4.0,#-0.4,#
							dopa_out_same 	= False,
							train_out_dopa 	= False,
							dHigh_out		= 3.0,#0.0,#
							dMid_out		= 0.01,#0.2,#
							dNeut_out		= -0.01,#-0.3,#
							dLow_out		= -3.0,#-0.5,#
							protocol		= 'toy2D',#'digit',#
							name 			= 'toy2D_lrOut_5e-10_2',
							n_runs 			= 1,
							n_epi_crit		= 500,
							n_epi_fine 		= 0,
							n_epi_dopa		= 0,				
							n_epi_post 		= 0,				
							t				= 0.00001,#1.0,#0.1,#
							A_hid			= 3.,
							A_out			= 3.,
							lr_hid			= 5e-4,#5e-3,
							lr_out			= 5e-10,
							batch_size 		= 50,
							block_feedback 	= False,
							n_hid_neurons	= 2,#49,#
							init_file		= '',
							lim_weights		= False,
							epsilon_xplr 	= 1.0,#1.0,#
							noise_xplr_hid	= 0.3,#0.2#
							noise_xplr_out	= 2e4,
							exploration		= True,
							compare_output 	= True,
							noise_activ		= 0.,
							pdf_method 		= 'fit',
							classifier		= 'neural',
							test_each_epi	= True,
							early_stop 		= False,
							verbose			= True,
							seed 			= 975 #np.random.randint(1000)
							)

""" load and pre-process training and testing images """
images_dict, labels_dict, ori_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																	A				= net.A_hid,
																	verbose 		= net.verbose,
																	digit_params 	= {	'dataset_train'		: 'train',
																						# 'classes' 			: np.array([ 4, 7, 9 ], dtype=int),
																						'classes' 			: np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype=int),
																						'dataset_path' 		: '/Users/raphaelholca/Documents/data-sets/MNIST',
																						'shuffle'			: False
																						},
																	gabor_params 	= {	'n_train' 			: 10000,
																						'n_test' 			: 10000,
																						'renew_trainset'	: False,
																						'target_ori' 		: 165.,
																						'excentricity' 		: 90.,#3.0,#1.5,
																						'noise_pixel'		: 0.0,
																						'rnd_phase' 		: False,
																						'rnd_freq' 			: False,
																						'im_size'			: 28
																						},
																	toy2D_params	= {	'n_points'			: 2000,
																						'separability' 		: '1D', #'1D'#'2D'#'non_linear'
																						'data_distrib' 		: 'uniform' #'uniform' #'normal' #'bimodal'
																						}
																	)

net.train(images_dict, labels_dict, images_params)

perf_dict = net.test(images_dict, labels_dict)

ex.save_net(net)

an.assess(	net,
			show_W_act		= True, 
			sort			= None, 
			target 			= None,
			test_all_ori 	= False
			)

print '\nrun name:\t' + net.name
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(net._train_start))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(net._train_stop))
print 'train time:\t' + time.strftime("%H:%M:%S", time.gmtime(net.runtime))





















