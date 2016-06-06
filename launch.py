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
net = hebbian_net.Network(	dHigh 			= 1.6,#0.0,#
							dMid 			= 0.0,#0.01,#
							dNeut 			= -0.08,#-0.5,#
							dLow 			= -0.4,#-4.0,#
							dopa_out_same 	= False,
							train_out_dopa 	= False,
							dHigh_out		= 0.0,#2.0,#
							dMid_out		= 0.2,#0.,#
							dNeut_out		= -0.3,#-0.,#
							dLow_out		= -0.5,#-0.2,#
							ach_1 			= 50.0,
							ach_2 			= 0.5,
							ach_3 			= 0.0,
							ach_4 			= 0.0,
							ach_func 		= 'sigmoidal', #'linear', 'exponential', 'polynomial', 'sigmoidal', 'handmade', 'preset'
							protocol		= 'digit', #'toy_data',#'gabor',#'digit',#
							name 			= 'digit_ACh_pretrain300_lr_1e-3_sigm-release_avg5',
							dopa_release 	= False, 
							ach_release		= True, 
							n_runs 			= 1,
							n_epi_crit		= 300,
							n_epi_fine 		= 0,
							n_epi_perc		= 700,
							n_epi_post 		= 0,
							t_hid			= 1.0,#3e0,#
							t_out			= 0.1,#1.0,#
							A				= 1.0e3,
							lr_hid			= 1e-3,#5e-6,#
							lr_out			= 5e-7,#5e-5,#
							batch_size 		= 50,
							block_feedback 	= False,
							n_hid_neurons	= 49,#16,#
							weight_init 	= 'input',
							init_file		= '',
							lim_weights		= False,
							log_weights 	= True,#False,#
							epsilon_xplr 	= 1.0,
							noise_xplr_hid	= 0.3,
							noise_xplr_out	= 2e4,#2e2,#
							exploration		= True,
							compare_output 	= True,
							noise_activ		= 0.0,
							pdf_method 		= 'fit',
							classifier		= 'neural_prob',
							RF_classifier 	= 'svm',
							test_each_epi	= True,
							early_stop 		= False,
							verbose			= True,
							seed 			= 977 #np.random.randint(1000)
							)

""" load and pre-process training and testing images """
images_dict, labels_dict, ori_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																	A				= net.A,
																	verbose 		= net.verbose,
																	digit_params 	= {	'dataset_train'		: 'train',
																						'classes' 			: np.array([ 1, 4, 9 ], dtype=int),
																						# 'classes' 			: np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype=int),
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
																						'im_size'			: 50
																						},
																	toy_data_params	= {	'dimension' 		: '2D', #'2D' #'3D'
																						'n_points'			: 2000,
																						'separability' 		: '1D', #'1D'#'2D'#'non_linear'
																						'data_distrib' 		: 'uniform' #'uniform' #'normal' #'multimodal'
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





















