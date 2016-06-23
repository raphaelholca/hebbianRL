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
import datetime
import hebbian_net
import helper.external as ex
import helper.assess_network as an

hebbian_net = reload(hebbian_net)
ex = reload(ex)
an = reload(an)

""" create Hebbian neural network """
net = hebbian_net.Network(	dHigh 				= 1.6,
							dMid 				= 0.0,
							dNeut 				= -0.08,
							dLow 				= -0.8,
							dopa_out_same 		= False,
							train_out_dopa 		= False,
							dHigh_out			= 0.0,
							dMid_out			= 0.2,
							dNeut_out			= -0.3,
							dLow_out			= -0.5,
							ach_1 				= 16.0,
							ach_2 				= 9.0,
							ach_3 				= 0.0,
							ach_4 				= 0.0,
							ach_func 			= 'sigmoidal', #'linear', 'exponential', 'polynomial', 'sigmoidal', 'handmade', 'preset'
							ach_avg 			= 20,
							protocol			= 'digit', #'toy_data', 'gabor', 'digit'
							name 				= 'long_DA',
							dopa_release 		= True, 
							ach_release			= False, 
							n_runs 				= 10,
							n_epi_crit			= 0,
							n_epi_fine 			= 0,
							n_epi_perc			= 300,
							n_epi_post 			= 0,
							t_hid				= 1.0,
							t_out				= 0.1,
							A					= 1.0e3,
							lr_hid				= 5e-4,
							lr_out				= 5e-7,
							batch_size 			= 50,
							block_feedback 		= False,
							shuffle_datasets	= True,
							n_hid_neurons		= 49,
							weight_init 		= 'input',
							init_file			= 'digit_pretrain_lr_5e-4_long',
							lim_weights			= True,
							log_weights 		= True,
							epsilon_xplr 		= 1.0,
							noise_xplr_hid		= 0.2,
							noise_xplr_out		= 2e4,
							exploration			= True,
							compare_output 		= True,
							noise_activ			= 0.0,
							pdf_method 			= 'fit',
							classifier			= 'neural_prob',
							RF_classifier 		= 'data',
							test_each_epi		= True,
							early_stop 			= False,
							verbose				= True,
							seed 				= 985 #np.random.randint(1000)
							)

""" load and pre-process training and testing images """
images_dict, labels_dict, ori_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																	A				= net.A,
																	verbose 		= net.verbose,
																	digit_params 	= {	'dataset_train'		: 'train',
																						# 'classes' 			: np.array([ 1, 4, 9 ], dtype=int),
																						'classes' 			: np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype=int),
																						'dataset_path' 		: '/Users/raphaelholca/Documents/data-sets/MNIST',
																						'shuffle'			: True
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

CM_all, perf_all = net.test(images_dict['test'], labels_dict['test'])

ex.save_net(net)

an.assess(	net,
			show_W_act		= True, 
			sort			= None, 
			target 			= None,
			test_all_ori 	= False,
			images 			= images_dict['train'],
			labels 			= labels_dict['train']
			)

print '\nrun name:\t' + net.name
print 'start time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(net._train_start))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(net._train_stop))
print 'train time:\t' +  str(datetime.timedelta(seconds=net.runtime))





















