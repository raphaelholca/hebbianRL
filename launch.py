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
import support.hebbianRL as rl
import support.external as ex
import support.plots as pl

rl = reload(rl)
ex = reload(ex)
pl = reload(pl)

""" create Hebbian neural network """
net = rl.Network(	dopa_values		= {	'dHigh' 	: 4.5,
										'dMid' 		: 0.02,
										'dNeut' 	: -0.1,
										'dLow' 		: -2.0,
										},
					name 			= 'test',
					n_run 			= 1,		
					n_epi_crit		= 10,				
					n_epi_dopa		= 10,				
					t				= 0.1, 							
					A 				= 1.2,
					n_hid_neurons	= 49,
					lim_weights		= False,
					lr				= 0.01,
					noise_std		= 0.2, 					#digit: 4.0 	; gabor: 0.2 (?)
					exploration		= True,
					pdf_method 		= 'fit',
					n_batch 		= 20,
					protocol		= 'digit',
					classifier		= 'actionNeurons',
					pre_train		= None,					#'digit_479_16'
					test_each_epi	= True,
					SVM				= False,
					save_data		= True,
					verbose			= True,
					show_W_act		= False,
					sort 			= None,
					target			= None,
					seed 			= 995 					#np.random.randint(1000)
					)

""" load and pre-process training and testing images """
images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task = ex.load_images(net,
																																		classes 		= np.array([ 4, 7, 9 ], dtype=int),
																																		dataset_train	= 'test',
																																		dataset_path 	= '/Users/raphaelholca/Documents/data-sets/MNIST',
																																		gabor_params 	= {	'target_ori' 	: 85.,
																																							'excentricity' 	: 3.,
																																							'noise_crit'	: 0.,
																																							'noise_train'	: 0.,
																																							'noise_test'	: 0.2,
																																							'im_size'		: 28,
																																							}
																																		)



tic = time.time()

net.train(	images, labels, orientations, 
			images_test, labels_test, orientations_test, 
			images_task, labels_task, orientations_task, 
			)

allCMs, allPerf, perc_correct_W_act = net.test(images, labels)

net.assess(images, labels)

net.save()



print '\n\nstart time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time:\t' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(time.time()))
print 'run time:\t' + time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))




