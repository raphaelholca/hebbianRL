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
																																		rActions 		= np.array(['a','b','c'], dtype='|S1'),
																																		dataset_train	= 'test',
																																		dataset_path 	= '/Users/raphaelholca/Documents/data-sets/MNIST',
																																		)


kwargs = {
'dataset'		: 'test'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination

'target_ori' 	: 85. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
'excentricity' 	: 3. 				,# degree range within wich to test the network (on each side of target orientation)
'noise_crit'	: 0. 				,# noise injected in the gabor filter for the pre-training (critical period)
'noise_train'	: 0. 				,# noise injected in the gabor filter for the training
'noise_test'	: 0.2 				,# noise injected in the gabor filter for the testing
'im_size'		: 28 				,# side of the gabor filter image (total pixels = im_size * im_size)

}

""" load and pre-process images """
ex.checkClassifier(net.classifier)
print 'seed: ' + str(net.seed) + '\n'
if not net.save_data: print "!!! ----- not saving data ----- !!! \n"
np.random.seed(net.seed)

global images, labels, orientations
global images_test, labels_test, orientations_test
global images_task, labels_task, orientations_task

""" parameter exploration """
tic = time.time()

if net.save_data: net.name = ex.checkdir(net, OW_bool=True) #create saving directory

allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = net.train(	images, labels, orientations, 
																					images_test, labels_test, orientations_test, 
																					images_task, labels_task, orientations_task, 
																					kwargs, **kwargs)

print '\n\nstart time: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(time.time()))
print 'run time: ' + time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))




