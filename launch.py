import os
import matplotlib
if 'Documents' in os.getcwd():
	matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers

import numpy as np
import pickle
import struct
import pypet
import time
import support.hebbianRL as rl
import support.external as ex
import support.plots as pl
import support.mnist as mnist
from scipy.optimize import basinhopping
from scipy.optimize import minimize
from scipy.optimize import brute
# from sknn.mlp import Regressor, Layer

rl = reload(rl)
ex = reload(ex)
pl = reload(pl)

def get_images():
	return images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task

def pypet_RLnetwork(traj):
	images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task = get_images()
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)

	try:
		allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(None, images, labels, orientations, 
																						images_test, labels_test, orientations_test, 
																						images_task, labels_task, orientations_task, 
																						None, parameter_dict, **parameter_dict)
	except ValueError:
		allPerf = np.ones(kwargs['nRun'])*-1
		perc_correct_W_act = np.ones(kwargs['nRun'])*-1

	traj.f_add_result('RLnetwork.$', 
					perc_W_act=perc_correct_W_act, 
					perf=allPerf, 
					comment='param exploration')

	return np.round(perc_correct_W_act,3), np.round(np.mean(allPerf),2)

""" parameters """
kwargs = {
'nRun' 			: 1					,# number of runs
'nEpiCrit'		: 7 				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
'nEpiDopa'		: 5					,# number of 'adult' episodes in each run (episodes when reward is not required for learning)
't_hid'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for hidden layer 
't_act'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for action layer 
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'test'			,# name of the folder where to save results
'dataset'		: 'test'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
'nHidNeurons'	: 49				,# number of hidden neurons
'lim_weights'	: False 			,# whether to artificially limit the value of weights. Used during parameter exploration
'lr'			: 0.01 				,# learning rate during 'critica period' (pre-training, nEpiCrit)
'e_greedy'		: False 			,# whether to use an epsilon-greedy approach to noise injection
'epsilon'		: 1.0 				,# probability of taking an exploratory decisions, range: [0,1]
'noise_std'		: 0.2 				,# parameter of the standard deviation of the normal distribution from which noise is drawn						digit: 4.0 	; gabor: 0.2 (?)
'proba_predict'	: False				,# whether the reward prediction is probabilistic (True) or deterministic/binary (False)
'exploration' 	: True				,# whether to take take explorative decisions (True) or not (False)
'pdf_method' 	: 'fit'				,# method used to approximate the pdf; valid: 'fit', 'subsample', 'full'

'dHigh' 		: 2.5 				,# learning rate increase for unexpected reward																	digit: 4.5	; gabor: 2.0
'dMid' 			: 6.00 				,# learning rate increase for correct reward prediction															digit: 0.02	; gabor: ---
'dNeut' 		: 6.				,# learning rate increase for correct no reward prediction														digit: -0.1	; gabor: ---
'dLow' 			: -1.5				,# learning rate increase for incorrect reward prediction														digit: -2.0	; gabor: 0.0

'a_0'			: 0.0				,# parameters for the RPE function
'a_1'			: 0.0				,
'a_2'			: 0.0				,
'a_3'			: 0.0				,

'nBatch' 		: 20 				,# mini-batch size
'protocol'		: 'digit'			,# training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination)
'target_ori' 	: 85. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
'excentricity' 	: 3. 				,# degree range within wich to test the network (on each side of target orientation)
'noise_crit'	: 0. 				,# noise injected in the gabor filter for the pre-training (critical period)
'noise_train'	: 0. 				,# noise injected in the gabor filter for the training
'noise_test'	: 0.2 				,# noise injected in the gabor filter for the testing
'im_size'		: 28 				,# side of the gabor filter image (total pixels = im_size * im_size)
'classifier'	: 'actionNeurons'	,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass', 'bayesian'
'temp_xplr'		: 1e-3				,# temperature for exploration in neural network-based parameter exploration
'pre_train'		: 'None'			,# initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization #'digit_479_16'
'test_each_epi'	: True 				,# whether to test the network's performance at each episode
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'save_data'		: True				,# whether to save data to disk
'verbose'		: True				,# whether to create text output
'show_W_act'	: False				,# whether to display W_act weights on the weight plots
'sort' 			: None				,# sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'
'target'		: None				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 995, #np.random.randint(1000), 	# seed of the random number generator

'comment'		: ''

}

""" parameters of the RPE function """
kwargs['RPE_function'] = 'two_lin' 		# RPE value; valid: 'neural' (function approx.), 'discrete', or callable function, e.g.: ex.polynomial, ex.tanh
# RPE_function_params = [3.11, 10000., 0.05, 0.148]				# parameters of the RPE function, if RPE_function is a callable function
# RPE_function_params = [2.4, 1000., 0.05, -2.0]				# parameters of the RPE function, if RPE_function is a callable function
# RPE_function_params = [0.2, 0.4, 0.2, 0.6]	
RPE_function_params = [0.0, 0.0, 0.0, 0.0]	

""" load and pre-process images """
ex.checkClassifier(kwargs['classifier'])
print 'seed: ' + str(kwargs['seed']) + '\n'
if not kwargs['save_data']: print "!!! ----- not saving data ----- !!! \n"
np.random.seed(kwargs['seed'])

global images, labels, orientations
global images_test, labels_test, orientations_test
global images_task, labels_task, orientations_task

if kwargs['protocol'] == 'digit':
	# kwargs['classes'] 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')

	# kwargs['classes'] 	= np.array([ 0 , 1 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b'], dtype='|S1')

	kwargs['classes'] 	= np.array([ 4 , 7 , 9 ], dtype=int)
	kwargs['rActions'] 	= np.array(['a','b','c'], dtype='|S1')

	# kwargs['classes'] 	= np.array([ 4 , 9 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b'], dtype='|S1')

	_, idx = np.unique(kwargs['rActions'], return_index=True)
	lActions = kwargs['rActions'][np.sort(idx)]
	ex.set_global(lActions, kwargs['rActions'], kwargs['classes'], kwargs)

	imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
	if kwargs['verbose']: print 'loading train images...'
	images, labels = mnist.read_images_from_mnist(classes = kwargs['classes'], dataset = kwargs['dataset'], path = imPath)
	images, labels = ex.evenLabels(images, labels)
	images = ex.normalize(images, kwargs['A']*np.size(images,1))

	if kwargs['verbose']: print 'loading test images...'
	test_dataset='test' if kwargs['dataset']=='train' else 'train'
	images_test, labels_test = mnist.read_images_from_mnist(classes = kwargs['classes'], dataset = test_dataset, path = imPath)
	images_test, labels_test = ex.evenLabels(images_test, labels_test)
	images_test, labels_test = ex.shuffle([images_test, labels_test])
	images_test = ex.normalize(images_test, kwargs['A']*np.size(images_test,1))
	
	orientations = None
	images_task = None
	labels_task = None
	orientations_task = None
	orientations_test = None

elif kwargs['protocol'] == 'gabor':
	if kwargs['verbose']: print 'creating gabor training images...'
	
	kwargs['classes'] 	= np.array([ 0 , 1 ], dtype=int)
	kwargs['rActions'] 	= np.array(['a','b'], dtype='|S1')

	_, idx = np.unique(kwargs['rActions'], return_index=True)
	lActions = kwargs['rActions'][np.sort(idx)]
	ex.set_global(lActions, kwargs['rActions'], kwargs['classes'])

	n_train = 50000
	n_test = 1000

	orientations = np.random.random(n_train)*180 #orientations of gratings (in degrees)
	images, labels = ex.generate_gabors(orientations, kwargs['target_ori'], kwargs['im_size'], kwargs['noise_crit'], kwargs['A'])

	orientations_task = np.random.random(n_train)*kwargs['excentricity']*2 + kwargs['target_ori'] - kwargs['excentricity'] #orientations of gratings (in degrees)
	images_task, labels_task = ex.generate_gabors(orientations_task, kwargs['target_ori'], kwargs['im_size'], kwargs['noise_train'], kwargs['A'])

	orientations_test = np.random.random(n_test)*kwargs['excentricity']*2 + kwargs['target_ori'] - kwargs['excentricity'] #orientations of gratings (in degrees)
	images_test, labels_test = ex.generate_gabors(orientations_test, kwargs['target_ori'], kwargs['im_size'], kwargs['noise_test'], kwargs['A'])

""" parameter exploration """
tic = time.time()

if kwargs['save_data']: kwargs['runName'] = ex.checkdir(kwargs, OW_bool=True) #create saving directory
allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(	RPE_function_params, 
																					images, labels, orientations, 
																					images_test, labels_test, orientations_test, 
																					images_task, labels_task, orientations_task, 
																					None, kwargs, **kwargs)

print '\n\nstart time: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(time.time()))
print 'run time: ' + time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))




