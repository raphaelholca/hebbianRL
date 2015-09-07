import numpy as np
import pickle
import os
import struct
import recon as rc
import support.hebbianRL as rl
import support.external as ex
import support.mnist as mnist
import support.grating as gr

rc = reload(rc)
rl = reload(rl)
ex = reload(ex)
gr = reload(gr)

""" 
experimental variables

classes (int) 	: class of the MNIST dataset to use to train the network
rActions (str)	: for each class of MNIST, the action that is rewarded. Capital letters indicates a class that is paired with ACh release.
"""

""" parameters """
kwargs = {
'nRun' 			: 1					,# number of runs
'nEpiCrit'		: 5 				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)		#50
'nEpiDopa'		: 0					,# number of 'adult' episodes in each run (episodes when reward is not required for learning)				#20
't_hid'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for hidden layer
't_act'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for action layer
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'gabor-2'			,# name of the folder where to save results
'dataset'		: 'train'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
'nHidNeurons'	: 49				,# number of hidden neurons
'lr'			: 0.005 			,# learning rate during 'critica period' (pre-training, nEpiCrit)
'aHigh' 		: 0.0 				,# learning rate increase for relevance signal (high ACh) outside of critical period
'aPairing'		: 1.0 				,# strength of ACh signal for pairing protocol
'dHigh' 		: 7.0 				,# learning rate increase for unexpected reward
'dMid' 			: 0.01 				,# learning rate increase for correct reward prediction
'dNeut' 		: -0.2				,# learning rate increase for correct no reward prediction
'dLow' 			: -2.0				,# learning rate increase for incorrect reward predictio
'nBatch' 		: 20 				,# mini-batch size
'protocol'		: 'gabor'			,# training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination)
'target_ori' 	: 135. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
'excentricity' 	: 20. 				,# range within wich to test the network (on each side of target orientation)
'classifier'	: 'actionNeurons'	,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass'
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'bestAction' 	: False				,# whether to take predicted best action (True) or take random actions (False)
'createOutput'	: True				,# whether to create plots, save data, etc. (set to False when using pypet)
'showPlots'		: False				,# whether to display plots
'show_W_act'	: True				,# whether to display W_act weights on the weight plots
'sort' 			: None				,# sorting methods for weights when displaying. Legal value: None, 'class', 'tSNE'
'target'		: None 				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 168#np.random.randint(1000) 	# seed of the random number generator
}

""" load and pre-process images """
ex.checkClassifier(kwargs['classifier'])
print 'seed: ' + str(kwargs['seed']) + '\n'

if kwargs['protocol'] == 'digit':
	kwargs['classes'] 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
	kwargs['rActions'] 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')

	# kwargs['classes'] 	= np.array([ 0,  1 , 2 , 3 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b','c','d'], dtype='|S1')

	# kwargs['classes'] 	= np.array([ 4 , 7 , 9 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b','c'], dtype='|S1')

	# kwargs['classes'] 	= np.array([ 4 , 9 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b'], dtype='|S1')

	imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
	print 'loading train images...'
	images, labels = mnist.read_images_from_mnist(classes = kwargs['classes'], dataset = kwargs['dataset'], path = imPath)
	images, labels = ex.evenLabels(images, labels, kwargs['classes'])
	images = ex.normalize(images, kwargs['A']*np.size(images,1))

	print 'loading test images...'
	test_dataset='test' if kwargs['dataset']=='train' else 'train'
	images_test, labels_test = mnist.read_images_from_mnist(classes = kwargs['classes'], dataset = test_dataset, path = imPath)
	images_test, labels_test = ex.evenLabels(images_test, labels_test, kwargs['classes'])
	images_test, labels_test = ex.shuffle([images_test, labels_test])
	images_test = ex.normalize(images_test, kwargs['A']*np.size(images_test,1))
	
	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba = rl.RLnetwork(images=images, labels=labels, images_test=images_test, labels_test=labels_test, kwargs=kwargs, **kwargs)


elif kwargs['protocol'] == 'gabor':
	print 'creating gabor training images...'
	
	kwargs['classes'] 	= np.array([ 0 , 1 ], dtype=int)
	kwargs['rActions'] 	= np.array(['a','b'], dtype='|S1')
	target_ori = kwargs['target_ori']
	excentricity = kwargs['excentricity']
	n_train = 10000
	orientations = np.random.random(n_train)*180 #orientations of gratings (in degrees)
	labels = np.zeros(n_train, dtype=int)
	labels[orientations>=target_ori] = 0
	labels[orientations<target_ori] = 0
	images = gr.gabor(size=28, lambda_freq=5, theta=orientations, sigma=28./5., phase=0, noise=0) #np.random.random(n_train)
	images = ex.normalize(images, kwargs['A']*np.size(images,1))

	n_task = 10000
	orientations_task = np.random.random(n_task)*excentricity*2 + target_ori - excentricity #orientations of gratings (in degrees)
	labels_task = np.zeros(n_task, dtype=int)
	labels_task[orientations_task>=target_ori] = 0
	labels_task[orientations_task<target_ori] = 0
	images_task = gr.gabor(size=28, lambda_freq=5, theta=orientations_task, sigma=28./5., phase=0, noise=0)
	images_task = ex.normalize(images_task, kwargs['A']*np.size(images,1))

	n_test = 100
	orientations_test = np.random.random(n_test)*180
	labels_test = np.zeros(n_test, dtype=int)
	labels_test[orientations_test>=target_ori] = 0
	labels_test[orientations_test<target_ori] = 0
	images_test = gr.gabor(size=28, lambda_freq=5, theta=orientations_test, sigma=28./5., phase=0, noise=0)
	images_test = ex.normalize(images_test, kwargs['A']*np.size(images_test,1))

	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba = rl.RLnetwork(images=images, labels=labels, orientations=orientations, images_task=images_task, labels_task=labels_task, orientations_task=orientations_task, images_test=images_test, labels_test=labels_test, orientations_test=orientations_test, kwargs=kwargs, **kwargs)




