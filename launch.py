import numpy as np
import pickle
import os
import struct
import pypet
import support.hebbianRL as rl
import support.external as ex
import support.mnist as mnist

rl = reload(rl)
ex = reload(ex)

def get_images():
	return images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task

def pypet_RLnetwork(traj):
	images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task = get_images()
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)

	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba = rl.RLnetwork(	images, labels, orientations, 
																			images_test, labels_test, orientations_test, 
																			images_task, labels_task, orientations_task, 
																			parameter_dict, **parameter_dict)

	traj.f_add_result('RLnetwork.$', 
					perc_W_act=perc_correct_W_act, 
					perf=np.mean(allPerf), 
					comment='exploring dopa for action weights')

	return np.round(perc_correct_W_act,3), np.round(np.mean(allPerf),2)

""" parameters """
kwargs = {
'nRun' 			: 1					,# number of runs
'nEpiCrit'		: 5 				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)		#50
'nEpiDopa'		: 5					,# number of 'adult' episodes in each run (episodes when reward is not required for learning)				#20
't_hid'			: 0.0001 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for hidden layer
't_act'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for action layer
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'digit-perf'			,# name of the folder where to save results
'dataset'		: 'train'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
'nHidNeurons'	: 16					,# number of hidden neurons
'lr'			: 0.005 			,# learning rate during 'critica period' (pre-training, nEpiCrit)
'aHigh' 		: 0.0 				,# learning rate increase for relevance signal (high ACh) outside of critical period
'aPairing'		: 1.0 				,# strength of ACh signal for pairing protocol
'dHigh' 		: 4.5 				,# learning rate increase for unexpected reward
'dMid' 			: 0. 				,# learning rate increase for correct reward prediction
'dNeut' 		: -0.1				,# learning rate increase for correct no reward prediction
'dLow' 			: -2.0				,# learning rate increase for incorrect reward predictio
'nBatch' 		: 20 				,# mini-batch size
'protocol'		: 'digit'			,# training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination)
'target_ori' 	: 85. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
'excentricity' 	: 10. 				,# degree range within wich to test the network (on each side of target orientation)
'noise_crit'	: 0. 				,# noise injected in the gabor filter for the pre-training (critical period)
'noise_train'	: 0. 				,# noise injected in the gabor filter for the training
'noise_test'	: 0. 				,# noise injected in the gabor filter for the testing
'im_size'		: 28 				,# side of the gabor filter image (total pixels = im_size * im_size)
'classifier'	: 'actionNeurons'	,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass'
'pypet_xplr'	: False				,# whether to compute pypet-based parameter exploration
'test_each_epi'	: True 				,# whether to test the network's performance at each episode
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'bestAction' 	: False				,# whether to take predicted best action (True) or take random actions (False)
'createOutput'	: True				,# whether to create plots, save data, etc. (set to False when using pypet)
'showPlots'		: False				,# whether to display plots
'show_W_act'	: True				,# whether to display W_act weights on the weight plots
'sort' 			: None				,# sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'
'target'		: None 				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 992#np.random.randint(1000) 	# seed of the random number generator
}

""" parameters for exploration """
explore_dict = {
'dHigh'			:	np.arange(0., 6.1, 1.5).tolist(),
# 'dHigh'			:	np.arange(0.00, 0.041, 0.01).tolist(),
'dMid'			:	np.arange(0.0, 0.81, 0.2).tolist(),
# 'dMid'			:	np.arange(0.00, 0.21, 0.05).tolist(),
'dNeut'			:	np.round(np.arange(-0.4, 0.1, 0.1),1).tolist(),
# 'dLow'			:	np.arange(-2.0, 0.1, 0.5).tolist()
'dLow'			:	np.arange(-4.0, 0.1, 1.0).tolist()
}

""" load and pre-process images """
ex.checkClassifier(kwargs['classifier'])
print 'seed: ' + str(kwargs['seed']) + '\n'
np.random.seed(kwargs['seed'])

global images, labels, orientations
global images_test, labels_test, orientations_test
global images_task, labels_task, orientations_task

if kwargs['protocol'] == 'digit':
	# kwargs['classes'] 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')

	# kwargs['classes'] 	= np.array([ 0,  1 , 2 , 3 ], dtype=int)
	# kwargs['rActions'] 	= np.array(['a','b','c','d'], dtype='|S1')

	kwargs['classes'] 	= np.array([ 4 , 7 , 9 ], dtype=int)
	kwargs['rActions'] 	= np.array(['a','b','c'], dtype='|S1')

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
	
	orientations = None
	images_task = None
	labels_task = None
	orientations_task = None
	orientations_test = None

elif kwargs['protocol'] == 'gabor':
	print 'creating gabor training images...'
	
	kwargs['classes'] 	= np.array([ 0 , 1 ], dtype=int)
	kwargs['rActions'] 	= np.array(['a','b'], dtype='|S1')

	n_train = 10000
	n_test = 1000
	
	orientations = np.random.random(n_train)*180 #orientations of gratings (in degrees)
	images, labels = ex.generate_gabors(orientations, kwargs['target_ori'], kwargs['im_size'], kwargs['noise_crit'], kwargs['A'])

	orientations_task = np.random.random(n_train)*kwargs['excentricity']*2 + kwargs['target_ori'] - kwargs['excentricity'] #orientations of gratings (in degrees)
	images_task, labels_task = ex.generate_gabors(orientations_task, kwargs['target_ori'], kwargs['im_size'], kwargs['noise_train'], kwargs['A'])

	orientations_test = np.random.random(n_test)*kwargs['excentricity']*2 + kwargs['target_ori'] - kwargs['excentricity'] #orientations of gratings (in degrees)
	images_test, labels_test = ex.generate_gabors(orientations_test, kwargs['target_ori'], kwargs['im_size'], kwargs['noise_test'], kwargs['A'])


if not kwargs['pypet_xplr']:
	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba = rl.RLnetwork(	images, labels, orientations, 
																				images_test, labels_test, orientations_test, 
																				images_task, labels_task, orientations_task, 
																				kwargs, **kwargs)
else:
	""" launch simulation with pypet """
	import matplotlib
	matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers

	env = pypet.Environment(trajectory = 'xplr',
							comment = 'testing with pypet...',
							log_stdout=False,
							add_time = False,
							multiproc = True,
							ncores = 12,
							filename='output/' + kwargs['runName'] + '/perf.hdf5',
							overwrite_file=False)

	traj = env.v_trajectory
	ex.add_parameters(traj, kwargs)

	explore_dict = pypet.cartesian_product(explore_dict, tuple(explore_dict.keys())) #if not all entry of dict need be explored through cartesian product replace tuple(.) only with relevant dict keys in tuple
	explore_dict['runName'] = ex.set_run_names(explore_dict, kwargs['runName'])
	traj.f_explore(explore_dict)

	#save parameters to file
	kwargs_save = kwargs.copy()
	for k in explore_dict.keys():
		if k != 'runName':
			kwargs_save[k] = np.unique(explore_dict[k])
	outName	= 'output/' + kwargs['runName']
	if not os.path.exists(outName): os.mkdir(outName)
	ex.save_data(None, None, kwargs_save, save_weights=False)

	#run the simuation
	env.f_run(pypet_RLnetwork)

	env.f_disable_logging() #disable logging and close all log-files




