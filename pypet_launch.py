import pypet
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import support.hebbianRL as rl
import support.external as ex
import support.mnist as mnist

rl = reload(rl)
ex = reload(ex)


""" 
experimental variables

classes (int) 	: class of the MNIST dataset to use to train the network
rActions (str)	: for each class of MNIST, the action that is rewarded. Capital letters indicates a class that is paired with ACh release.
"""

""" pypet-specific implementation """
def add_parameters(traj, kwargs):
	for k in kwargs.keys():
		traj.f_add_parameter(k, kwargs[k])

def add_exploration(traj, runName):
	explore_dict = {
	't_hid'			:	[10., 1., 0.1, 0.01, 0.001] ,# np.arange(-2., 20.1, 2.0).tolist(),
	't_act'			:	[10., 1., 0.1, 0.01, 0.001] ,# np.round(np.arange(-2.0, 9.1, 1.0),1).tolist(),
	# 'dNeut'			:	[ 0.,  -0.04, -0.08, -0.12, -0.16, -0.2] ,# np.round(np.arange(-0.6, 0.51, 0.1),1).tolist()
	# 'dLow'			:	[ -0.25, -0.5, -1.5] # np.arange(-9.0,2.1, 1.0).tolist()
	}

	explore_dict = pypet.cartesian_product(explore_dict, ('t_hid', 't_act'))
	explore_dict['runName'] = set_run_names(explore_dict, runName)
	traj.f_explore(explore_dict)

def set_run_names(explore_dict, runName):
	nXplr = len(explore_dict[explore_dict.keys()[0]])
	runName_list = [runName for _ in range(nXplr)]
	for n in range(nXplr):
		for k in explore_dict.keys():
			runName_list[n] += '_'
			runName_list[n] += k
			runName_list[n] += str(explore_dict[k][n]).replace('.', ',')
	return runName_list


def pypet_RLnetwork(traj):
	images, labels = get_images()
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)

	# try:
	allCMs, allPerf, perc_correct_W_act = rl.RLnetwork(images=images, labels=labels, kwargs=parameter_dict, **parameter_dict)
	# except ValueError:
	# 	print "----- NaN in computations -----"
	# 	allCMs, allPerf, perc_correct_W_act = -np.inf, -np.inf, -np.inf
	
	traj.f_add_result('RLnetwork.$', 
					perc_W_act=perc_correct_W_act, 
					perf=np.mean(allPerf), 
					comment='exploring dopa for action weights')

	return np.round(perc_correct_W_act,3), np.round(np.mean(allPerf),2)

def postproc(traj, result_list):
	""" TODO? """
	return

def get_images():
	""" load and pre-process images """
	images, labels = mnist.read_images_from_mnist(classes = classes, dataset = kwargs['dataset'], path = imPath)
	images, labels = ex.evenLabels(images, labels, classes)

	return images, labels

""" parameters """
kwargs = {
'nRun' 			: 3					,# number of runs
'nEpiCrit'		: 4					,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
'nEpiDopa'		: 0					,# number of 'adult' episodes in each run (episodes when reward is not required for learning)
't_hid'			: 0.001 			,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition)
't_act'			: 0.001 			,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition)
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'temp'			,# name of the folder where to save results
'dataset'		: 'train'			,# MNIST dataset to use; legal values: 'test', 'train' ##use train for actual results
'nHidNeurons'	: 49				,# number of hidden neurons
'lr'			: 0.005 			,# learning rate during 'critica period' (pre-training, nEpiCrit)
'aHigh' 		: 0.0 				,# learning rate increase for relevance signal (high ACh) outside of critical period
'aPairing'		: 1.0 				,# strength of ACh signal for pairing protocol
'dHigh' 		: 8.0 				,# learning rate increase for unexpected reward
'dMid' 			: 0.0 				,# learning rate increase for correct reward prediction
'dNeut' 		: -0.2				,# learning rate increase for correct no reward prediction
'dLow' 			: -3.0				,# learning rate increase for incorrect reward prediction
'nBatch' 		: 20 				,# mini-batch size
'classifier'	: 'actionNeurons'	,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass'
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'bestAction' 	: False				,# whether to take predicted best action (True) or take random actions (False)
'createOutput'	: True				,# whether to create plots, save data, etc. (set to False when using pypet)
'showPlots'		: False				,# whether to display plots
'show_W_act'	: True				,# whether to display W_act weights on the weight plots
'sort' 			: False				,# whether to sort weights by their class when displaying
'target'		: None 				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 992#np.random.randint(1000) 				# seed of the random number generator
}

classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
rActions 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')

# classes 	= np.array([ 3,  4 , 5 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['0','B','0','0','0','0'], dtype='|S1')
# rActions 	= np.array(['a','B','c','d','e','f'], dtype='|S1')
# rActions 	= np.array(['a','b','c','d','e','f'], dtype='|S1')

# classes 	= np.array([ 1 , 4 , 9 ], dtype=int)
# rActions 	= np.array(['a','b','c'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['a','b','c'], dtype='|S1')

# classes 	= np.array([ 4 , 9 ], dtype=int)
# rActions 	= np.array(['a','b'], dtype='|S1')

kwargs['classes'] 	= classes
kwargs['rActions'] 	= rActions

ex.checkClassifier(kwargs['classifier'])
print 'seed: ' + str(kwargs['seed']) + '\n'
print 'loading data...'
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'

""" launch simulation with pypet """
filename = os.path.join('output/' + kwargs['runName'], 'perf.hdf5')
env = pypet.Environment(trajectory = 'ach',
						comment = 'testing with pypet...',
						log_stdout=False,
						add_time = False,
						multiproc = True,
						ncores = 10,
						filename=filename,
						overwrite_file=False)

traj = env.v_trajectory
add_parameters(traj, kwargs)
add_exploration(traj, kwargs['runName'])

#run the simuation
env.f_run(pypet_RLnetwork)
env.f_disable_logging() #disable logging and close all log-files






