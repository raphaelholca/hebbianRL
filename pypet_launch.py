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
	# 'dMid'			:	np.arange(-0.25, 1.26, 0.25).tolist(),
	'dHigh'			:	np.arange(-2.0, 20.1, 2.0).tolist(),
	# 'dNeut'			:	np.round(np.arange(-0.125, 0.026, 0.025),3).tolist()
	'dLow'			:	np.arange(-22.0,0.1, 2.0).tolist()
	}

	explore_dict = pypet.cartesian_product(explore_dict, ('dHigh', 'dLow'))# 'dMid', 'dNeut'
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

	try:
		allCMs, allPerf, perc_correct_W_act = rl.RLnetwork(images=images, labels=labels, kwargs=parameter_dict, **parameter_dict)
	except ValueError:
		print "----- NaN in computations -----"
		allCMs, allPerf, perc_correct_W_act = -np.inf, -np.inf, -np.inf
	
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
'nRun' 			: 5				,# number of runs
'nEpiCrit'		: 2				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
'nEpiAch'		: 0				,# number of ACh episodes in each run (episodes when ACh only is active)
'nEpiProc'		: 2				,# number of 'procedural learning' episodes (to initialize the action weights after critical period)
'nEpiDopa'		: 2				,# number of 'adult' episodes in each run (episodes when reward is not required for learning)
'A' 			: 1.2			,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'dHigh_dLow_4',# name of the folder where to save results
'dataset'		: 'train'		,# MNIST dataset to use; legal values: 'test', 'train' ##use train for actual results
'nHidNeurons'	: 49			,# number of hidden neurons
'lrCrit'		: 0.005 		,# learning rate during 'critica period' (pre-training, nEpiCrit)
'lrAdlt'		: 0.005			,# learning rate after the end of the 'critica period' (adult/training, nEpiAch and nEpiDopa)
'aHigh' 		: 6. 			,# learning rate increase for relevance signal (high ACh) outside of critical period
'aLow'			: 1. 			,# learning rate increase without relevant signal (no ACh)
'dMid' 			: 0.0 			,# learning rate increase for correct reward prediction
'dHigh' 		: 5.0			,# learning rate increase for unexpected reward (high dopamine) outside of critical period
'dNeut' 		: 0.0			,# learning rate increase for no reward, when none predicted
'dLow' 			: -8.			,# learning rate increase for incorrect reward prediction (low dopamine)
'nBatch' 		: 20 			,# mini-batch size
'classifier'	: 'actionNeurons'	,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass'
'SVM'			: True			,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'bestAction' 	: False			,# whether to take predicted best action (True) or take random actions (False)
'feedback'		: True			,# whether to feedback activation of classification neurons to hidden neurons
'balReward'		: False			,# whether reward should sum to the same value for stim. that are always rewarded and stim. that are rewarded for specific actions
'createOutput'	: False			,# whether to create plots, save data, etc. (set to False when using pypet)
'showPlots'		: False			,# whether to display plots
'show_W_act'	: True			,# whether to display W_act weights on the weight plots
'sort' 			: False			,# whether to sort weights by their class when displaying
'target'		: 4 			,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 992#np.random.randint(1000) 				# seed of the random number generator
}

classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
rActions 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')

# classes 	= np.array([ 3,  4 , 5 , 7 , 8 , 9 ], dtype=int)
# rActions 	= np.array(['0','B','0','0','0','0'], dtype='|S1')
# rActions 	= np.array(['a','B','c','d','e','f'], dtype='|S1')
# rActions 	= np.array(['a','b','c','d','e','f'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['a','c','a'], dtype='|S1')

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
env = pypet.Environment(trajectory = 'dMid',
						comment = 'testing with pypet...',
						log_stdout=False,
						add_time = False,
						multiproc = True,
						ncores = 8,
						filename=filename,
						overwrite_file=False)

traj = env.v_trajectory
add_parameters(traj, kwargs)
add_exploration(traj, kwargs['runName'])

#run the simuation
env.f_run(pypet_RLnetwork)
env.f_disable_logging() #disable logging and close all log-files






