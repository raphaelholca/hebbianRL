import matplotlib
matplotlib.use('Agg') #to avoid sending plots to screen when working on the servers

import numpy as np
import pickle
import os
import struct
import pypet
import support.hebbianRL as rl
import support.external as ex
import support.mnist as mnist
from sknn.mlp import Regressor, Layer

rl = reload(rl)
ex = reload(ex)

def get_images():
	return images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task

def pypet_RLnetwork(traj):
	images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task = get_images()
	parameter_dict = traj.parameters.f_to_dict(short_names=True, fast_access=True)

	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(images, labels, orientations, 
																						images_test, labels_test, orientations_test, 
																						images_task, labels_task, orientations_task, 
																						None, parameter_dict, **parameter_dict)

	traj.f_add_result('RLnetwork.$', 
					perc_W_act=perc_correct_W_act, 
					perf=np.mean(allPerf), 
					comment='param exploration')

	return np.round(perc_correct_W_act,3), np.round(np.mean(allPerf),2)

""" parameters """
kwargs = {
'nRun' 			: 1					,# number of runs
'nEpiCrit'		: 0 				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
'nEpiDopa'		: 3					,# number of 'adult' episodes in each run (episodes when reward is not required for learning)
't_hid'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for hidden layer 
't_act'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for action layer 
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'test_none'			,# name of the folder where to save results
'dataset'		: 'train'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
'nHidNeurons'	: 16				,# number of hidden neurons
'lim_weights'	: False 			,# whether to artificially limit the value of weights. Used during parameter exploration
'lr'			: 0.01 				,# learning rate during 'critica period' (pre-training, nEpiCrit)
'e_greedy'		: False 			,# whether to use an epsilon-greedy approach to noise injection
'epsilon'		: 1.0 				,# probability of taking an exploratory decisions, range: [0,1]
'noise_std'		: 0.2 				,# parameter of the standard deviation of the normal distribution from which noise is drawn						digit: 4.0 	; gabor: 0.2 (?)
'proba_predict'	: True 			,# whether the reward prediction is probabilistic (True) or deterministic/binary (False)
'exploration' 	: True				,# whether to take take explorative decisions (True) or not (False)
'pdf_method' 	: 'fit'				,# method used to approximate the pdf; valid: 'fit', 'subsample', 'full'
'aHigh' 		: 0.0 				,# learning rate increase for relevance signal (high ACh) outside of critical period
'aPairing'		: 1.0 				,# strength of ACh signal for pairing protocol
'dHigh' 		: 3.0 				,# learning rate increase for unexpected reward																	digit: 4.5	; gabor: 2.0
'dMid' 			: 0.00 				,# learning rate increase for correct reward prediction															digit: 0.02	; gabor: ---
'dNeut' 		: -0.				,# learning rate increase for correct no reward prediction														digit: -0.1	; gabor: ---
'dLow' 			: -1.				,# learning rate increase for incorrect reward prediction														digit: -2.0	; gabor: 0.0
'nBatch' 		: 20 				,# mini-batch size
'protocol'		: 'digit'			,# training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination)
'target_ori' 	: 85. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
'excentricity' 	: 3. 				,# degree range within wich to test the network (on each side of target orientation)
'noise_crit'	: 0. 				,# noise injected in the gabor filter for the pre-training (critical period)
'noise_train'	: 0. 				,# noise injected in the gabor filter for the training
'noise_test'	: 0.2 				,# noise injected in the gabor filter for the testing
'im_size'		: 28 				,# side of the gabor filter image (total pixels = im_size * im_size)
'classifier'	: 'bayesian'		,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass', 'bayesian'
'param_xplr'	: 'None' 		,# method for parameter exploration; valid values are: 'None', 'pypet', 'neural_net'
'pre_train'		: 'digit_479_16'	,# initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization
'test_each_epi'	: True 			,# whether to test the network's performance at each episode
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'save_data'		: False				,# whether to save data to disk
'verbose'		: True				,# whether to create text output
'show_W_act'	: True				,# whether to display W_act weights on the weight plots
'sort' 			: None				,# sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'
'target'		: None 				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 995, #np.random.randint(1000), 	# seed of the random number generator

'a_0'			: 0.,
'a_1'			: 0.,
'a_2'			: 1.0,
'a_3'			: 8.,

'comment'		: ''

}

""" parameters for exploration """
explore_dict = {
'a_0'			:	np.arange(-0.1, 0.11, 0.05).tolist(),
'a_1'			:	np.arange(-2., 2.1, 1.0).tolist(), #np.arange(-0.004, 0.0041, 0.002).tolist(),
'a_2'			:	np.arange(-1., 1.1, 0.5).tolist(), #np.arange(-0.004, 0.0041, 0.002).tolist(),
'a_3'			:	np.arange(0., 8.1, 2.0).tolist(),
# 'dHigh'			:	np.arange(0., 6.1, 1.5).tolist(),
# 'dMid'			:	np.round(np.arange(-0.4, 0.41, 0.2),1).tolist(), #np.arange(-0.004, 0.0041, 0.002).tolist(),
# 'dNeut'			:	np.round(np.arange(-0.4, 0.01, 0.1),1).tolist(), #np.arange(-0.004, 0.0041, 0.002).tolist(),
# 'dLow'			:	np.arange(-1.5, 0.51, 0.5).tolist(),
# 'noise_std'		:	[0.005, 0.01, 0.05]
}

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
	ex.set_global(lActions, kwargs['rActions'], kwargs['classes'])

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

if kwargs['param_xplr'] == 'None':
	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(	images, labels, orientations, 
																						images_test, labels_test, orientations_test, 
																						images_task, labels_task, orientations_task, 
																						None, kwargs, **kwargs)

elif kwargs['param_xplr'] == 'neural_net':
	nn_regressor = Regressor(
	    layers=[
	        Layer("Rectifier", 	units=5),
	        Layer("Linear", 	units=1)],
	    learning_rate=0.02,
	    n_iter=1)

	n_iter = 200
	n_sample = 100
	sample_input = np.zeros((n_iter*n_sample, 3)) #[prediction_error, best_DA, performance]
	for i_iter in range(n_iter):
		print "training hebbian network..."
		allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(	images, labels, orientations, 
																							images_test, labels_test, orientations_test, 
																							images_task, labels_task, orientations_task, 
																							nn_regressor, kwargs, **kwargs)

		best_DA = np.array([])
		for i in np.arange(-1,1.1,0.2):
			X = np.ones((120, 2))*i
			X[:,0]=np.arange(-6,6,0.1)
			best_DA = np.append(best_DA, X[ np.argmax( nn_regressor.predict(X) ), 0 ] )
		
		print 'run ' + str(i_iter+1) + '/' + str(n_iter) + '; perf: ' + str(np.round(allPerf[0],3)*100) + '%' + '   ; best_DA: ' + str(np.round(best_DA,1))
		sample_idx = np.random.choice(np.size(nn_input,0),size=n_sample)
		sample_input[i_iter*n_sample: (i_iter+1)*n_sample, :2] = nn_input[sample_idx,:]
		sample_input[i_iter*n_sample: (i_iter+1)*n_sample, 2] = np.ones(n_sample)*allPerf
		print "training regressor neural net... \n"
		nn_regressor.fit(nn_input, np.ones(np.size(nn_input,0))*allPerf)
		# import pdb; pdb.set_trace()

	pickle.dump(sample_input, open('output/' + kwargs['runName'] + '/sample_input', 'w'))
	pickle.dump(nn_regressor, open('output/' + kwargs['runName'] + '/nn_regressor', 'wb'))

elif kwargs['param_xplr'] == 'pypet':
	""" launch simulation with pypet for parameter exploration """
	env = pypet.Environment(trajectory = 'xplr',
							comment = 'testing with pypet...',
							log_stdout=False,
							add_time = False,
							multiproc = True,
							ncores = 6,
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
	ex.save_data(None, None, None, None, kwargs_save, save_weights=False)

	#run the simuation
	env.f_run(pypet_RLnetwork)

	env.f_disable_logging() #disable logging and close all log-files




