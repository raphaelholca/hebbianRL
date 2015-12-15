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
from sknn.mlp import Regressor, Layer

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
'nEpiCrit'		: 0 				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)
'nEpiDopa'		: 2					,# number of 'adult' episodes in each run (episodes when reward is not required for learning)
't_hid'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for hidden layer 
't_act'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for action layer 
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 'noProba_2d_pypet'		,# name of the folder where to save results
'dataset'		: 'train'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
'nHidNeurons'	: 16				,# number of hidden neurons
'lim_weights'	: True 				,# whether to artificially limit the value of weights. Used during parameter exploration
'lr'			: 0.01 				,# learning rate during 'critica period' (pre-training, nEpiCrit)
'e_greedy'		: False 			,# whether to use an epsilon-greedy approach to noise injection
'epsilon'		: 1.0 				,# probability of taking an exploratory decisions, range: [0,1]
'noise_std'		: 0.2 				,# parameter of the standard deviation of the normal distribution from which noise is drawn						digit: 4.0 	; gabor: 0.2 (?)
'proba_predict'	: False				,# whether the reward prediction is probabilistic (True) or deterministic/binary (False)
'exploration' 	: False				,# whether to take take explorative decisions (True) or not (False)
'pdf_method' 	: 'fit'				,# method used to approximate the pdf; valid: 'fit', 'subsample', 'full'
'aHigh' 		: 0.0 				,# learning rate increase for relevance signal (high ACh) outside of critical period
'aPairing'		: 1.0 				,# strength of ACh signal for pairing protocol

'dHigh' 		: 2.5 				,# learning rate increase for unexpected reward																	digit: 4.5	; gabor: 2.0
'dMid' 			: 6.00 				,# learning rate increase for correct reward prediction															digit: 0.02	; gabor: ---
'dNeut' 		: 6.				,# learning rate increase for correct no reward prediction														digit: -0.1	; gabor: ---
'dLow' 			: -1.5				,# learning rate increase for incorrect reward prediction														digit: -2.0	; gabor: 0.0

'a_0'			: 3.2				,# parameters for the RPE function
'a_1'			: -2.7				,

'nBatch' 		: 20 				,# mini-batch size
'protocol'		: 'digit'			,# training protocol. Possible values: 'digit' (MNIST classification), 'gabor' (orientation discrimination)
'target_ori' 	: 85. 				,# target orientation around which to discriminate clock-wise vs. counter clock-wise
'excentricity' 	: 3. 				,# degree range within wich to test the network (on each side of target orientation)
'noise_crit'	: 0. 				,# noise injected in the gabor filter for the pre-training (critical period)
'noise_train'	: 0. 				,# noise injected in the gabor filter for the training
'noise_test'	: 0.2 				,# noise injected in the gabor filter for the testing
'im_size'		: 28 				,# side of the gabor filter image (total pixels = im_size * im_size)
'classifier'	: 'bayesian'		,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass', 'bayesian'
'param_xplr'	: 'pypet' 			,# method for parameter exploration; valid values are: 'None', 'pypet', 'neural_net', 'basinhopping', 'gridsearch', 'minimize'
'temp_xplr'		: 1e-3				,# temperature for exploration in neural network-based parameter exploration
'pre_train'		: 'digit_479_16'	,# initialize weights with pre-trained weights saved to file; use '' or 'None' for random initialization
'test_each_epi'	: False 			,# whether to test the network's performance at each episode
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'save_data'		: False				,# whether to save data to disk
'verbose'		: False				,# whether to create text output
'show_W_act'	: False				,# whether to display W_act weights on the weight plots
'sort' 			: None				,# sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'
'target'		: None				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 995, #np.random.randint(1000), 	# seed of the random number generator

'comment'		: ''

}

""" parameters of the RPE function """
kwargs['RPE_function'] = 'tanh' 		# RPE value; valid: 'neural' (function approx.), 'discrete', or callable function, e.g.: ex.polynomial, ex.tanh
# RPE_function_params = [3.11, 10000., 0.05, 0.148]				# parameters of the RPE function, if RPE_function is a callable function
RPE_function_params = [3.11, 0.148]				# parameters of the RPE function, if RPE_function is a callable function

""" parameters for exploration """
explore_dict = {
# 'dHigh'			:	np.arange(0., 6.1, 1.5).tolist(),
# 'dMid'			:	np.round(np.arange(-0.4, 0.41, 0.2),1).tolist(), #np.arange(-0.004, 0.0041, 0.002).tolist(),
# 'dNeut'			:	np.round(np.arange(-0.4, 0.01, 0.1),1).tolist(), #np.arange(-0.004, 0.0041, 0.002).tolist(),
# 'dLow'			:	np.arange(-1.5, 0.51, 0.5).tolist(),
# 'noise_std'		:	[0.005, 0.01, 0.05]

'a_0'			:	np.round(np.arange(2., 4.1, 0.2),1).tolist(),
'a_1'			:	np.round(np.arange(-4., -1.9, 0.2),1).tolist(),

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

if kwargs['param_xplr'] == 'None':
	if kwargs['save_data']: kwargs['runName'] = ex.checkdir(kwargs, OW_bool=True) #create saving directory
	allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(	RPE_function_params, 
																						images, labels, orientations, 
																						images_test, labels_test, orientations_test, 
																						images_task, labels_task, orientations_task, 
																						None, kwargs, **kwargs)


elif kwargs['param_xplr'] == 'basinhopping' or kwargs['param_xplr'] == 'minimize' or kwargs['param_xplr'] == 'gridsearch':
	print "optimizing function parameters with: \'" + kwargs['param_xplr'] + "\'\n"
	print kwargs['runName'] + '\n'
	kwargs['runName'] = ex.checkdir(kwargs, OW_bool=True)

	func = rl.RLnetwork
	# RPE_function_params_0 = [0.1, 1.0, -1., 4.] ## polynomial init # <-- the number of parameters sets the order of the polynomial function to use
	# RPE_function_params_0 = [2., 10., 0.05, 0.5] ## tanh init
	RPE_function_params_0 = [3.50012704, -2.76669199] ## tanh_2d init

	args_tuple = (images, labels, orientations, 
					images_test, labels_test, orientations_test, 
					images_task, labels_task, orientations_task, 
					None, kwargs)

	for k in ['classes', 'rActions', 'nRun', 'nEpiCrit', 'nEpiDopa', 't_hid', 't_act', 'A', 'runName', 'dataset', 'nHidNeurons', 'lim_weights', 'lr', 'e_greedy', 'epsilon', 'noise_std', 'proba_predict', 'exploration', 'RPE_function', 'pdf_method', 'aHigh', 'aPairing', 'dHigh', 'dMid', 'dNeut', 'dLow', 'a_0', 'a_1', 'nBatch', 'protocol', 'target_ori', 'excentricity', 'noise_crit', 'noise_train', 'noise_test', 'im_size', 'classifier', 'param_xplr', 'temp_xplr', 'pre_train', 'test_each_epi', 'SVM', 'save_data', 'verbose', 'show_W_act', 'sort', 'target', 'seed', 'comment']:
		args_tuple += (kwargs[k],)

	if kwargs['param_xplr'] == 'basinhopping':
		bounds_max = [10.,5.]
		bounds_min = [1.,-2.] 
		bounds_accept_test = ex.basinhopping_bounds(xmax=bounds_max, xmin=bounds_min)
		optim_results = basinhopping( 	func, 
										RPE_function_params_0,
										# T=0.1, 
										stepsize=1.5,
										accept_test=bounds_accept_test,
										niter_success=15,
										disp=True,
										callback=ex.bh_callback,
										minimizer_kwargs={												# local minimizer options
											'args':args_tuple,
											'method':'Nelder-Mead',
											'options':{	'disp':True,
														'xtol': 0.05, 
														'ftol': 0.05,
													},	
											}
										)
	
	elif kwargs['param_xplr'] == 'gridsearch':
		optim_results = brute(	func,
								args=args_tuple,
								ranges=[(0., 4.), (-4.1, -0.1)],
								Ns=7,
								full_output=True,
								disp=True,
								# finish=None
								)

	elif kwargs['param_xplr'] == 'minimize':
		optim_results = minimize( 	func, 
									RPE_function_params_0,
									args_tuple,
									method='Nelder-Mead',			#<- Melder-Neav: doesn't rely on gradient evaluation; good for noisy functions
									options={	'disp':True,
												'xtol': 0.01, 
												'ftol': 0.01,
											},	
									)

	pickle.dump(optim_results, open('output/' + kwargs['runName'] + '/optim_results', 'w'))
	ex.save_data(None, None, None, None, kwargs, save_weights=False)


elif kwargs['param_xplr'] == 'neural_net':
	nn_regressor = Regressor(		#[prediction_error, tried_DA_values]
	    layers=[
	        Layer("Rectifier", 	units=5),
	        Layer("Linear", 	units=1)],
	    learning_rate=1e-8,
	    n_iter=1)

	n_iter = 15 # number of training iterations
	n_sample = 1000 # number of sample input to save 
	sample_input_save = np.zeros((n_iter*n_sample, 3)) #[prediction_error, best_DA, performance]
	perf_save = np.array([])
	all_DA_save = np.zeros((121,105,n_iter))

	kwargs['runName'] = ex.checkdir(kwargs, OW_bool=True)
	pickle.dump(nn_regressor, open('output/' + kwargs['runName'] + '/nn_regressor' + '/nn_epi_0', 'w'))

	for i_iter in range(n_iter):
		print "training hebbian network..."
		try:
			allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba, nn_input = rl.RLnetwork(	None, images, labels, orientations, 
																							images_test, labels_test, orientations_test, 
																							images_task, labels_task, orientations_task, 
																							nn_regressor, kwargs, **kwargs)
		except ValueError: 
			allPerf = np.zeros(1)

		perf_save = np.append(perf_save, allPerf[0])
		all_DA = np.zeros(shape=(121,105))
		for rpe_idx, rpe in enumerate(np.arange(-1,1.1,0.02)):
			X = np.ones((121, 2))*rpe
			X[:,1]=np.arange(-6,6.1,0.1)
			all_DA[:, rpe_idx] = nn_regressor.predict(X)[:,0]
		pl.regressor_prediction(all_DA, i_iter, kwargs, perf=allPerf[0], nn_input=nn_input)
		all_DA_save[:,:,i_iter] = all_DA

		
		sample_idx = np.random.choice(np.size(nn_input,0),size=n_sample)
		sample_input_save[i_iter*n_sample: (i_iter+1)*n_sample, :2] = nn_input[sample_idx,:]
		sample_input_save[i_iter*n_sample: (i_iter+1)*n_sample, 2] = np.ones(n_sample)*allPerf
		
		print "training regressor neural net..."
		nn_regressor.fit(nn_input, np.ones(np.size(nn_input,0))*allPerf)
		
		if i_iter+1==n_iter:
			for rpe_idx, rpe in enumerate(np.arange(-1,1.1,0.02)):
				X = np.ones((121, 2))*rpe
				X[:,1]=np.arange(-6,6.1,0.1)
				all_DA[:, rpe_idx] = nn_regressor.predict(X)[:,0]
			pl.regressor_prediction(all_DA, i_iter+1, kwargs)

		print 'run ' + str(i_iter+1) + '/' + str(n_iter) + '; perf: ' + str(np.round(allPerf[0],3)*100) + '%' + '   ; all_DA: ' + str(np.round(X[np.argmax(all_DA[:,::10],0), 1],1)) + ' \n'

		pickle.dump(nn_regressor, open('output/' + kwargs['runName'] + '/nn_regressor' + '/nn_epi_' + str(i_iter+1), 'w'))

	#save results to file
	pickle.dump(sample_input_save, open('output/' + kwargs['runName'] + '/sample_input', 'w'))
	pickle.dump(all_DA_save, open('output/' + kwargs['runName'] + '/best_DA_epi', 'w'))
	pickle.dump(perf_save, open('output/' + kwargs['runName'] + '/perf_epi', 'w'))
	pl.perf_progress({'000': perf_save}, kwargs)
	ex.save_data(None, None, None, None, kwargs, save_weights=False)

elif kwargs['param_xplr'] == 'pypet':
	""" launch simulation with pypet for parameter exploration """
	env = pypet.Environment(trajectory = 'xplr',
							comment = 'testing with pypet...',
							log_stdout=False,
							add_time = False,
							multiproc = True,
							ncores = 10,
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

print '\n\nstart time: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(tic))
print 'end time: ' + time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime(time.time()))
print 'run time: ' + time.strftime("%H:%M:%S", time.gmtime(time.time()-tic))




