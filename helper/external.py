""" 
Author: Raphael Holca-Lamarre
Date: 23/10/2014

Support functions for Hebbian neural network
"""

import numpy as np
import pickle
import os
import sys
import shutil
import numba
import grating as gr
import time
import datetime
import struct
from array import array

gr = reload(gr)

def load_images(protocol, A, verbose=True, digit_params={}, gabor_params={}, toy_data_params={}, load_test=True, normalize_im=True):
	""" 
	Load images training and testing images 

		Args:
			protocol (str): experimental protocol, maybe 'digit', 'gabor' or 'toy'
			A (float): normalization constant for the images
			verbose (bool, optional): whether to print comments to console
			digit_params (dict): parameters for loading the MNIST dataset. These are:
				classes (numpy array): classes of the MNIST dataset to load
				dataset_train (str): name of the dataset to load for training the network; maybe 'test' or 'train'
				dataset_path (str): path of the MNIST dataset
			gabor_params (dict): parameters for creating the gabor patches. These are:
				n_train (int): number of training images
				n_test (int): number of testing images
				target_ori (float): target orientation around which to discriminate clock-wise vs. counter clock-wise
				excentricity (float): degree range within wich to test the network (on each side of target orientation)
				noise_pixel (float): noise injected in the pixels of gabor filter
				rnd_phase (bool): whether to use random phase (True) or use set phase
				im_size (int): side of the gabor filter image (total pixels = im_size * im_size)
			load_test (bool, optional): whether to load test images (True) or not (False). Default: True
			normalize_im (bool, optional): whether to normalize images. Default: True

		returns:
			(2D numpy array): training images
			(numpy array): training labels
			(2D numpy array): test images 
			(numpy array): testing labels 
			(2D numpy array): training images for gabor task 
			(numpy array): training labels

	"""

	if protocol == 'digit':
		if digit_params=={}:
			print "*** no digit parameters provided, falling back on default ***"
			digit_params={	'dataset_train'	: 'train',
							'classes' 		: np.array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 ], dtype=int),
							'dataset_path' 	: '/Users/raphaelholca/Documents/data-sets/MNIST',
							}

		if verbose: print 'loading train images...'
		images, labels = read_images_from_mnist(classes=digit_params['classes'], dataset=digit_params['dataset_train'], path=digit_params['dataset_path'])
		images, labels = even_labels(images, labels, digit_params['classes'])
		if normalize_im: images = normalize(images, A)

		if load_test:
			if verbose: print 'loading test images...'
			dataset_test = 'test' if digit_params['dataset_train']=='train' else 'train'
			images_test, labels_test = read_images_from_mnist(classes=digit_params['classes'], dataset=dataset_test ,  path=digit_params['dataset_path'])
		
			images_test, labels_test = even_labels(images_test, labels_test, digit_params['classes'])
			if normalize_im: images_test = normalize(images_test, A)
		else:
			images_test = None
			labels_test = None

		images_task = None
		labels_task = None
		images_params = digit_params

		orientations = None
		orientations_test = None
		orientations_task = None

	elif protocol == 'gabor':
		if gabor_params=={}:
			print "*** no gabor parameters provided, falling back on default ***"
			gabor_params 	= {	'n_train' 			: 10000,
								'n_test' 			: 10000,
								'renew_trainset'	: False,
								'target_ori' 		: 165.,
								'excentricity' 		: 90.,#3.0,#1.5,
								'noise_pixel'		: 0.0,
								'rnd_phase' 		: False,
								'rnd_freq' 			: False,
								'im_size'			: 50
								}
		if verbose: print 'creating gabor training images...'
		gabor_params['target_ori'] %= 180.

		orientations = np.random.random(gabor_params['n_train'])*180 #orientations of gratings (in degrees)
		phase = np.random.random(gabor_params['n_train']) if gabor_params['rnd_phase'] else 0.25
		freq = np.random.random(gabor_params['n_train'])*0.1+5. if gabor_params['rnd_freq'] else 5.
		images, labels = generate_gabors(orientations, gabor_params['target_ori'], gabor_params['im_size'], phase=phase, freq=freq)

		if not gabor_params['renew_trainset']:
			orientations_task = np.random.random(gabor_params['n_train'])*gabor_params['excentricity']*2 + gabor_params['target_ori'] - gabor_params['excentricity'] 
			phase_task = np.random.random(gabor_params['n_train']) if gabor_params['rnd_phase'] else 0.25
			freq_task = np.random.random(gabor_params['n_train'])*5.+2. if gabor_params['rnd_freq'] else 5.
			images_task, labels_task = generate_gabors(orientations_task, gabor_params['target_ori'], gabor_params['im_size'], phase=phase_task, freq=freq_task)
		else:
			orientations_task, images_task, labels_task = None, None, None

		if load_test:
			orientations_test = np.random.random(gabor_params['n_test'])*gabor_params['excentricity']*2 + gabor_params['target_ori'] - gabor_params['excentricity']
			phase_test = np.random.random(gabor_params['n_test']) if gabor_params['rnd_phase'] else 0.25
			freq_test = np.random.random(gabor_params['n_test'])*5.+2. if gabor_params['rnd_freq'] else 5.
			images_test, labels_test = generate_gabors(orientations_test, gabor_params['target_ori'], gabor_params['im_size'], phase=phase_test, freq=freq_test)
		else:
			orientations_test, images_test, labels_test = None, None, None

		images_params = gabor_params

	elif protocol == 'toy_data':
		if verbose: print 'creating toy training images...'

		images, images_test, labels, labels_test = generate_toy_data(protocol, A, toy_data_params)
		
		images_task = None
		labels_task = None

		orientations = None
		orientations_test = None
		orientations_task = None

		images_params = toy_data_params

	return {'train':images, 'test':images_test, 'task':images_task}, {'train':labels, 'test':labels_test, 'task':labels_task}, {'train':orientations, 'test':orientations_test, 'task':orientations_task}, images_params

def read_images_from_mnist(classes, dataset = "train", path = '/Users/raphaelholca/Documents/data-sets/MNIST'):
    """ Import the MNIST data set """

    if not os.path.exists(path): #in case the code is running on the server
        path = '/mnt/antares_raid/home/raphaelholca/Documents/data-sets/MNIST'


    if dataset == 'train':
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == 'test':
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'test' or 'train'"

    flbl = open(fname_lbl, 'rb')
    #magic_nr, size = struct.unpack(">II", flbl.read(8))
    struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    #magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    size, rows, cols = struct.unpack(">IIII", fimg.read(16))[1:4]
    img = array("B", fimg.read())
    fimg.close()

    ind = [ k for k in xrange(size) if lbl[k] in classes ]
    images = np.zeros(shape=(len(ind), rows*cols))
    labels = np.zeros(shape=(len(ind)), dtype=int)
    for i in xrange(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i] = lbl[ind[i]]

    return images, labels

def even_labels(images, labels, classes):
	"""
	Even out images and labels distribution so that they are evenly distributed over the labels.

	Args:
		images (numpy array): images
		labels (numpy array): labels constant
		classes (numpy array): classes of the MNIST dataset used to train the network

	returns:
		numpy array: evened-out images
		numpy array: evened-out labels
	"""

	n_classes = len(classes)
	n_digits, bins = np.histogram(labels, bins=10, range=(0,9))
	m = np.min(n_digits[n_digits!=0])
	images_even = np.zeros((m*n_classes, np.size(images,1)))
	labels_even = np.zeros(m*n_classes, dtype=int)
	for i, c in enumerate(classes):
		images_even[i*m:(i+1)*m,:] = images[labels==c,:][0:m,:]
		labels_even[i*m:(i+1)*m] = labels[labels==c][0:m]
	images, labels = np.copy(images_even), np.copy(labels_even)
	
	return images, labels

def checkdir(name, protocol, overwrite=True):
	"""
	Checks if directory exits. If not, creates it. If yes, asks whether to overwrite. If user choose not to overwrite, execution is terminated

	Args:
		name (str): name of the network, used to save data to disk
		protocol (str): experimental protocol
		overwrite (bool, optional): whether to overwrite existing directory
	"""

	if os.path.exists('output/' + name):
		if overwrite: overwrite='yes'
		else: overwrite = raw_input('Folder \''+name+'\' already exists. Overwrite? (y/n/<new name>) ')
		if overwrite in ['n', 'no', 'not', ' ', '']:
			sys.exit('Folder exits - not overwritten')
		elif overwrite in ['y', 'yes']:
			shutil.rmtree('output/' + name)
		else:
			name = overwrite
			checkdir(name)
			return name
	os.makedirs('output/' + name)

	return name

def shuffle_datasets(images_dict, labels_dict):
	""" shuffle test & train datasets """

	#concatenate images and labels
	images_conca = np.concatenate((images_dict['train'], images_dict['test']), axis=0)
	labels_conca = np.concatenate((labels_dict['train'], labels_dict['test']), axis=0)

	#shuffle images and labels
	shuffle_idx = np.arange(len(labels_conca))
	np.random.shuffle(shuffle_idx)
	images_conca = images_conca[shuffle_idx,:]
	labels_conca = labels_conca[shuffle_idx]

	#split concatenated images and labels into train and test datasets
	split_idx = len(labels_dict['train'])
	images_train, images_test = images_conca[:split_idx,:], images_conca[split_idx:,:]
	labels_train, labels_test = labels_conca[:split_idx], labels_conca[split_idx:]

	return images_train, images_test, labels_train, labels_test

def shuffle(arrays):
	"""
	Shuffles the passed vectors according to the same random order

	Args:
		arrays (list): list of arrays to shuffle

	returns:
		list: list of shuffled arrays
	"""

	rnd_idx = np.arange(len(arrays[0]))
	np.random.shuffle(rnd_idx)
	shuffled_arrays = []
	for a in arrays:
		if len(np.shape(a))==1:
			shuffled_arrays.append(a[rnd_idx])
		elif len(np.shape(a))==2:
			shuffled_arrays.append(a[rnd_idx,:])

	return shuffled_arrays

def generate_gabors(orientations, target_ori, im_size, noise_pixel=0., phase=0.25, freq=5.):
	"""
	Calling function to generate gabor filters

	Args:
		orientations (numpy array): 1-D array of orientations of gratings (in degrees) (one grating is created for each orientation provided)
		target_ori (float): target orientation around which to discriminate clock-wise vs. counter clock-wise
		im_size (int): side of the gabor filter image (total pixels = im_size * im_size)
		noise_pixel (int,optional): noise level to add to the pixels of Gabor patch; represents the standard deviation of the Gaussian distribution from which noise is drawn; range: (0, inf
		phase (float, list or numpy array, optional): phase of the filter; range: [0, 1)

	returns:
		numpy array: gabor filters of size: (len(orientations), im_size*im_size)
		numpy array: labels (clock-wise / counter clock-wise) of each gabor filter
	"""

	images = gr.gabor(size=im_size, freq=freq, theta=orientations, sigma=0.2, phase=phase, noise_pixel=noise_pixel)

	if type(orientations) is not np.ndarray and type(orientations) is not list:
		orientations = np.array([orientations])		

	orientations = relative_orientations(orientations, target_ori)

	labels = np.zeros(len(orientations), dtype=int)
	labels[orientations<=0] = 0
	labels[orientations>0] = 1

	return images, labels

def relative_orientations(orientations, target_ori):
	""" converts absolute orienations to orientations relative to the target orientation """

	orientations -= target_ori
	orientations[orientations>90.] -= 180.
	orientations[orientations<-90.] += 180.

	return orientations

def save_net(net):
	""" Print parameters of Network object to human-readable file and save Network to disk """
		
	""" save network to file """
	n_file = open(os.path.join('output', net.name, 'Network'), 'w')
	pickle.dump(net, n_file)
	n_file.close()

	save_file = os.path.join('output', net.name, net.name + '_params.txt')
	if hasattr(net, 'runtime'):
		print_params(vars(net), save_file, runtime=net.runtime)
	else:
		print_params(vars(net), save_file)

def print_params(param_dict, save_file, runtime=None):
	""" print parameters """
	tab_length = 25

	params_to_print = ['dHigh', 'dMid', 'dNeut', 'dLow', 'dopa_values', 'dopa_out_same', 'train_out_dopa', 'dopa_values_out', 'dHigh_out', 'dMid_out', 'dNeut_out', 'dLow_out', 'ach_values', 'ach_1', 'ach_2', 'ach_3', 'ach_4', 'ach_func', 'ach_avg', 'protocol', 'name', 'dopa_release', 'ach_release', 'n_runs', 'n_epi_crit', 'n_epi_fine', 'n_epi_perc', 'n_epi_post', 't_hid', 't_out', 'A','lr_hid', 'lr_out', 'batch_size', 'block_feedback', 'shuffle_datasets', 'n_hid_neurons', 'weight_init', 'init_file', 'lim_weights', 'log_weights', 'epsilon_xplr', 'noise_xplr_hid', 'noise_xplr_out', 'exploration', 'compare_output', 'noise_activ', 'pdf_method', 'classifier', 'test_each_epi', 'early_stop', 'verbose', 'seed', 'images_params']

	
	param_file = open(save_file, 'w')
	time_str = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
	time_line = ('created on\t: %s\n' % time_str).expandtabs(tab_length)
	param_file.write(time_line)

	if runtime is not None:
		runtime_str = str(datetime.timedelta(seconds=runtime))
		runtime_line = ('runtime\t: %s\n\n' % runtime_str).expandtabs(tab_length)
		param_file.write(runtime_line)
	else:
		param_file.write('\n')

	for p in params_to_print:
		if p in param_dict.keys():
			if not isinstance(param_dict[p], dict):
				line = ('%s \t: %s\n' %( p, str(param_dict[p]) )).expandtabs(tab_length)
				param_file.write(line)
			else:
				for ik, k in enumerate(sorted(param_dict[p].keys())):
					if ik==0:
						line = ('%s \t: %s: %s\n' % (p, k, str(param_dict[p][k]))).expandtabs(tab_length)
					else:
						line = ('\t  %s: %s\n' % (k, str(param_dict[p][k]))).expandtabs(tab_length)
					param_file.write(line)
	param_file.close()

def normalize(images, A):
	"""
	Normalize each image to the sum of its pixel value (equivalent to feedforward inhibition)

	Args:

		images (numpy array): image to normalize
		A (int): normalization constant

	returns:
		numpy array: normalized images
	"""

	return (A-images.shape[1])*images/np.sum(images,1)[:,np.newaxis] + 1.

def softmax(activ, implementation='numba', t=1.):
	"""
	Softmax function (equivalent to lateral inhibition, or winner-take-all)

	Args:
		activ (numpy array): activation of neurons to be fed to the function; should be (training examples x neurons)
		vectorial (str, optional): which implementation to use ('vectorial', 'iterative', 'numba')
		t (float): temperature parameter; determines the sharpness of the softmax, or the strength of the competition

	returns:
		numpy array: the activation fed through the softmax function
	"""

	if len(np.shape(activ))==1:
		activ = np.reshape(activ, (1,-1))

	#vectorial
	if implementation=='vectorial':
		activ_norm = np.copy(activ - np.max(activ,1)[:,np.newaxis])
		activ_SM = np.exp((activ_norm)/t) / np.sum(np.exp((activ_norm)/t), 1)[:,np.newaxis]

	#iterative
	elif implementation=='iterative':
		activ_SM = np.zeros_like(activ)
		for i in range(np.size(activ,0)):
			scale = 0
			I = np.copy(activ[i,:])
			if (I[np.argmax(I)] > 700):
			    scale  = I[np.argmax(I)] - 700
			if (I[np.argmin(I)] < -740 + scale):
			    I[np.argmin(I)] = -740 + scale
			activ_SM[i,:] = np.exp((I-scale)/t) / np.sum(np.exp((I-scale)/t))

	#iterative with numba
	elif implementation=='numba':
		activ_SM = np.zeros_like(activ)
		activ_SM = softmax_numba(activ, activ_SM, t=t)
	
	return activ_SM

@numba.njit
def softmax_numba(activ, activ_SM, t=1.):
	"""
	Numba implementation of the softmax function
	"""

	for ex in range(activ.shape[0]):
		sum_tot = 0.
		ex_max=np.max(activ[ex,:])
		for i in range(activ.shape[1]): #compute exponential
			activ_SM[ex,i] = np.exp((activ[ex,i]-ex_max)/t)
			sum_tot += activ_SM[ex,i]
		for i in range(activ.shape[1]): #divide by sum of exponential
			activ_SM[ex,i] /= sum_tot

	return activ_SM

def propagate_layerwise(X, W, SM=True, t=1., log_weights=True):
	"""
	One propagation step

	Args:
		X (numpy array): input vector to the neurons of layer 1
		W (numpy matrix): weight matrix; shape: (input neurons x hidden neurons)
		SM (bool, optional): whether to pass the activation throught the Softmax function. Default: True
		t (float, optional): temperature parameter for the softmax function (only passed to the function, not used here). Default: 1.0
		log_weights (bool, optional): whether to take the logarithm of the weight. Default: True

	returns:
		numpy array: the activation of the hidden neurons
	"""

	if log_weights:
		activ = np.dot(X, np.log(W))
	else:
		activ = np.dot(X, W)
	if SM: activ = softmax(activ, t=t)
	return activ

@numba.njit
def disinhibition(post_neurons, lr, dopa, ach, post_neurons_lr):
	"""
	support function for numba implementation of learning_step() 
	"""
	for b in range(post_neurons.shape[0]):
		for pn in range(post_neurons.shape[1]):
			post_neurons_lr[b, pn] = post_neurons[b, pn] * lr * dopa[b] * ach[b]

	return post_neurons_lr

@numba.njit
def regularization(dot, post_neurons, W, sum_ar):
	"""
	support function for numba implementation of learning_step() 
	"""
	for j in range(post_neurons.shape[1]):
		for i in range(post_neurons.shape[0]):
			sum_ar[j] += post_neurons[i,j]
	
	for i in range(dot.shape[0]):
		for j in range(dot.shape[1]):
			dot[i,j] -= W[i,j] * sum_ar[j]

	return dot

def reward_delivery(labels, actions):
	"""
	Computes the reward based on the action taken and the label of the current input

	Args:
		labels (numpy array): image labels
		actions (numpy array): action taken

	returns:
		numpy array: 1 (reward) for correct label-action pair, otherwise 0
	"""

	if actions is not None:
		reward = np.zeros(len(labels), dtype=int)
		reward[labels==actions] = 1
	else:
		reward = None

	return reward

def reward_prediction(explorative, compare_output, best_action=None, action_taken=None):
	"""
	Computes reward prediction based on the best (greedy) action and the action taken

	Args:
		explorative (numpy array): contains 1s for trials where noise is injected (exploratory) and 0s otherwise
		compare_output (bool): whether to compare the value of greedy and taken action to determine if the trial is exploratory
		best_action (numpy array, optional): best (greedy) action for each trial of a batch. Default: None
		action_taken (numpy array, optional): action taken for each trial of a batch. Default: None

	returns:
		numpy array: reward prediction, either deterministic or expected value (depending on proba_predict)
	"""

	if compare_output:
		return best_action == action_taken
	else:
		return ~explorative

def compute_dopa(predicted_reward, reward, dopa_values):
	"""
	Computes the dopa signal based on the actual and predicted rewards

	Args:
		predicted_reward (numpy array, bool): predicted reward (True, False)
		reward (numpy array, int): reward received (0, 1)
		dopa_values (dict): dopa value for unpredicted reward, must includes keys: 'dHigh', 'dMid', 'dNeut' and 'dLow'

	returns:
		numpy array: array of dopamine release value
	"""

	dopa = np.zeros(len(reward))

	dopa[np.logical_and(predicted_reward==0, reward==1)] = dopa_values['dHigh']		#unpredicted reward
	dopa[np.logical_and(predicted_reward==1, reward==1)] = dopa_values['dMid']		#correct reward prediction
	dopa[np.logical_and(predicted_reward==0, reward==0)] = dopa_values['dNeut']		#correct no reward prediction
	dopa[np.logical_and(predicted_reward==1, reward==0)] = dopa_values['dLow']		#incorrect reward prediction

	return dopa

def no_difference(best, alte, diff_tol=0.005, confidence='0.95'):
	""" 
	test that there are no statistical difference in the performance of models with different VTA values 

	Args:
		best (numpy array): performance measure for best model
		alte (numpy array): performance measure for alternate model
		diff_tol (float, optional): difference that we are willing to tolerante in the difference of models
		confidence (str, optional): confidence level; allowed values: '0.99', '0.95', and '0.90'

	returns:
		(bool): whether or not there is a difference in the performance; True: no statistical difference (reject null hypothesis); False: statistical difference (cannot reject null hypothesis)
	"""

	conf_to_z = {'0.99':1.96, '0.95':1.645, '0.90':1.28}
	z_value = conf_to_z[confidence]

	best_num = float(len(best))
	best_avg = np.mean(best)
	best_std = np.std(best)
	
	alte_num = float(len(alte))
	alte_avg = np.mean(alte)
	alte_std = np.std(alte)

	diff_std = np.sqrt( ( (best_num-1)*(best_std**2) + (alte_num-1)*(alte_std**2) / (best_num+alte_num-2) ) * ( 1/best_num +  1/alte_num) )
	# diff_std = np.sqrt( best_avg*(1-best_avg)/best_num + alte_avg*(1-alte_avg)/alte_num )

	t = np.abs( (best_avg-alte_avg-diff_tol) / diff_std )

	return t > z_value 

def exploration(epsilon_xplr, batch_size):
	""" 
	Returns an array determining whether a trial will be exploratory or not. The values in the array are 1 for exploratory and 0 for explotative trials. The probability of having an exploratory (1) trial is determined by epsilon_xplr. The size of the array is determined by the batch size. 
	"""
	explorative_trials = np.zeros(batch_size, dtype=bool)
	explorative_proba = np.random.random(size=batch_size)
	explorative_trials[explorative_proba < epsilon_xplr] = 1

	return explorative_trials

def generate_toy_data(protocol, A, toy_data_params):
	""" generate 2D and 3D toy data to test model """

	if toy_data_params['data_distrib']=='multimodal':
		return multimodal_toy_data(protocol, A, toy_data_params)

	if toy_data_params['dimension'] == '2D':
		if toy_data_params['data_distrib']=='uniform':
			images = np.abs(np.random.random((toy_data_params['n_points'], 2)))
			images_test = np.abs(np.random.random((toy_data_params['n_points'], 2)))
		elif toy_data_params['data_distrib']=='normal':
			images = np.abs(np.random.normal(loc=0.5, scale=0.15, size=(toy_data_params['n_points'], 2)))
			images_test = np.abs(np.random.normal(loc=0.5, scale=0.15, size=(toy_data_params['n_points'], 2)))
	if toy_data_params['dimension'] == '3D':
		if toy_data_params['data_distrib']=='uniform':
			images_tmp = np.random.random((toy_data_params['n_points']*2, 2))
			images_tmp = images_tmp[images_tmp[:,0]+images_tmp[:,1]<=1.0]
			images = np.zeros((len(images_tmp), 3))
			images[:,:2] = images_tmp
			images[:,2] = 1.0 - np.sum(images,1)

			images_tmp = np.random.random((toy_data_params['n_points']*2, 2))
			images_tmp = images_tmp[images_tmp[:,0]+images_tmp[:,1]<=1.0]
			images_test = np.zeros((len(images_tmp), 3))
			images_test[:,:2] = images_tmp
			images_test[:,2] = 1.0 - np.sum(images_test,1)

		elif toy_data_params['data_distrib']=='normal':
			raise NotImplementedError ("normal distribution not implemented for 3D data")
	
	images = normalize(images, A)
	images_test = normalize(images_test, A)

	labels = toy_labeling(images, toy_data_params, A)
	labels_test = toy_labeling(images_test, toy_data_params, A)

	images, labels = even_labels(images, labels, np.unique(labels))
	images_test, labels_test = even_labels(images_test, labels_test, np.unique(labels))

	return images, images_test, labels, labels_test

def toy_labeling(images, toy_data_params, A):
	""" labels toy data """

	labels = np.zeros(np.size(images,0), dtype=int)

	if toy_data_params['dimension'] == '2D':
		if toy_data_params['separability'] == '1D':
			labels[images[:,0] >= (A-np.size(images,1)) * 2./3. + 1.] = 1
		elif toy_data_params['separability'] == '2D':
			labels[images[:,0] >= images[:,1]] = 1
		elif toy_data_params['separability'] == 'non_linear':
			labels[images[:,0] >= (A-np.size(images,1)) * 2./3. + 1. ] = 1
			labels[images[:,0] <  (A-np.size(images,1)) * 1./3. + 1. ] = 1
			# labels[images[:,0] >= 0.5*np.cos(images[:,1]/0.2)+(images[:,1]**2)/1.5+0.5 ] = 1
		else:
			raise ValueError('invalid separability method')

	if toy_data_params['dimension'] == '3D':
		if toy_data_params['separability'] == '1D':
			labels[images[:,0] >= (A-np.size(images,1)) * 1./2. + 1.] = 1
		elif toy_data_params['separability'] == '2D':
			labels[images[:,0] - images[:,1] > 0] = 1
		# elif toy_data_params['separability'] == 'non_linear':
		# 	labels[images[:,0] >= (A-np.size(images,1)) * 2./3. + 1. ] = 1
		# 	labels[images[:,0] <  (A-np.size(images,1)) * 1./3. + 1. ] = 1
		# 	# labels[images[:,0] >= 0.5*np.cos(images[:,1]/0.2)+(images[:,1]**2)/1.5+0.5 ] = 1
		else:
			raise ValueError('invalid separability method')

	return labels

def multimodal_toy_data(protocol, A, toy_data_params):
	""" generate multimodal toy data """

	if toy_data_params['dimension'] == '2D':
		b1 = np.random.normal(loc=0.6, scale=0.01, size=(toy_data_params['n_points']/2, 1))
		b2 = np.random.normal(loc=0.3, scale=0.10, size=(toy_data_params['n_points']/2, 1))
		b1 = np.concatenate((b1,1.-b1), axis=1)
		b2 = np.concatenate((b2,1.-b2), axis=1)
		images = np.clip(np.concatenate((b1, b2), axis=0), 0, 1)
		labels = np.zeros(2*(toy_data_params['n_points']/2), dtype=int)
		labels[:toy_data_params['n_points']/2]=1

		b1 = np.random.normal(loc=0.6, scale=0.01, size=(toy_data_params['n_points']/2, 1))
		b2 = np.random.normal(loc=0.2, scale=0.10, size=(toy_data_params['n_points']/2, 1))
		b1 = np.concatenate((b1,1.-b1), axis=1)
		b2 = np.concatenate((b2,1.-b2), axis=1)
		images_test = np.clip(np.concatenate((b1, b2), axis=0), 0 ,1)
		labels_test = np.zeros(2*(toy_data_params['n_points']/2), dtype=int)
		labels_test[:toy_data_params['n_points']/2]=1
	elif toy_data_params['dimension'] == '3D':
		raise NotImplementedError ("multimodal distribution not implemented for 3D data")


	images = normalize(images, A)
	images_test = normalize(images_test, A)

	images, labels = even_labels(images, labels, np.unique(labels))
	images_test, labels_test = even_labels(images_test, labels_test, np.unique(labels))

	return images, images_test, labels, labels_test

def set_labels2idx(classes):
	""" creates a numpy array to convert labels to indexes """
	labels2idx = np.zeros(10, dtype=int)
	for ic, c in enumerate(classes):
		labels2idx[c] = ic
	return labels2idx

def ach_linear(rel_perf, ach_1, ach_2, ach_3=None, ach_4=None):
	""" linear relation between relative perfomance and ACh release """
	#exploration range: ach_1: [-100.0, -50.0, -20.0, -10.0, 0.0, 10, 20.0, 50.0]
	#					ach_2: [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
	#results:			ach_1 = XX
	#					ach_2 = XX

	return ach_1 * (rel_perf - 1.0) + ach_2 + 1.0

def ach_exponential(rel_perf, ach_1, ach_2, ach_3=None, ach_4=None):
	""" exponential relation between relative perfomance and ACh release """
	#exploration range: ach_1: [-8.0, -5.0, -2.0, 0.0, 2.0, 5.0, 8.0, 10.0, 12.0, 15.0],
	#					ach_2: [-0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.0],
	#results:			ach_1 = XX
	#					ach_2 = XX

	return np.exp(ach_1 * (-rel_perf + 1.0)) + ach_2

def ach_polynomial(rel_perf, ach_1, ach_2, ach_3, ach_4):
	""" polynomial relation between relative perfomance and ACh release """
	#exploration range: ach_1: [-0.25, 0.0, 0.5]
	#					ach_2: [-20.0, -10.0, 0.0, 10.0, 20.0]
	#					ach_3: [-1000.0, -100.0, 0.0, 100.0, 1000.0]
	#					ach_4: [-1000.0, -500.0, -100.0, 0.0, 1000.0]
	#results:			ach_1 = XX
	#					ach_2 = XX
	#					ach_3 = XX
	#					ach_4 = XX

	return 1.0+ach_1 + ach_2*(rel_perf-1.0) + ach_3*(rel_perf-1.0)**2 + ach_4*(rel_perf-1.0)**3

def ach_sigmoidal(rel_perf, ach_1, ach_2=None, ach_3=None, ach_4=None):
	""" sigmoidal relation between relative perfomance and ACh release """
	
	# return 2.0/(1.0+np.exp(ach_1*(rel_perf-1.0)))
	return ach_2*(2.0/(1.0+np.exp(ach_1*(rel_perf-1.0))))

def ach_handmade(rel_perf, ach_1=None, ach_2=None, ach_3=None, ach_4=None):
	""" handmade relation between relative perfomance and ACh release """

	# return ((((rel_perf-1)*-1)+1)**15)*3
	return 3*(np.exp(ach_1 * (-rel_perf + 1.0)) + ach_2)












