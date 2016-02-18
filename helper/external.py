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
import struct
from array import array

gr = reload(gr)

def load_images(protocol, A, verbose=True, digit_params={}, gabor_params={}):
	""" 
	Load images training and testing images 

		Args:
			protocol (str): experimental protocol, maybe 'digit' or 'gabor'
			A (float): normalization constant for the images
			verbose (bool, optional): whether to print comments to console
			digit_params (dict): parameters for loading the MNIST dataset. These are:
				classes (numpy array): classes of the MNIST dataset to load
				dataset_train (str): name of the dataset to load for training the network; maybe 'test' or 'train'
				dataset_path (str): path of the MNIST dataset
				shuffle (bool): whether to mix train and test dataset and split them up again
			gabor_params (dict): parameters for creating the gabor patches. These are:
				n_train (int): number of training images
				n_test (int): number of testing images
				target_ori (float): target orientation around which to discriminate clock-wise vs. counter clock-wise
				excentricity (float): degree range within wich to test the network (on each side of target orientation)
				noise_crit (float): noise injected in the gabor filter for the pre-training (critical period)
				noise_train (float): noise injected in the gabor filter for the training
				noise_test (float): noise injected in the gabor filter for the testing
				im_size (int): side of the gabor filter image (total pixels = im_size * im_size)

		returns:
			(2D numpy array): training images
			(numpy array): training labels
			(2D numpy array): test images 
			(numpy array): testing labels 
			(2D numpy array): training images for gabor task 
			(numpy array): training labels

	"""

	if protocol == 'digit':
		if verbose: print 'loading train images...'
		images, labels = read_images_from_mnist(classes=digit_params['classes'], dataset=digit_params['dataset_train'], path=digit_params['dataset_path'])

		if verbose: print 'loading test images...'
		dataset_test = 'test' if digit_params['dataset_train']=='train' else 'train'
		images_test, labels_test = read_images_from_mnist(classes=digit_params['classes'], dataset=dataset_test ,  path=digit_params['dataset_path'])

		if digit_params['shuffle']:
			n_images_train = len(labels)
			images_all = np.append(images, images_test, axis=0)
			labels_all = np.append(labels, labels_test)
			images_rdn, labels_rdn = shuffle([images_all, labels_all])

			images = images_rdn[:n_images_train, :]
			labels = labels_rdn[:n_images_train]
			images_test = images_rdn[n_images_train:, :]
			labels_test = labels_rdn[n_images_train:]
		
		images, labels = even_labels(images, labels, digit_params['classes'])
		images = normalize(images, A*np.size(images,1))

		images_test, labels_test = even_labels(images_test, labels_test, digit_params['classes'])
		images_test = normalize(images_test, A*np.size(images_test,1))

		images_task = None
		labels_task = None
		images_params = digit_params

	elif protocol == 'gabor':
		if verbose: print 'creating gabor training images...'
		gabor_params['target_ori'] %= 180.

		orientations = np.random.random(gabor_params['n_train'])*180 #orientations of gratings (in degrees)
		images, labels = generate_gabors(orientations, gabor_params['target_ori'], gabor_params['im_size'], gabor_params['noise'], A)

		orientations_task = np.random.random(gabor_params['n_train'])*gabor_params['excentricity']*2 + gabor_params['target_ori'] - gabor_params['excentricity'] 
		images_task, labels_task = generate_gabors(orientations_task, gabor_params['target_ori'], gabor_params['im_size'], gabor_params['noise'], A)

		orientations_test = np.random.random(gabor_params['n_test'])*gabor_params['excentricity']*2 + gabor_params['target_ori'] - gabor_params['excentricity']
		images_test, labels_test = generate_gabors(orientations_test, gabor_params['target_ori'], gabor_params['im_size'], gabor_params['noise'], A)

		images_params = gabor_params

	return {'train':images, 'test':images_test, 'task':images_task}, {'train':labels, 'test':labels_test, 'task':labels_task}, images_params

def read_images_from_mnist(classes, dataset = "train", path = '/Users/raphaelholca/Documents/data-sets/MNIST'):
    """ Import the MNIST data set """

    if not os.path.exists(path): #in case the code is running on the server
        path = '/mnt/antares_raid/home/raphaelholca/Documents/data-sets/MNIST'


    if dataset is "train":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset is "test":
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

def generate_gabors(orientations, target_ori, im_size, noise, A, phase=0.25):
	"""
	Calling function to generate gabor filters

	Args:
		orientations (numpy array): 1-D array of orientations of gratings (in degrees) (one grating is created for each orientation provided)
		target_ori (float): target orientation around which to discriminate clock-wise vs. counter clock-wise
		im_size (int): side of the gabor filter image (total pixels = im_size * im_size)
		noise (int): noise level to add to Gabor patch; represents the standard deviation of the Gaussian distribution from which noise is drawn; range: (0, inf
		A (float): input normalization constant
		phase (float, list or numpy array, optional): phase of the filter; range: [0, 1]

	returns:
		numpy array: gabor filters of size: (len(orientations), im_size*im_size)
		numpy array: labels (clock-wise / counter clock-wise) of each gabor filter
	"""

	images = gr.gabor(size=im_size, lambda_freq=im_size/5., theta=orientations, sigma=im_size/5., phase=phase, noise=noise)

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
	
	net.name = checkdir(net.name, net.protocol, overwrite=True)
	
	""" save network to file """
	n_file = open(os.path.join('output', net.name, 'Network'), 'w')
	pickle.dump(net, n_file)
	n_file.close()

	save_file = os.path.join('output', net.name, net.name + '_params.txt')
	print_params(vars(net), save_file)

def print_params(param_dict, save_file):
	""" print parameters """
	tab_length = 25

	params_to_print = ['dHigh', 'dMid', 'dNeut', 'dLow', 'dopa_values', 'protocol', 'name', 'n_runs', 'n_epi_crit', 'n_epi_dopa', 't', 'A', 'lr', 'batch_size', 'n_hid_neurons', 'init_file', 'lim_weights', 'noise_std',
		'exploration', 'pdf_method', 'classifier', 'test_each_epi', 'verbose', 'seed', 'images_params']

	
	param_file = open(save_file, 'w')
	time_str = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
	time_line = ('created on\t: %s\n\n' % time_str).expandtabs(tab_length)
	param_file.write(time_line)

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

def propagate_layerwise(input, W, SM=True, t=1.):
	"""
	One propagation step

	Args:
		input (numpy array): input vector to the neurons of layer 1
		W (numpy matrix): weight matrix; shape: (input neurons x hidden neurons)
		SM (bool, optional): whether to pass the activation throught the Softmax function
		t (float): temperature parameter for the softmax function (only passed to the function, not used here)

	returns:
		numpy array: the activation of the hidden neurons
	"""

	activ = np.dot(input, np.log(W))
	if SM: activ = softmax(activ, t=t)
	return activ

@numba.njit
def disinhibition(post_neurons, lr, dopa, post_neurons_lr):
	"""
	support function for numba implementation of learning_step() 
	"""
	for b in range(post_neurons.shape[0]):
		for pn in range(post_neurons.shape[1]):
			post_neurons_lr[b, pn] = post_neurons[b, pn] * lr * dopa[b]

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

	reward = np.zeros(len(labels), dtype=int)
	reward[labels==actions] = 1

	return reward

def reward_prediction(best_action, action_taken, proba_predict=False, posterior=None):
	"""
	Computes reward prediction based on the best (greedy) action and the action taken

	Args:
		best_action (numpy array): best (greedy) action for each trial of a batch
		action_taken (numpy array): action taken for each trial of a batch
		proba_predict (boolean, optional): whether reward prediction is probabilistic (i.e., the expected value of the reward) or deterministic (i.e., binary)
		posterior (numpy array, optional): posterior probability of the bayesian decoder 

	returns:
		numpy array: reward prediction, either deterministic or expected value (depending on proba_predict)
	"""

	if not proba_predict:
		reward_prediction = best_action==action_taken #binary reward prediction
	else:
		reward_prediction = posterior[range(np.size(action_taken,0)), val2idx(action_taken)] #expected value of the reward for the action taken

	return reward_prediction

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
	dopa[np.logical_and(predicted_reward==1, reward==1)] = dopa_values['dMid']			#correct reward prediction
	dopa[np.logical_and(predicted_reward==0, reward==0)] = dopa_values['dNeut']		#correct no reward prediction
	dopa[np.logical_and(predicted_reward==1, reward==0)] = dopa_values['dLow']			#incorrect reward prediction

	return dopa

















