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
import time	
import grating as gr
import helper.mnist as mnist

gr = reload(gr)

def load_images(net, classes, dataset_train, dataset_path, gabor_params={}):
	# target orientation around which to discriminate clock-wise vs. counter clock-wise
	# degree range within wich to test the network (on each side of target orientation)
	# noise injected in the gabor filter for the pre-training (critical period)
	# noise injected in the gabor filter for the training
	# noise injected in the gabor filter for the testing
	# side of the gabor filter image (total pixels = im_size * im_size)
	if net.protocol == 'digit':

		imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
		if net.verbose: print 'loading train images...'
		images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset_train, path = imPath)
		images, labels = evenLabels(images, labels, classes)
		images = normalize(images, net.A*np.size(images,1))

		if net.verbose: print 'loading test images...'
		test_dataset='test' if dataset_train=='train' else 'train'
		images_test, labels_test = mnist.read_images_from_mnist(classes = classes, dataset = test_dataset, path = imPath)
		images_test, labels_test = evenLabels(images_test, labels_test, classes)
		images_test, labels_test = shuffle([images_test, labels_test])
		images_test = normalize(images_test, net.A*np.size(images_test,1))
		
		orientations = None
		images_task = None
		labels_task = None
		orientations_task = None
		orientations_test = None

	elif net.protocol == 'gabor':
		if net.verbose: print 'creating gabor training images...'
		set_global(classes)

		n_train = 50000
		n_test = 1000

		orientations = np.random.random(n_train)*180 #orientations of gratings (in degrees)
		images, labels = generate_gabors(orientations, gabor_params['target_ori'], gabor_params['im_size'], gabor_params['noise_crit'], net.A)

		orientations_task = np.random.random(n_train)*gabor_params['excentricity']*2 + gabor_params['target_ori'] - gabor_params['excentricity'] #orientations of gratings (in degrees)
		images_task, labels_task = generate_gabors(orientations_task, gabor_params['target_ori'], gabor_params['im_size'], gabor_params['noise_train'], net.A)

		orientations_test = np.random.random(n_test)*gabor_params['excentricity']*2 + gabor_params['target_ori'] - gabor_params['excentricity'] #orientations of gratings (in degrees)
		images_test, labels_test = generate_gabors(orientations_test, gabor_params['target_ori'], gabor_params['im_size'], gabor_params['noise_test'], net.A)

	return images, labels, orientations, images_test, labels_test, orientations_test, images_task, labels_task, orientations_task

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

@numba.njit
def normalize_numba(images, A):
	"""
	numba-optimized version of the normalize function; Normalize each image to the sum of its pixel value (equivalent to feedforward inhibition)

	Args:

		images (numpy array): image to normalize
		A (int): normalization constant

	returns:
		numpy array: normalized images
	"""
	A_i = (A-images.shape[1])
	for im in range(images.shape[0]):
		sum_px = 0
		for px in range(images.shape[1]):
			sum_px += images[im,px]
		for px in range(images.shape[1]):
			images[im,px] = A_i*images[im,px]/sum_px + 1.

	return images

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

	labels = np.zeros(len(orientations), dtype=int)
	labels[orientations<=target_ori] = 0
	labels[orientations>target_ori] = 1
	images = gr.gabor(size=im_size, lambda_freq=im_size/5., theta=orientations, sigma=im_size/5., phase=phase, noise=noise)
	images = normalize(images, A*np.size(images,1))

	return images, labels

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

def evenLabels(images, labels, classes):
	"""
	Even out images and labels distribution so that they are evenly distributed over the labels.

	Args:
		images (numpy array): images
		labels (numpy array): labels constant

	returns:
		numpy array: evened-out images
		numpy array: evened-out labels
	"""

	nClasses = len(classes)
	nDigits, bins = np.histogram(labels, bins=10, range=(0,9))
	m = np.min(nDigits[nDigits!=0])
	images_even = np.zeros((m*nClasses, np.size(images,1)))
	labels_even = np.zeros(m*nClasses, dtype=int)
	for i, c in enumerate(classes):
		images_even[i*m:(i+1)*m,:] = images[labels==c,:][0:m,:]
		labels_even[i*m:(i+1)*m] = labels[labels==c][0:m]
	images, labels = np.copy(images_even), np.copy(labels_even)
	return images, labels

def propagate_layerwise(bInput, W_in, SM=True, t=1.):
	"""
	One propagation step

	Args:
		bInput (numpy array): input vector to the neurons of layer 1
		W_in (numpy matrix): weight matrix; shape: (input neurons x hidden neurons)
		SM (bool, optional): whether to pass the activation throught the Softmax function
		t (float): temperature parameter for the softmax function (only passed to the function, not used here)

	returns:
		numpy array: the activation of the hidden neurons
	"""

	hidNeurons = np.dot(bInput, np.log(W_in))
	if SM: hidNeurons = softmax(hidNeurons, t=t)
	return hidNeurons

@numba.njit
def disinhibition(postNeurons, lr, disinhib, postNeurons_lr):
	"""
	support function for numba implementation of learning_step() 
	"""
	for b in range(postNeurons.shape[0]):
		for pn in range(postNeurons.shape[1]):
			postNeurons_lr[b, pn] = postNeurons[b, pn] * lr * disinhib[b]

	return postNeurons_lr

@numba.njit
def regularization(dot, postNeurons, W, sum_ar):
	"""
	support function for numba implementation of learning_step() 
	"""
	for j in range(postNeurons.shape[1]):
		for i in range(postNeurons.shape[0]):
			sum_ar[j] += postNeurons[i,j]
	
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
		proba_predict (boolean): whether reward prediction is probabilistic (i.e., the expected value of the reward) or deterministic (i.e., binary)
		posterior (numpy array): posterior probability of the bayesian decoder 

	returns:
		numpy array: reward prediction, either deterministic or expected value (depending on proba_predict)
	"""

	if not proba_predict:
		reward_prediction = best_action==action_taken #binary reward prediction
	else:
		reward_prediction = posterior[range(np.size(action_taken,0)), val2idx(action_taken)] #expected value of the reward for the action taken

	return reward_prediction

def compute_dopa(predicted_reward, bReward, dopa_values):
	"""
	Computes the dopa signal based on the actual and predicted rewards

	Args:
		predicted_reward (numpy array, bool): predicted reward (True, False)
		bReward (numpy array, int): reward received (0, 1)
		dHigh (numpy array): dopa value for unpredicted reward
		dMid (numpy array): dopa value for correct reward prediction
		dNeut (numpy array): dopa value for correct no reward prediction
		dLow (numpy array): dopa value for incorrect reward prediction

	returns:
		numpy array: array of dopamine release value
	"""

	dopa = np.zeros(len(bReward))

	dopa[np.logical_and(predicted_reward==0, bReward==1)] = dopa_values['dHigh']			#unpredicted reward
	dopa[np.logical_and(predicted_reward==1, bReward==1)] = dopa_values['dMid']			#correct reward prediction
	dopa[np.logical_and(predicted_reward==0, bReward==0)] = dopa_values['dNeut']			#correct no reward prediction
	dopa[np.logical_and(predicted_reward==1, bReward==0)] = dopa_values['dLow']			#incorrect reward prediction

	return dopa

def save_data(net, W_in, W_act, perf, slopes=None, save_weights=True):
	"""
	Save passed data to file. Use pickle for weights and ConfigObj for the setting parameters 

	Args:
		W_in (numpy array): weight matrix to be saved to pickle file
		W_act (numpy array): weight matrix to be saved to pickle file
		perf (list): performance at each episode of the training
		slopes (dict): dictionary of various measurements of slope values
	"""

	if save_weights:
		pFile = open('output/' + net.name + '/W_in', 'w')
		pickle.dump(W_in, pFile)
		pFile.close()

		pFile = open('output/' + net.name + '/W_act', 'w')
		pickle.dump(W_act, pFile)
		pFile.close()

		pFile = open('output/' + net.name + '/perf_epi', 'w')
		pickle.dump(perf, pFile)
		pFile.close()

		pFile = open('output/' + net.name + '/slopes', 'w')
		pickle.dump(slopes, pFile)
		pFile.close()

def checkdir(net, OW_bool=True):
	"""
	Checks if directory exits. If not, creates it. If yes, asks whether to overwrite. If user choose not to overwrite, execution is terminated

	Args:
		kwargs (dict): parameters of the model
	"""

	if os.path.exists('output/' + net.name):
		if OW_bool: overwrite='yes'
		else: overwrite = raw_input('Folder \''+net.name+'\' already exists. Overwrite? (y/n/<new name>) ')
		if overwrite in ['n', 'no', 'not', ' ', '']:
			sys.exit('Folder exits - not overwritten')
		elif overwrite in ['y', 'yes']:
			shutil.rmtree('output/' + net.name)
		else:
			net.name = overwrite
			checkdir(net.name)
			return net.name
	os.makedirs('output/' + net.name)
	if net.protocol=='digit' and net.save_data==True:
		os.makedirs('output/' + net.name + '/RFs')
	if net.protocol=='gabor' and net.save_data==True:
		os.makedirs('output/' + net.name + '/TCs')

	return net.name

def shuffle(arrays):
	"""
	Shuffles the passed vectors according to the same random order

	Args:
		arrays (list): list of arrays to shuffle

	returns:
		list: list of shuffled arrays
		numpy array: indices of the random shuffling
	"""

	rndIdx = np.arange(len(arrays[0]))
	np.random.shuffle(rndIdx)
	shuffled_arrays = []
	for a in arrays:
		if len(np.shape(a))==1:
			shuffled_arrays.append(a[rndIdx])
		elif len(np.shape(a))==2:
			shuffled_arrays.append(a[rndIdx,:])

	return shuffled_arrays#, rndIdx


def computeCM(classResults, labels_test, classes):
	"""
	Computes the confusion matrix for a set of classification results

	Args:
		classResults (numpy array): result of the classifcation task
		labels_test (numpy array): labels of the test dataset
		classes (numpy array): all classes of the MNIST dataset used in the current run

	returns:
		numpy array: confusion matrix of shape (actual class x predicted class)
	"""

	nClasses = len(classes)
	confusMatrix = np.zeros((nClasses, nClasses))
	for ilabel,label in enumerate(classes):
		for iclassif, classif in enumerate(classes):
			classifiedAs = np.sum(np.logical_and(labels_test==label, classResults==classif))
			overTot = np.sum(labels_test==label)
			confusMatrix[ilabel, iclassif] = float(classifiedAs)/overTot
	
	return confusMatrix


















