
""" support functions for hebbian network and neural classifier """

import numpy as np
import accel
import pickle
import os
import sys
import shutil
import numba
import time	
import string
from configobj import ConfigObj
import grating as gr

gr = reload(gr)

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
		classes (numpy array): all classes of the MNIST dataset used in the current run

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

def propL1(bInput, W_in, SM=True, t=1.):
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

	hidNeurons = np.dot(bInput, accel.log(W_in))
	if SM: hidNeurons = softmax(hidNeurons, t=t)
	return hidNeurons

def learningStep(preNeurons, postNeurons, W, lr, disinhib=None, numba=True):
	"""
	One learning step for the hebbian network

	Args:
		preNeurons (numpy array): activation of the pre-synaptic neurons
		postNeurons (numpy array): activation of the post-synaptic neurons
		W (numpy array): weight matrix
		lr (float): learning rate
		disinhib (numpy array, optional): learning rate increase for the effect of acetylcholine and dopamine

	returns:
		numpy array: change in weight; must be added to the weight matrix W
	"""
	if disinhib is None or disinhib.shape[0]!=postNeurons.shape[0]: disinhib=np.ones(postNeurons.shape[0])

	if numba:
		postNeurons_lr = disinhibition(postNeurons, lr, disinhib, np.zeros_like(postNeurons))
		dot = np.dot(preNeurons.T, postNeurons_lr)
		return regularization(dot, postNeurons_lr, W, np.zeros(postNeurons_lr.shape[1]))
	else:
		postNeurons_lr = postNeurons * (lr * disinhib[:,np.newaxis]) #adds the effect of dopamine and acetylcholine to the learning rate  
	return (np.dot(preNeurons.T, postNeurons_lr) - np.sum(postNeurons_lr, 0)*W)


@numba.njit
def disinhibition(postNeurons, lr, disinhib, postNeurons_lr):
	"""
	support function for numba implementation of learningStep() 
	"""
	for b in range(postNeurons.shape[0]):
		for pn in range(postNeurons.shape[1]):
			postNeurons_lr[b, pn] = postNeurons[b, pn] * lr * disinhib[b]

	return postNeurons_lr

@numba.njit
def regularization(dot, postNeurons, W, sum_ar):
	"""
	support function for numba implementation of learningStep() 
	"""
	for j in range(postNeurons.shape[1]):
		for i in range(postNeurons.shape[0]):
			sum_ar[j] += postNeurons[i,j]
	
	for i in range(dot.shape[0]):
		for j in range(dot.shape[1]):
			dot[i,j] -= W[i,j] * sum_ar[j]

	return dot

def track_perf(perf, classes, bLabels, pred_bLabels, decay_param=0.001):
	"""
	Tracks performance for all classes using a weighted average

	Args:
		perf (numpy array): matrix of performances of shape (nClasses x 2), containing in perf[:,0] the number of correct trials and in perf[:,1] the number of all trials
		classes (numpy array): all classes of the MNIST dataset used in the current run
		bLabels (numpy array): image labels
		pred_bLabels (numpy array): predictated stimulus class (i.e., label predicted by the network)
		decay_param (float, optional): decay parameter of the weighted average (~0, all values equally considered; ~1, only last value considered; 0.1: ~50 values; 0.01: 500; 0.001: ~5000)

	returns:
		numpy array: updated performances (i.e., count of correct and total trials)
	"""

	#commented out code doesn't require label knowledge

	for ic, c in enumerate(classes):
		# if sum(pred_bLabels==c) != 0:
		if sum(bLabels==c) != 0:
			# perf[ic,0] = np.sum(bLabels[pred_bLabels==c] == pred_bLabels[pred_bLabels==c])*decay_param + perf[ic,0]*(1-decay_param)
			# perf[ic,1] = np.sum(pred_bLabels==c)*decay_param + perf[ic,1]*(1-decay_param)

			perf[ic,0] = np.sum(bLabels[bLabels==c] == pred_bLabels[bLabels==c])*decay_param + perf[ic,0]*(1-decay_param)
			perf[ic,1] = np.sum(bLabels==c)*decay_param + perf[ic,1]*(1-decay_param)
	return perf

def compute_reward(labels, actions):
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

def compute_dopa(bPredictActions, bActions, bReward, dHigh, dMid, dNeut, dLow):
	"""
	Computes the dopa signal based on the actual and predicted rewards

	Args:
		bPredictActions (numpy array): predicted best action
		bActions (numpy array): action taken
		bReward (numpy array): reward received
		dHigh (numpy array): dopa value for unpredicted reward
		dMid (numpy array): dopa value for correct reward prediction
		dNeut (numpy array): dopa value for correct no reward prediction
		dLow (numpy array): dopa value for incorrect reward prediction

	returns:
		numpy array: array of dopamine release value
	"""

	dopa = np.zeros(len(bActions))

	dopa[np.logical_and(bPredictActions!=bActions, bReward==1)] = dHigh			#unpredicted reward
	dopa[np.logical_and(bPredictActions==bActions, bReward==1)] = dMid			#correct reward prediction
	dopa[np.logical_and(bPredictActions!=bActions, bReward==0)] = dNeut			#correct no reward prediction
	dopa[np.logical_and(bPredictActions==bActions, bReward==0)] = dLow			#incorrect reward prediction

	return dopa

def compute_dopa_2(rPredicted, rActual, dHigh, dMid, dLow):
	"""
	Computes the dopa signal based on the difference between predicted and actual rewards

	Args:
		rPredicted (numpy array): predicted rewards for current batch
		rActual (numpy array): received rewards for current batch
		dHigh (numpy array): dopa value for unpredicted reward
		dMid (numpy array): dopa value for correct reward prediction
		dLow (numpy array): dopa value for incorrect reward prediction

	returns:
		numpy array: array of dopamine release value
	"""

	error = np.round(rActual-rPredicted)

	dopa = np.zeros(len(rActual))

	dopa[error  > 0] = dHigh		#unpredicted reward
	dopa[error == 0] = dMid			#correct reward prediction
	dopa[error  < 0] = dLow			#incorrect reward prediction

	return dopa


def compute_ach(perf, pred_bLabels_idx, aHigh, rActions=None, aPairing=1.):
	"""
	Computes the ach signal based on stimulus difficulty (average classification performance)

	Args:
		perf (numpy array): average performance over n batches
		pred_bLabels_idx (numpy array): index of predictated stimulus class (i.e., index of the predicted digit label)
		aHigh (numpy array): parameter of the exponential decay function relating perfomance to ach release
		rActions (numpy array): rewarded action
		aPairing (float) : strength of ACh release for stimulus pairing protocol; ach_labels is set to aPairing for capital letters in rActions

	returns:
		numpy array: array of acetylcholine release value for each training example of the current batch
		numpy array: array of acetylcholine release value for each of the digit label
	"""

	perf_ratio = np.zeros(np.size(perf,0))
	mask = perf[:,1]!=0 #avoid division by zero
	perf_ratio[mask] = perf[mask,0]/perf[mask,1]
	if np.mean(perf_ratio)==0:  #avoid division by zero
		ach_labels = np.ones(np.size(perf,0))
	else: 
		perc_mean =  perf_ratio/np.mean(perf_ratio)
		ach_labels = np.exp(aHigh*(-perc_mean+1))
	if rActions!=None and aPairing!=1.: ach_labels[np.array([char.isupper() for char in rActions])] = aPairing
	return ach_labels[pred_bLabels_idx], ach_labels

def Q_learn(Q, state, action, reward, Q_LR=0.01):
	"""
	Q-learner for state-action pairs

	Args;
		Q (numpy array): look up table of Q values (state x action)
		state (numpy array): index of the states visited in the current batch
		action (numpy array): index of the actions taken in the current batch
		reward (numpy array): reward value obtained in the current batch
		Q_LR (float): learning rate (0 < LR < 1)

	returns:
		numpy array: Q look-up table
	"""

	for b in range(len(state)):
		Q[state[b], action[b]] = (1 - Q_LR) * Q[state[b], action[b]] + Q_LR*reward[b]

	return Q

def save_data(W_in, W_act, perf, slopes, args, save_weights=True):
	"""
	Save passed data to file. Use pickle for weights and ConfigObj for the setting parameters 

	Args:
		W_in (numpy array): weight matrix to be saved to pickle file
		W_act (numpy array): weight matrix to be saved to pickle file
		perf (list): performance at each episode of the training
		slopes (dict): dictionary of various measurements of slope values
		args (dict): arguments to the hebbianRL function to be saved to ConfigObj
	"""

	if save_weights:
		pFile = open('output/' + args['runName'] + '/W_in', 'w')
		pickle.dump(W_in, pFile)
		pFile.close()

		pFile = open('output/' + args['runName'] + '/W_act', 'w')
		pickle.dump(W_act, pFile)
		pFile.close()

		pFile = open('output/' + args['runName'] + '/perf_epi', 'w')
		pickle.dump(perf, pFile)
		pFile.close()

		pFile = open('output/' + args['runName'] + '/slopes', 'w')
		pickle.dump(slopes, pFile)
		pFile.close()


	settingFile = ConfigObj()
	settingFile.filename = 'output/' + args['runName'] + '/settings.txt'
	for k in sorted(args.keys()):
		if type(args[k]) == type(np.array(0)): #convert numpy array to list
			args[k] = list(args[k])
		settingFile[k] = args[k]
	
	settingFile.write()

def load_data(runs_list, path='../output/'):
	"""
	Loads data from files for specified runs

	Args:
		runs_list (dict or list): list of the runs to load from files; should be something like: runs = ['control_49-small', 'dopa_49-small']
		path (string, optional): path to the folders containing the data

	returns:
		dict: dictionary of dictionaries filled with data loaded from file
	"""

	if type(runs_list) not in [list, np.array, dict]:
		runs_list = [runs_list] 

	runs = {}
	for r in runs_list: runs[r]=dict()
	for k in runs.keys():
		runName = k
		datapath = path + runName

		pFile = open(path + runName + '/W_in', 'r')
		runs[k]['W_in'] = pickle.load(pFile)
		pFile.close()

		pFile = open(path + runName + '/W_act', 'r')
		runs[k]['W_act'] = pickle.load(pFile)
		pFile.close()

		pFile = open(path + runName + '/classResults', 'r')
		runs[k]['classResults'] = pickle.load(pFile)
		pFile.close()

		pFile = open(path + runName + '/RFclass', 'r')
		runs[k]['RFclass'] = pickle.load(pFile)
		pFile.close()

		pFile = open(path + runName + '/perf_epi', 'r')
		runs[k]['perf_epi'] = pickle.load(pFile)
		pFile.close()

		pFile = open(path + runName + '/slopes', 'r')
		runs[k]['slopes'] = pickle.load(pFile)
		pFile.close()

		settingFile = ConfigObj(datapath+'/settings.txt')

		runs[k]['nRun'] 			= int(settingFile['nRun'])
		runs[k]['runName'] 			= runName
		runs[k]['nEpiCrit'] 		= int(settingFile['nEpiCrit'])
		runs[k]['nEpiDopa'] 		= int(settingFile['nEpiDopa'])
		runs[k]['t_hid'] 			= float(settingFile['t_hid'])
		runs[k]['t_act'] 			= float(settingFile['t_act'])
		runs[k]['A'] 				= float(settingFile['A'])
		runs[k]['dataset'] 			= settingFile['dataset']
		runs[k]['nHidNeurons'] 		= int(settingFile['nHidNeurons'])
		runs[k]['lr'] 				= float(settingFile['lr'])
		runs[k]['aHigh'] 			= float(settingFile['aHigh'])
		runs[k]['aPairing'] 		= float(settingFile['aPairing'])
		runs[k]['dHigh'] 			= float(settingFile['dHigh'])
		runs[k]['dMid'] 			= float(settingFile['dMid'])
		runs[k]['dNeut'] 			= float(settingFile['dNeut'])
		runs[k]['dLow'] 			= float(settingFile['dLow'])
		runs[k]['nBatch'] 			= int(settingFile['nBatch'])
		runs[k]['protocol'] 		= settingFile['protocol']
		runs[k]['target_ori'] 		= float(settingFile['target_ori'])
		runs[k]['excentricity'] 	= float(settingFile['excentricity'])
		runs[k]['noise_crit'] 		= float(settingFile['noise_crit'])
		runs[k]['noise_train'] 		= float(settingFile['noise_train'])
		runs[k]['noise_test'] 		= float(settingFile['noise_test'])
		runs[k]['im_size'] 			= int(settingFile['im_size'])
		runs[k]['classifier'] 		= settingFile['classifier']
		runs[k]['pypet_xplr'] 		= conv_bool(settingFile['pypet_xplr'])
		runs[k]['test_each_epi'] 	= conv_bool(settingFile['test_each_epi'])
		runs[k]['SVM'] 				= conv_bool(settingFile['SVM'])
		runs[k]['bestAction'] 		= conv_bool(settingFile['bestAction'])
		runs[k]['createOutput'] 	= conv_bool(settingFile['createOutput'])
		runs[k]['showPlots'] 		= conv_bool(settingFile['showPlots'])
		runs[k]['show_W_act'] 		= conv_bool(settingFile['show_W_act'])
		runs[k]['sort'] 			= settingFile['sort']
		runs[k]['target'] 			= settingFile['target']
		runs[k]['seed'] 			= int(settingFile['seed'])

		runs[k]['classes'] 			= np.array(map(int, settingFile['classes']))
		runs[k]['rActions'] 		= np.array(settingFile['rActions'])
		runs[k]['nClasses'] 		= len(runs[k]['classes'])

	return runs

def checkdir(runName, OW_bool=True):
	"""
	Checks if directory exits. If not, creates it. If yes, asks whether to overwrite. If user choose not to overwrite, execution is terminated

	Args:
		runName (str): name of the folder where to save the data
	"""

	if os.path.exists('output/' + runName):
		if OW_bool: overwrite='yes'
		else: overwrite = raw_input('Folder \''+runName+'\' already exists. Overwrite? (y/n/<new name>) ')
		if overwrite in ['n', 'no', 'not', ' ', '']:
			sys.exit('Folder exits - not overwritten')
		elif overwrite in ['y', 'yes']:
			if os.path.exists('output/' + runName + '/RFs'):
				shutil.rmtree('output/' + runName + '/RFs')
			if os.path.exists('output/' + runName + '/TCs'):
				shutil.rmtree('output/' + runName + '/TCs')
			shutil.rmtree('output/' + runName)
		else:
			runName = overwrite
			checkdir(runName)
			return runName
	os.makedirs('output/' + runName)
	os.makedirs('output/' + runName + '/RFs')
	os.makedirs('output/' + runName + '/TCs')
	print 'run:  ' + runName
	return runName

def checkClassifier(classifier):
	"""
	Checks if classifier has correct value. If not, raise an error.

	Args:
		classifier (str): name of the classifier
	"""

	if classifier not in ['neuronClass', 'SVM', 'actionNeurons']:
		raise ValueError( '\'' + classifier +  '\' not a legal classifier value. Legal values are: \'neuronClass\', \'SVM\', \'actionNeurons\'.')


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

def val2idx(actionVal, lActions):
	"""
	Returns the index of the action (int) for each provided value (str)

	Args:
		actionVal (numpy array of str): array of 1-char long strings representing the value of the chosen action for an input image 
		lActions (numpy array): possible legal actions

	returns:
		numpy array of int: array of int representing the index of the chosen action for an input image
	"""

	actionIdx = np.zeros_like(actionVal, dtype=int)
	for i,v in enumerate(lActions):
		actionIdx[actionVal==v] = i

	return actionIdx

def labels2actionVal(labels, classes, rActions):
	"""
	returns the the correct action value (str) for each provided label (int)

	Args:
		labels (numpy array): labels of the input images
		classes (numpy array): all classes of the MNIST dataset used in the current run
		rActions (numpy array of str): reward actions associated with each of the classes of MNIST

	returns:
		numpy array str: rewarded action value for each images. Returns empty space ' ' if provided label is not part of the considered classes
	"""

	actionVal = np.empty(len(labels), dtype='|S1')
	for i in range(len(classes)):
		actionVal[labels==classes[i]] = rActions[i]
	actionVal[actionVal=='']=' '
	return actionVal

def actionVal2labels(actionVal, classes, rActions):
	"""
	returns the class labels (int) for each action value (str). If more than one label corresponds to the same action value, than a list of list is returned, with the inside list containing all correct labels for the action value.

	Args:
		actionVal (numpy array of str): array of 1-char long strings representing the value of the chosen action for an input image 
		classes (numpy array): all classes of the MNIST dataset used in the current run
		rActions (numpy array): rewarded actions for each class (may be rActions_z)

	returns:
		list: label associated with each action value
	"""

	labels=[]
	for act in actionVal:
		labels.append(list(classes[act==rActions]))
	return labels

def label2idx(classes, labels):
	"""
	Creates a vector of length identical to labels but with the index of the label rather than its class label (int)

	Args:
		classes (numpy array): all classes of the MNIST dataset used in the current run
		labels (numpy array): labels of the input images

	returns:
		numpy array str: rewarded action value for each images
	"""

	actionIdx = np.ones(len(labels), dtype=int)
	for i,c in enumerate(classes):
		actionIdx[labels==c] = i
	return actionIdx

def computeCM(classResults, labels_test, classes):
	"""
	Computes the confusion matrix for a set of classification results

	Args:
		classResults (numpy array): result of the classifcation task
		labels_test (numpy array): labels of the test dataset
		classes (numpy array): all classes of the MNIST dataset used in the current run (or rAction, the correct action associated with each digit class)

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

	# import pdb; pdb.set_trace()
	
	return confusMatrix

def conv_bool(bool_str):
	"""
	Converts a string ('True', 'False') value to boolean (True, False)

	Args:
		bool_str (str): string to convert

	returns:
		bool: boolean value of the string
	"""

	if bool_str=='True': return True
	elif bool_str=='False': return False
	else: return None

def rand_ACh(nClasses):
	"""
	Randommly assigns a class to ACh release (creates an array of lower case characters of length nClasses and randomly sets one character to upper case, which triggers ACh release).

	Args:
		nClasses (int): number of digit classes

	returns:
		int: target digit class
		numpy array: array of rewarded actions
		numpy array: array of rewarded actions
		numpy array: array of legal actions
	"""
	target = np.random.randint(nClasses)
	rActions = np.array(list(string.ascii_lowercase)[:nClasses])
	rActions[target] = rActions[target].upper()
	rActions_z = np.copy(rActions)
	lActions = np.copy(rActions)
	print 'target digit: ' + str(target)

	return target, rActions, rActions_z, lActions

""" pypet-specific support functions """
def add_parameters(traj, kwargs):
	for k in kwargs.keys():
		traj.f_add_parameter(k, kwargs[k])

def set_run_names(explore_dict, runName):
	nXplr = len(explore_dict[explore_dict.keys()[0]])
	runName_list = [runName for _ in range(nXplr)]
	for n in range(nXplr):
		for k in explore_dict.keys():
			runName_list[n] += '_'
			runName_list[n] += k
			runName_list[n] += str(explore_dict[k][n]).replace('.', ',')
	return runName_list


















