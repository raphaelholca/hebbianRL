
""" support functions for hebbian network and neural classifier """

import numpy as np
import accel
import pickle
import os
import sys
import shutil
import time	
from configobj import ConfigObj

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

def softmax(activ, vectorial=True, t=1.):
	"""
	Softmax function (equivalent to lateral inhibition, or winner-take-all)

	Args:
		activ (numpy array): activation of neurons to be fed to the function; should be (training examples x neurons)
		vectorial (bool, optional): whether to use vectorial of iterative method
		t (float): temperature parameter; determines the sharpness of the softmax, or the strength of the competition

	returns:
		numpy array: the activation fed through the softmax function
	"""
	
	#vectorial
	if vectorial:
		# scale = np.clip(np.max(activ,1)-700, 0, np.inf)
		# tmpRND=np.random.rand(np.shape(activ)[0],np.shape(activ)[1])/100000
		# activ+=tmpRND #add a random offset to insure that there is only a single min
		# activ[activ==np.min(activ,1)[:,np.newaxis]] = np.clip(np.min(activ,1), -740+scale, np.inf)
		# activ-=tmpRND
		# return np.exp((activ-scale[:,np.newaxis])/t) / np.sum(np.exp((activ-scale[:,np.newaxis])/t), 1)[:,np.newaxis]

		activ_norm = np.copy(activ - np.max(activ,1)[:,np.newaxis])
		return np.exp((activ_norm)/t) / np.sum(np.exp((activ_norm)/t), 1)[:,np.newaxis]

	#iterative
	else:
		activ_SM = np.zeros_like(activ)
		for i in range(np.size(activ,0)):
			scale = 0
			I = np.copy(activ[i,:])
			if (I[np.argmax(I)] > 700):
			    scale  = I[np.argmax(I)] - 700
			if (I[np.argmin(I)] < -740 + scale):
			    I[np.argmin(I)] = -740 + scale
			activ_SM[i,:] = np.exp((I-scale)/t) / np.sum(np.exp((I-scale)/t))
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
	One propagation step from input to hidden layer

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

def propL2_learn(classes, labels):
	"""
	One propagation step from hidden to classification layer, during learning (activation determined by the labels)

	Args:
		classes (numpy array): all classes of the MNIST dataset used in the current run
		labels (numpy matrix): labels associated with the input

	returns:
		numpy array: the activation of the classification neurons
	"""

	classNeurons = np.zeros((len(labels), len(classes)))
	labelsIdx = label2idx(classes, labels)
	classNeurons[np.arange(len(labels)),labelsIdx] = 1.0
	return classNeurons

def propL2_class(hidNeurons, W_class):
	"""
	One propagation step from hidden to classification layer, during classification (activation determined by the feedforward input)

	Args:
		hidNeurons (numpy array): activation of the hidden neurons, i.e., the input to the classification layer
		W_class (numpy matrix): weight matrix; shape: (hidden neurons x classification neurons)

	returns:
		numpy array: the activation of the classification neurons
	"""

	return	np.dot(hidNeurons, W_class)

def learningStep(preNeurons, postNeurons, W, lr, ach=np.zeros(1), dopa=np.zeros(1)):
	"""
	One learning step for the hebbian network

	Args:
		preNeurons (numpy array): activation of the pre-synaptic neurons
		postNeurons (numpy array): activation of the post-synaptic neurons
		W (numpy array): weight matrix
		lr (float): learning rate
		ach(numpy array, optional): learning rate increase for the effect of acetylcholine
		dopa(numpy array, optional): learning rate increase for the effect of dopamine

	returns:
		numpy array: change in weight; must be added to the weight matrix W
	"""

	postNeurons_lr = postNeurons * (lr + ach[:,np.newaxis] + dopa[:,np.newaxis]) #adds the effect of dopamine and acetylcholine to the learning rate  
	return (np.dot(preNeurons.T, postNeurons_lr) - np.sum(postNeurons_lr, 0)*W)

def compute_reward(labels, classes, actions, rActions):
	"""
	Computes the reward based on the action taken and the label of the current input

	Args:
		labels (numpy array): image labels
		classes (numpy array): all classes of the MNIST dataset used in the current run
		actions (numpy array): action taken
		rActions (numpy array): rewarded action

	returns:
		numpy array: reward for the label and action pair
	"""
	reward = np.zeros(len(labels), dtype=int)
	for i in range(len(classes)):
		reward[np.logical_and(labels==classes[i], actions==rActions[i])] = 1 #reward correct state-action pairs
		reward[np.logical_and(labels==classes[i], '1'==rActions[i])] = 2 #reward states that are always rewarded
		reward[np.logical_and(labels==classes[i], '0'==rActions[i])] = -1 #do not reward states that are never rewarded

	return reward

def save_data(runName, W_in, W_act, W_class, seed, nRun, classes, rActions, dataset, A, nEpiCrit, nEpiProc, nEpiAdlt, nHidNeurons, lrCrit, lrAdlt, aHigh, aLow, dHigh, dMid, dNeut, dLow, nBatch, bestAction, feedback, SVM, classifier):
	"""
	Save passed data to file. Use pickle for weights and ConfigObj for the setting parameters 

	Args:
		runName (str): name of the current experiment/folder where to save the data
		W_save (numpy array): weight matrix to be saved to pickle file
		rest of args: setting parameters to be saved to ConfigObj
	"""

	pFile = open('output/' + runName + '/W_in', 'w')
	pickle.dump(W_in, pFile)
	pFile.close()

	pFile = open('output/' + runName + '/W_act', 'w')
	pickle.dump(W_act, pFile)
	pFile.close()

	if W_class:
		pFile = open('output/' + runName + '/W_class', 'w')
		pickle.dump(W_class, pFile)
		pFile.close()

	settingFile = ConfigObj()
	settingFile.filename 			= 'output/' + runName + '/settings.txt'
	settingFile['classes'] 			= list(classes)
	settingFile['rActions'] 		= list(rActions)
	settingFile['nRun'] 			= nRun
	settingFile['nEpiCrit']			= nEpiCrit
	settingFile['nEpiProc']			= nEpiProc
	settingFile['nEpiAdlt']			= nEpiAdlt 
	settingFile['A'] 				= A
	settingFile['runName'] 			= runName
	settingFile['dataset'] 			= dataset
	settingFile['nHidNeurons'] 		= nHidNeurons
	settingFile['lrCrit']			= lrCrit
	settingFile['lrAdlt']			= lrAdlt
	settingFile['aHigh'] 			= aHigh
	settingFile['aLow'] 			= aLow
	settingFile['dHigh'] 			= dHigh
	settingFile['dMid'] 			= dMid
	settingFile['dNeut'] 			= dNeut
	settingFile['dLow'] 			= dLow
	settingFile['nBatch'] 			= nBatch
	settingFile['classifier'] 		= classifier
	settingFile['SVM'] 				= SVM
	settingFile['bestAction'] 		= bestAction
	settingFile['feedback'] 		= feedback
	settingFile['seed'] 			= seed
	
	settingFile.write()

def load_data(runs):
	"""
	Loads data from files for specified runs

	Args:
		runs (dict): dictionary of dictionaries of the runs to load from files

	returns:
		dict: orinigal dictionary filled with data loaded from file
	"""

	for k in runs.keys():
		runName = k
		datapath = '../output/' + runName

		pFile = open('../output/' + runName + '/W_in', 'r')
		runs[k]['W_in'] = pickle.load(pFile)
		pFile.close()

		pFile = open('../output/' + runName + '/W_act', 'r')
		runs[k]['W_act'] = pickle.load(pFile)
		pFile.close()

		pFile = open('../output/' + runName + '/classResults', 'r')
		runs[k]['classResults'] = pickle.load(pFile)
		pFile.close()

		pFile = open('../output/' + runName + '/RFclass', 'r')
		runs[k]['RFclass'] = pickle.load(pFile)
		pFile.close()

		settingFile = ConfigObj(datapath+'/settings.txt')

		runs[k]['runName'] 			= runName
		runs[k]['classes'] 			= map(int, settingFile['classes'])
		runs[k]['rActions'] 		= settingFile['rActions']
		runs[k]['nEpiCrit'] 		= int(settingFile['nEpiCrit'])
		runs[k]['nEpiProc'] 		= int(settingFile['nEpiProc'])
		runs[k]['nEpiAdlt'] 		= int(settingFile['nEpiAdlt'])
		runs[k]['nHidNeurons'] 		= int(settingFile['nHidNeurons'])
		runs[k]['bestAction'] 		= conv_bool(settingFile['bestAction'])
		runs[k]['feedback'] 		= conv_bool(settingFile['feedback'])
		runs[k]['lrCrit'] 			= float(settingFile['lrCrit'])
		runs[k]['lrAdlt'] 			= float(settingFile['lrAdlt'])
		runs[k]['aHigh'] 			= float(settingFile['aHigh'])
		runs[k]['aLow']				= float(settingFile['aLow'])
		runs[k]['dHigh'] 			= float(settingFile['dHigh'])
		runs[k]['dMid'] 			= float(settingFile['dMid'])
		runs[k]['dNeut'] 			= float(settingFile['dNeut'])
		runs[k]['dLow'] 			= float(settingFile['dLow'])
		runs[k]['classifier'] 		= settingFile['classifier']
		runs[k]['dataset'] 			= settingFile['dataset']
		runs[k]['nRun'] 			= int(settingFile['nRun'])
		runs[k]['nBatch'] 			= int(settingFile['nBatch'])
		runs[k]['A'] 				= float(settingFile['A'])

		runs[k]['nClasses'] = len(runs[k]['classes'])

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
			shutil.rmtree('output/' + runName)
		else:
			runName = overwrite
			checkdir(runName)
			return runName
	os.makedirs('output/' + runName)
	os.makedirs('output/' + runName + '/RFs')
	print 'run:  ' + runName
	return runName

def checkClassifier(classifier):
	"""
	Checks if classifier has correct value. If not, raise an error.

	Args:
		classifier (str): name of the classifier
	"""

	if classifier not in ['neural', 'SVM', 'neuronClass']:
		raise ValueError( '\'' + classifier +  '\' not a legal classifier value. Legal values are: \'neural\', \'SVM\', \'neuronClass\'.')


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
		shuffled_arrays.append(a[rndIdx,:])

	return shuffled_arrays#, rndIdx

def val2idx(actionVal, lActions):
	"""
	Creates a vector of length identical to actionVal but with the index of the action (int) rather than their value (str)

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
	Creates a new vector of length identical to labels but with the correct action value (str) rather than the label (int)

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

def label2idx(classes, labels):
	"""
	Creates a vector of length identical to labels but with the index of the label rather than its class

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





