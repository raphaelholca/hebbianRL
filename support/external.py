
""" support functions for hebbian network and neural classifier """

import numpy as np
import accel
import pickle
import os
import sys
import shutil
import time	
import string
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

def softmax(activ, vectorial=True, t=1., disinhib=np.ones(1)):
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
		activ_norm = np.copy(activ - np.max(activ,1)[:,np.newaxis])
		activ_SM = np.exp((activ_norm)/t) / np.sum(np.exp((activ_norm)/t), 1)[:,np.newaxis]
		return activ_SM

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

###deprecated?###
# def propL2_learn(classes, labels):
# 	"""
# 	One propagation step from hidden to classification layer, during learning (activation determined by the labels)

# 	Args:
# 		classes (numpy array): all classes of the MNIST dataset used in the current run
# 		labels (numpy matrix): labels associated with the input

# 	returns:
# 		numpy array: the activation of the classification neurons
# 	"""

# 	classNeurons = np.zeros((len(labels), len(classes)))
# 	labelsIdx = label2idx(classes, labels)
# 	classNeurons[np.arange(len(labels)),labelsIdx] = 1.0
# 	return classNeurons

###deprecated?###
# def propL2_class(hidNeurons, W_class):
# 	"""
# 	One propagation step from hidden to classification layer, during classification (activation determined by the feedforward input)

# 	Args:
# 		hidNeurons (numpy array): activation of the hidden neurons, i.e., the input to the classification layer
# 		W_class (numpy matrix): weight matrix; shape: (hidden neurons x classification neurons)

# 	returns:
# 		numpy array: the activation of the classification neurons
# 	"""

# 	return	np.dot(hidNeurons, W_class)

def learningStep(preNeurons, postNeurons, W, lr, disinhib=np.ones(1)):
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

	postNeurons_lr = postNeurons * (lr * disinhib[:,np.newaxis]) #adds the effect of dopamine and acetylcholine to the learning rate  
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

def save_data(W_in, W_act, args):
	"""
	Save passed data to file. Use pickle for weights and ConfigObj for the setting parameters 

	Args:
		W_in (numpy array): weight matrix to be saved to pickle file
		W_act (numpy array): weight matrix to be saved to pickle file
		args (dict): arguments to the hebbianRL function to be saved to ConfigObj
	"""

	pFile = open('output/' + args['runName'] + '/W_in', 'w')
	pickle.dump(W_in, pFile)
	pFile.close()

	pFile = open('output/' + args['runName'] + '/W_act', 'w')
	pickle.dump(W_act, pFile)
	pFile.close()

	settingFile = ConfigObj()
	settingFile.filename 			= 'output/' + args['runName'] + '/settings.txt'
	for k in sorted(args.keys()):
		if type(args[k]) == type(np.array(0)): #convert numpy array to list
			args[k] = list(args[k])
		settingFile[k] = args[k]
	
	settingFile.write()

def load_data(runs_list, path='../output/'):
	"""
	Loads data from files for specified runs

	Args:
		runs_list (dict): list of the runs to load from files; should be something like: runs = ['control_49-small', 'dopa_49-small']
		path (string, optional): path to the folders containing the data

	returns:
		dict: dictionary of dictionaries filled with data loaded from file
	"""

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

		settingFile = ConfigObj(datapath+'/settings.txt')

		runs[k]['runName'] 			= runName
		runs[k]['classes'] 			= np.array(map(int, settingFile['classes']))
		runs[k]['rActions'] 		= np.array(settingFile['rActions'])
		runs[k]['nEpiCrit'] 		= int(settingFile['nEpiCrit'])
		runs[k]['target'] 			= settingFile['target']
		runs[k]['nEpiProc'] 		= int(settingFile['nEpiProc'])
		runs[k]['nEpiAch'] 			= int(settingFile['nEpiAch'])
		runs[k]['nEpiDopa'] 		= int(settingFile['nEpiDopa'])
		runs[k]['nHidNeurons'] 		= int(settingFile['nHidNeurons'])
		runs[k]['bestAction'] 		= conv_bool(settingFile['bestAction'])
		runs[k]['feedback'] 		= conv_bool(settingFile['feedback'])
		runs[k]['lrCrit'] 			= float(settingFile['lrCrit'])
		runs[k]['lrAdlt'] 			= float(settingFile['lrAdlt'])
		runs[k]['aHigh'] 			= float(settingFile['aHigh'])
		runs[k]['aLow']				= float(settingFile['aLow'])
		runs[k]['dHigh'] 			= float(settingFile['dHigh'])
		try:
			runs[k]['dMid'] 		= float(settingFile['dMid'])
		except KeyError:
			pass#print 'while loading data: no dMid provided'
		runs[k]['dNeut'] 			= float(settingFile['dNeut'])
		runs[k]['dLow'] 			= float(settingFile['dLow'])
		runs[k]['classifier'] 		= settingFile['classifier']
		runs[k]['dataset'] 			= settingFile['dataset']
		runs[k]['nRun'] 			= int(settingFile['nRun'])
		runs[k]['nBatch'] 			= int(settingFile['nBatch'])
		runs[k]['A'] 				= float(settingFile['A'])

		runs[k]['nClasses'] = len(runs[k]['classes'])

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

	import pdb; pdb.set_trace()###
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

def actionVal2labels(actionVal, classes, rActions):
	"""
	returns a list of length identical to actionVal but with class labels (int) rather than action value (str). If more than one label corresponds to the same action value, than a list of list is returned, with the inside list containing all correct labels for the action value.

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



