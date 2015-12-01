""" Hebbian learning-based classifier used to assess the performance of the representation """

import numpy as np
import matplotlib.pyplot as pyplot
import support.mnist as mnist
import support.external as ex
import support.plots as pl
import support.bayesian_decoder as bc
import pickle
import re
from sklearn.svm import SVC

ex = reload(ex)
pl = reload(pl)
bc = reload(bc)

def actionNeurons(W_in_save, W_act_save, images_test, labels_test, kwargs, save_data, verbose):
	"""
	evaluates the quality of a representation using the action neurons of the network.

	Args:
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		W_act_save (numpy array) : weight matrix from hidden to classification layer; shape = (hidden x class)
		images_test (numpy array): test images
		labels_test (numpy array): test labels
		kwargs (dict): parameters of the model
		save_data (bool, optional) : whether save data
		verbose (bool, optional) : whether to display text ouput
	"""

	runName = kwargs['runName']
	classes = kwargs['classes']
	rActions = kwargs['rActions']
	nHidNeurons = kwargs['nHidNeurons']
	t_hid = kwargs['t_hid']

	if verbose: print "\nassessing performance..."

	""" variable initialization """
	allCMs = []
	allPerf = []

	""" process labels for plot """
	rActions_uni, idx = np.unique(rActions, return_index=True)
	sorter = idx.argsort()
	rActions_uni = rActions_uni[sorter]
	labels_print = ex.actionVal2labels(rActions_uni)
	for i_l, l in enumerate(labels_print): labels_print[i_l] = re.sub("[^0-9 ]", "", str(l))
	labels_print = np.array(labels_print)
	labels_print[rActions_uni=='z'] = 'z'

	for iw in sorted(W_in_save.keys()):
		if verbose: print 'run: ' + str(int(iw)+1)
		W_in = W_in_save[iw]
		W_act = W_act_save[iw]

		""" testing of the classifier """
		hidNeurons = ex.propL1(images_test, W_in, t=t_hid)
		actNeurons = ex.propL1(hidNeurons, W_act)
		classIdx = np.argmax(actNeurons, 1)
		classResults = rActions[classIdx]
		
		""" compute classification performance """
		correct_classif = float(np.sum(classResults==ex.labels2actionVal(labels_test)))
		allPerf.append(correct_classif/len(labels_test))
		CM = ex.computeCM(classResults, ex.labels2actionVal(labels_test), np.unique(rActions))
		CM = CM[sorter,:]
		CM = CM[:,sorter]
		allCMs.append(CM)

	""" print and save performance measures """
	if save_data:
		print_save(allCMs, allPerf, rActions_uni, runName, verbose)
	return allCMs, allPerf

def SVM(runName, W_in_save, images_train, labels_train, classes, nDimStates, A, train_dataset, save_data, verbose, SM=True):
	"""
	evaluates the quality of a representation using an SVM. Trains an SVM on the images transformed into the representation, then assesses performance.

	Args:
		runName (str) : name of the folder where to save results
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		images_train (numpy array): training image of the MNIST dataset
		labels_train (numpy array): training labels of the MNIST dataset
		classes (numpy array): all classes of the MNIST dataset used in the current run
		nDimStates (int) : number of dimensions of the states (size of images)
		A (int): normalization constant
		train_dataset (str): name of the dataset used for training
		save_data (bool, optional) : whether save data
		verbose (bool, optional) : whether to display text ouput
		SM (bool, optional): whether to pass the activation throught the Softmax function
	"""

	if verbose: print "\nassessing performance..."

	""" load and pre-process images """
	if verbose: print 'assessing performance with SVM...'
	if  train_dataset=='train': test_dataset='test'
	else: test_dataset='train'
	images_test, labels_test = mnist.read_images_from_mnist(classes=classes, dataset=test_dataset)
	images_test = ex.normalize(images_test, A*nDimStates)
	images_test, labels_test = ex.evenLabels(images_test, labels_test, classes)
	images_train, labels_train = ex.shuffle([images_train, labels_train])

	""" variable initialization """
	allCMs = []
	allPerf = []

	""" train and test SVM for all runs """
	for iw in sorted(W_in_save.keys()):
		""" load weight matrix and transform images into representation """
		print 'run: ' + str(int(iw)+1)
		W_in = W_in_save[iw][0:nDimStates,:]
		hidNeurons_train = ex.propL1(images_train, W_in, SM=SM)
		hidNeurons_test = ex.propL1(images_test, W_in, SM=SM)

		""" train SVM """
		svm_repres = SVC(kernel="rbf", C=1000.0, gamma=0.25) #kernel="linear" #C=1000000000.0 #C=1000.0
		svm_repres.fit(hidNeurons_train, labels_train)

		""" test SVM """
		classResults = svm_repres.predict(hidNeurons_test)
		allPerf.append(float(np.sum(classResults==labels_test))/len(labels_test))
		allCMs.append(ex.computeCM(classResults, labels_test, classes))

	""" print and save performance measures """
	if save_data:
		print_save(allCMs, allPerf, classes, runName, verbose)
	return allCMs, allPerf

def neuronClass(runName, W_in_save, classes, RFproba, nDimStates, A, images_test, labels_test, save_data, verbose):
	"""
	evaluates the quality of a representation using the class of the most activated neuron as the classification result

	Args:
		runName (str) : name of the folder where to save results
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		RFproba (numpy array) : probability of that a RF belongs to a certain class (of the MNIST dataset)
		nDimStates (int) : number of dimensions of the states (size of images)
		A (int): normalization constant
		save_data (bool, optional) : whether save data
		verbose (bool, optional) : whether to display text ouput
	"""

	if verbose: print "\nassessing performance..."

	""" variable initialization """
	allCMs = []
	allPerf = []

	""" make image class prediction based on the class of most activated neuron """
	for i, iw in enumerate(sorted(W_in_save.keys())):
		""" load weight matrix and find most activated neuron """
		if verbose: print 'run: ' + str(int(iw)+1)
		W_in = W_in_save[iw][0:nDimStates,:]
		neuronC = np.argmax(RFproba[i],1) #class of each neuron
		argmaxActiv = np.argmax(ex.propL1(images_test, W_in, SM=False),1)
		classResults = neuronC[argmaxActiv]

		""" compute classification performance """
		allPerf.append(float(np.sum(classResults==labels_test))/len(labels_test))
		allCMs.append(ex.computeCM(classResults, labels_test, classes))

	""" print and save performance measures """
	if save_data:
		print_save(allCMs, allPerf, classes, runName, verbose)
	return allCMs, allPerf

def bayesian(W_in_save, images, labels, images_test, labels_test, kwargs, save_data=None, verbose=None):
	"""
	evaluates the performance of the newtork using a bayesian decoder

	Args:
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		images (numpy array): train images
		labels (numpy array): train labels
		images_test (numpy array): test images
		labels_test (numpy array): test labels
		kwargs (dict): parameters of the model
	"""

	runName = kwargs['runName']
	classes = kwargs['classes']
	rActions = kwargs['rActions']
	t_hid = kwargs['t_hid']
	pdf_method = kwargs['pdf_method']
	if verbose is None: verbose = kwargs['verbose']
	if save_data is None: save_data = kwargs['save_data']

	if verbose: print "\nassessing performance..."

	""" variable initialization """
	allCMs = []
	allPerf = []

	""" process labels for plot """
	rActions_uni, idx = np.unique(rActions, return_index=True)
	sorter = idx.argsort()
	rActions_uni = rActions_uni[sorter]
	labels_print = ex.actionVal2labels(rActions_uni)
	for i_l, l in enumerate(labels_print): labels_print[i_l] = re.sub("[^0-9 ]", "", str(l))
	labels_print = np.array(labels_print)
	labels_print[rActions_uni=='z'] = 'z'

	for iw in sorted(W_in_save.keys()):
		if verbose: print 'run: ' + str(int(iw)+1)
		W_in = W_in_save[iw]

		""" compute pdf """
		pdf_marginals, pdf_evidence, pdf_labels = bc.pdf_estimate(images, labels, W_in, kwargs)

		""" testing of the classifier """
		posterior = bc.bayesian_decoder(ex.propL1(images_test, W_in, t=t_hid), pdf_marginals, pdf_evidence, pdf_labels, pdf_method)
		classIdx = np.argmax(posterior, 1)
		classResults = rActions[classIdx]
		
		""" compute classification performance """
		correct_classif = float(np.sum(classResults==ex.labels2actionVal(labels_test)))
		allPerf.append(correct_classif/len(labels_test))
		CM = ex.computeCM(classResults, ex.labels2actionVal(labels_test), np.unique(rActions))
		CM = CM[sorter,:]
		CM = CM[:,sorter]
		allCMs.append(CM)

	""" print and save performance measures """
	if verbose:
		print_save(allCMs, allPerf, rActions_uni, runName, verbose)
	return allCMs, allPerf

def print_save(allCMs, allPerf, classes, runName, verbose):
	""" print and save performance measures """
	avgCM = np.mean(allCMs,0)
	steCM = np.std(allCMs,0)/np.sqrt(np.shape(allCMs)[0])
	avgPerf = np.mean(allPerf)
	stePerf = np.std(allPerf)/np.sqrt(len(allPerf))

	pFile = open('output/' + runName + '/classResults', 'w')
	pDict = {'allCMs':allCMs, 'avgCM':avgCM, 'steCM':steCM, 'allPerf':allPerf, 'avgPerf':avgPerf, 'stePerf':stePerf}
	pickle.dump(pDict, pFile)
	pFile.close()

	perf_print = ''
	perf_print += '\naverage confusion matrix:' + '\n'
	c_str = ''
	for c in classes: c_str += str(c).rjust(6)
	perf_print += c_str + '\n'
	perf_print += '-'*(len(c_str)+3) + '\n'
	perf_print += str(np.round(avgCM,2)) + '\n'
	perf_print += '\naverage correct classification:' + '\n'
	perf_print += str(np.round(100*avgPerf,2)) + ' +/- ' + str(np.round(100*stePerf,2)) + ' %' + '\n'
	if len(allPerf)>1:
		perf_print += '\nof which best performance is:' + '\n'
		perf_print += str(np.round(100*(np.max(allPerf)),2)) + '%' + ' (run ' + str(np.argmax(allPerf)) + ')' + '\n'
		perf_print += 'and worse performance is:' + '\n'
		perf_print += str(np.round(100*(np.min(allPerf)),2)) + '%' + ' (run ' + str(np.argmin(allPerf)) + ')' + '\n'

	if verbose: print perf_print

	perf_file = open('./output/' + runName + '/' +runName+ '_perf.txt', 'w')
	perf_file.write(perf_print)
	perf_file.close()

	fig = pl.plotCM(avgCM, classes)
	pyplot.savefig('./output/' + runName + '/' +runName+ '_avgCM.pdf')
	pyplot.close(fig)















