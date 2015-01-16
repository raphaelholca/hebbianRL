""" Hebbian learning-based classifier used to assess the performance of the representation """

import numpy as np
import matplotlib.pyplot as pyplot
import support.mnist as mnist
import support.external as ex
import support.plots as pl
import pickle
from sklearn.svm import SVC
ex = reload(ex)
pl = reload(pl)

def neural(runName, W_in_save, W_class_save, classes, rActions, nHidNeurons, nDimStates, A, train_dataset, show=True):
	"""
	evaluates the quality of a representation using a neural classifier

	Args:
		runName (str) : name of the folder where to save results
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		W_class_save (numpy array) : weight matrix from hidden to classification layer; shape = (hidden x class)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		images (numpy array): image to normalize
		rActions (numpy array of str): reward actions associated with each of the classes of MNIST
		nHidNeurons (int): number of hidden neurons
		nDimStates (int) : number of dimensions of the states (size of images)
		A (int): normalization constant
		show (bool, optional) : whether to display the confusion matrix (True) or not (False)
	"""

	""" load and pre-process images """
	print "assessing performance with neural classifier..."
	if  train_dataset=='train': test_dataset='test'
	else: test_dataset='test'  #'train' ##
	images, labels = mnist.read_images_from_mnist(classes=classes, dataset=test_dataset)
	images = ex.normalize(images, A*nDimStates)
	images, labels = ex.evenLabels(images, labels, classes)

	""" variable initialization """
	allCMs = []
	allPerf = []

	for iw in sorted(W_in_save.keys()):
		print 'run: ' + str(int(iw)+1)
		W_in = W_in_save[iw][0:nDimStates,:]
		W_class = W_class_save[iw]

		""" testing of the classifier """
		hidNeurons = ex.propL1(images, W_in)
		classNeurons = ex.propL2_class(hidNeurons, W_class)
		classIdx = np.argmax(classNeurons, 1)
		classResults = classes[classIdx]
		
		""" compute classification performance """
		allPerf.append(float(np.sum(classResults==labels))/len(labels))
		allCMs.append(ex.computeCM(classResults, labels, classes))

	""" print and save performance measures """
	print_save(allCMs, allPerf, classes, runName, show)

def SVM(runName, W_in_save, images_train, labels_train, classes, nDimStates, A, train_dataset, show=True, SM=False):
	"""
	evaluates the quality of a representation using an SVM

	Args:
		runName (str) : name of the folder where to save results
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		images_train (numpy array): training image of the MNIST dataset
		labels_train (numpy array): training labels of the MNIST dataset
		classes (numpy array): all classes of the MNIST dataset used in the current run
		nDimStates (int) : number of dimensions of the states (size of images)
		A (int): normalization constant
		train_dataset (str): name of the dataset used for learning the weights ('train' or 'test')
		show (bool, optional): whether to display the confusion matrix (True) or not (False)
		SM (bool, optional): whether to pass the activation throught the Softmax function
	"""

	""" load and pre-process images """
	print 'assessing performance with SVM...'
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
	print_save(allCMs, allPerf, classes, runName, show)

def neuronClass(runName, W_in_save, classes, RFproba, nDimStates, A, show=True):
	"""
	evaluates the quality of a representation using the class of the most activated neuron as the classification result

	Args:
		runName (str) : name of the folder where to save results
		W_in_save (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		RFproba (numpy array) : probability of that a RF belongs to a certain class (of the MNIST dataset)
		nDimStates (int) : number of dimensions of the states (size of images)
		A (int): normalization constant
		show (bool, optional) : whether to display the confusion matrix (True) or not (False)
	"""

	""" load and pre-process images """
	print "assessing performance with neuron class..."
	imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
	images, labels = mnist.read_images_from_mnist(classes=classes, dataset='test', path=imPath)
	images = ex.normalize(images, A*nDimStates)
	images, labels = ex.evenLabels(images, labels, classes)

	""" variable initialization """
	allCMs = []
	allPerf = []

	""" make image class prediction based on the class of most activated neuron """
	for i, iw in enumerate(sorted(W_in_save.keys())):
		""" load weight matrix and find most activated neuron """
		print 'run: ' + str(int(iw)+1)
		W_in = W_in_save[iw][0:nDimStates,:]
		neuronC = np.argmax(RFproba[i],1)
		argmaxActiv = np.argmax(ex.propL1(images, W_in, SM=False),1)
		classResults = neuronC[argmaxActiv]

		""" compute classification performance """
		allPerf.append(float(np.sum(classResults==labels))/len(labels))
		allCMs.append(ex.computeCM(classResults, labels, classes))

	""" print and save performance measures """
	print_save(allCMs, allPerf, classes, runName, show)

def print_save(allCMs, allPerf, classes, runName, show):
	""" print and save performance measures """
	avgCM = np.mean(allCMs,0)
	steCM = np.std(allCMs,0)/np.sqrt(np.shape(allCMs)[0])
	avgPerf = np.mean(allPerf)
	stePerf = np.std(allPerf)/np.sqrt(len(allPerf))

	pFile = open('output/' + runName + '/classResults', 'w')
	pDict = {'allCMs':allCMs, 'avgCM':avgCM, 'steCM':steCM, 'allPerf':allPerf, 'avgPerf':avgPerf, 'stePerf':stePerf}
	pickle.dump(pDict, pFile)
	pFile.close()

	print '\naverage confusion matrix:'
	c_str = ''
	for c in classes: c_str += ' '*5 + str(c)
	print c_str
	print '-'*(len(c_str)+3)
	print np.round(avgCM,2)
	print '\naverage correct classification:'
	print str(np.round(100*avgPerf,1)) + ' +/- ' + str(np.round(100*stePerf,1)) + '%'

	fig = pl.plotCM(avgCM, classes)
	pyplot.savefig('./output/' + runName + '/' +runName+ '_avgCM.png')
	if show:
		pyplot.show(block=False)
	else:
		pyplot.close(fig)















