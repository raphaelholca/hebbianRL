""" routine to classify and build a histogram of the distribution of the classes of the weights (RFs) of the representation  """

import numpy as np
import plots as pl
import external as ex
import matplotlib.pyplot as pyplot
import pickle
pl = reload(pl)
ex = reload(ex)

def hist(runName, W, classes, nDimStates, proba=False, show=True):
	"""
	computes the class of the weight of each neuron using a SVM (i.e., classify the weight matrix according to an SVM trained on the MNIST dataset)

	Args:
		runName (str) : name of the folder where to save results
		W (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		nDimStates (int) : number of dimensions of the states (size of images)
		proba (bool, optional) : whether to compute RF class histogram as the sum of the class probability or as the sum of the argmax of the class (winner-take-all)
		show (bool, optional) : whether to the histogram of the weight class distribution (True) or not (False)
	"""

	nClasses = 10
	nRun = len(W.keys())

	#load classifier from file; parameters of the model from:
	#http://peekaboo-vision.blogspot.co.uk/2010/09/mnist-for-ever.html
	#svm_mnist = SVC(kernel="rbf", C=2.8, gamma=.0073, probability=True, verbose=True)
	print "computing RF classes..."
	pfile = open('support/SVM-MNIST-proba', 'r')
	svm_mnist = pickle.load(pfile)
	pfile.close()

	RFproba = []
	perf = np.zeros((nRun,nClasses))
	RFsharp = np.zeros((nRun,nClasses))
	RFclass = np.zeros((nRun,nClasses))
	for i,r in enumerate(sorted(W.keys())):
		print 'run: ' + str(i+1)
		RFproba.append(np.round(svm_mnist.predict_proba(W[r][:nDimStates,:].T),2))
		if proba:
			RFclass[i,:] = np.sum(RFproba[i],0)
		else:
			RFclass[i,:], _ = np.histogram(np.argmax(RFproba[i],1), bins=nClasses, range=(-0.5,9.5))

	RFclass_mean = np.mean(RFclass, 0)
	RFclass_ste = np.std(RFclass, 0)/np.sqrt(np.size(RFclass,0))

	pRFclass = {'RFproba':RFproba, 'RFclass_all':RFclass, 'RFclass_mean':RFclass_mean, 'RFclass_ste':RFclass_ste}

	pfile = open('output/'+runName+'/RFclass', 'w')
	pickle.dump(pRFclass, pfile)
	pfile.close()

	fig = pl.plotHist(RFclass_mean[classes], classes, h_err=RFclass_ste[classes])
	pyplot.savefig('./output/'+runName+'/' +runName+ '_RFhist.png')
	if show:
		pyplot.show(block=False)
	else:
		pyplot.close(fig)

	return RFproba, RFclass

def plot(runName, W, RFproba, target=None, W_act=None, sort=False):
	print "ploting RFs..."
	for i,r in enumerate(sorted(W.keys())):
		print 'run: ' + str(i+1)
		if sort: #sort weights according to their class
			RFclass = np.argmax(RFproba[i],1)
			sort_idx = np.array([x for (y,x) in sorted(zip(RFclass, np.arange(len(RFclass))), key=lambda pair: pair[0])])
			W[r] = W[r][:,sort_idx] 
			RFproba[i] 	= np.array([x for (y,x) in sorted(zip(RFclass, RFproba[i]), key=lambda pair: pair[0])])
		target_pass=None
		if target:
			T_idx = np.argwhere(np.argmax(RFproba[i],1)==target)
			target_pass = np.zeros((np.size(W[r],0),1,1))
			target_pass[T_idx,:,:]=1.0
		W_act_pass=None
		if W_act:
			W_act_pass = W_act[r]
		fig = pl.plotRF(W[r], target=target_pass, W_act=W_act_pass)
		pyplot.savefig('output/' + runName + '/RFs/' +runName+ '_' + str(r).zfill(3))
		pyplot.close(fig)

def selectivity(W, RFproba, images, labels, classes):
	
	acti = ex.propL1(images, W, SM=False)
	nNeurons = np.size(acti,1)
	nClasses = len(classes)
	best_neuron = np.argmax(acti, 1)
	RFclass = np.argmax(RFproba,1)
	select_neuron = np.zeros(nNeurons)
	select_class = np.zeros(nClasses)
	for n in range(nNeurons):
		all_acti, _ = np.histogram(labels[best_neuron==n], bins=10, range=(0,9))
		select_neuron[n] = float(all_acti[RFclass[n]])/np.sum(all_acti)
	for i, c in enumerate(classes):
		select_class[i] = np.mean(select_neuron[RFclass==c])

	return select_class, select_neuron

















