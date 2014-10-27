""" routine to classify and build a histogram of the distribution of the classes of the weights (RFs) of the representation  """

def hist(runName, W, classes, show=True):
	"""
	computes the class of the weight of each neuron using a SVM (i.e., classify the weight matrix according to an SVM trained on the MNIST dataset)

	Args:
		runName (str) : name of the folder where to save results
		W (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
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
		RFproba.append(np.round(svm_mnist.predict_proba(W[r].T),2))
		RFclass[i,:], _ = np.histogram(np.argmax(RFproba[i],1), bins=nClasses, range=(-0.5,9.5))

	RFclass_mean = np.mean(RFclass, 0)
	RFclass_ste = np.std(RFclass, 0)/np.sqrt(np.size(RFclass,0))

	pRFclass = {'RFclass_all':RFclass, 'RFclass_mean':RFclass_mean, 'RFclass_ste':RFclass_ste}

	pfile = open('output/'+runName+'/RFclass', 'w')
	pickle.dump(pRFclass, pfile)
	pfile.close()

	fig = pl.plotHist(RFclass_mean[classes], classes, h_err=RFclass_ste)
	pyplot.savefig('./output/'+runName+'/RFhist.png')
	if show:
		pyplot.show(block=False)
	else:
		pyplot.close(fig)

def sharp():
	#TODO
	return

















