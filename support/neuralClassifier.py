""" Hebbian learning-based classifier used to assess the performance of the representation """

def performance(runName, W_in_save, W_class_save, classes, rActions, nHidNeurons, nDimStates, A, show=True):
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
	print "assessing performance..."
	imPath = '../data-sets/MNIST'
	images, labels = mnist.read_images_from_mnist(classes=classes, dataset='test', path=imPath)
	images = ex.normalize(images, A)
	images, labels = ex.evenLabels(images, labels, classes)

	""" variable initialization """
	nClasses = len(classes)
	allCMs = []
	allPerf = []

	""" training of the neural classifier """
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

	""" print and save """
	avgCM = np.mean(allCMs,0)
	steCM = np.std(allCMs,0)/np.sqrt(np.shape(allCMs)[0])
	avgPerf = np.mean(allPerf)
	stePerf = np.std(allPerf)/np.sqrt(len(allPerf))

	pFile = open('output/' + runName + '/classResults', 'w')
	pDict = {'allCMs':allCMs, 'avgCM':avgCM, 'steCM':steCM, 'allPerf':allPerf, 'avgPerf':avgPerf, 'stePerf':stePerf}
	pickle.dump(pDict, pFile)
	pFile.close()

	print '\naverage confusion matrix:'
	print '   0     1     2     3     4     5     6     7     8     9    '
	print '--------------------------------------------------------------'
	print np.round(avgCM,2)
	print '\naverage correct classification:'
	print str(np.round(100*avgPerf,1)) + ' +/- ' + str(np.round(100*stePerf,1)) + '%'

	fig = pl.plotCM(avgCM, classes)
	pyplot.savefig('./output/' + runName + '/avgCM.png')
	if show:
		pyplot.show(block=False)
	else:
		pyplot.close(fig)






















