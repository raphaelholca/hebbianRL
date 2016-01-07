""" routine to classify and build a histogram of the distribution of the classes of the weights (RFs) of the representation  """

import numpy as np
import plots as pl
import external as ex
import matplotlib.pyplot as pyplot
import support.mnist as mnist
import pickle
pl = reload(pl)
ex = reload(ex)

def hist(net, W, classes, images, labels, protocol, n_bins=10, SVM=True, save_data=True, verbose=True, lr_ratio=1.0, rel_classes=np.array([False])):
	"""
	computes the class of the weight (RF) of each neuron. Can be used to compute the selectivity index of a neuron: use SVM=False and lr_ratio=1.0. Selectivity is measured as # of preferred stimulus example that activate the neuron / # all stimulus example that activate the neuron

	Args:
		W (numpy array) : weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		images (numpy array) : images of the MNIST dataset used for training
		labels (numpy array) : labels corresponding to the images of the MNIST dataset
		n_bins (int, optional): number of bins in the histogram
		SVM (bool, optional) : whether to compute the class of the weight of each neuron using an SVM (i.e., classify the weight matrix according to an SVM trained on the MNIST dataset) (True) or based on the number of example of each class that activates a neuron (a weight is classified as a '9' if '9' is the most frequent class to activate the neuron) (False) - SVM = False will not work with ACh signaling
		save_data (bool, optional) : whether save data
		verbose (bool, optional) : whether to display text ouput
		lr_ratio (float, optional) : the ratio between ach signal and normal learning rate
		rel_classes (numpy array, optional) : the classes relevant in the training protocol (i.e., those not equal to '0')

	return:
		RFproba (numpy array) : probability that a each RF belongs to a certain class. For SVM=True, this probability is computed by predict_proba of scikit-learn. For SVM=False, the probability is computed as the # of stimuli from a digit class that activate the neuron / total # of stimuli that activate the neuron (shape= nRun x n_hid_neurons x 10). This can be used to compute the selectivity index of a neuron (when SVM=False abd lr_ratio=1.0) by taking np.max(RFproba,2)
		RFclass (numpy array) : count of weights/RFs responsive of each digit class (shape= nRun x 10)
		RFselec (numpy array) : mean selectivity index for all RFs of a digit class. Computed as the mean of RFproba for each class
	"""

	if verbose: print "\ncomputing RF classes..."
	nRun = len(W.keys())
	nNeurons = np.size(W['000'],1)

	if SVM:
		#load classifier from file; parameters of the model from:
		#http://peekaboo-vision.blogspot.co.uk/2010/09/mnist-for-ever.html
		#svm_mnist = SVC(kernel="rbf", C=2.8, gamma=.0073, probability=True, verbose=True)
		pfile = open('support/SVM-MNIST-proba', 'r')
		svm_mnist = pickle.load(pfile)
		pfile.close()

	RFproba = np.zeros((nRun,nNeurons,n_bins))
	RFclass = np.zeros((nRun,n_bins))
	RFselec = np.zeros((nRun,n_bins))
	for i,r in enumerate(sorted(W.keys())):
		if verbose: print 'run: ' + str(i+1)
		if SVM:
			RFproba[r,:,:] = np.round(svm_mnist.predict_proba(W[r].T),2)
		else:
			mostActiv = np.argmax(ex.propL1(images, W[r]),1)
			for n in range(nNeurons):
				RFproba[int(r),n,:] = np.histogram(labels[mostActiv==n], bins=n_bins, range=(-0.5,9.5))[0]
				RFproba[int(r),n,rel_classes] *= lr_ratio #to balance the effect of ACh
				RFproba[int(r),n,:]/= np.sum(RFproba[int(r),n,:])+1e-20 #+1e-20 to avoid divide zero error
		RFclass[i,:], _ = np.histogram(np.argmax(RFproba[i],1), bins=n_bins, range=(-0.5,9.5))
		for c in range(n_bins):
			RFselec[i,c] = np.mean(np.max(RFproba[i],1)[np.argmax(RFproba[i],1)==c])

	RFclass_mean = np.mean(RFclass, 0)
	RFclass_ste = np.std(RFclass, 0)/np.sqrt(np.size(RFclass,0))

	pRFclass = {'RFproba':RFproba, 'RFclass_all':RFclass, 'RFclass_mean':RFclass_mean, 'RFclass_ste':RFclass_ste, 'RFselec':RFselec}

	if save_data:
		pfile = open('output/'+ net.name +'/RFclass', 'w')
		pickle.dump(pRFclass, pfile)
		pfile.close()

		if protocol=='digit':
			bin_names = classes
		elif protocol=='gabor':
			bin_size = 180./n_bins
			bin_names = np.zeros(n_bins, dtype='|S3')
			for i in range(n_bins):
				bin_names[i] = str(int(bin_size*i + bin_size/2.))
		fig = pl.plotHist(RFclass_mean[classes], bin_names, h_err=RFclass_ste[classes])
		pyplot.savefig('./output/'+net.name+'/' +net.name+ '_RFhist.pdf')
		pyplot.close(fig)

	return RFproba, RFclass, RFselec

def plot(net, W, RFproba, target=None, W_act=None, sort=None, notsame=None, verbose=True):
	if verbose: print "\nploting RFs..."

	if sort=='tSNE':
		if np.mod(np.sqrt(np.size(W['000'],1)),1)!=0:
			print '!! number of neurons not square; using class sorting for display !!'
			sort='class'

	for i,r in enumerate(sorted(W.keys())):
		W_sort = np.copy(W[r])
		if verbose: print 'run: ' + str(i+1)
		if sort=='class': #sort weights according to their class
			RFclass = np.argmax(RFproba[i],1)
			sort_idx = np.array([x for (y,x) in sorted(zip(RFclass, np.arange(len(RFclass))), key=lambda pair: pair[0])])
			W_sort = W[r][:,sort_idx] 
			RFproba[i] = np.array([x for (y,x) in sorted(zip(RFclass, RFproba[i]), key=lambda pair: pair[0])])
		elif sort=='tSNE':
			W_sort = tSNE_sort(W_sort)
		target_pass=None
		if target:
			T_idx = np.argwhere(np.argmax(RFproba[i],1)==target[r])
			target_pass = np.zeros((np.size(W_sort,0),1,1))
			target_pass[T_idx,:,:]=1.0
		W_act_pass=None
		if W_act:
			W_act_pass = W_act[r]
		if notsame:
			notsame_pass = notsame[r]
		else: 
			notsame_pass = np.array([])

		fig = pl.plotRF(W_sort, target=target_pass, W_act=W_act_pass, notsame=notsame_pass)
		pyplot.savefig('output/' + net.name + '/RFs/' +net.name+ '_' + str(r).zfill(3)+'.png')
		pyplot.close(fig)

def clockwise(r):
	return list(r[0]) + clockwise(list(reversed(zip(*r[1:])))) if r else []

def tSNE_sort(W):
	"""
	sorts weight using t-Distributed Stochastic Neighbor Embedding (t-SNE)

	Args:
		W (numpy array) : weights from input to hidden neurons

	returns:
		(numpy array) : sorted weights
	"""
	import tsne
	import numpy.ma as ma

	Y = tsne.tsne(W.T, 2 , 50).T
	side = int(np.sqrt(np.size(W,1)))
	n_hid_neurons = np.size(W,1)

	n0_min, n1_min = np.min(Y,1)
	n0_max, n1_max = np.max(Y,1)
	n0_grid = np.hstack(np.linspace(n0_min, n0_max, side)[:,np.newaxis]*np.ones((side,side)))
	n1_grid = np.hstack(np.linspace(n1_max, n1_min, side)[np.newaxis,:]*np.ones((side,side)))

	sorted_idx = np.zeros(n_hid_neurons, dtype=int)
	mask = np.zeros(n_hid_neurons, dtype=bool)

	iterate_clockwise = map(int, clockwise(list(np.reshape(np.linspace(0,n_hid_neurons-1,n_hid_neurons),(side,side)))))

	for i in iterate_clockwise:
		c_point = np.array([n0_grid[i], n1_grid[i]])[:,np.newaxis]
		dist = ma.array(np.sum((Y-c_point)**2,0), mask=mask)
		min_idx = np.argmin(dist)
		sorted_idx[i] = min_idx
		mask[min_idx] = True

	return W[:,sorted_idx]

def selectivity(W, RFproba, images, labels, classes):
	"""
	computes the selectivity of a neuron, using the already computed RFproba. This RFproba must have been computed using hist() with SVM=False and lr_ratio=1.0

	Args:
		W (numpy array) : weights from input to hidden neurons
		RFproba (numpy array) : 
	"""
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

















