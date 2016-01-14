""" Support functions to assess network; functions to plot receptive fields, histogram of the distribution of the classes of the weights (RFs) of the representation, etc.  """

import numpy as np
import external as ex
import grating as gr 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import helper.mnist as mnist
import pickle
ex = reload(ex)
gr = reload(gr)


def hist(name, W, classes, images, labels, n_bins=10, SVM=False, save_data=True, verbose=True):
	"""
	computes the class of the weight (RF) of each neuron. Can be used to compute the selectivity index of a neuron: use SVM=False. Selectivity is measured as # of preferred stimulus example that activate the neuron / # all stimulus example that activate the neuron

	Args:
		name (str): name of the network, used to save to file
		W (numpy array): weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		images (numpy array): images of the MNIST dataset used for training
		labels (numpy array): labels corresponding to the images of the MNIST dataset
		n_bins (int, optional): number of bins in the histogram
		SVM (bool, optional): whether to compute the class of the weight of each neuron using an SVM (i.e., classify the weight matrix according to an SVM trained on the MNIST dataset) (True) or based on the number of example of each class that activates a neuron (a weight is classified as a '9' if '9' is the most frequent class to activate the neuron) (False) - SVM = False will not work with ACh signaling
		save_data (bool, optional): whether save data
		verbose (bool, optional): whether to display text ouput

	return:
		RFproba (numpy array): probability that a each RF belongs to a certain class. For SVM=True, this probability is computed by predict_proba of scikit-learn. For SVM=False, the probability is computed as the # of stimuli from a digit class that activate the neuron / total # of stimuli that activate the neuron (shape= nRun x n_hid_neurons x 10). This can be used to compute the selectivity index of a neuron by taking np.max(RFproba,2)
		RFclass (numpy array): count of neurons responsive of each digit class (shape= nRun x 10)
		RFselec (numpy array): mean selectivity index for all RFs of a digit class. Computed as the mean of RFproba for each class
	"""

	if verbose: print "\ncomputing RF classes..."
	nRun = len(W)
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
			mostActiv = np.argmax(ex.propagate_layerwise(images, W[r]),1)
			for n in range(nNeurons):
				RFproba[int(r),n,:] = np.histogram(labels[mostActiv==n], bins=n_bins, range=(-0.5,9.5))[0]
				RFproba[int(r),n,:]/= np.sum(RFproba[int(r),n,:])+1e-20 #+1e-20 to avoid divide zero error
		RFclass[i,:], _ = np.histogram(np.argmax(RFproba[i],1), bins=n_bins, range=(-0.5,9.5))
		for c in range(n_bins):
			RFselec[i,c] = np.mean(np.max(RFproba[i],1)[np.argmax(RFproba[i],1)==c])

	RFclass_mean = np.mean(RFclass, 0)
	RFclass_ste = np.std(RFclass, 0)/np.sqrt(np.size(RFclass,0))

	pRFclass = {'RFproba':RFproba, 'RFclass_all':RFclass, 'RFclass_mean':RFclass_mean, 'RFclass_ste':RFclass_ste, 'RFselec':RFselec}

	if save_data:
		pfile = open('output/'+ name +'/RFclass', 'w')
		pickle.dump(pRFclass, pfile)
		pfile.close()

		bin_names = classes
		fig = plot_hist(RFclass_mean[classes], bin_names, h_err=RFclass_ste[classes])
		plt.savefig('./output/'+name+'/' +name+ '_RFhist.pdf')
		plt.close(fig)

	return RFproba, RFclass, RFselec

def hist_gabor(n_bins, name, hid_W_runs, t, target_ori, save_data, verbose):
	""" Computes the distribution of orientation preference of neurons in the network. """
	pref_ori = gr.preferred_orientations(hid_W_runs, t, target_ori, name)
	RFproba = np.zeros((len(hid_W_runs), np.size(hid_W_runs['000'],0), 2), dtype=int)
	for r in pref_ori.keys():
		RFproba[int(r),:,:][pref_ori[r]<=target_ori] = [1,0]
		RFproba[int(r),:,:][pref_ori[r]>target_ori] = [0,1]

	h_all = np.zeros((len(pref_ori.keys()), n_bins))
	for r in pref_ori.keys():
		h_all[int(r), :] = np.histogram(pref_ori[r], n_bins, range=(0.,180.))[0]
	h_mean = np.mean(h_all,0)
	h_ste = np.std(h_all,0)/np.sqrt(len(pref_ori.keys()))

	if save_data:
		bin_size = 180./n_bins
		bin_names = np.zeros(n_bins, dtype='|S3')
		for i in range(n_bins):
			bin_names[i] = str(int(bin_size*i + bin_size/2.))

		fig = plot_hist(h_mean, bin_names, h_err=h_ste)
		plt.savefig('./output/'+name+'/' +name+ '_RFhist.pdf')
		plt.close(fig)

	return RFproba

def selectivity(W, RFproba, images, labels, classes):
	"""
	computes the selectivity of a neuron, using the already computed RFproba. This RFproba must have been computed using hist() with SVM=False and lr_ratio=1.0

	Args:
		W (numpy array) : weights from input to hidden neurons
		RFproba (numpy array) : 
	"""
	acti = ex.propagate_layerwise(images, W, SM=False)
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

""" initialize color maps """
n_colors = 9
my_blues = [plt.get_cmap('YlGnBu')(1.*i/n_colors) for i in range(n_colors)]
my_reds = [plt.get_cmap('YlOrRd')(1.*i/n_colors) for i in range(n_colors)]
cm_pastel = [plt.get_cmap('Paired')(1.*i/n_colors) for i in range(n_colors)]

def plot_all_RF(name, W, RFproba, target=None, W_act=None, sort=None, not_same=None, verbose=True):
	""" Plot the RFs of neurons """
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
		if not_same:
			notsame_pass = not_same[r]
		else: 
			notsame_pass = np.array([])

		fig = plot_single_RF(W_sort, target=target_pass, W_act=W_act_pass, not_same=notsame_pass)
		plt.savefig('output/' + name + '/RFs/' + name+ '_' + str(r).zfill(3)+'.png')
		plt.close(fig)

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

def plot_single_RF(W, target=None, W_act=None, cmap='Greys', not_same=np.array([])):
	""" plots of the weights, with superimposed colouring for target digit and out layer weights """
	
	#plot parameters
	n_hid_neurons = np.size(W,1)
	v = int(np.sqrt(n_hid_neurons))
	h = int(np.ceil(float(n_hid_neurons)/v))
	Wmin = np.min(W)
	Wmax = np.max(W)

	#create a transparent colormap
	cmap_trans = plt.get_cmap('binary') 
	cmap_trans._init()
	alphas = np.linspace(0., 1., cmap_trans.N+3)
	cmap_trans._lut[:,-1] = alphas

	im_size = int(np.sqrt(np.size(W,0)))

	#plot figure
	fig, ax = plt.subplots(figsize=(h,v))
	for i in range(np.size(W,1)):
		plt.subplot(v,h,i+1)
		if type(target)!=type(None) and target[i]!=0:
			plt.imshow(target[i], cmap=cmap, vmin=0., vmax=3, extent=(0,im_size,0,im_size))
		plt.imshow(np.reshape(W[:,i], (im_size,im_size)), interpolation='nearest', cmap=cmap_trans, extent=(0,im_size,0,im_size), vmin=Wmin)
		if type(W_act)!=type(None):
			if i in not_same:
				plt.imshow([[0]], cmap='RdYlBu', vmin=0., vmax=3, extent=(im_size,im_size+2,0,im_size))
			plt.imshow(W_act[i,:][:,np.newaxis], interpolation='nearest', cmap='binary', extent=(im_size,im_size+2,0,im_size))
			plt.imshow([[0.]], interpolation='nearest', cmap='binary', alpha=0, extent=(0,im_size+2,0,im_size))
		plt.xticks([])
		plt.yticks([])

	#plot parameters
	fig.patch.set_facecolor('white')
	plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)

	return fig

def plot_CM(confusMatrix, classes):
	""" plots the confusion matrix, with color on the diagonal, and with the alphas indicating the magnitude of the error """

	#create a transparent colormap
	nClasses = len(classes)
	cmap_trans = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','white'],256) 
	cmap_trans._init()
	alphas = np.linspace(1.0, 0, cmap_trans.N+3)
	cmap_trans._lut[:,-1] = alphas

	#creates the background color matrix
	colorMatrix = np.ones_like(confusMatrix)
	np.fill_diagonal(colorMatrix, -1.0)

	#plot the matrix and number values
	sH = 1.0+0.5*nClasses
	sV = 0.9+0.5*nClasses
	fig, ax = plt.subplots(figsize=(sH,sV))
	ax.imshow(colorMatrix, interpolation='nearest', cmap='RdYlGn_r', vmin=-1.2, vmax=1.2)
	ax.imshow(confusMatrix, interpolation='nearest', cmap=cmap_trans, vmin=-0.0, vmax=1)
	for i in range(nClasses):
		for j in range(nClasses):
			perc = int(np.round(confusMatrix[i,j],2)*100)
			ax.annotate(perc, xy=(0, 0),  xycoords='data', xytext=(j, i), textcoords='data', size=15, ha='center', va='center')

	#plot parameters
	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xticks(np.arange(nClasses))
	ax.set_yticks(np.arange(nClasses))
	ax.set_xticklabels(classes, fontsize=18)
	ax.set_yticklabels(classes, fontsize=18)
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_xlabel('prediction', fontsize=17)
	ax.set_ylabel('label', fontsize=18)
	plt.tight_layout()
	# fig.subplots_adjust(top=0, bottom=0, right=1, left=0)

	return fig

def plot_hist(h, bins, h_err=None):
	""" plots the histogram of receptive field class distribution """

	fig, ax = plt.subplots(figsize=(1+0.5*len(h),3))
	Xs = np.arange(len(h))
	y_max = np.ceil(np.sum(h))
	ax.bar(Xs, h, yerr=h_err, color=my_blues[6], ecolor=my_blues[7])

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xticks(Xs+0.5)
	ax.set_xticklabels(bins)
	s = np.where(y_max>4, 2,1)
	ax.set_yticks(np.arange(y_max+1, step=s))
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_xlabel('class', fontsize=18)
	ax.set_ylabel('neuron count', fontsize=18)
	plt.tight_layout()

	return fig

def perf_progress(net, perf, nEpi=None):
	"""
	plots the progression of the error rate over training episodes
	"""

	fig, ax = plt.subplots()
	plt.gca().set_color_cycle(cm.Paired(i) for i in np.linspace(0,0.9,10))

	for r in perf.keys():
		X = np.arange( len(perf[r][net.n_epi_crit:]) )+1
		ax.plot(X, perf[r][net.n_epi_crit:]*100, lw=3)

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.set_ylim([50,100])
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_xlabel('episodes after dopa', fontsize=18)
	ax.set_ylabel('% correct', fontsize=18)
	plt.tight_layout()

	plt.savefig('output/' + net.name + '/' + net.name+ '_progress.pdf')
	plt.close(fig)	















