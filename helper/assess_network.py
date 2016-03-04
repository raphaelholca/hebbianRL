""" Support functions to assess network; functions to plot receptive fields, histogram of the distribution of the classes of the weights (RFs) of the representation, etc.  """

import os
import numpy as np
import external as ex
import grating as gr 
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import pickle

ex = reload(ex)
gr = reload(gr)

def assess(net, curve_method='with_noise', slope_binned=False, show_W_act=True, sort=None, target=None, save_path=''):
	"""
	Assess network properties: plot weights, compute weight distribution, compute tuning curves, save data, etc.

	Args:
		net (Network object): network object to assess
		curve_method (str, optional): method to use to compute the tuning curves
		slope_binned (bool, optional): whether to bin slopes when plotting the difference before and after training
		save_data (bool, optional): whether to save data to disk. Default: True
		show_W_act (bool, optional): whether to display out_W weights on the weight plots. Default:True
		sort (str, optional): sorting methods for weights when displaying. Valid value: None, 'class', 'tSNE'. Default: None
		target (int, optional): target digit (to be used to color plots). Use None if not desired. Default: None
		save_path (str, optional): path where to save data
	"""

	""" create saving directory """
	if save_path=='': save_path=os.path.join('.', 'output', net.name)
	if not os.path.exists(save_path):
		os.makedirs(save_path)
	if not os.path.exists(os.path.join(save_path, 'RFs')):
		os.makedirs(os.path.join(save_path, 'RFs'))
	if net.protocol=='gabor' and not os.path.exists(os.path.join(save_path, 'TCs')):
		os.makedirs(os.path.join(save_path, 'TCs'))
	RFproba = net.RF_info['RFproba']

	""" plot and save confusion matrices """
	print_save_CM(net.perf_dict, net.name, net.classes, net.verbose, True, save_path)

	""" plot RF properties """
	plot_RF_info(net, save_path, curve_method=curve_method, slope_binned=slope_binned)		

	""" compute correct weight assignment in the ouput layer """
	if net._train_class_layer:
		net.correct_out_W = np.zeros(net.n_runs)
		not_same = {}
		for r in range(net.n_runs):
			same = np.argmax(RFproba[r],1) == net.classes[np.argmax(net.out_W_trained[r],1)]
			not_same[r] = np.argwhere(~same)
			net.correct_out_W[r] = np.sum(same)

		if net.verbose: 
			print 'correct out weight assignment:\n' + str(np.mean(net.correct_out_W)) + ' of ' + str(net.n_hid_neurons)
	else:
		not_same = None
		correct_out_W = 0.

	""" plot weights """
	if show_W_act: W_act_pass=net.out_W_trained
	else: W_act_pass=None
	plot_all_RF(net.name, net.hid_W_trained, RFproba, target=target, W_act=W_act_pass, sort=sort, not_same=not_same, verbose=net.verbose, save_path=save_path)	
	
	""" plot performance progression """
	plot_perf_progress(net.name, net.perf_train_prog, net.perf_test_prog, net.n_epi_crit, epi_start=0, save_path=save_path)

	""" plot performance at various orientations """
	if net.protocol=='gabor': perf_all_ori(net, save_path=save_path)

def hist(name, W, classes, images, labels, n_bins=10, save_data=True, verbose=True, save_path=''):
	"""
	computes the class of the weight (RF) of each neuron. Can be used to compute the selectivity index of a neuron. Selectivity is measured as # of preferred stimulus example that activate the neuron / # all stimulus example that activate the neuron

	Args:
		name (str): name of the network, used to save to file
		W (numpy array): weight matrix from input to hidden layer; shape = (input x hidden)
		classes (numpy array): all classes of the MNIST dataset used in the current run
		images (numpy array): images of the MNIST dataset used for training
		labels (numpy array): labels corresponding to the images of the MNIST dataset
		n_bins (int, optional): number of bins in the histogram
		save_data (bool, optional): whether save data
		verbose (bool, optional): whether to display text ouput
		save_path (str, optional): path where to save data

	return:
		RF_info (dict): dictionary of data array relative to receptive field properties of the neurons
	"""

	if verbose: print "computing RF classes..."
	n_runs = np.size(W,0)
	n_neurons = np.size(W,2)

	RFproba = np.zeros((n_runs,n_neurons,n_bins))
	RFclass = np.zeros((n_runs,n_bins))
	RFselec = np.zeros((n_runs,n_bins))
	for r in range(n_runs):
		if verbose: print 'run: ' + str(r+1)
		mostActiv = np.argmax(ex.propagate_layerwise(images, W[r]),1)
		for n in range(n_neurons):
			RFproba[int(r),n,:] = np.histogram(labels[mostActiv==n], bins=n_bins, range=(-0.5,9.5))[0]
			RFproba[int(r),n,:]/= np.sum(RFproba[int(r),n,:])+1e-20 #+1e-20 to avoid divide zero error
		RFclass[r,:], _ = np.histogram(np.argmax(RFproba[r],1), bins=n_bins, range=(-0.5,9.5))
		for c in range(n_bins):
			RFselec[r,c] = np.mean(np.max(RFproba[r],1)[np.argmax(RFproba[r],1)==c])

	RFclass_mean = np.mean(RFclass, 0)
	RFclass_ste = np.std(RFclass, 0)/np.sqrt(np.size(RFclass,0))

	RF_info = {'RFproba':RFproba, 'RFclass_all':RFclass, 'RFclass_mean':RFclass_mean, 'RFclass_ste':RFclass_ste, 'RFselec':RFselec}

	return RF_info

def plot_RF_info(net, save_path, curve_method='no_softmax', slope_binned=False):
	""" plot RF properties """
	
	if net.protocol=='digit':
		fig = plot_hist(net.RF_info['RFclass_mean'][net.classes], net.classes, h_err=net.RF_info['RFclass_ste'][net.classes])
		plt.savefig(os.path.join(save_path, net.name+'_RFhist.pdf'))
		plt.close(fig)

	elif net.protocol=='gabor':
		RF_info = hist_gabor(net.name, net.hid_W_naive, net.hid_W_trained, net.t, net.images_params, True, True, save_path=save_path, curve_method=curve_method)
		_ = gr.slope_difference(RF_info['slopes_naive']['all_dist_from_target'], RF_info['slopes_naive']['all_slope_at_target'], RF_info['slopes']['all_dist_from_target'], RF_info['slopes']['all_slope_at_target'], net.name, plot=True, save_path=save_path, slope_binned=slope_binned)
		# bin_edge = np.arange(-90,91,2.5)[::2]
		# bin_mid = np.arange(-90,91,2.5)[1::2]
		# bin_edge = np.arange(-90,91,10)[::2]
		# bin_mid = np.arange(-90,91,10)[1::2]
		bin_edge = np.arange(-90,91,5)[1::2]
		bin_mid = np.arange(-90,91,5)[::2]
		bin_num = len(bin_edge)-1

		n_runs = np.size(RF_info['pref_ori'],0)
		h_all = np.zeros((n_runs, bin_num))
		for r in range(n_runs):
			h_all[r, :] = np.histogram(RF_info['pref_ori'][r,:], bin_edge)[0] #, range=(0.,180.)
		h_mean = np.mean(h_all,0)
		h_ste = np.std(h_all,0)/np.sqrt(n_runs)

		fig = plot_hist(h_mean, map(str,bin_mid), h_err=h_ste)
		plt.savefig(os.path.join(save_path, net.name+'_RFhist.pdf'))
		plt.close(fig)

def hist_gabor(name, hid_W_naive, hid_W_trained, t, images_params, save_data, verbose, save_path='', curve_method='basic'):
	""" Computes the distribution of orientation preference of neurons in the network. """
	
	#compute RFs info for the naive network
	curves_naive, pref_ori_naive = gr.tuning_curves(hid_W_naive, t, images_params, name, curve_method=curve_method, plot=False, save_path=save_path)#no_softmax
	slopes_naive = gr.slopes(hid_W_naive, curves_naive, pref_ori_naive, t, images_params['target_ori'], name, plot=False, save_path=save_path)

	#compute RFs info for the trained network
	curves, pref_ori = gr.tuning_curves(hid_W_trained, t, images_params, name, curve_method=curve_method, plot=save_data, save_path=save_path)
	slopes = gr.slopes(hid_W_trained, curves, pref_ori, t, images_params['target_ori'], name, plot=False, save_path=save_path)

	RFproba = gabor_RFproba(hid_W_trained, pref_ori)
	
	RF_info = {'RFproba':RFproba, 'curves':curves, 'pref_ori':pref_ori, 'pref_ori_naive':pref_ori_naive, 'slopes':slopes, 'slopes_naive':slopes_naive}
	
	return RF_info

def gabor_RFproba(W, pref_ori):
	""" computes to which orientation class each stimulus corresponds to """
	RFproba = np.zeros((np.size(W,0), np.size(W,2), 2), dtype=int)
	
	n_runs = np.size(pref_ori,0)
	for r in range(n_runs):
		RFproba[r,:,:][pref_ori[r,:]<=0] = [1,0]
		RFproba[r,:,:][pref_ori[r,:]>0] = [0,1]

	return RFproba

def selectivity(W, RFproba, images, labels, classes):
	"""
	computes the selectivity of a neuron, using the already computed RFproba. This RFproba must have been computed using hist()

	Args:
		W (numpy array) : weights from input to hidden neurons
		RFproba (numpy array) : 
	"""
	acti = ex.propagate_layerwise(images, W, SM=False)
	n_neurons = np.size(acti,1)
	n_classes = len(classes)
	best_neuron = np.argmax(acti, 1)
	RFclass = np.argmax(RFproba,1)
	select_neuron = np.zeros(n_neurons)
	select_class = np.zeros(n_classes)
	for n in range(n_neurons):
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

def print_save_CM(perf_dict, name, classes, verbose, save_data, save_path):
	""" print and save performance measures """

	CM_avg = perf_dict['CM_avg']
	perf_all = perf_dict['perf_all']
	perf_avg = perf_dict['perf_avg']
	perf_ste = perf_dict['perf_ste']

	if verbose or save_data:
		perf_print = ''
		perf_print += '\naverage confusion matrix:' + '\n'
		c_str = ''
		for c in classes: c_str += str(c).rjust(6)
		perf_print += c_str + '\n'
		perf_print += '-'*(len(c_str)+3) + '\n'
		perf_print += str(np.round(CM_avg,2)) + '\n'
		perf_print += '\naverage correct classification:' + '\n'
		perf_print += str(np.round(100*perf_avg,2)) + ' +/- ' + str(np.round(100*perf_ste,2)) + ' %' + '\n'
		if len(perf_all)>1:
			perf_print += '\nof which best performance is:' + '\n'
			perf_print += str(np.round(100*(np.max(perf_all)),2)) + '%' + ' (run ' + str(np.argmax(perf_all)) + ')' + '\n'
			perf_print += 'and worse performance is:' + '\n'
			perf_print += str(np.round(100*(np.min(perf_all)),2)) + '%' + ' (run ' + str(np.argmin(perf_all)) + ')' + '\n'

		print perf_print

	if save_data:
		perf_file = open(os.path.join(save_path, name+ '_perf.txt'), 'w')
		perf_file.write(perf_print)
		perf_file.close()

		fig = plot_CM(CM_avg, classes)
		plt.savefig(os.path.join(save_path, name+ '_avgCM.pdf'))
		plt.close(fig)


def plot_all_RF(name, W, RFproba, target=None, W_act=None, sort=None, not_same=None, verbose=True, save_path=''):
	""" Plot the RFs of neurons """
	if verbose: print "\nploting RFs..."

	if sort=='tSNE':
		if np.mod(np.sqrt(np.size(W['000'],1)),1)!=0:
			print '!! number of neurons not square; using class sorting for display !!'
			sort='class'

	n_runs = np.size(W,0)
	for r in range(n_runs):
		W_sort = np.copy(W[r])
		if verbose: print 'run: ' + str(r+1)
		if sort=='class': #sort weights according to their class
			RFclass = np.argmax(RFproba[r],1)
			sort_idx = np.array([x for (y,x) in sorted(zip(RFclass, np.arange(len(RFclass))), key=lambda pair: pair[0])])
			W_sort = W[r][:,sort_idx] 
			RFproba[r] = np.array([x for (y,x) in sorted(zip(RFclass, RFproba[r]), key=lambda pair: pair[0])])
		elif sort=='tSNE':
			W_sort = tSNE_sort(W_sort)
		target_pass=None
		if target:
			T_idx = np.argwhere(np.argmax(RFproba[r],1)==target[r])
			target_pass = np.zeros((np.size(W_sort,0),1,1))
			target_pass[T_idx,:,:]=1.0
		W_act_pass=None
		if W_act is not None:
			W_act_pass = W_act[r]
		if not_same:
			notsame_pass = not_same[r]
		else: 
			notsame_pass = np.array([])

		fig = plot_single_RF(W_sort, target=target_pass, W_act=W_act_pass, not_same=notsame_pass)
		if not os.path.exists(os.path.join(save_path, 'RFs')):
			os.makedirs(os.path.join(save_path, 'RFs'))
		plt.savefig(os.path.join(save_path, 'RFs', name+ '_' + str(r).zfill(3)+'.png'))
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

def plot_single_RF(W, target=None, W_act=None, cmap='binary', not_same=np.array([]), vmin=None, vmax=None):
	""" plots of the weights, with superimposed colouring for target digit and out layer weights """
	
	#plot parameters
	n_hid_neurons = np.size(W,1)
	v = int(np.sqrt(n_hid_neurons))
	h = int(np.ceil(float(n_hid_neurons)/v))

	#create a transparent colormap
	cmap_trans = plt.get_cmap(cmap) 
	cmap_trans._init()
	alphas = np.linspace(0., 1., cmap_trans.N+3)
	cmap_trans._lut[:,-1] = alphas

	im_size = int(np.sqrt(np.size(W,0)))

	#plot figure
	fig, ax = plt.subplots(figsize=(h,v))
	for i in range(np.size(W,1)):
		plt.subplot(v,h,i+1)
		if target is not None and target[i]!=0:
			plt.imshow(target[i], cmap=cmap, vmin=0., vmax=3, extent=(0,im_size,0,im_size))
		plt.imshow(np.reshape(W[:,i], (im_size,im_size)), interpolation='nearest', cmap=cmap_trans, extent=(0,im_size,0,im_size), vmin=vmin, vmax=vmax)
		if W_act is not None:
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
	n_classes = len(classes)
	cmap_trans = mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['white','white'],256) 
	cmap_trans._init()
	alphas = np.linspace(1.0, 0, cmap_trans.N+3)
	cmap_trans._lut[:,-1] = alphas

	#creates the background color matrix
	colorMatrix = np.ones_like(confusMatrix)
	np.fill_diagonal(colorMatrix, -1.0)

	#plot the matrix and number values
	sH = 1.0+0.5*n_classes
	sV = 0.9+0.5*n_classes
	fig, ax = plt.subplots(figsize=(sH,sV))
	ax.imshow(colorMatrix, interpolation='nearest', cmap='RdYlGn_r', vmin=-1.2, vmax=1.2)
	ax.imshow(confusMatrix, interpolation='nearest', cmap=cmap_trans, vmin=-0.0, vmax=1)
	for i in range(n_classes):
		for j in range(n_classes):
			perc = int(np.round(confusMatrix[i,j],2)*100)
			ax.annotate(perc, xy=(0, 0),  xycoords='data', xytext=(j, i), textcoords='data', size=15, ha='center', va='center')

	#plot parameters
	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xticks(np.arange(n_classes))
	ax.set_yticks(np.arange(n_classes))
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
	y_max = np.max(h)
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
	ax.set_ylabel('avg. neuron count', fontsize=18)
	plt.tight_layout()

	return fig

def plot_perf_progress(name, perf_train, perf_test, dopa_start, epi_start=0, save_path=''):
	"""
	plots the progression of the error rate over training episodes

	Args:
		name (str): name of the network, used for saving purposes
		perf_train (numpy array): training performance at each episode of each run
		perf_test (numpy array): testing performance at each episode of each run
		epi_start (int, optional): episode at which to start the plot (used epi_start=n_epi_crit to plot only after statistical pre-training). Default: 0 
	"""
	plot_train = True
	plot_mean = True

	fig, ax = plt.subplots()
	plt.gca().set_color_cycle(cm.Paired(i) for i in np.linspace(0,0.9,10))
	alpha_all = 0.35 if plot_mean else 1.0
	n_epi_plot = len(perf_train[0, epi_start:])

	n_runs = np.size(perf_train, 0)
	for r in range(n_runs):
		#vline for start of dopa
		if epi_start < dopa_start and n_epi_plot > dopa_start:
			plt.axvline(x=dopa_start, ymin=0, ymax=100, lw=2.5, ls='--', c='k', alpha=0.25)
		
		X = np.arange( len(perf_train[r, epi_start:]) )+1
		
		#progression curves for all runs
		if plot_train: 
			ax.plot(X, perf_train[r, epi_start:]*100, lw=3, ls=':', alpha=alpha_all)
		if perf_test is not None:
			ax.plot(X, perf_test[r, epi_start:]*100, lw=3, ls='-', alpha=alpha_all)

	#mean progression curves
	if plot_mean:
		if plot_train: 
			ax.plot(X, np.mean(perf_train[:, epi_start:], 0)*100, lw=3, ls=':', alpha=1.0, c='k')
		if perf_test is not None:
				ax.plot(X, np.mean(perf_test[:, epi_start:], 0)*100, lw=3, ls='-', alpha=1.0, c='k')

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.set_xticks(np.arange(1, len(perf_train[0,epi_start:])+1))
	# ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_ylim([70.,100.])
	if n_epi_plot>0: ax.set_xticks(np.arange(0, n_epi_plot+1, np.clip(n_epi_plot/10, 1, 10000)))
	ax.set_xlabel('training episodes', fontsize=18)
	ax.set_ylabel('% correct', fontsize=18)
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, name+'_progress.pdf'))
	plt.close(fig)	


def perf_all_ori(net, save_path=''):
	""" assess performance of network at various orientations """
	# ori_to_tests = np.array([-60., -20., -10., -5., 0., +5., +10., +20., +60.])
	ori_to_tests = np.array([-60., -30., -20., -10., -5., -2., -1., 0., +1., +2., +5., +10., +20., +30., +60.])
	# ori_to_tests = np.array([-20., -5., 0.])
	verbose = True

	if save_path=='':
		save_path=os.path.join('output', net.name)

	#save original variables:
	n_runs_ori 			= net.n_runs
	n_epi_crit_ori		= net.n_epi_crit
	n_epi_dopa_ori 		= net.n_epi_dopa
	lr_hid_ori 			= net.lr_hid
	lr_out_ori			= net.lr_out
	init_file_ori		= net.init_file
	test_each_epi_ori	= net.test_each_epi
	verbose_ori			= net.verbose
	target_ori_ori 		= net.images_params['target_ori']
	hid_W_ori 			= np.copy(net.hid_W)
	out_W_ori 			= np.copy(net.out_W)
	hid_W_trained_ori 	= np.copy(net.hid_W_trained)
	out_W_trained_ori 	= np.copy(net.out_W_trained)
	#set training variables:
	net.n_runs 			= 1
	net.n_epi_crit		= 1
	net.n_epi_dopa		= 0
	net.lr_hid 			= 0
	net.lr_out 			*= 100
	net.init_file 		= 'NO_INIT'
	net.test_each_epi 	= False
	net.verbose 	 	= False

	perf_at_ori = np.zeros((n_runs_ori, len(ori_to_tests)))

	#test each orientation
	for i_ori, ori in enumerate(ori_to_tests):
		if net.verbose: print "\n----------------------------------------------------------------------------------------"
		else: print
		
		#load and pre-process training and testing images
		gabor_params = net.images_params
		gabor_params['target_ori'] = target_ori_ori + ori	
		if verbose: print "test orientation: " + str(ori) + " (" + str(gabor_params['target_ori']) + ")"
		
		np.random.seed(0)
		images_dict, labels_dict, images_params = ex.load_images(	protocol 		= net.protocol,
																	A 				= net.A,
																	verbose 		= net.verbose,
																	gabor_params 	= gabor_params
																	)

		#test each run
		for i_run in range(n_runs_ori):
			#re-initialize output weights
			np.random.seed(0)
			net.out_W = (np.random.random_sample(size=(net.n_hid_neurons, net.n_out_neurons))/1000+1.0)/net.n_hid_neurons

			#load weights for each runs
			net.hid_W = hid_W_trained_ori[i_run,:,:]

			#train net
			net.train(images_dict, labels_dict, images_params)

			#test net
			perf_at_ori[i_run, i_ori] = net.test(images_dict, labels_dict, during_training=True)

			if verbose: print "performance: " + str(perf_at_ori[i_run, i_ori])

	print
	print perf_at_ori

	#reset changed variables
	net.n_runs 			= n_runs_ori
	net.n_epi_crit		= n_epi_crit_ori
	net.n_epi_dopa 		= n_epi_dopa_ori
	net.lr_hid 			= lr_hid_ori
	net.lr_out			= lr_out_ori
	net.init_file		= init_file_ori
	net.test_each_epi	= test_each_epi_ori
	net.verbose			= verbose_ori
	net.hid_W 			= np.copy(hid_W_ori)
	net.out_W 			= np.copy(out_W_ori)
	net.hid_W_trained 	= np.copy(hid_W_trained_ori)
	net.out_W_trained  	= np.copy(out_W_trained_ori)
	net.images_params['target_ori'] = target_ori_ori 

	""" plot of performance for different orientations """
	fig, ax = plt.subplots()
				
	ax.scatter(np.ones_like(perf_at_ori)*ori_to_tests, perf_at_ori, alpha=0.2)
	ax.errorbar(ori_to_tests, np.mean(perf_at_ori,0), yerr=np.std(perf_at_ori,0)/np.sqrt(n_runs_ori), marker='o', ms=5, ls='-', lw=1.5, c='r', mfc='r', mec='r', ecolor='r', mew=1)

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_xlabel('angle from target (deg)', fontsize=18)
	ax.set_ylabel('% correct', fontsize=18)
	ax.set_xlim([-90,90])
	ax.tick_params(axis='both', which='major', direction='out', labelsize=16)
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, 'perf_all_ori.pdf'))












