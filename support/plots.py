""" support plottting functions """

import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import external as ex
import matplotlib.pyplot as plt
ex = reload(ex)

""" initialize color maps """
NUM_COLORS = 9
my_blues = [plt.get_cmap('YlGnBu')(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
my_reds = [plt.get_cmap('YlOrRd')(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]
cm_pastel = [plt.get_cmap('Paired')(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

def plotRF(W, target=None, W_act=None, cmap='Greys', notsame=np.array([])):
	""" 
	plots of the weights, with superimposed colouring for target digit and L2 weights 
	"""
	#plot parameters
	nHidNeurons = np.size(W,1)
	v = int(np.sqrt(nHidNeurons))
	h = int(np.ceil(float(nHidNeurons)/v))
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
			if i in notsame:
				plt.imshow([[0]], cmap='RdYlBu', vmin=0., vmax=3, extent=(im_size,im_size+2,0,im_size))
			plt.imshow(W_act[i,:][:,np.newaxis], interpolation='nearest', cmap='binary', extent=(im_size,im_size+2,0,im_size))
			plt.imshow([[0.]], interpolation='nearest', cmap='binary', alpha=0, extent=(0,im_size+2,0,im_size))
		plt.xticks([])
		plt.yticks([])

	#plot parameters
	fig.patch.set_facecolor('white')
	plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)

	return fig

def plotCM(confusMatrix, classes):
	""" 
	plots the confusion matrix, with color on the diagonal, and with the alphas indicating the magnitude of the
	error 
	"""

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

def plotHist(h, bins, h_err=None):
	"""
	plots the histogram of receptive field class distribution
	"""
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

def perf_progress(perf, kwargs, nEpi=None):
	"""
	plots the progression of the error rate over training episodes
	"""
	runName = kwargs['runName']
	nEpiCrit = kwargs['nEpiCrit']
	nEpiDopa = kwargs['nEpiDopa']

	fig, ax = plt.subplots()
	plt.gca().set_color_cycle(cm.Paired(i) for i in np.linspace(0,0.9,10))

	for r in perf.keys():
		if kwargs['param_xplr'] == 'neural_net':
			X = np.arange( len(perf[r][nEpiCrit:]) )
		else:
			X = np.arange( len(perf[r][nEpiCrit:]) )+1
		ax.plot(X, perf[r][nEpiCrit:]*100, lw=3)

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.set_ylim([50,100])
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	if kwargs['param_xplr'] == 'neural_net':
		ax.set_xlabel('training episodes', fontsize=18)
	else:
		ax.set_xlabel('episodes after dopa', fontsize=18)
	ax.set_ylabel('% correct', fontsize=18)
	plt.tight_layout()

	plt.savefig('output/' + runName + '/' + runName+ '_progress.pdf')
	plt.close(fig)	


def plot_noise_proba(W_in, images, kwargs):
	"""
	plots the probability that noise injection changes the most active hidden neurons.
	"""

	runName = kwargs['runName']
	t_hid = kwargs['t_hid']
	noise_std = kwargs['noise_std']
	nHidNeurons = np.size(W_in,1)

	hidNeurons = ex.propL1(images, W_in, SM=False)
	hidNeurons = np.sort(hidNeurons,1)[:, ::-1]
	hidNeurons_noise = hidNeurons + np.random.normal(0, noise_std, np.shape(hidNeurons))

	hidNeurons = ex.softmax(hidNeurons, t=t_hid)
	hidNeurons_noise = ex.softmax(hidNeurons_noise, t=t_hid)

	proba_argmax = np.histogram(np.argmax(hidNeurons_noise,1), bins=nHidNeurons, range=(0,nHidNeurons))[0]
	proba_argmax = proba_argmax / float(np.sum(proba_argmax))

	hidNeurons_noise = np.sort(hidNeurons_noise,1)[:, ::-1]

	mean_activ = np.mean(hidNeurons,0)
	mean_activ_noise = np.mean(hidNeurons_noise,0)

	""" plot of probability of most activated neuron after noise injection """
	fig, ax = plt.subplots()

	ax.bar(np.arange(nHidNeurons), proba_argmax, width=0.8, color=my_blues[6], edgecolor=my_blues[6])

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.set_xlim([0,nHidNeurons])
	ax.set_ylim([0,1])
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_xlabel('neuron sorted by activ. before noise', fontsize=18)
	ax.set_ylabel('prob. most activ. neuron after noise', fontsize=18)
	plt.tight_layout()

	plt.savefig('output/' + runName + '/' + runName+ '_activ_prob.pdf')
	plt.close(fig)

	""" plot of distribution of activation in network after noise injection """
	fig, ax = plt.subplots()

	ax.bar(np.arange(nHidNeurons), mean_activ, width=0.4, color=my_blues[6], edgecolor=my_blues[6])
	ax.bar(np.arange(nHidNeurons)+.5, mean_activ_noise, width=0.4, color=my_reds[6], edgecolor=my_reds[6])

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.set_xlim([0,nHidNeurons])
	ax.set_ylim([0,1])
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_xlabel('neuron sorted by activation', fontsize=18)
	ax.set_ylabel('activation', fontsize=18)
	plt.tight_layout()

	plt.savefig('output/' + runName + '/' + runName+ '_activation.pdf')
	plt.close(fig)

def regressor_prediction(all_DA, epi, kwargs, perf=None, nn_input=None):
	"""
	plot performance prediction by neural regressor

	Args:
		all_DA (numpy array): array of saved performance predictions, size: ( n_DA x n_RPE )
	"""

	if epi!=0 or True:
		vmax = 1.0
		vmin = 0.0
	else:
		vmax = np.max(all_DA)
		vmin = np.min(all_DA)

	plt.figure()
	plt.imshow(all_DA[:,:], vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest', cmap='autumn')
	plt.colorbar(orientation='vertical')
	plt.plot(np.argmax(all_DA[:,:],0), c='k', lw=3)
	if nn_input is not None:
		plt.scatter((nn_input[:,0]+1)*50, (nn_input[:,1]+6)*10, marker='x', color='w', s=15)

	plt.xticks([0,25,50,75,100], ['-1', '-0.5', '0', '+0.5', '+1'])
	plt.yticks([0,30,60,90,120], ['-6', '-3', '0', '+3', '+6'])

	plt.xlim(0,100)
	plt.ylim(0,120)

	if perf is not None:
		plt.title('epi ' + str(epi) + ' -- ' + str(np.round(perf,3)*100))
	else:
		plt.title('epi ' + str(epi))
	plt.xlabel('RPE')
	plt.ylabel('DA')

	plt.savefig('output/' + kwargs['runName'] + '/regressor_prediction/epi_' + str(epi) + '.png')
	plt.close()

























