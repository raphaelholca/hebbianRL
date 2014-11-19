""" plottting functions for the hebbian network and neural classifier """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

""" initialize color maps """
NUM_COLORS = 9
my_blues = [plt.get_cmap('YlGnBu')(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

def plotRF(W, target=None):
	""" plot weights """
	#plot parameters
	nHidNeurons = np.size(W,1)
	nDimStates = 784
	v = int(np.sqrt(nHidNeurons))
	h = int(np.ceil(float(nHidNeurons)/v))

	#create a transparent colormap
	cmap_trans = plt.get_cmap('binary')#mpl.colors.LinearSegmentedColormap.from_list('my_cmap',['w','k'],256) 
	cmap_trans._init()
	if type(target)!=type(None):
		alphas = np.linspace(0., 1., cmap_trans.N+3)
		cmap_trans._lut[:,-1] = alphas

	#plot figure
	fig, ax = plt.subplots(figsize=(h,v))
	for i in range(np.size(W,1)):
		plt.subplot(v,h,i+1)
		if type(target)!=type(None):
			plt.imshow(target[i], cmap='Blues', vmin=0., vmax=3, extent=(0,nDimStates,nDimStates,0))
		plt.imshow(np.reshape(W[:nDimStates,i], (28,28)), interpolation='nearest', cmap=cmap_trans)
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
	# fig.subplots_adjust(top=1, right=1)

	return fig

def plotHist(h, bins, h_err=None):
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





