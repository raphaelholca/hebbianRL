""" plottting functions for the hebbian network and neural classifier """

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

def plotRF(W, e=''):
	""" plot weights """
	#plot parameters
	nHidNeurons = np.size(W,1)
	nDimStates = 784
	v = int(np.sqrt(nHidNeurons))
	h = int(np.ceil(float(nHidNeurons)/v))
	
	#plot figure
	fig, ax = plt.subplots()
	for i in range(np.size(W,1)):
		plt.subplot(v,h,i+1)
		plt.imshow(np.reshape(W[:nDimStates,i], (28,28)), interpolation='nearest')
		plt.xticks([])
		plt.yticks([])

	#plot parameters
	fig.patch.set_facecolor('white')
	plt.suptitle('episode ' + e)

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
	fig, ax = plt.subplots(figsize=(sV,sH))
	ax.imshow(colorMatrix, interpolation='nearest', cmap='RdYlGn_r', vmin=-1.2, vmax=1.2)
	ax.imshow(confusMatrix, interpolation='nearest', cmap=cmap_trans, vmin=-0.0, vmax=1)
	for i in range(nClasses):
		for j in range(nClasses):
			perc = int(confusMatrix[i,j]*100)
			ax.annotate(perc, xy=(0, 0),  xycoords='data', xytext=(j, i), textcoords='data', size=15, ha='center', va='center')

	#plot parameters
	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_visible(False)
	ax.spines['bottom'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xticks(np.arange(nClasses))
	ax.set_yticks(np.arange(nClasses))
	ax.set_xticklabels(classes, fontsize=20)
	ax.set_yticklabels(classes, fontsize=20)
	ax.xaxis.set_ticks_position('none')
	ax.yaxis.set_ticks_position('none')
	ax.set_xlabel('classification', fontsize=20)
	ax.set_ylabel('label', fontsize=20)
	plt.tight_layout()
	# fig.subplots_adjust(top=1, right=1)

	return fig

def plotHist(h, bins, h_err=None):
	fig, ax = plt.subplots(figsize=(4,4))
	# ax.hist(h,bins)
	ax.bar(bins, h, width=bins[1]-bins[0])

	fig.patch.set_facecolor('white')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_xticks(bins+(bins[1]-bins[0])/2.)
	ax.set_xticklabels(bins)
	ax.tick_params(axis='both', which='major', direction='out', labelsize=17)
	ax.xaxis.set_ticks_position('bottom')
	ax.yaxis.set_ticks_position('left')
	ax.set_xlabel('digit class', fontsize=20)
	ax.set_ylabel('neuron count', fontsize=20)
	plt.tight_layout()

	return fig





