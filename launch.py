import numpy as np
import pickle
import os
import struct
import recon as rc
import support.hebbianRL as rl
import support.external as ex
import support.mnist as mnist

rc = reload(rc)
rl = reload(rl)
ex = reload(ex)

def transform(W_bp, images, name):
	"""transform images in the representation learned by the first conv layer of a back-prop conv network"""
	print 'convolving ' + name + ' images...'
	if os.path.exists('images_subs_'+name):
		f=open('images_subs_'+name, 'r')
		images_subs=pickle.load(f)
		f.close()
	
	else:
		print 'computing new conlvolution...'
		from scipy.signal import convolve2d
		from skimage.measure import block_reduce
		images_square = np.reshape(images, (np.size(images,0),28,28))
		images_conv = np.zeros((np.size(images,0), np.size(W_bp,0), 32, 32))
		images_subs = np.zeros((np.size(images,0), np.size(W_bp,0), 32/2, 32/2))
		for f in range(np.size(W_bp,0)):
			for i in range(np.size(images,0)):
				images_conv[i,f,:,:] = convolve2d(images_square[i,:,:], W_bp[f,0,:,:], mode='full', boundary='fill', fillvalue=0)
				images_subs[i,f,:,:] = block_reduce(images_conv[i,f,:,:], block_size=(2, 2), func=np.max)

		images_subs = np.reshape(images_subs, (np.size(images_subs,0), np.size(images_subs,1), (32/2)**2))
		images_subs = np.reshape(images_subs, (np.size(images_subs,0), np.size(images_subs,1)*((32/2)**2)))

		f=open('images_subs_'+name, 'w')
		pickle.dump(images_subs, f)
		f.close()

	return images_subs


""" 
experimental variables

classes (int) 	: class of the MNIST dataset to use to train the network
rActions (str)	: for each class of MNIST, the action that is rewarded. Capital letters indicates a class that is paired with ACh release.
"""

classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
rActions 	= np.array(['a','b','c','d','e','f','g','h','i','j'], dtype='|S1')

# classes 	= np.array([ 0,  1 , 2 , 3 ], dtype=int)
# rActions 	= np.array(['a','b','c','d'], dtype='|S1')

# classes 	= np.array([ 4 , 7 , 9 ], dtype=int)
# rActions 	= np.array(['a','b','c'], dtype='|S1')

# classes 	= np.array([ 4 , 9 ], dtype=int)
# rActions 	= np.array(['a','b'], dtype='|S1')

""" parameters """
kwargs = {
'nRun' 			: 1				,# number of runs
'nEpiCrit'		: 2 				,# number of 'critical period' episodes in each run (episodes when reward is not required for learning)		#50
'nEpiDopa'		: 0				,# number of 'adult' episodes in each run (episodes when reward is not required for learning)				#20
't_hid'			: 1. 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for hidden layer
't_act'			: 0.1 				,# temperature of the softmax function (t<<1: strong competition; t>=1: weak competition) for action layer
'A' 			: 1.2				,# input normalization constant. Will be used as: (input size)*A; for images: 784*1.2=940.8
'runName' 		: 't-print'		,# name of the folder where to save results
'dataset'		: 'test'			,# dataset to use; possible values: 'test': MNIST test, 'train': MNIST train, 'grating': orientation discrimination
'nHidNeurons'	: 49				,# number of hidden neurons
'lr'			: 0.005 			,# learning rate during 'critica period' (pre-training, nEpiCrit)
'aHigh' 		: 0.0 				,# learning rate increase for relevance signal (high ACh) outside of critical period
'aPairing'		: 1.0 				,# strength of ACh signal for pairing protocol
'dHigh' 		: 7.0 				,# learning rate increase for unexpected reward
'dMid' 			: 0.01 				,# learning rate increase for correct reward prediction
'dNeut' 		: -0.2				,# learning rate increase for correct no reward prediction
'dLow' 			: -2.0				,# learning rate increase for incorrect reward predictio
'nBatch' 		: 20 				,# mini-batch size
'classifier'	: 'actionNeurons'	,# which classifier to use for performance assessment. Possible values are: 'actionNeurons', 'SVM', 'neuronClass'
'SVM'			: False				,# whether to use an SVM or the number of stimuli that activate a neuron to determine the class of the neuron
'bestAction' 	: False				,# whether to take predicted best action (True) or take random actions (False)
'createOutput'	: True				,# whether to create plots, save data, etc. (set to False when using pypet)
'showPlots'		: False				,# whether to display plots
'show_W_act'	: True				,# whether to display W_act weights on the weight plots
'sort' 			: None				,# sorting methods for weights when displaying. Legal value: None, 'class', 'tSNE'
'target'		: None 				,# target digit (to be used to color plots). Use None if not desired
'seed' 			: 168#np.random.randint(1000) 	# seed of the random number generator
}

kwargs['classes'] 	= classes
kwargs['rActions'] 	= rActions

# decrease = 0.5
# kwargs['dHigh'] *= decrease
# kwargs['dMid'] *= decrease
# kwargs['dNeut'] *= decrease
# kwargs['dLow'] *= decrease

""" load and pre-process images """
ex.checkClassifier(kwargs['classifier'])
print 'seed: ' + str(kwargs['seed']) + '\n'
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'

print 'loading train data...'
images, labels = mnist.read_images_from_mnist(classes = kwargs['classes'], dataset = kwargs['dataset'], path = imPath)
images, labels = ex.evenLabels(images, labels, classes)

print 'loading test data...'
test_dataset='test' if kwargs['dataset']=='train' else 'train'
images_test, labels_test = mnist.read_images_from_mnist(classes = kwargs['classes'], dataset = test_dataset, path = imPath)
images_test, labels_test = ex.evenLabels(images_test, labels_test, classes)

# f = open('weights_bp', 'r')
# W_bp = pickle.load(f)
# f.close()
# W_bp+=1. ##to ensure no negative weights
# images = transform(W_bp, images, 'train')
# images_test = transform(W_bp, images_test, 'test')

images_test, labels_test = ex.shuffle([images_test, labels_test])
images_test = ex.normalize(images_test, kwargs['A']*np.size(images_test,1))
allCMs, allPerf, perc_correct_W_act, W_in, W_act, RFproba = rl.RLnetwork(images=images, labels=labels, images_test=images_test, labels_test=labels_test, kwargs=kwargs, **kwargs)

# import matplotlib.pyplot as plt
# import support.plots as pl
# pl = reload(pl)
# for i in range(np.size(W_bp,0)):
# 	pl.plotRF(W_in[256*i : 256*(i+1), :])
# plt.show(block=False)



