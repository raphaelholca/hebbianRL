"""
Author: Raphael Holca-Lamarre
Date: 26/05/2015

This function creates a convolutional hebbian neural network
"""

import numpy as np
import matplotlib.pyplot as plt
import recon as rc
import sys
import pickle
import numba
import time
import support.accel as accel
import support.mnist as mnist
import support.external as ex
from progressbar import ProgressBar

ex = reload(ex)
rc = reload(rc)

""" support functions """

def add_padding(images, pad_size, pad_value=0.):
	""" 
	Adds padding of value 1 around images and reshapes images vector to 3D (1D all images + 2D spatial dimensions).

	Args:
		images (2D numpy array): images to add the padding to; size = (images_num x nPixels_1D)
		pad_size (int): amount of padding to add on each 4 sides of the 2D image
		pad_value (float, optional): value of the padding to add

	returns:
		3D numpy array; size = (images_num x nPixels_2D x nPixels_2D)
	"""

	images_num = np.size(images, 0)
	sqrtPixels = int(np.sqrt(np.size(images,1)))
	images = np.reshape(images, (images_num, sqrtPixels, sqrtPixels))
	images_padded = np.ones((images_num, sqrtPixels+pad_size*2, sqrtPixels+pad_size*2))*pad_value

	images_padded[:, pad_size-1:sqrtPixels+pad_size-1, pad_size-1:sqrtPixels+pad_size-1] = images

	return images_padded

@numba.njit
def get_conv_input(image, conv_input, conv_side):
	"""
	Gets the input to the convolving weight matrix

	Args:
		image (2D numpy array): image to get the input from; size = (images_side x images_side)
		conv_input (2D numpy array): empty input array to be filled; size = (conv_neuronNum x conv_side**2)
		conv_side (int): size of the convolutional filter

	returns:
		2D numpy array; size = (conv_neuronNum x conv_side**2)
	"""
	images_side = image.shape[0]

	im=0
	for i in range(images_side-conv_side+1):
		for j in range(images_side-conv_side+1):
			select = image[i:i+conv_side, j:j+conv_side]
			km=0
			for k in range(conv_side):
				for l in range(conv_side):
					conv_input[im,km] = select[k,l]
					km+=1
			im+=1
	return conv_input	

def subsampling(conv_activ, conv_mapSide, conv_mapNum, subS_mapSide):
	FM = np.reshape(conv_activ, (conv_mapSide, conv_mapSide, conv_mapNum))
	SSM = np.zeros((subS_mapSide, subS_mapSide, conv_mapNum))
	ite = np.arange(0, conv_mapSide, 2)
	SSM = subsampling_numba(FM, SSM, ite)
	SSM = ex.softmax( np.reshape(SSM, (subS_mapSide**2, conv_mapNum) ), t=1. )
	subS_activ = np.reshape(SSM, (-1))[np.newaxis,:]

	return subS_activ

@numba.njit
def subsampling_numba(FM, SSM, ite):
	"""
	Subsamples the convolutional feature maps

	Args:
		FM (3D numpy array): feature maps; size = (cMaps_side x cMaps_side x nFeatureMaps)
		SSM (3D numpy array): empty subsampled feature maps to be filled; size = (cMaps_side/2 x cMaps_side/2 x nFeatureMaps)
		ite (1D numpy array): iterator used over FM (contains np.arange(0, cMaps_side, 2))

	returns:
		3D numpy array; subsampled feature maps; size = (cMaps_side/2 x cMaps_side/2 nFeatureMaps)
	"""
	for f in range(FM.shape[2]):
		for im in range(ite.shape[0]):
			i=ite[im]
			for jm in range(ite.shape[0]):
				j=ite[jm]
				select = FM[i:i+2,j:j+2,f]
				tmp_sum=0
				for k in range(2):
					for l in range(2):
						tmp_sum += select[k,l]
				SSM[im,jm,f] = tmp_sum
	return SSM

def shuffle_images(images, labels):
	"""
	Shuffles images and labels
	"""

	rdnIndex = range(images.shape[0])
	np.random.shuffle(rdnIndex)
	rndImages = images[rdnIndex,:,:]
	rndLabels = labels[rdnIndex]

	return rndImages, rndLabels

def load_images(classes, dataset, imPath, pad_size):
	"""
	Loads and pads images
	"""

	print "importing and pre-processing " + dataset + " images..."
	images, labels = mnist.read_images_from_mnist(classes = classes, dataset = dataset, path = imPath)
	images, labels = ex.evenLabels(images, labels, classes)
	# images = ex.normalize_numba(images, 1080)
	images = add_padding(images, pad_size=pad_size, pad_value=0)
	images += 1e-5 #to avoid division by zero error when the convolving filter takes as input a patch of images that is filled with 0s

	return images, labels

def propagate(image, conv_W, feedF_W, class_W, A, t, size_params, noise=False, noise_distrib=50):
	"""
	propagates a single image through the network and return its classification
	"""
	#get params from size_params dict
	conv_neuronNum = size_params['conv_neuronNum']
	conv_filterSide = size_params['conv_filterSide']
	conv_mapSide = size_params['conv_mapSide']
	conv_mapNum = size_params['conv_mapNum']
	subS_mapSide = size_params['subS_mapSide']

	#get input to the convolutional filter
	conv_input = np.zeros((conv_neuronNum, conv_filterSide**2))
	conv_input = get_conv_input(image, conv_input, conv_filterSide)
	conv_input = ex.normalize_numba(conv_input, A)

	#activate convolutional feature maps
	conv_activ = ex.propL1(conv_input, conv_W, SM=True, t=t)

	#subsample feature maps
	subS_activ = subsampling(conv_activ, conv_mapSide, conv_mapNum, subS_mapSide)

	#activate feedforward layer
	feedF_activ = ex.propL1(subS_activ, feedF_W, SM=False)

	#add noise
	if noise:
		feedF_activ_noise = feedF_activ + np.random.uniform(0, noise_distrib, np.shape(feedF_activ))
		feedF_activ_noise = ex.softmax(feedF_activ_noise, t=t)
		class_activ_noise = ex.propL1(feedF_activ_noise, class_W, SM=True, t=0.001)
	
	feedF_activ = ex.softmax(feedF_activ, t=t)

	#activate classification layer
	class_activ = ex.propL1(feedF_activ, class_W, SM=True, t=0.001)

	if not noise:
		return np.argmax(class_activ), conv_input, conv_activ, subS_activ, feedF_activ, class_activ
	else:
		return np.argmax(class_activ), conv_input, conv_activ, subS_activ, feedF_activ_noise, class_activ, class_activ_noise


""" define parameters """
# classes 	= np.array([ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ], dtype=int)
# classes 	= np.array([0,1], dtype=int)
classes 	= np.array([4,7,9], dtype=int)
nClasses = len(classes)

runName 			= 't_1'
nEpi 				= 50
nCrit 				= 5
DA 					= True
A 					= 200.
lr 					= 1e-5
dataset 			= 'test'
nBatch 				= 196  #196 112 49
conv_mapNum			= 12
conv_filterSide		= 5
feedF_neuronNum 	= 49
class_neuronNum 	= nClasses
seed 				= 970#np.random.randint(1000) #970

np.random.seed(seed)
print '\n' + 'run name: ' + runName + ' -- seed: ' + str(seed) + '\n'

""" load and pre-process images """
pad_size = (conv_filterSide-1)/2
imPath = '/Users/raphaelholca/Documents/data-sets/MNIST'
images, labels = load_images(classes, dataset, imPath, pad_size)
dataset_test='test' if dataset=='train' else 'train'
images_test, labels_test = load_images(classes, dataset_test, imPath, pad_size)

##reduce size of training set
# images, labels = shuffle_images(images, labels)
# images = images[:10000, :,:]
# labels = labels[:10000]
##

##duplicate size of test dataset to test effect of size
# images_ =np.zeros((images.shape[0]*6, images.shape[1], images.shape[2]))
# labels_ =np.zeros(labels.shape[0]*6, dtype=int)
# for i in range(6):
# 	images_[images.shape[0]*i:images.shape[0]*(i+1), :, :] = images
# 	labels_[labels.shape[0]*i:labels.shape[0]*(i+1)] = labels
# images = images_
# labels = labels_
##

""" initialize variables """
images_num = np.size(images, 0)
images_side = np.size(images,2)
conv_neuronNum = (images_side-conv_filterSide+1)**2
conv_mapSide = int(np.sqrt(conv_neuronNum))
subS_mapSide = conv_mapSide/2
size_params = {'conv_neuronNum':conv_neuronNum, 'conv_filterSide':conv_filterSide, 'conv_mapSide':conv_mapSide, 'conv_mapNum':conv_mapNum, 'subS_mapSide':subS_mapSide}

""" initialize weights """
if False: #load pre-trained weights from file
	# f = open('w_corners', 'r')
	# conv_W = pickle.load(f)
	# f.close()

	f = open('weights', 'r')
	weights = pickle.load(f)
	f.close()

	conv_W = weights['conv_W']
	feedF_W = weights['feedF_W']
	class_W = weights['class_W']
else: #random initialization
	conv_W = np.random.random_sample(size=(conv_filterSide**2, conv_mapNum)) + A/(conv_filterSide**2) + 2.5 ##0.5
	feedF_W = np.random.random_sample(size=((subS_mapSide**2)*conv_mapNum, feedF_neuronNum))/1000 + float(subS_mapSide**2)/((subS_mapSide**2)*conv_mapNum) + 0.6
	class_W = (np.random.random_sample(size=(feedF_neuronNum, class_neuronNum))/1000+1.0)/feedF_neuronNum

""" training network """
print "training network..."
imcount = 0
for e in range(nEpi):
	print '\ntrain epi: ' + str(e+1) + '/' + str(nEpi)
	rndImages, rndLabels = shuffle_images(images, labels)

	correct_train = 0.
	last_neuron_class = np.zeros((feedF_neuronNum, class_neuronNum))
	dopa_save = []

	pbar_epi = ProgressBar()
	for i in pbar_epi(range(rndImages.shape[0])):
		imcount+=1
		
		if e<nCrit or not DA:
			classif, conv_input, conv_activ, subS_activ, feedF_activ, class_activ = propagate(rndImages[i,:,:], conv_W, feedF_W, class_W, A, 0.01, size_params, noise=False)

		if e>=nCrit and DA: #DOPA
			classif, conv_input, conv_activ, subS_activ, feedF_activ, class_activ, class_activ_noise = propagate(rndImages[i,:,:], conv_W, feedF_W, class_W, A, 0.01, size_params, noise=True)

			reward = ex.compute_reward(ex.label2idx(classes, [rndLabels[i]]), np.argmax(class_activ_noise))

			# dopa = ex.compute_dopa([np.argmax(class_activ)], [np.argmax(class_activ_noise)], reward, dHigh=7.0, dMid=0.01, dNeut=-0.0, dLow=-2.0) #parameters from old network
			dopa = ex.compute_dopa([np.argmax(class_activ)], [np.argmax(class_activ_noise)], reward, dHigh=2.0, dMid=0.00, dNeut=-0.02, dLow=-0.5) #OK paramters for feedforward layer alone
			# dopa = ex.compute_dopa([np.argmax(class_activ)], [np.argmax(class_activ_noise)], reward, dHigh=2.0, dMid=1.00, dNeut=-0.02, dLow=-0.5) #testing parameters for convolutional layer

			dopa_save.append(dopa[0])
		else: dopa = None

		last_neuron_class[np.argmax(feedF_activ), np.argwhere(rndLabels[i]==classes)] += 1 ##will create a problem when noise is added

		# learn weights...
		#...of the convolutional matrices
		if imcount<60000:
			for b in range(conv_neuronNum/nBatch):
				dW_conv = ex.learningStep(conv_input[b*nBatch:(b+1)*nBatch-1, :], conv_activ[b*nBatch:(b+1)*nBatch-1, :], conv_W, lr=lr*0.1, disinhib=dopa)
				conv_W += dW_conv
				conv_W = np.clip(conv_W, 1e-10, np.inf)

		#...of the feedforward layer
		dW_FF = ex.learningStep(subS_activ, feedF_activ, feedF_W, lr=lr*3600, disinhib=dopa)
		feedF_W += dW_FF
		feedF_W = np.clip(feedF_W, 1e-10, np.inf)

		#...of the classification layer
		dW_class = ex.learningStep(feedF_activ, class_activ, class_W, lr=0.005)
		if np.argmax(class_activ) == ex.label2idx(classes, [rndLabels[i]]):
			dW_class *= 0.75
			correct_train+=1
		else:
			dW_class *= -0.5
		class_W += dW_class
		class_W = np.clip(class_W, 1e-10, np.inf)

		if np.isnan(class_W).any(): import pdb; pdb.set_trace()

	print 'train error: ' + str(np.round((1.-correct_train/rndImages.shape[0])*100,2)) +'%'

	step = labels_test.shape[0]/1005
	rndImages_test, rndLabels_test = shuffle_images(images_test, labels_test)
	images_test_short = rndImages_test[::step]
	labels_test_short = rndLabels_test[::step]
	correct_test = 0.
	for i in range(images_test_short.shape[0]):
		classif = propagate(images_test_short[i,:,:], conv_W, feedF_W, class_W, A, 0.01, size_params)[0]
		if classif == ex.label2idx(classes, [labels_test_short[i]]): correct_test += 1.
	print 'approx. test error: ' + str(np.round((1.-correct_test/images_test_short.shape[0])*100,2)) +'%'

	correct_Wout = np.sum(np.argmax(last_neuron_class,1)==np.argmax(class_W,1))
	print 'correct W_out assignment: ' + str(correct_Wout) + '/' + str(feedF_neuronNum)

""" test network """
print '\ntesting network...'
step = images_test.shape[0]/8920
images_test=images_test[::step,:,:]
labels_test=labels_test[::step]

correct = 0.
pbar_epi = ProgressBar()
for i in pbar_epi(range(images_test.shape[0])):
	classif = propagate(images_test[i,:,:], conv_W, feedF_W, class_W, A, 0.01, size_params)[0]
	if classif == ex.label2idx(classes, [labels_test[i]]): correct += 1.
print 'test error: ' + str(np.round((1.-correct/images_test.shape[0])*100,2)) +'%'


""" plot convolutional filter """
# nRows = int(np.sqrt(conv_mapNum))
# nCols = np.ceil(conv_mapNum/float(nRows))
# fig = plt.figure(figsize=(nCols,nRows))
# for f in range(conv_mapNum):
# 	plt.subplot(nRows, nCols, f)
# 	plt.imshow(np.reshape(conv_W[:,f], (conv_filterSide,conv_filterSide)), interpolation='nearest', cmap='Greys', vmin=np.min(conv_W), vmax=np.max(conv_W))
# 	# plt.imshow(np.reshape(conv_W[:,f], (conv_filterSide,conv_filterSide)), interpolation='nearest', cmap='Greys', vmin=np.min(conv_W[:,f]), vmax=np.max(conv_W[:,f]))
# 	plt.xticks([])
# 	plt.yticks([])
# fig.patch.set_facecolor('white')
# plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)
# plt.show(block=False)

""" plot output neuron RF reconstruction """
# nRows = int(np.sqrt(feedF_neuronNum))
# nCols = feedF_neuronNum/nRows
# fig = plt.figure(figsize=(nCols,nRows))
# for n in range(feedF_neuronNum):
# 	plt.subplot(nRows, nCols, n)
# 	W = np.reshape(feedF_W[:,n], (subS_mapSide, subS_mapSide, conv_mapNum))
# 	rc.recon(conv_W, W, display_all=False)
# 	plt.xticks([])
# 	plt.yticks([])
# fig.patch.set_facecolor('white')
# plt.subplots_adjust(left=0., right=1., bottom=0., top=1., wspace=0., hspace=0.)
# plt.show(block=False)
	



































